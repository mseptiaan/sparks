// sparks-eval scores a model on a JSONL set of typed questions and searches, per
// question name, for the calibration temperature that minimises NLL.
//
// Each line of the cases file is one state with several named questions:
//
//	{"state":"...",
//	 "questions":{"pattern":{"type":"choice","instructions":"...","criteria":{...}}, ...},
//	 "expected":{"pattern":"merchant_risk","review_priority":"2","signals_support_alert":"true"}}
//
// expected values are criteria keys: the choice key, the score level
// ("1".."n" = criteria index + 1), or the noul key ("true"/"false", mapped to
// the model's yes/no labels before scoring).
//
// The criteria form builds the model prompt (instructions + criteria text) and
// derives choice options / score range from criteria. The older typed form
// with explicit question/options/min/max still works:
// {"state":"...", "question":{"type":"noul","question":"..."}, "expected":"yes"}
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mseptiaan/sparks/sparks"
)

type rawCase struct {
	State     string                  `json:"state"`
	Question  *evalQuestion           `json:"question"`
	Questions map[string]evalQuestion `json:"questions"`
	Expected  json.RawMessage         `json:"expected"`
}

// evalQuestion accepts the typed form (explicit question/options/min/max) and
// the criteria form (instructions + criteria) emitted by eval/gen_cases.py.
// The criteria form carries the threshold text the model must apply, so the
// prompt is built from it: instructions plus the criteria descriptions.
type evalQuestion struct {
	Type         sparks.Kind     `json:"type"`
	Question     string          `json:"question"`
	Options      []string        `json:"options,omitempty"`
	Min          int             `json:"min,omitempty"`
	Max          int             `json:"max,omitempty"`
	Instructions string          `json:"instructions,omitempty"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

func (q evalQuestion) build(name string) (sparks.Question, error) {
	if q.Question != "" {
		return sparks.Question{Type: q.Type, Question: q.Question, Options: q.Options, Min: q.Min, Max: q.Max}, nil
	}
	// Criteria form builds the exact production prompt via CompileQuestion,
	// so eval scores what sparks serves.
	c, err := sparks.CompileQuestion(name, sparks.WireQuestion{
		Type: q.Type, Instructions: q.Instructions, Criteria: q.Criteria,
	})
	if err != nil {
		return sparks.Question{}, err
	}
	return c.Q, nil
}

// mapExpected translates file labels to model labels: noul true/false to
// yes/no, and score digits ("1".."n") to legend indices ("0".."n-1") when
// the question was built from a criteria list.
func mapExpected(q sparks.Question, e string) (string, error) {
	if q.Type == sparks.KindNoul {
		return noulExpected(q, e), nil
	}
	if q.Type == sparks.KindScore && len(q.Levels) > 0 {
		d, err := strconv.Atoi(e)
		if err != nil || d < 1 || d > len(q.Levels) {
			return "", fmt.Errorf("score expected %q (want level 1..%d)", e, len(q.Levels))
		}
		return strconv.Itoa(d - 1), nil
	}
	return e, nil
}

// noulExpected maps the criteria key ("true"/"false") to the model's yes/no
// labels. Values already in yes/no form pass through untouched.
func noulExpected(q sparks.Question, e string) string {
	if q.Type == sparks.KindNoul {
		switch e {
		case "true":
			return "yes"
		case "false":
			return "no"
		}
	}
	return e
}

type item struct {
	Name     string
	State    string
	Q        sparks.Question
	Expected string
}

type metrics struct {
	N                                        int
	Acc, MeanConf, NLL, Brier, ECE, MeanMass float64
}

// progress reports completed work with ETA on stderr so stdout stays clean
// for piped results. Call add from workers; it throttles renders.
// On a TTY it draws a single-line \r bar; on a pipe/file it prints
// newline-terminated lines at each 10% step so captures still show life.
type progress struct {
	total int
	done  atomic.Int64
	start time.Time
	label atomic.Value // string: current phase, e.g. "T=1.00"
	tty   bool

	mu      sync.Mutex
	last    time.Time
	needNL  bool // TTY bar mid-line, needs \n before other stderr output
	nextPct int  // next percent step to print in non-TTY mode
}

func newProgress(total int) *progress {
	p := &progress{total: total, start: time.Now(), nextPct: 10}
	p.last = p.start
	if fi, err := os.Stderr.Stat(); err == nil {
		p.tty = fi.Mode()&os.ModeCharDevice != 0
	}
	return p
}

func (p *progress) add() {
	n := p.done.Add(1)
	p.mu.Lock()
	defer p.mu.Unlock()
	now := time.Now()
	frac := float64(n) / float64(max(p.total, 1))
	if p.tty {
		// Always show the first tick (fast tiny runs) and the final one.
		if n > 1 && n < int64(p.total) && now.Sub(p.last) < 200*time.Millisecond {
			return
		}
	} else {
		pct := int(frac * 100)
		if n < int64(p.total) && pct < p.nextPct && now.Sub(p.last) < 5*time.Second {
			return
		}
		for p.nextPct <= pct {
			p.nextPct += 10
		}
	}
	p.last = now
	p.renderLocked(n, now, frac)
}

// breakLine ends a mid-line TTY bar so the next stderr write (e.g. a fatal
// error) starts on a fresh line. No-op when nothing is mid-line.
func (p *progress) breakLine() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.needNL {
		fmt.Fprint(os.Stderr, "\n")
		p.needNL = false
	}
}

func (p *progress) renderLocked(n int64, now time.Time, frac float64) {
	const width = 30
	filled := int(frac * float64(width))
	bar := strings.Repeat("#", filled) + strings.Repeat("-", width-filled)
	elapsed := now.Sub(p.start)
	rate := float64(n) / max(elapsed.Seconds(), 1e-9)
	eta := time.Duration(float64(p.total-int(n)) / max(rate, 1e-9) * float64(time.Second))
	doneAt := now.Add(eta).Format("15:04:05")
	label, _ := p.label.Load().(string)
	if label != "" {
		label = " " + label
	}
	if !p.tty {
		fmt.Fprintf(os.Stderr, "progress: %d/%d (%d%%)%s | %5.1f/s | elapsed %s | ETA %s (~%s)\n",
			n, p.total, int(frac*100), label, rate, shortDur(elapsed), shortDur(eta), doneAt)
		return
	}
	fmt.Fprintf(os.Stderr, "\r[%s] %d/%d (%5.1f%%)%s | %5.1f/s | elapsed %s | ETA %s (~%s)",
		bar, n, p.total, frac*100, label, rate, shortDur(elapsed), shortDur(eta), doneAt)
	if n >= int64(p.total) {
		fmt.Fprint(os.Stderr, "\n")
		p.needNL = false
	} else {
		p.needNL = true
	}
}

func shortDur(d time.Duration) string {
	d = d.Round(time.Second)
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second
	if h > 0 {
		return fmt.Sprintf("%dh%02dm%02ds", h, m, s)
	}
	if m > 0 {
		return fmt.Sprintf("%dm%02ds", m, s)
	}
	return fmt.Sprintf("%ds", s)
}

func main() {
	cases := flag.String("cases", "eval/cases.jsonl", "JSONL file of test cases")
	url := flag.String("backend-url", "http://127.0.0.1:8080", "chat server base URL (must expose POST /v1/chat/completions with `logprobs`)")
	model := flag.String("model", "", "model name sent to the backend")
	apiKey := flag.String("api-key", "", "optional bearer token")
	rot := flag.Int("rotations", 3, "option orderings averaged per choice")
	workers := flag.Int("workers", 4, "parallel questions in flight (lower on metered APIs)")
	agreeMargin := flag.Float64("agree-margin", 0.3, "two-rotation agreement gate: skip remaining rotations when rotations 0 and 1 agree above this margin (0 disables)")
	skip := flag.String("skip", "", "comma-separated question names to exclude from model scoring (code-computed via compose rules or unanswerable, e.g. review_priority,signals_support_alert)")
	reasoning := flag.Bool("reasoning", false, "enable provider thinking (high effort); default false disables thinking so the first token is the answer label")
	showProg := flag.Bool("progress", true, "show progress bar with ETA on stderr")
	debug := flag.Bool("debug", false, "log backend request JSON and raw response to stderr")
	flag.Parse()

	items, err := load(*cases)
	if err != nil {
		log.Fatal(err)
	}

	items = skipItems(items, parseSkip(*skip))
	if len(items) == 0 {
		log.Fatal("all questions skipped")
	}
	// Retries sit under the cache with quota-friendly waits (metered APIs
	// answer bursts with 429 rather than queueing).
	var backend sparks.Backend = &sparks.RetryingBackend{
		Inner: &sparks.ChatCompat{BaseURL: *url, Model: *model, APIKey: *apiKey,
			Reasoning: *reasoning, Debug: *debug},
		MaxRetries: 6, BaseWait: 2 * time.Second, MaxWait: 45 * time.Second,
	}
	backend = sparks.NewCachedBackend(backend)
	names := groupNames(items)
	fmt.Printf("%d questions over %d question types, rotations=%d\n", len(items), len(names), *rot)

	temps := []float64{0.5, 0.75, 1, 1.5, 2, 3, 5, 7, 10}
	type best struct {
		T, NLL float64
		Res    []sparks.Result
	}
	bests := map[string]*best{}
	for _, n := range names {
		bests[n] = &best{NLL: math.Inf(1)}
	}
	var prog *progress
	if *showProg && !*debug {
		prog = newProgress(len(items) * len(temps))
	}
	var totalUsage sparks.Usage // cache hits report zero, so this sums real consumption
	var totalSkipped int64
	for _, temp := range temps {
		d := sparks.NewDecider(backend, sparks.Config{Rotations: *rot, AgreeMargin: *agreeMargin, Temperature: temp})
		if prog != nil {
			prog.label.Store(fmt.Sprintf("T=%.2f", temp))
		}
		res, err := run(context.Background(), d, items, *workers, prog)
		totalSkipped += d.Skipped.Load()
		for _, r := range res {
			totalUsage.PromptTokens += r.Usage.PromptTokens
			totalUsage.CompletionTokens += r.Usage.CompletionTokens
			totalUsage.CachedTokens += r.Usage.CachedTokens
		}
		if prog != nil {
			prog.breakLine()
		}
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("\n== T = %.2f ==\n", temp)
		fmt.Printf(
			"%-24s %4s %7s %9s %7s %7s %7s %9s\n",
			"question",
			"n",
			"acc",
			"meanConf",
			"NLL",
			"Brier",
			"ECE",
			"labelMass",
		)
		for _, n := range names {
			m := score(items, res, n)
			fmt.Printf("%-24s %4d %7.3f %9.3f %7.3f %7.3f %7.3f %9.3f\n",
				n, m.N, m.Acc, m.MeanConf, m.NLL, m.Brier, m.ECE, m.MeanMass)
			if m.NLL < bests[n].NLL {
				*bests[n] = best{T: temp, NLL: m.NLL, Res: res}
			}
		}
	}

	fmt.Println("\n== Best temperature per question (by NLL) ==")
	for _, n := range names {
		edge := bests[n].T == temps[len(temps)-1]
		fmt.Printf("%-24s T=%.2f%s\n", n, bests[n].T, map[bool]string{true: "  <-- at grid edge, NLL may still fall: widen", false: ""}[edge])
	}
	fmt.Println("\n== Recall per expected class at each question's best T ==")
	for _, n := range names {
		fmt.Printf("%s (T=%.2f)\n", n, bests[n].T)
		for _, line := range recall(items, bests[n].Res, n) {
			fmt.Println("  " + line)
		}
	}
	fmt.Println("\nNote: with few cases the temperature estimate is noisy; use 100+ per question type.")
	fmt.Printf("Tokens consumed: %d prompt (%d cached) + %d completion; rotations skipped by gates: %d\n", totalUsage.PromptTokens, totalUsage.CachedTokens, totalUsage.CompletionTokens, totalSkipped)
}

func load(path string) ([]item, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []item
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for line := 1; sc.Scan(); line++ {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var c rawCase
		if err := json.Unmarshal(sc.Bytes(), &c); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		if len(c.Questions) > 0 {
			var exp map[string]string
			if err := json.Unmarshal(c.Expected, &exp); err != nil {
				return nil, fmt.Errorf("%s:%d: expected must be an object when questions is set: %w", path, line, err)
			}
			keys := make([]string, 0, len(c.Questions))
			for k := range c.Questions {
				keys = append(keys, k)
			}
			slices.Sort(keys)
			for _, k := range keys {
				e, ok := exp[k]
				if !ok {
					return nil, fmt.Errorf("%s:%d: no expected value for question %q", path, line, k)
				}
				qq, err := c.Questions[k].build(k)
				if err != nil {
					return nil, fmt.Errorf("%s:%d: %w", path, line, err)
				}
				expVal, err := mapExpected(qq, e)
				if err != nil {
					return nil, fmt.Errorf("%s:%d: %w", path, line, err)
				}
				out = append(out, item{Name: k, State: c.State, Q: qq, Expected: expVal})
			}
			continue
		}
		if c.Question == nil {
			return nil, fmt.Errorf("%s:%d: neither question nor questions present", path, line)
		}
		var e string
		if err := json.Unmarshal(c.Expected, &e); err != nil {
			return nil, fmt.Errorf("%s:%d: expected must be a string: %w", path, line, err)
		}
		qq, err := c.Question.build("default")
		if err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		expVal, err := mapExpected(qq, e)
		if err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		out = append(out, item{Name: "default", State: c.State, Q: qq, Expected: expVal})
	}
	return out, sc.Err()
}

// parseSkip splits the -skip flag into a name set.
func parseSkip(s string) map[string]bool {
	out := map[string]bool{}
	for _, n := range strings.Split(s, ",") {
		if n = strings.TrimSpace(n); n != "" {
			out[n] = true
		}
	}
	return out
}

// skipItems drops code-computed or unanswerable questions from model scoring:
// review_priority comes from compose rules, signals_support_alert has neither
// model signal nor code rule, so both only burn backend calls here.
func skipItems(items []item, skip map[string]bool) []item {
	if len(skip) == 0 {
		return items
	}
	out := make([]item, 0, len(items))
	for _, it := range items {
		if !skip[it.Name] {
			out = append(out, it)
		}
	}
	return out
}

func groupNames(items []item) []string {
	seen := map[string]bool{}
	var names []string
	for _, it := range items {
		if !seen[it.Name] {
			seen[it.Name] = true
			names = append(names, it.Name)
		}
	}
	slices.Sort(names)
	return names
}

func run(ctx context.Context, d *sparks.Decider, items []item, workers int, prog *progress) ([]sparks.Result, error) {
	workers = max(workers, 1)
	res := make([]sparks.Result, len(items))
	var (
		wg       sync.WaitGroup
		mu       sync.Mutex
		firstErr error
	)
	sem := make(chan struct{}, workers)
	for i := range items {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			defer func() {
				if prog != nil {
					prog.add()
				}
			}()
			cctx, cancel := context.WithTimeout(ctx, 90*time.Second)
			defer cancel()
			r, err := d.Ask(cctx, items[i].State, items[i].Q)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("%s: %w", items[i].Name, err)
				}
				mu.Unlock()
				return
			}
			res[i] = r
		}()
	}
	wg.Wait()
	return res, firstErr
}

func score(items []item, res []sparks.Result, name string) metrics {
	var m metrics
	type bin struct{ n, correct, conf float64 }
	var bins [10]bin
	for i, it := range items {
		if it.Name != name {
			continue
		}
		r := res[i]
		hit := 0.0
		if r.Best == it.Expected {
			hit = 1
		}
		m.N++
		m.Acc += hit
		m.MeanConf += r.Confidence
		m.MeanMass += r.LabelMass
		m.NLL += -math.Log(math.Max(r.Probs[it.Expected], 1e-9))
		for k, p := range r.Probs { // multi-class Brier
			y := 0.0
			if k == it.Expected {
				y = 1
			}
			m.Brier += (p - y) * (p - y)
		}
		bi := int(math.Min(r.Confidence*10, 9))
		bins[bi].n++
		bins[bi].correct += hit
		bins[bi].conf += r.Confidence
	}
	n := float64(m.N)
	if n == 0 {
		return m
	}
	for _, b := range bins {
		if b.n > 0 {
			m.ECE += b.n / n * math.Abs(b.correct/b.n-b.conf/b.n)
		}
	}
	m.Acc /= n
	m.MeanConf /= n
	m.MeanMass /= n
	m.NLL /= n
	m.Brier /= n
	return m
}

func recall(items []item, res []sparks.Result, name string) []string {
	tot, ok := map[string]int{}, map[string]int{}
	for i, it := range items {
		if it.Name != name {
			continue
		}
		tot[it.Expected]++
		if res[i].Best == it.Expected {
			ok[it.Expected]++
		}
	}
	keys := make([]string, 0, len(tot))
	for k := range tot {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, fmt.Sprintf("%-28s %3d/%-3d  %.2f", k, ok[k], tot[k], float64(ok[k])/float64(tot[k])))
	}
	return out
}
