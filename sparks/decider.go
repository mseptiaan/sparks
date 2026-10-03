package sparks

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

type Config struct {
	Rotations   int     // option orderings averaged per Choice (bias fix); default 3
	Temperature float64 // calibration: probs ∝ p^(1/T); default 1 (fit it with sparks-eval)
	// MinMargin enables adaptive rotations: when rotation 0 alone separates
	// the top two options by at least this calibrated-probability margin,
	// the rest are skipped. 0 disables. Saves backend calls on easy cases.
	MinMargin float64
	// AgreeMargin enables the two-rotation agreement gate (0 disables).
	// Rotations 0 and 1 are scored first; when they pick the same option
	// and their average top-2 margin clears it, the rest are skipped.
	// Unlike the single-rotation MinMargin gate, a pure position bias
	// cannot pass: it picks different options per rotation and forces
	// the full fan-out. Measure on your cases before enabling.
	AgreeMargin float64
	System      string
}

type Decider struct {
	B   Backend
	Cfg Config
	// Skipped counts rotations saved by the MinMargin/AgreeMargin gates.
	Skipped atomic.Int64
}

func NewDecider(b Backend, cfg Config) *Decider {
	if cfg.Rotations <= 0 {
		cfg.Rotations = 3
	}
	if cfg.Temperature <= 0 {
		cfg.Temperature = 1
	}
	if cfg.System == "" {
		cfg.System = defaultSystem
	}
	return &Decider{B: b, Cfg: cfg}
}

// callBackend scores one user text against the system prompt.
func (d *Decider) callBackend(ctx context.Context, user string) (map[string]float64, Usage, error) {
	return d.B.TopLogProbs(ctx, d.Cfg.System, user)
}

func (d *Decider) Ask(ctx context.Context, state string, q Question) (Result, error) {
	if err := q.Validate(); err != nil {
		return Result{}, err
	}
	switch q.Type {
	case KindNoul:
		return d.Noul(ctx, state, q.Question)
	case KindChoice:
		return d.Choice(ctx, state, q.Question, q.Options)
	case KindScore:
		if len(q.Levels) > 0 {
			return d.ScoreLevels(ctx, state, q.Question, q.Levels)
		}
		return d.Score(ctx, state, q.Question, q.Min, q.Max)
	}
	return Result{}, fmt.Errorf("unknown question type %q", q.Type)
}

// Decide answers many questions against one state, concurrently.
func (d *Decider) Decide(ctx context.Context, state string, qs map[string]Question) (map[string]Result, error) {
	out := make(map[string]Result, len(qs))
	var mu sync.Mutex
	var wg sync.WaitGroup
	var firstErr error
	for name, q := range qs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := d.Ask(ctx, state, q)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if firstErr == nil {
					firstErr = fmt.Errorf("question %q: %w", name, err)
				}
				return
			}
			out[name] = r
		}()
	}
	wg.Wait()
	return out, firstErr
}

// Noul answers a yes/no question. It is a 2-option Choice scored in both
// orders, so "A"/"B" position bias cancels out.
func (d *Decider) Noul(ctx context.Context, state, question string) (Result, error) {
	probs, mass, usage, err := d.choiceProbs(ctx, state, question, []string{"Yes", "No"}, 2)
	if err != nil {
		return Result{}, err
	}
	p := map[string]float64{"yes": probs[0], "no": probs[1]}
	best := "no"
	if p["yes"] >= p["no"] {
		best = "yes"
	}
	return Result{Type: KindNoul, Best: best, Probs: p, Confidence: p[best], LabelMass: mass, Usage: usage}, nil
}

func (d *Decider) Choice(ctx context.Context, state, question string, options []string) (Result, error) {
	if n := len(options); n < 2 || n > 26 {
		return Result{}, fmt.Errorf("choice needs 2..26 options, got %d", n)
	}
	probs, mass, usage, err := d.choiceProbs(ctx, state, question, options, d.Cfg.Rotations)
	if err != nil {
		return Result{}, err
	}
	p := make(map[string]float64, len(options))
	bi := 0
	for i, o := range options {
		p[o] = probs[i]
		if probs[i] > probs[bi] {
			bi = i
		}
	}
	return Result{Type: KindChoice, Best: options[bi], Probs: p, Confidence: probs[bi], LabelMass: mass, Usage: usage}, nil
}

func (d *Decider) Score(ctx context.Context, state, question string, min, max int) (Result, error) {
	if min < 0 || max > 9 || min >= max {
		return Result{}, fmt.Errorf("score range must satisfy 0 <= min < max <= 9, got %d..%d", min, max)
	}
	labels := make([]string, 0, max-min+1)
	for v := min; v <= max; v++ {
		labels = append(labels, strconv.Itoa(v))
	}
	prompt := scoreUser(state, question, min, max)
	lps, usage, err := d.callBackend(ctx, prompt)
	if err != nil {
		return Result{}, err
	}
	vals, mass := labelLogProbs(lps, labels)
	probs := softmaxT(vals, d.Cfg.Temperature)
	p := make(map[string]float64, len(labels))
	var ev float64
	bi := 0
	for i, l := range labels {
		p[l] = probs[i]
		ev += float64(min+i) * probs[i]
		if probs[i] > probs[bi] {
			bi = i
		}
	}
	return Result{Type: KindScore, Best: labels[bi], Probs: p, Confidence: probs[bi], Value: &ev, LabelMass: mass, Usage: usage}, nil
}

// choiceProbs scores `rotations` cyclic shifts of the option order and
// averages the (temperature-scaled) distributions, mapped back to the
// original option order.
func (d *Decider) choiceProbs(ctx context.Context, state, question string, options []string, rotations int) ([]float64, float64, Usage, error) {
	n := len(options)
	if rotations > n {
		rotations = n
	}
	if rotations < 1 {
		rotations = 1
	}
	labels := letters(n)
	type res struct {
		p     []float64 // in original option order
		mass  float64
		usage Usage
		err   error
	}
	scoreOne := func(k int) res {
		shift := k * n / rotations
		order := make([]int, n) // position -> original index
		ordered := make([]string, n)
		for pos := range n {
			order[pos] = (pos + shift) % n
			ordered[pos] = options[order[pos]]
		}
		lps, usage, err := d.callBackend(ctx, choiceUser(state, question, ordered))
		if err != nil {
			return res{err: err}
		}
		vals, mass := labelLogProbs(lps, labels)
		pp := softmaxT(vals, d.Cfg.Temperature)
		orig := make([]float64, n)
		for pos, idx := range order {
			orig[idx] = pp[pos]
		}
		return res{p: orig, mass: mass, usage: usage}
	}
	// Adaptive: score rotation 0 alone; when its top-2 margin already
	// clears MinMargin, the remaining rotations would only confirm it.
	// Otherwise rotation 0 is reused as the first fan-out result.
	results := make([]res, rotations)
	ready := make([]bool, rotations)
	if d.Cfg.MinMargin > 0 && rotations > 1 {
		if first := scoreOne(0); first.err == nil {
			if margin(first.p) >= d.Cfg.MinMargin {
				d.Skipped.Add(int64(rotations - 1))
				return first.p, first.mass, first.usage, nil
			}
			results[0] = first
			ready[0] = true
		}
	}
	// Agreement gate: score rotations 0 and 1 up front. A pure position
	// bias picks different options in each and forces the full fan-out;
	// only questions both rotations settle the same way skip the rest.
	if d.Cfg.AgreeMargin > 0 && rotations > 2 && !ready[0] && !ready[1] {
		var first [2]res
		var gwg sync.WaitGroup
		for k := 0; k < 2; k++ {
			gwg.Add(1)
			go func() {
				defer gwg.Done()
				first[k] = scoreOne(k)
			}()
		}
		gwg.Wait()
		if first[0].err == nil && first[1].err == nil {
			results[0], results[1] = first[0], first[1]
			ready[0], ready[1] = true, true
			avg2 := make([]float64, n)
			for i := range avg2 {
				avg2[i] = (first[0].p[i] + first[1].p[i]) / 2
			}
			if argmax(first[0].p) == argmax(first[1].p) && margin(avg2) >= d.Cfg.AgreeMargin {
				d.Skipped.Add(int64(rotations - 2))
				usage := first[0].usage
				usage.PromptTokens += first[1].usage.PromptTokens
				usage.CompletionTokens += first[1].usage.CompletionTokens
				usage.CachedTokens += first[1].usage.CachedTokens
				return avg2, (first[0].mass + first[1].mass) / 2, usage, nil
			}
		}
	}
	var wg sync.WaitGroup
	for k := range rotations {
		if ready[k] {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[k] = scoreOne(k)
		}()
	}
	wg.Wait()
	avg := make([]float64, n)
	var mass float64
	var usage Usage
	for _, r := range results {
		if r.err != nil {
			return nil, 0, Usage{}, r.err
		}
		for i, v := range r.p {
			avg[i] += v / float64(rotations)
		}
		mass += r.mass / float64(rotations)
		usage.PromptTokens += r.usage.PromptTokens
		usage.CompletionTokens += r.usage.CompletionTokens
		usage.CachedTokens += r.usage.CachedTokens
	}
	return avg, mass, usage, nil
}

// argmax is the index of the largest probability (first wins ties).
func argmax(p []float64) int {
	bi := 0
	for i, v := range p {
		if v > p[bi] {
			bi = i
		}
	}
	return bi
}

// margin is the gap between the top two probabilities (0 for <2 options).
func margin(p []float64) float64 {
	top1, top2 := 0.0, 0.0
	for _, v := range p {
		if v > top1 {
			top2, top1 = top1, v
		} else if v > top2 {
			top2 = v
		}
	}
	return top1 - top2
}

const logFloor = -30.0 // used when a label is outside the backend's top-K

// labelLogProbs collects, per label, the log of the summed probability of all
// tokens that equal the label after trimming spaces (" A" and "A" both count).
func labelLogProbs(lps map[string]float64, labels []string) (vals []float64, mass float64) {
	sum := make(map[string]float64, len(labels))
	for tok, lp := range lps {
		sum[strings.TrimSpace(tok)] += math.Exp(lp)
	}
	vals = make([]float64, len(labels))
	for i, l := range labels {
		if p := sum[l]; p > 0 {
			vals[i] = math.Log(p)
			mass += p
		} else {
			vals[i] = logFloor
		}
	}
	return vals, mass
}

// softmaxT is softmax(vals / T): temperature scaling for calibration.
func softmaxT(vals []float64, t float64) []float64 {
	m := math.Inf(-1)
	for _, v := range vals {
		m = max(m, v/t)
	}
	out := make([]float64, len(vals))
	var z float64
	for i, v := range vals {
		out[i] = math.Exp(v/t - m)
		z += out[i]
	}
	for i := range out {
		out[i] /= z
	}
	return out
}
