// sparks: HTTP server for typed decisions (Typesafe-compatible wire format).
//
//	POST /v1/decide
//	{"state": "...", "questions": {"refund": {"type":"noul","instructions":"..."}, ...}}
//
// Answers come back as {model, usage, answers}, with one choice/score/noul
// answer per question ID. See https://docs.typesafe.ai/primitives.
//
// Calibration (temperature) is per question name: sparks-eval finds each
// question's best T, and the server applies it via -temps. Names without an
// override use the default. Calibration is only valid for the exact
// (backend, rotations, question text) triple used during eval: reword a
// question and you must refit its T.
package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/mseptiaan/sparks/sparks"
)

type decideRequest struct {
	State     any                            `json:"state"`
	Questions map[string]sparks.WireQuestion `json:"questions"`
}

// decideResponse mirrors the Typesafe SystemOne response envelope: answers
// keyed by question ID, plus model and usage metadata.
type decideResponse struct {
	Model   string            `json:"model"`
	Usage   responseUsage     `json:"usage"`
	Answers map[string]answer `json:"answers"`
}

type responseUsage struct {
	// Real backend token counts when reported; null when the backend
	// stays silent (some local servers omit usage). CachedTokens is a
	// subset of InputTokens billed at the cached rate, if any.
	InputTokens  *int `json:"input_tokens"`
	OutputTokens *int `json:"output_tokens"`
	CachedTokens *int `json:"cached_tokens,omitempty"`
}

// answer is one Typesafe-shaped answer (choice, score, or noul),
// discriminated by Type.
type answer struct {
	Type          sparks.Kind        `json:"type"`
	Choice        *string            `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Noul          *float64           `json:"noul,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Legend        map[string]any     `json:"legend,omitempty"`
}

// toAnswer shapes an internal result as its Typesafe answer. Confidence is
// currently the best-label probability; the spread-normalized Typesafe
// confidence formulas are a separate slice.
func toAnswer(c sparks.Compiled, r sparks.Result) answer {
	switch c.Q.Type {
	case sparks.KindChoice:
		return answer{Type: sparks.KindChoice, Choice: &r.Best, Confidence: &r.Confidence, Probabilities: r.Probs}
	case sparks.KindScore:
		legend := make(map[string]any, len(c.Legend))
		for i, d := range c.Legend {
			legend[strconv.Itoa(i)] = d
		}
		return answer{Type: sparks.KindScore, Score: r.Value, Confidence: &r.Confidence, Probabilities: r.Probs, Legend: legend}
	default:
		p := r.Probs["yes"]
		return answer{Type: sparks.KindNoul, Noul: &p}
	}
}

// parseTemps parses "name=T,..." into a default temperature and per-question
// overrides. The reserved name "default" sets the fallback for question IDs
// without an override; absent, it is 1.0. Every T must be finite and > 0.
func parseTemps(s string) (float64, map[string]float64, error) {
	def := 1.0
	overrides := map[string]float64{}
	if strings.TrimSpace(s) == "" {
		return def, overrides, nil
	}
	for _, kv := range strings.Split(s, ",") {
		name, val, ok := strings.Cut(kv, "=")
		name = strings.TrimSpace(name)
		val = strings.TrimSpace(val)
		if !ok || name == "" || val == "" {
			return 0, nil, fmt.Errorf("bad entry %q (want name=T)", kv)
		}
		t, err := strconv.ParseFloat(val, 64)
		if err != nil || math.IsNaN(t) || math.IsInf(t, 0) || t <= 0 {
			return 0, nil, fmt.Errorf("bad temperature for %q: %q (want T > 0)", name, val)
		}
		if name == "default" {
			def = t
		} else {
			overrides[name] = t
		}
	}
	return def, overrides, nil
}

type server struct {
	deciders     map[float64]*sparks.Decider // one per distinct temperature, shared backend
	defTemp      float64
	overrides    map[string]float64
	modelName    string
	sem          chan struct{}
	auth         string
	timeout      time.Duration
	maxQuestions int
	reqs         atomic.Uint64
}

func newServer(b sparks.Backend, rotations int, minMargin float64, agreeMargin float64, defTemp float64, overrides map[string]float64, modelName string, maxInflight int, auth string, timeout time.Duration, maxQuestions int) *server {
	seen := map[float64]bool{defTemp: true}
	for _, t := range overrides {
		seen[t] = true
	}
	dec := make(map[float64]*sparks.Decider, len(seen))
	for t := range seen {
		dec[t] = sparks.NewDecider(b, sparks.Config{Rotations: rotations, MinMargin: minMargin, AgreeMargin: agreeMargin, Temperature: t})
	}
	return &server{
		deciders: dec, defTemp: defTemp, overrides: overrides, modelName: modelName,
		sem: make(chan struct{}, maxInflight), auth: auth,
		timeout: timeout, maxQuestions: maxQuestions,
	}
}

// tempFor resolves the calibration temperature for a question name.
func (s *server) tempFor(name string) float64 {
	if t, ok := s.overrides[name]; ok {
		return t
	}
	return s.defTemp
}

func (s *server) serveDecide(w http.ResponseWriter, r *http.Request) {
	id := s.reqs.Add(1)
	start := time.Now()
	n := 0
	status := http.StatusOK
	defer func() {
		log.Printf("req=%d decide status=%d questions=%d elapsed=%s", id, status, n, time.Since(start).Round(time.Millisecond))
	}()
	fail := func(code int, msg string) {
		status = code
		http.Error(w, msg, code)
	}

	if s.auth != "" {
		got := r.Header.Get("Authorization")
		if subtle.ConstantTimeCompare([]byte(got), []byte("Bearer "+s.auth)) != 1 {
			fail(http.StatusUnauthorized, "unauthorized")
			return
		}
	}

	var req decideRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		fail(http.StatusBadRequest, "bad json: "+err.Error())
		return
	}
	if len(req.Questions) == 0 {
		fail(http.StatusBadRequest, "no questions")
		return
	}
	if len(req.Questions) > s.maxQuestions {
		fail(http.StatusBadRequest, fmt.Sprintf("too many questions: got %d, max %d", len(req.Questions), s.maxQuestions))
		return
	}
	state, err := sparks.RenderState(req.State)
	if err != nil {
		fail(http.StatusBadRequest, err.Error())
		return
	}
	names := make([]string, 0, len(req.Questions))
	compiled := make(map[string]sparks.Compiled, len(req.Questions))
	for name, wq := range req.Questions {
		c, err := sparks.CompileQuestion(name, wq)
		if err != nil {
			fail(http.StatusBadRequest, err.Error())
			return
		}
		if err := c.Q.Validate(); err != nil {
			fail(http.StatusBadRequest, fmt.Sprintf("question %q: %v", name, err))
			return
		}
		compiled[name] = c
		names = append(names, name)
	}
	slices.Sort(names)
	n = len(names)

	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	default:
		w.Header().Set("Retry-After", "1")
		fail(http.StatusTooManyRequests, "too many requests")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.timeout)
	defer cancel()

	// Group by resolved temperature so each question is scored with the
	// calibration eval found for it.
	groups := map[float64]map[string]sparks.Question{}
	for _, name := range names {
		t := s.tempFor(name)
		if groups[t] == nil {
			groups[t] = map[string]sparks.Question{}
		}
		groups[t][name] = compiled[name].Q
	}

	out := make(map[string]answer, len(names))
	var usage sparks.Usage
	reported := false
	var mu sync.Mutex
	var wg sync.WaitGroup
	var firstErr error
	for t, qs := range groups {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := s.deciders[t].Decide(ctx, state, qs)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if firstErr == nil {
					firstErr = err
				}
				return
			}
			for k, v := range res {
				out[k] = toAnswer(compiled[k], v)
				usage.PromptTokens += v.Usage.PromptTokens
				usage.CompletionTokens += v.Usage.CompletionTokens
				usage.CachedTokens += v.Usage.CachedTokens
				if v.Usage != (sparks.Usage{}) {
					reported = true
				}
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		if ctx.Err() != nil {
			fail(http.StatusGatewayTimeout, "backend timeout")
		} else {
			fail(http.StatusBadGateway, firstErr.Error())
		}
		return
	}

	resp := decideResponse{Model: s.modelName, Answers: out}
	if reported {
		in, co, cached := usage.PromptTokens, usage.CompletionTokens, usage.CachedTokens
		resp.Usage = responseUsage{InputTokens: &in, OutputTokens: &co, CachedTokens: &cached}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// overrideFlag renders the -temps map in stable order for startup logging.
func overrideFlag(overrides map[string]float64) string {
	names := make([]string, 0, len(overrides))
	for name := range overrides {
		names = append(names, name)
	}
	slices.Sort(names)
	parts := make([]string, 0, len(names))
	for _, name := range names {
		parts = append(parts, fmt.Sprintf("%s=%.2f", name, overrides[name]))
	}
	return strings.Join(parts, ",")
}

func main() {
	addr := flag.String("addr", ":8088", "listen address")
	url := flag.String("backend-url", "http://127.0.0.1:8080", "chat server base URL (must expose POST /v1/chat/completions with `logprobs`)")
	model := flag.String("model", "", "model name sent to the backend")
	modelName := flag.String("model-name", "sparks", "engine name reported as model in responses")
	apiKey := flag.String("api-key", "", "optional bearer token for the backend")
	topK := flag.Int("top-k", 20, "logprobs requested per call (Fireworks caps at 5; missing labels get a floor)")
	rot := flag.Int("rotations", 3, "option orderings averaged per choice")
	adaptiveMargin := flag.Float64("adaptive-margin", 0, "UNSAFE: skip remaining rotations when rotation 0 margin clears it (0 disables; measured to lock in confident position-bias errors that evade review)")
	agreeMargin := flag.Float64("agree-margin", 0.3, "UNSAFE: skip remaining rotations when rotations 0 and 1 agree with margin above it (0 disables; measured accuracy-neutral at 0.3 on 200 Netra cases, re-measure on yours)")
	temps := flag.String("temps", "", "calibration temperatures: default=T,name=T,... (fit with sparks-eval; default is 1.0)")
	reasoning := flag.Bool("reasoning", false, "enable provider thinking (high effort); default false disables thinking so the first token is the answer label")
	timeout := flag.Duration("timeout", 60*time.Second, "per-request backend budget")
	maxInflight := flag.Int("max-inflight", 32, "concurrent decide requests before returning 429")
	maxQuestions := flag.Int("max-questions", 32, "max questions per decide request")
	auth := flag.String("auth-token", "", "if set, clients must send Authorization: Bearer <token>")
	backendRetries := flag.Int("backend-retries", 3, "retries on retryable backend failures (429/5xx)")
	backendTimeout := flag.Duration("backend-timeout", 60*time.Second, "HTTP timeout per backend call")
	debug := flag.Bool("debug", false, "log backend request JSON and raw response to stderr")
	flag.Parse()

	defTemp, overrides, err := parseTemps(*temps)
	if err != nil {
		log.Fatal(err)
	}
	if *topK < 1 || *topK > 20 {
		log.Fatalf("bad -top-k %d (want 1..20)", *topK)
	}
	if *maxInflight < 1 {
		log.Fatalf("bad -max-inflight %d (want >= 1)", *maxInflight)
	}
	if *maxQuestions < 1 {
		log.Fatalf("bad -max-questions %d (want >= 1)", *maxQuestions)
	}
	var backend sparks.Backend = &sparks.ChatCompat{
		BaseURL: *url, Model: *model, APIKey: *apiKey, TopK: *topK,
		Reasoning: *reasoning,
		Client:    &http.Client{Timeout: *backendTimeout}, Debug: *debug,
	}
	if *backendRetries > 0 {
		backend = &sparks.RetryingBackend{Inner: backend}
	}
	srv := newServer(backend, *rot, *adaptiveMargin, *agreeMargin, defTemp, overrides, *modelName, *maxInflight, *auth, *timeout, *maxQuestions)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok")) })
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok")) })
	mux.HandleFunc("POST /v1/decide", srv.serveDecide)

	httpSrv := &http.Server{
		Addr: *addr, Handler: mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      *timeout + 15*time.Second,
		IdleTimeout:       60 * time.Second,
	}

	authState := "disabled"
	if *auth != "" {
		authState = "enabled"
	}
	go func() {
		log.Printf("sparks listening on %s (model %s, backend %s, rotations %d, temps {default=%.2f,%s}, timeout %s, max-inflight %d, auth %s)",
			*addr, *modelName, *url, *rot, defTemp, overrideFlag(overrides), timeout.String(), *maxInflight, authState)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()
	shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}
