package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mseptiaan/sparks/sparks"
)

// stubBackend returns a fixed distribution over the A/B/C answer labels.
type stubBackend struct {
	probs map[string]float64
	usage sparks.Usage
	err   error
	calls atomic.Int64
}

func (s *stubBackend) TopLogProbs(context.Context, string, string) (map[string]float64, sparks.Usage, error) {
	s.calls.Add(1)
	if s.err != nil {
		return nil, sparks.Usage{}, s.err
	}
	return s.probs, s.usage, nil
}

// chatStubBackend records the system text Decider sends.
type chatStubBackend struct {
	probs  map[string]float64
	usage  sparks.Usage
	mu     sync.Mutex
	calls  int
	system string
}

func (s *chatStubBackend) TopLogProbs(_ context.Context, system, _ string) (map[string]float64, sparks.Usage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	s.system = system
	return s.probs, s.usage, nil
}

func TestDecideSendsSystemText(t *testing.T) {
	be := &chatStubBackend{probs: map[string]float64{"A": -0.05, "B": -3.0},
		usage: sparks.Usage{PromptTokens: 40, CompletionTokens: 2}}
	srv := testServer(be)
	rec := postDecide(t, srv, `{"state":"s","questions":{
		"q": {"type":"noul","instructions":"Q?"}}}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("code %d: %s", rec.Code, rec.Body.String())
	}
	be.mu.Lock()
	defer be.mu.Unlock()
	if be.calls != 2 {
		t.Fatalf("calls = %d, want 2 (both noul orders)", be.calls)
	}
	if !strings.Contains(be.system, "precise decision engine") {
		t.Fatalf("system = %q", be.system)
	}
}

func testServer(b sparks.Backend) *server {
	return newServer(b, 1, 0, 0, 1.0, map[string]float64{"hot": 0.5, "flat": 5.0}, "test-engine", 8, "", 5*time.Second, 4)
}

func postDecide(t *testing.T, srv *server, body string, hdr map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/decide", strings.NewReader(body))
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	srv.serveDecide(rec, req)
	return rec
}

func decodeEnvelope(t *testing.T, rec *httptest.ResponseRecorder) map[string]json.RawMessage {
	t.Helper()
	var env struct {
		Model   string                     `json:"model"`
		Usage   map[string]any             `json:"usage"`
		Answers map[string]json.RawMessage `json:"answers"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	if env.Model != "test-engine" {
		t.Fatalf("model %q", env.Model)
	}
	return env.Answers
}

func envelopeUsage(t *testing.T, rec *httptest.ResponseRecorder) (any, int, int) {
	t.Helper()
	var env struct {
		Usage *struct {
			Input  *int `json:"input_tokens"`
			Output *int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	if env.Usage == nil || env.Usage.Input == nil || env.Usage.Output == nil {
		return nil, 0, 0
	}
	return true, *env.Usage.Input, *env.Usage.Output
}

func TestDecideReportsUsage(t *testing.T) {
	be := &stubBackend{probs: map[string]float64{"A": -0.05, "B": -3.0},
		usage: sparks.Usage{PromptTokens: 100, CompletionTokens: 1, CachedTokens: 60}}
	srv := testServer(be)
	rec := postDecide(t, srv, `{"state":"s","questions":{
		"c": {"type":"choice","instructions":"Q?","criteria":{"x":null,"y":null}},
		"n": {"type":"noul","instructions":"Q?"}}}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("code %d: %s", rec.Code, rec.Body.String())
	}
	// rotations=1: choice 1 call + noul 2 calls.
	if ok, in, out := envelopeUsage(t, rec); ok == nil || in != 300 || out != 3 {
		t.Fatalf("usage in=%d out=%d", in, out)
	}
	var env struct {
		Usage *struct {
			Cached *int `json:"cached_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	if env.Usage == nil || env.Usage.Cached == nil || *env.Usage.Cached != 180 {
		t.Fatalf("cached usage: %s", rec.Body.String())
	}
}

func TestTempFor(t *testing.T) {
	srv := testServer(&stubBackend{})
	if got := srv.tempFor("hot"); got != 0.5 {
		t.Fatalf("override: %v", got)
	}
	if got := srv.tempFor("other"); got != 1.0 {
		t.Fatalf("default: %v", got)
	}
}

func TestDecideNullUsageWhenSilent(t *testing.T) {
	srv := testServer(&stubBackend{probs: map[string]float64{"A": -0.1, "B": -0.1}})
	rec := postDecide(t, srv, `{"state":"s","questions":{"q":{"type":"noul","instructions":"Q?"}}}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("code %d", rec.Code)
	}
	if ok, _, _ := envelopeUsage(t, rec); ok != nil {
		t.Fatalf("silent backend must yield null usage: %s", rec.Body.String())
	}
}

func TestDecideChoiceAndTemps(t *testing.T) {
	// Strongly favors A: low T => high confidence, high T => near uniform.
	be := &stubBackend{probs: map[string]float64{"A": -0.01, "B": -4.0}}
	srv := testServer(be)
	rec := postDecide(t, srv, `{"state":"s","questions":{
		"hot": {"type":"choice","instructions":"Pick?","criteria":{"x":"first","y":"second"}},
		"flat": {"type":"choice","instructions":"Pick?","criteria":{"x":"first","y":null}},
		"plain": {"type":"choice","instructions":"Pick?","criteria":{"x":null,"y":null}}}}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("code %d: %s", rec.Code, rec.Body.String())
	}
	got := decodeEnvelope(t, rec)
	if len(got) != 3 {
		t.Fatalf("answers %v", got)
	}
	type choiceOut struct {
		Type          string             `json:"type"`
		Choice        string             `json:"choice"`
		Confidence    float64            `json:"confidence"`
		Probabilities map[string]float64 `json:"probabilities"`
	}
	var prev float64 = 2
	for _, name := range []string{"hot", "plain", "flat"} {
		var c choiceOut
		if err := json.Unmarshal(got[name], &c); err != nil {
			t.Fatal(err)
		}
		if c.Type != "choice" || c.Choice != "x" {
			t.Fatalf("%s: %+v", name, c)
		}
		if len(c.Probabilities) != 2 {
			t.Fatalf("%s probs %v", name, c.Probabilities)
		}
		if c.Confidence >= prev {
			t.Fatalf("confidence should fall with T: %s=%v", name, c.Confidence)
		}
		prev = c.Confidence
	}
}

func TestDecideScoreLegend(t *testing.T) {
	be := &stubBackend{probs: map[string]float64{"A": -0.05, "B": -3.0, "C": -3.0}}
	srv := testServer(be)
	rec := postDecide(t, srv, `{"state":"s","questions":{
		"sev": {"type":"score","instructions":"How severe?",
			"criteria":["Cosmetic",{"en":"Degraded"},"Blocking"]}}}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("code %d: %s", rec.Code, rec.Body.String())
	}
	got := decodeEnvelope(t, rec)
	var s struct {
		Type          string             `json:"type"`
		Score         float64            `json:"score"`
		Confidence    float64            `json:"confidence"`
		Legend        map[string]any     `json:"legend"`
		Probabilities map[string]float64 `json:"probabilities"`
	}
	if err := json.Unmarshal(got["sev"], &s); err != nil {
		t.Fatal(err)
	}
	if s.Type != "score" {
		t.Fatalf("type %q", s.Type)
	}
	if s.Legend["0"] != "Cosmetic" {
		t.Fatalf("legend %v", s.Legend)
	}
	var ev float64
	for i := range 3 {
		k := string(rune('0' + i))
		ev += float64(i) * s.Probabilities[k]
	}
	if s.Score < ev-1e-9 || s.Score > ev+1e-9 {
		t.Fatalf("score %v != expected %v", s.Score, ev)
	}
	if s.Score > 0.5 {
		t.Fatalf("strong-A stub should score near 0, got %v", s.Score)
	}
}

func TestDecideNoulIsProbability(t *testing.T) {
	be := &stubBackend{probs: map[string]float64{"A": -0.05, "B": -3.0}}
	srv := testServer(be)
	rec := postDecide(t, srv, `{"state":"s","questions":{
		"refund": {"type":"noul","instructions":"Refund?",
			"criteria":{"true":"asks for money back"}}}}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("code %d: %s", rec.Code, rec.Body.String())
	}
	got := decodeEnvelope(t, rec)
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(got["refund"], &raw); err != nil {
		t.Fatal(err)
	}
	if _, hasConf := raw["confidence"]; hasConf {
		t.Fatalf("noul must not carry confidence: %v", raw)
	}
	var n struct {
		Type string  `json:"type"`
		Noul float64 `json:"noul"`
	}
	if err := json.Unmarshal(got["refund"], &n); err != nil {
		t.Fatal(err)
	}
	if n.Type != "noul" {
		t.Fatalf("type %+v", n)
	}
	// The stub favors letter A in both rotations ([Yes,No] then [No,Yes]),
	// so rotation averaging must cancel it to ~0.5. This pins the
	// position-bias cancellation, not just the response shape.
	if n.Noul < 0.49 || n.Noul > 0.51 {
		t.Fatalf("noul %+v", n)
	}
}

func TestDecideStateShapes(t *testing.T) {
	srv := testServer(&stubBackend{probs: map[string]float64{"A": -0.1, "B": -0.1}})
	if rec := postDecide(t, srv, `{"questions":{"q":{"type":"noul","instructions":"Q?"}}}`, nil); rec.Code != http.StatusBadRequest {
		t.Fatalf("missing state: code %d, want 400", rec.Code)
	}
	if rec := postDecide(t, srv, `{"state":null,"questions":{"q":{"type":"noul","instructions":"Q?"}}}`, nil); rec.Code != http.StatusBadRequest {
		t.Fatalf("null state: code %d, want 400", rec.Code)
	}
	// Structured state is serialized, not rejected.
	for _, st := range []string{`"plain text"`, `{"a":1,"b":[true,null]}`, `[1,"x"]`} {
		rec := postDecide(t, srv, `{"state":`+st+`,"questions":{"q":{"type":"noul","instructions":"Q?"}}}`, nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("state %s: code %d, want 200: %s", st, rec.Code, rec.Body.String())
		}
	}
}

func TestDecideValidation(t *testing.T) {
	srv := testServer(&stubBackend{})
	cases := map[string]string{
		"bad json":      `{"state":`,
		"no question":   `{"state":"s","questions":{}}`,
		"empty choice":  `{"state":"s","questions":{"q":{"type":"choice","instructions":"Q?","criteria":{}}}}`,
		"one option":    `{"state":"s","questions":{"q":{"type":"choice","instructions":"Q?","criteria":{"x":"only"}}}}}`,
		"one level":     `{"state":"s","questions":{"q":{"type":"score","instructions":"Q?","criteria":["solo"]}}}`,
		"empty noul":    `{"state":"s","questions":{"q":{"type":"noul"}}}`,
		"bad noul crit": `{"state":"s","questions":{"q":{"type":"noul","instructions":"Q?","criteria":[]}}}`,
		"bad type":      `{"state":"s","questions":{"q":{"type":"wat","instructions":"Q?"}}}`,
	}
	for name, body := range cases {
		if rec := postDecide(t, srv, body, nil); rec.Code != http.StatusBadRequest {
			t.Fatalf("%s: code %d, want 400: %s", name, rec.Code, rec.Body.String())
		}
	}
	rec := postDecide(t, srv, `{"state":"s","questions":{
		"a":{"type":"noul","instructions":"Q?"},"b":{"type":"noul","instructions":"Q?"},
		"c":{"type":"noul","instructions":"Q?"},"d":{"type":"noul","instructions":"Q?"},
		"e":{"type":"noul","instructions":"Q?"}}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("too many: code %d, want 400", rec.Code)
	}
}

func TestDecideAuth(t *testing.T) {
	srv := newServer(&stubBackend{probs: map[string]float64{"A": -0.1, "B": -0.1}},
		1, 0, 0, 1.0, nil, "test-engine", 8, "s3cr3t", 5*time.Second, 4)
	body := `{"state":"s","questions":{"q":{"type":"noul","instructions":"Q?"}}}`
	if rec := postDecide(t, srv, body, nil); rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing: code %d, want 401", rec.Code)
	}
	if rec := postDecide(t, srv, body, map[string]string{"Authorization": "Bearer wrong"}); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong: code %d, want 401", rec.Code)
	}
	if rec := postDecide(t, srv, body, map[string]string{"Authorization": "Bearer s3cr3t"}); rec.Code != http.StatusOK {
		t.Fatalf("correct: code %d, want 200: %s", rec.Code, rec.Body.String())
	}
}

func TestDecideBackendFailureIs502(t *testing.T) {
	be := &stubBackend{err: &sparks.BackendError{StatusCode: 503, Status: "503 Service Unavailable"}}
	rec := postDecide(t, testServer(be), `{"state":"s","questions":{"q":{"type":"noul","instructions":"Q?"}}}`, nil)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("code %d, want 502", rec.Code)
	}
}

func TestDecideOverloadIs429(t *testing.T) {
	srv := testServer(&stubBackend{probs: map[string]float64{"A": -0.1, "B": -0.1}})
	srv.sem <- struct{}{} // fill the semaphore (cap 8) so the request trips 429
	for range 7 {
		srv.sem <- struct{}{}
	}
	rec := postDecide(t, srv, `{"state":"s","questions":{"q":{"type":"noul","instructions":"Q?"}}}`, nil)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("code %d, want 429", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Fatal("want Retry-After header")
	}
}

func TestRetryingBackend(t *testing.T) {
	be := &stubBackend{err: &sparks.BackendError{StatusCode: 503, Status: "503"}}
	r := &sparks.RetryingBackend{Inner: be, MaxRetries: 2, BaseWait: time.Millisecond, MaxWait: 2 * time.Millisecond}
	if _, _, err := r.TopLogProbs(context.Background(), "sys", "p"); err == nil {
		t.Fatal("want error after retries")
	}
	if got := be.calls.Load(); got != 3 {
		t.Fatalf("calls %d, want 1+2 retries", got)
	}

	be409 := &stubBackend{err: &sparks.BackendError{StatusCode: 409, Status: "409"}}
	r409 := &sparks.RetryingBackend{Inner: be409, MaxRetries: 3, BaseWait: time.Millisecond}
	if _, _, err := r409.TopLogProbs(context.Background(), "sys", "p"); err == nil {
		t.Fatal("want error")
	}
	if got := be409.calls.Load(); got != 1 {
		t.Fatalf("non-retryable calls %d, want 1", got)
	}
}
