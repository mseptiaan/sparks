package sparks

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"regexp"
	"testing"
)

var optLine = regexp.MustCompile(`(?m)^([A-Z])\. (.+)$`)

// fakeBackend simulates a model that "knows" the target option text, with an
// optional bias toward the letter A.
type fakeBackend struct {
	target string
	biasA  float64
}

func (f fakeBackend) TopLogProbs(_ context.Context, _, user string) (map[string]float64, Usage, error) {
	ms := optLine.FindAllStringSubmatch(user, -1)
	w := map[string]float64{}
	var total float64
	for _, m := range ms {
		v := 0.1
		if m[2] == f.target {
			v += 0.3
		}
		if m[1] == "A" {
			v += f.biasA
		}
		w[m[1]] = v
		total += v
	}
	out := map[string]float64{}
	for l, v := range w {
		out[" "+l] = math.Log(v / total * 0.95) // leading-space variant, like real tokenizers
	}
	return out, Usage{}, nil
}

func TestChoiceMapsRotationsBack(t *testing.T) {
	d := NewDecider(fakeBackend{target: "billing"}, Config{Rotations: 3})
	r, err := d.Choice(context.Background(), "s", "q", []string{"bug", "billing", "other"})
	if err != nil {
		t.Fatal(err)
	}
	if r.Best != "billing" {
		t.Fatalf("best = %q, want billing; probs=%v", r.Best, r.Probs)
	}
	var sum float64
	for _, p := range r.Probs {
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("probs sum to %v", sum)
	}
}

func TestRotationsCancelPositionBias(t *testing.T) {
	// Strong bias toward letter A: without rotation the first option wins.
	b := fakeBackend{target: "c", biasA: 0.6}
	opts := []string{"a", "b", "c"}

	noRot := NewDecider(b, Config{Rotations: 1})
	r1, _ := noRot.Choice(context.Background(), "s", "q", opts)
	if r1.Best != "a" {
		t.Fatalf("expected biased answer 'a' with 1 rotation, got %q", r1.Best)
	}
	rot := NewDecider(b, Config{Rotations: 3})
	r3, _ := rot.Choice(context.Background(), "s", "q", opts)
	if r3.Best != "c" {
		t.Fatalf("expected 'c' with 3 rotations, got %q (%v)", r3.Best, r3.Probs)
	}
}

func TestNoul(t *testing.T) {
	d := NewDecider(fakeBackend{target: "Yes", biasA: 0.5}, Config{})
	r, err := d.Noul(context.Background(), "s", "q")
	if err != nil {
		t.Fatal(err)
	}
	if r.Best != "yes" || r.Probs["yes"] <= r.Probs["no"] {
		t.Fatalf("got %+v", r)
	}
}

type digitBackend struct{ probs map[string]float64 }

func (f digitBackend) TopLogProbs(context.Context, string, string) (map[string]float64, Usage, error) {
	out := map[string]float64{}
	for k, v := range f.probs {
		out[k] = math.Log(v)
	}
	return out, Usage{}, nil
}

func TestScoreExpectedValue(t *testing.T) {
	d := NewDecider(digitBackend{map[string]float64{"4": 0.6, "5": 0.4}}, Config{})
	r, err := d.Score(context.Background(), "s", "q", 1, 5)
	if err != nil {
		t.Fatal(err)
	}
	if r.Best != "4" || r.Value == nil || math.Abs(*r.Value-4.4) > 1e-6 {
		t.Fatalf("got %+v value=%v", r, r.Value)
	}
}

func TestTemperatureSoftens(t *testing.T) {
	b := digitBackend{map[string]float64{"1": 0.9, "2": 0.1}}
	sharp, _ := NewDecider(b, Config{Temperature: 1}).Score(context.Background(), "s", "q", 1, 2)
	soft, _ := NewDecider(b, Config{Temperature: 3}).Score(context.Background(), "s", "q", 1, 2)
	if !(soft.Confidence < sharp.Confidence) {
		t.Fatalf("T=3 should be less confident: %v vs %v", soft.Confidence, sharp.Confidence)
	}
}

func TestValidation(t *testing.T) {
	d := NewDecider(digitBackend{}, Config{})
	if _, err := d.Choice(context.Background(), "s", "q", []string{"only"}); err == nil {
		t.Fatal("want error for 1 option")
	}
	if _, err := d.Score(context.Background(), "s", "q", 5, 1); err == nil {
		t.Fatal("want error for bad range")
	}
}

func TestChatCompatParsesUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"logprobs":{"content":[{"top_logprobs":[{"token":"A","logprob":-0.1}]}]}}],`+
			`"usage":{"prompt_tokens":1234,"completion_tokens":1,"total_tokens":1235,`+
			`"prompt_tokens_details":{"cached_tokens":1100}}}`)
	}))
	defer srv.Close()
	_, u, err := (&ChatCompat{BaseURL: srv.URL}).TopLogProbs(context.Background(), "sys", "user")
	if err != nil {
		t.Fatal(err)
	}
	if u != (Usage{PromptTokens: 1234, CompletionTokens: 1, CachedTokens: 1100}) {
		t.Fatalf("usage = %+v", u)
	}
}

func TestCachedBackendZeroOnHit(t *testing.T) {
	inner := &countBackend{probs: map[string]float64{"A": -0.1}, usage: Usage{PromptTokens: 100, CompletionTokens: 1}}
	c := NewCachedBackend(inner)
	ctx := context.Background()
	if _, u, err := c.TopLogProbs(ctx, "sys", "user"); err != nil || u.PromptTokens != 100 {
		t.Fatalf("miss = %+v,%v", u, err)
	}
	if _, u, err := c.TopLogProbs(ctx, "sys", "user"); err != nil || u != (Usage{}) {
		t.Fatalf("hit should report zero usage: %+v,%v", u, err)
	}
	if inner.calls != 1 {
		t.Fatalf("calls = %d", inner.calls)
	}
}

type countBackend struct {
	probs map[string]float64
	usage Usage
	calls int
}

func (b *countBackend) TopLogProbs(context.Context, string, string) (map[string]float64, Usage, error) {
	b.calls++
	return b.probs, b.usage, nil
}

func TestDecideSumsRotationsUsage(t *testing.T) {
	b := &countBackend{
		probs: map[string]float64{"A": math.Log(0.9), "B": math.Log(0.05), "C": math.Log(0.05)},
		usage: Usage{PromptTokens: 100, CompletionTokens: 1},
	}
	d := NewDecider(b, Config{Rotations: 3})
	r, err := d.Choice(context.Background(), "s", "q", []string{"x", "y", "z"})
	if err != nil {
		t.Fatal(err)
	}
	if r.Usage != (Usage{PromptTokens: 300, CompletionTokens: 3}) {
		t.Fatalf("usage = %+v (calls %d)", r.Usage, b.calls)
	}
}

func TestAdaptiveRotations(t *testing.T) {
	ctx := context.Background()
	mk2 := func(pa, pb float64) *countBackend {
		return &countBackend{
			probs: map[string]float64{"A": math.Log(pa), "B": math.Log(pb)},
			usage: Usage{PromptTokens: 10},
		}
	}
	// Decisive rotation 0 short-circuits the rest.
	decisive := mk2(0.95, 0.05)
	d := NewDecider(decisive, Config{Rotations: 3, MinMargin: 0.5})
	if _, err := d.Choice(ctx, "s", "q", []string{"x", "y"}); err != nil {
		t.Fatal(err)
	}
	if decisive.calls != 1 {
		t.Fatalf("decisive calls = %d, want 1", decisive.calls)
	}
	// Close call runs all rotations (probe reused as rotation 0).
	closec := mk2(0.55, 0.45)
	d2 := NewDecider(closec, Config{Rotations: 3, MinMargin: 0.5})
	if _, err := d2.Choice(ctx, "s", "q", []string{"x", "y"}); err != nil {
		t.Fatal(err)
	}
	if closec.calls != 2 {
		t.Fatalf("close calls = %d, want 2", closec.calls)
	}
	three := &countBackend{
		probs: map[string]float64{"A": math.Log(0.5), "B": math.Log(0.3), "C": math.Log(0.2)},
		usage: Usage{PromptTokens: 10},
	}
	d3 := NewDecider(three, Config{Rotations: 3, MinMargin: 0.5})
	if _, err := d3.Choice(ctx, "s", "q", []string{"x", "y", "z"}); err != nil {
		t.Fatal(err)
	}
	if three.calls != 3 {
		t.Fatalf("three-way close calls = %d, want 3", three.calls)
	}
	// Disabled by default even when decisive.
	off := mk2(0.95, 0.05)
	d4 := NewDecider(off, Config{Rotations: 3})
	if _, err := d4.Choice(ctx, "s", "q", []string{"x", "y"}); err != nil {
		t.Fatal(err)
	}
	if off.calls != 2 {
		t.Fatalf("disabled calls = %d, want 2", off.calls)
	}
}

type countFake struct {
	fakeBackend
	calls int
}

func (b *countFake) TopLogProbs(ctx context.Context, sys, user string) (map[string]float64, Usage, error) {
	b.calls++
	return b.fakeBackend.TopLogProbs(ctx, sys, user)
}

func TestAgreeGate(t *testing.T) {
	ctx := context.Background()
	opts := []string{"a", "b", "c"}
	// Decisive content: both rotations pick the target, third call skipped.
	agree := &countFake{fakeBackend: fakeBackend{target: "c"}}
	d := NewDecider(agree, Config{Rotations: 3, AgreeMargin: 0.1})
	r, err := d.Choice(ctx, "s", "q", opts)
	if err != nil {
		t.Fatal(err)
	}
	if r.Best != "c" {
		t.Fatalf("best = %q, want c", r.Best)
	}
	if agree.calls != 2 {
		t.Fatalf("agree calls = %d, want 2", agree.calls)
	}
	if d.Skipped.Load() != 1 {
		t.Fatalf("skipped = %d, want 1", d.Skipped.Load())
	}
	// Pure position bias: rotations disagree, full fan-out runs.
	biased := &countFake{fakeBackend: fakeBackend{target: "c", biasA: 0.6}}
	d2 := NewDecider(biased, Config{Rotations: 3, AgreeMargin: 0.1})
	r2, err := d2.Choice(ctx, "s", "q", opts)
	if err != nil {
		t.Fatal(err)
	}
	if r2.Best != "c" {
		t.Fatalf("best = %q, want c", r2.Best)
	}
	if biased.calls != 3 {
		t.Fatalf("biased calls = %d, want 3", biased.calls)
	}
	// Disabled by default even when decisive.
	off := &countFake{fakeBackend: fakeBackend{target: "c"}}
	d3 := NewDecider(off, Config{Rotations: 3})
	if _, err := d3.Choice(ctx, "s", "q", opts); err != nil {
		t.Fatal(err)
	}
	if off.calls != 3 {
		t.Fatalf("disabled calls = %d, want 3", off.calls)
	}
}
