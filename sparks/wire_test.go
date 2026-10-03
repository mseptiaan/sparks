package sparks

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func mustCompile(t *testing.T, name, raw string) Compiled {
	t.Helper()
	var wq WireQuestion
	if err := json.Unmarshal([]byte(raw), &wq); err != nil {
		t.Fatal(err)
	}
	c, err := CompileQuestion(name, wq)
	if err != nil {
		t.Fatalf("compile %s: %v", name, err)
	}
	return c
}

func TestCompileChoice(t *testing.T) {
	c := mustCompile(t, "mood", `{"type":"choice","instructions":"How does the customer feel?",
		"criteria":{"angry":"Strong language or threats","calm":null,"excited":{"tone":"upbeat"}}}`)
	if len(c.ChoiceNames) != 3 || c.ChoiceNames[0] != "angry" || c.ChoiceNames[2] != "excited" {
		t.Fatalf("names not sorted: %v", c.ChoiceNames)
	}
	if c.Q.Type != KindChoice {
		t.Fatalf("type %q", c.Q.Type)
	}
	for _, want := range []string{"- angry: Strong language", "- calm\n", `"tone":"upbeat"`} {
		if !contains(c.Q.Question, want) {
			t.Fatalf("prompt missing %q:\n%s", want, c.Q.Question)
		}
	}
}

func TestCompileChoiceValidation(t *testing.T) {
	for _, raw := range []string{
		`{"type":"choice","criteria":{}}`,
		`{"type":"choice","criteria":{"only":"one"}}`,
		`{"type":"choice","criteria":[]}`,
		`{"type":"mystery"}`,
	} {
		var wq WireQuestion
		if err := json.Unmarshal([]byte(raw), &wq); err != nil {
			t.Fatal(err)
		}
		if _, err := CompileQuestion("q", wq); err == nil {
			t.Fatalf("want error for %s", raw)
		}
	}
}

func TestCompileScore(t *testing.T) {
	c := mustCompile(t, "sev", `{"type":"score","instructions":"How severe?",
		"criteria":["Cosmetic","Blocking issue"]}`)
	if len(c.Q.Levels) != 2 || c.Q.Levels[1] != "Blocking issue" {
		t.Fatalf("levels %v", c.Q.Levels)
	}
	if len(c.Legend) != 2 {
		t.Fatalf("legend %v", c.Legend)
	}
	// Null instructions get a fallback question; object levels are kept raw.
	c2 := mustCompile(t, "sev2", `{"type":"score","criteria":[{"en":"Bad"},{"en":"Good"}]}`)
	if c2.Q.Question == "" {
		t.Fatal("want fallback question")
	}
	if _, ok := c2.Legend[0].(map[string]any); !ok {
		t.Fatalf("legend should keep raw JSON: %v", c2.Legend[0])
	}
	if _, err := CompileQuestion("q", WireQuestion{Type: KindScore}); err == nil {
		t.Fatal("want error for missing criteria")
	}
}

func TestCompileNoul(t *testing.T) {
	c := mustCompile(t, "refund", `{"type":"noul","instructions":"Refund requested?",
		"criteria":{"true":"Customer asks for money back"}}`)
	if c.Q.Type != KindNoul || !contains(c.Q.Question, "Yes: Customer asks") {
		t.Fatalf("prompt:\n%s", c.Q.Question)
	}
	if _, err := CompileQuestion("q", WireQuestion{Type: KindNoul}); err == nil {
		t.Fatal("want error for empty noul")
	}
}

func TestRenderState(t *testing.T) {
	if s, err := RenderState("hello"); err != nil || s != "hello" {
		t.Fatalf("%q %v", s, err)
	}
	s, err := RenderState(map[string]any{"a": 1})
	if err != nil || s != `{"a":1}` {
		t.Fatalf("%q %v", s, err)
	}
	for _, bad := range []any{nil, "  ", ""} {
		if _, err := RenderState(bad); err == nil {
			t.Fatalf("want error for %v", bad)
		}
	}
}

func TestScoreLevels(t *testing.T) {
	d := NewDecider(fakeBackend{target: "Blocking issue"}, Config{})
	r, err := d.Ask(context.Background(), "s", Question{
		Type: KindScore, Question: "How severe?",
		Levels: []string{"Cosmetic", "Degraded", "Blocking issue"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Best != "2" {
		t.Fatalf("best %q", r.Best)
	}
	if len(r.Probs) != 3 {
		t.Fatalf("probs %v", r.Probs)
	}
	var sum float64
	for _, p := range r.Probs {
		sum += p
	}
	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("probs sum %v", sum)
	}
	if r.Value == nil || *r.Value < 1 || *r.Value > 2 {
		t.Fatalf("expected index near 2, got %v", r.Value)
	}
}

func contains(s, sub string) bool { return strings.Contains(s, sub) }
