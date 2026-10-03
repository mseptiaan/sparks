package sparks

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type chatStub struct {
	probs        map[string]float64
	usage        Usage
	system, user string
	calls        int
}

func (b *chatStub) TopLogProbs(_ context.Context, system, user string) (map[string]float64, Usage, error) {
	b.calls++
	b.system, b.user = system, user
	return b.probs, b.usage, nil
}

func TestChatCompatParsesChatShape(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"A"},`+
			`"logprobs":{"content":[{"token":"A","logprob":-0.01,`+
			`"top_logprobs":[{"token":"A","logprob":-0.01},{"token":"B","logprob":-5.0}]}]}}],`+
			`"usage":{"prompt_tokens":50,"completion_tokens":2}}`)
	}))
	defer srv.Close()
	m, u, err := (&ChatCompat{BaseURL: srv.URL}).TopLogProbs(context.Background(), "sys", "user")
	if err != nil {
		t.Fatal(err)
	}
	if m["A"] != -0.01 || m["B"] != -5.0 {
		t.Fatalf("probs = %v", m)
	}
	if u != (Usage{PromptTokens: 50, CompletionTokens: 2}) {
		t.Fatalf("usage = %+v", u)
	}
}

func TestDeciderSendsSystemAndUser(t *testing.T) {
	b := &chatStub{probs: map[string]float64{" A": -0.05, " B": -3.0}, usage: Usage{PromptTokens: 60, CompletionTokens: 2}}
	d := NewDecider(b, Config{Rotations: 1})
	r, err := d.Choice(context.Background(), "s", "Pick?", []string{"x", "y"})
	if err != nil {
		t.Fatal(err)
	}
	if r.Best != "x" {
		t.Fatalf("best = %q", r.Best)
	}
	if b.calls != 1 {
		t.Fatalf("calls = %d", b.calls)
	}
	if !strings.Contains(b.system, "precise decision engine") {
		t.Fatalf("system = %q", b.system)
	}
	if !strings.Contains(b.user, "Options:") || !strings.Contains(b.user, "Pick?") {
		t.Fatalf("user = %q", b.user)
	}
	if r.Usage != (Usage{PromptTokens: 60, CompletionTokens: 2}) {
		t.Fatalf("usage = %+v", r.Usage)
	}
}

func TestChatCompatReasoningFields(t *testing.T) {
	const resp = `{"choices":[{"message":{"content":"A"},` +
		`"logprobs":{"content":[{"token":"A","logprob":-0.01,` +
		`"top_logprobs":[{"token":"A","logprob":-0.01}]}]}}],` +
		`"usage":{"prompt_tokens":50,"completion_tokens":2}}`
	for _, tc := range []struct {
		reasoning bool
		want      map[string]any
	}{
		{false, map[string]any{"reasoning": map[string]any{"enabled": false}, "reasoning_effort": "none"}},
		{true, map[string]any{"reasoning": map[string]any{"effort": "high"}, "reasoning_effort": "high"}},
	} {
		var got map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
				t.Errorf("decode request: %v", err)
			}
			fmt.Fprint(w, resp)
		}))
		_, _, err := (&ChatCompat{BaseURL: srv.URL, Reasoning: tc.reasoning}).TopLogProbs(context.Background(), "sys", "user")
		srv.Close()
		if err != nil {
			t.Fatal(err)
		}
		for k := range tc.want {
			gotV, err := json.Marshal(got[k])
			if err != nil {
				t.Fatal(err)
			}
			wantV, _ := json.Marshal(tc.want[k])
			if string(gotV) != string(wantV) {
				t.Errorf("reasoning=%v: %s = %s, want %s", tc.reasoning, k, gotV, wantV)
			}
		}
	}
}
