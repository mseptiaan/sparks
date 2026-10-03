package main

import (
	"encoding/json"
	"testing"

	"github.com/mseptiaan/sparks/sparks"
)

func TestBuildChoiceFromCriteria(t *testing.T) {
	var q evalQuestion
	if err := json.Unmarshal([]byte(`{"type":"choice","instructions":"Pick one.",
		"criteria":{"b":"Bee.","a":"Ay.","c":"See."}}`), &q); err != nil {
		t.Fatal(err)
	}
	got, err := q.build("pattern")
	if err != nil {
		t.Fatal(err)
	}
	wantOpts := []string{"a", "b", "c"}
	if len(got.Options) != len(wantOpts) {
		t.Fatalf("options = %v, want %v", got.Options, wantOpts)
	}
	for i, o := range wantOpts {
		if got.Options[i] != o {
			t.Fatalf("options = %v, want %v", got.Options, wantOpts)
		}
	}
	for _, o := range wantOpts {
		if !contains(got.Question, o) {
			t.Fatalf("prompt missing criteria key %q:\n%s", o, got.Question)
		}
	}
}

func TestBuildNoulFromCriteria(t *testing.T) {
	var q evalQuestion
	if err := json.Unmarshal([]byte(`{"type":"noul","instructions":"Supported?",
		"criteria":{"true":"It holds.","false":"It does not."}}`), &q); err != nil {
		t.Fatal(err)
	}
	got, err := q.build("signals_support_alert")
	if err != nil {
		t.Fatal(err)
	}
	if got.Type != sparks.KindNoul {
		t.Fatalf("type = %v", got.Type)
	}
	if !contains(got.Question, "It holds.") || !contains(got.Question, "It does not.") {
		t.Fatalf("prompt missing criteria text:\n%s", got.Question)
	}
	if noulExpected(got, "true") != "yes" || noulExpected(got, "false") != "no" {
		t.Fatal("true/false must map to the model's yes/no labels")
	}
	if noulExpected(got, "yes") != "yes" {
		t.Fatal("yes/no form must pass through")
	}
}

func TestBuildScoreFromCriteria(t *testing.T) {
	var q evalQuestion
	if err := json.Unmarshal([]byte(`{"type":"score","instructions":"How fast?",
		"criteria":["Routine.","High.","Urgent."]}`), &q); err != nil {
		t.Fatal(err)
	}
	got, err := q.build("review_priority")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"Routine.", "High.", "Urgent."}
	if got.Type != sparks.KindScore || len(got.Levels) != len(want) {
		t.Fatalf("got %+v, want legend levels %v", got, want)
	}
	for i, l := range want {
		if got.Levels[i] != l {
			t.Fatalf("levels = %v, want %v", got.Levels, want)
		}
	}
}

func TestMapExpectedLegend(t *testing.T) {
	q := sparks.Question{Type: sparks.KindScore, Levels: []string{"a", "b", "c"}}
	for in, want := range map[string]string{"1": "0", "2": "1", "3": "2"} {
		got, err := mapExpected(q, in)
		if err != nil || got != want {
			t.Fatalf("mapExpected(%q) = %q,%v; want %q", in, got, err, want)
		}
	}
	for _, bad := range []string{"0", "4", "x", ""} {
		if _, err := mapExpected(q, bad); err == nil {
			t.Fatalf("want error for %q", bad)
		}
	}
}

func TestBuildRejectsBadCriteria(t *testing.T) {
	cases := []string{
		`{"type":"choice","instructions":"x","criteria":{"only":"one"}}`,
		`{"type":"choice","instructions":"x","criteria":["not","an","object"]}`,
		`{"type":"score","instructions":"x","criteria":{"not":"a list"}}`,
		`{"type":"score","instructions":"x","criteria":["only"]}`,
		`{"type":"bogus","instructions":"x"}`,
	}
	for _, c := range cases {
		var q evalQuestion
		if err := json.Unmarshal([]byte(c), &q); err != nil {
			t.Fatal(err)
		}
		if _, err := q.build("q"); err == nil {
			t.Fatalf("want error for %s", c)
		}
	}
}

func TestBuildPassesTypedFormThrough(t *testing.T) {
	var q evalQuestion
	if err := json.Unmarshal([]byte(`{"type":"choice","question":"Q?",
		"options":["x","y"]}`), &q); err != nil {
		t.Fatal(err)
	}
	got, err := q.build("q")
	if err != nil {
		t.Fatal(err)
	}
	if got.Question != "Q?" || len(got.Options) != 2 {
		t.Fatalf("typed form altered: %+v", got)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestSkipItems(t *testing.T) {
	mk := func(name string) item { return item{Name: name} }
	items := []item{mk("pattern"), mk("review_priority"), mk("signals_support_alert")}
	got := skipItems(items, parseSkip("review_priority,signals_support_alert"))
	if len(got) != 1 || got[0].Name != "pattern" {
		t.Fatalf("got %v, want only pattern", got)
	}
	if got := skipItems(items, parseSkip("")); len(got) != 3 {
		t.Fatalf("empty skip keeps all, got %d", len(got))
	}
	if got := skipItems(items, parseSkip("pattern,review_priority,signals_support_alert")); len(got) != 0 {
		t.Fatalf("full skip empties, got %d", len(got))
	}
}
