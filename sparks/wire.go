package sparks

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// Typesafe-compatible wire format for questions.
//
// Mirrors https://docs.typesafe.ai/primitives:
//   - choice criteria is a required object mapping option names to
//     descriptions (a description may be a string, object, array, or null).
//   - score criteria is a required non-empty ordered list with one entry per
//     score from zero.
//   - noul criteria is optional and describes the yes/no outcomes under the
//     "true"/"false" keys.
//
// instructions, every description, and state accept JSONContent: a string,
// object, array, or null. Objects and arrays are rendered as compact JSON.

// WireQuestion is one Typesafe-style question as it arrives in a request.
type WireQuestion struct {
	Type         Kind            `json:"type"`
	Instructions any             `json:"instructions,omitempty"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

// Compiled is a WireQuestion lowered to an executable Question plus the
// display metadata needed to shape a Typesafe-style answer.
type Compiled struct {
	Q           Question
	ChoiceNames []string // choice: criteria names, sorted
	Legend      []any    // score: raw level descriptions in order
}

// renderContent renders one JSONContent value for prompt inclusion.
// Strings pass through; objects and arrays become compact JSON; null is "".
func renderContent(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	default:
		b, err := json.Marshal(t)
		if err != nil {
			return ""
		}
		return string(b)
	}
}

// RenderState renders request state: strings pass through, objects and
// arrays become compact JSON. Null or blank state is a client error.
func RenderState(v any) (string, error) {
	if v == nil {
		return "", fmt.Errorf("state is required")
	}
	if s, ok := v.(string); ok {
		if strings.TrimSpace(s) == "" {
			return "", fmt.Errorf("state is required")
		}
		return s, nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("state must be text or a JSON object/array")
	}
	return string(b), nil
}

// CompileQuestion lowers one Typesafe-style question to an executable
// Question. Choice names are sorted for deterministic prompts.
func CompileQuestion(name string, wq WireQuestion) (Compiled, error) {
	instr := strings.TrimSpace(renderContent(wq.Instructions))
	switch wq.Type {
	case KindChoice:
		var crit map[string]any
		if err := json.Unmarshal(wq.Criteria, &crit); err != nil || len(crit) == 0 {
			return Compiled{}, fmt.Errorf("question %q: choice criteria must be a non-empty object", name)
		}
		names := make([]string, 0, len(crit))
		for k := range crit {
			names = append(names, k)
		}
		slices.Sort(names)
		if len(names) < 2 || len(names) > 26 {
			return Compiled{}, fmt.Errorf("question %q: choice needs 2..26 criteria names, got %d", name, len(names))
		}
		var b strings.Builder
		if instr != "" {
			b.WriteString(instr)
		}
		b.WriteString("\n\nCriteria:\n")
		for _, k := range names {
			if d := strings.TrimSpace(renderContent(crit[k])); d != "" {
				fmt.Fprintf(&b, "- %s: %s\n", k, d)
			} else {
				fmt.Fprintf(&b, "- %s\n", k)
			}
		}
		return Compiled{
			Q:           Question{Type: KindChoice, Question: strings.TrimSpace(b.String()), Options: names},
			ChoiceNames: names,
		}, nil
	case KindScore:
		var crit []any
		if err := json.Unmarshal(wq.Criteria, &crit); err != nil || len(crit) < 2 {
			return Compiled{}, fmt.Errorf("question %q: score criteria must be a list of at least 2 levels", name)
		}
		if len(crit) > 26 {
			return Compiled{}, fmt.Errorf("question %q: score needs at most 26 levels, got %d", name, len(crit))
		}
		levels := make([]string, len(crit))
		for i, c := range crit {
			if d := strings.TrimSpace(renderContent(c)); d != "" {
				levels[i] = d
			} else {
				levels[i] = fmt.Sprintf("level %d", i)
			}
		}
		q := instr
		if q == "" {
			q = "Which level best describes the state?"
		}
		return Compiled{
			Q:      Question{Type: KindScore, Question: q, Levels: levels},
			Legend: slices.Clone(crit),
		}, nil
	case KindNoul:
		var crit map[string]any
		if len(wq.Criteria) > 0 {
			if err := json.Unmarshal(wq.Criteria, &crit); err != nil {
				return Compiled{}, fmt.Errorf("question %q: noul criteria must be an object", name)
			}
		}
		text := instr
		if text == "" && len(crit) == 0 {
			return Compiled{}, fmt.Errorf("question %q: instructions or criteria required", name)
		}
		var b strings.Builder
		b.WriteString(text)
		if d := strings.TrimSpace(renderContent(crit["true"])); d != "" {
			b.WriteString("\n\nYes: " + d)
		}
		if d := strings.TrimSpace(renderContent(crit["false"])); d != "" {
			b.WriteString("\nNo: " + d)
		}
		return Compiled{Q: Question{Type: KindNoul, Question: strings.TrimSpace(b.String())}}, nil
	}
	return Compiled{}, fmt.Errorf("question %q: unknown type %q", name, wq.Type)
}

// ScoreLevels answers an ordered rubric: levels[i] describes score i and the
// result expectation is over 0-based level indices, matching the Typesafe
// Score answer (score, legend, probabilities). Labels are single letters, so
// up to 26 levels are supported. One backend call.
func (d *Decider) ScoreLevels(ctx context.Context, state, question string, levels []string) (Result, error) {
	if len(levels) < 2 || len(levels) > 26 {
		return Result{}, fmt.Errorf("score needs 2..26 levels, got %d", len(levels))
	}
	lps, usage, err := d.callBackend(ctx, choiceUser(state, question, levels))
	if err != nil {
		return Result{}, err
	}
	vals, mass := labelLogProbs(lps, letters(len(levels)))
	probs := softmaxT(vals, d.Cfg.Temperature)
	p := make(map[string]float64, len(levels))
	var ev float64
	bi := 0
	for i := range levels {
		p[strconv.Itoa(i)] = probs[i]
		ev += float64(i) * probs[i]
		if probs[i] > probs[bi] {
			bi = i
		}
	}
	return Result{Type: KindScore, Best: strconv.Itoa(bi), Probs: p, Confidence: probs[bi], Value: &ev, LabelMass: mass, Usage: usage}, nil
}
