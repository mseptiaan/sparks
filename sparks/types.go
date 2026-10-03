// Package sparks implements "typed decisions" (Noul, Choice, Score) on top of any
// language model that can return next-token log-probabilities. Nothing is
// generated: we do one forward pass and read the probabilities of the answer
// labels.
package sparks

import "fmt"

type Kind string

const (
	KindNoul   Kind = "noul"   // yes / no
	KindChoice Kind = "choice" // pick one of several options
	KindScore  Kind = "score"  // ordinal digit scale, e.g. 1..5
)

// Question is one typed question asked against a state.
type Question struct {
	Type     Kind     `json:"type"`
	Question string   `json:"question"`
	Options  []string `json:"options,omitempty"` // choice only (2..26)
	Min      int      `json:"min,omitempty"`     // score only, digit mode (0..9)
	Max      int      `json:"max,omitempty"`     // score only, digit mode (min < max <= 9)
	Levels   []string `json:"levels,omitempty"`  // score only, legend mode (2..26 level descriptions)
}

// Validate reports whether the question is well-formed (client error if not).
// Backend failures are never reported here.
func (q Question) Validate() error {
	switch q.Type {
	case KindNoul:
		return nil
	case KindChoice:
		if n := len(q.Options); n < 2 || n > 26 {
			return fmt.Errorf("choice needs 2..26 options, got %d", n)
		}
		return nil
	case KindScore:
		if len(q.Levels) > 0 {
			if len(q.Levels) < 2 || len(q.Levels) > 26 {
				return fmt.Errorf("score needs 2..26 levels, got %d", len(q.Levels))
			}
			return nil
		}
		if q.Min < 0 || q.Max > 9 || q.Min >= q.Max {
			return fmt.Errorf("score range must satisfy 0 <= min < max <= 9, got %d..%d", q.Min, q.Max)
		}
		return nil
	}
	return fmt.Errorf("unknown question type %q", q.Type)
}

// Result is the typed answer plus a full probability distribution.
//
//	noul:   Probs keys "yes" / "no"
//	choice: Probs keys are the option texts
//	score:  Probs keys are the digits "1".."5"; Value is the expected value
type Result struct {
	Type       Kind               `json:"type"`
	Best       string             `json:"best"`
	Probs      map[string]float64 `json:"probs"`
	Confidence float64            `json:"confidence"`      // probability of Best
	Value      *float64           `json:"value,omitempty"` // score only
	// LabelMass is the raw probability mass the model put on the answer labels.
	// Low values mean the model wanted to say something else (bad prompt).
	LabelMass float64 `json:"label_mass"`
	// Usage totals backend tokens consumed scoring this question.
	Usage Usage `json:"-"`
}
