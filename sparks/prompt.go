package sparks

import (
	"fmt"
	"strings"
)

const defaultSystem = "You are a precise decision engine. Use only the information in the state. " +
	"Reply with the answer label only, nothing else."

func letters(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = string(rune('A' + i))
	}
	return out
}

func choiceUser(state, question string, ordered []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "State:\n%s\n\nQuestion: %s\n\nOptions:\n", strings.TrimSpace(state), question)
	for i, o := range ordered {
		fmt.Fprintf(&b, "%c. %s\n", 'A'+i, o)
	}
	b.WriteString("\nAnswer with only the letter of the correct option.")
	return b.String()
}

func scoreUser(state, question string, min, max int) string {
	return fmt.Sprintf("State:\n%s\n\nQuestion: %s\n\nAnswer with only a single digit from %d to %d.",
		strings.TrimSpace(state), question, min, max)
}
