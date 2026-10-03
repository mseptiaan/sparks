package sparks

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// ChatCompat is the backend: an OpenAI-style POST /v1/chat/completions for
// single-token label logprobs. The server owns the chat template; the caller
// supplies system and user texts. Reasoning toggles provider thinking: off
// (default) sends reasoning {"enabled":false} with reasoning_effort "none",
// so the first token is the answer label; on sends effort "high" both ways.
// It needs max_tokens small and logprobs on — a model that thinks first
// puts prose in top-K.
type ChatCompat struct {
	BaseURL string // e.g. https://api.openai.com
	Model   string
	APIKey  string
	TopK    int // default 5
	// Reasoning enables provider thinking (default false).
	Reasoning bool
	Client    *http.Client
	Debug     bool // log request JSON and raw response to stderr
}

func (o *ChatCompat) TopLogProbs(ctx context.Context, system, user string) (map[string]float64, Usage, error) {
	k := o.TopK
	if k <= 0 {
		k = 5
	}
	client := o.Client
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	msg := func(role, content string) map[string]string {
		return map[string]string{"role": role, "content": content}
	}
	bodyMap := map[string]any{
		"model": o.Model,
		"messages": []map[string]string{
			msg("system", system),
			msg("user", user),
		},
		"temperature": 0, "max_completion_tokens": 2,
		"logprobs": true, "top_logprobs": k,
	}
	if o.Reasoning {
		bodyMap["reasoning"] = map[string]any{"effort": "high"}
		bodyMap["reasoning_effort"] = "high"
	} else {
		bodyMap["reasoning"] = map[string]any{"enabled": false}
		bodyMap["reasoning_effort"] = "none"
	}
	body, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, Usage{}, err
	}
	url := strings.TrimRight(o.BaseURL, "/") + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, Usage{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if o.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+o.APIKey)
	}
	resp, err := client.Do(req)
	if o.Debug {
		debugMu.Lock()
		fmt.Fprintf(os.Stderr, "=== POST %s ===\n--- request ---\n%s\n", url, body)
		debugMu.Unlock()
	}
	if err != nil {
		return nil, Usage{}, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if o.Debug {
		debugMu.Lock()
		fmt.Fprintf(os.Stderr, "--- response %s ---\n%s\n", resp.Status, raw)
		debugMu.Unlock()
	}
	if resp.StatusCode != http.StatusOK {
		return nil, Usage{}, &BackendError{
			StatusCode: resp.StatusCode,
			Status:     resp.Status,
			Body:       truncate(strings.TrimSpace(string(raw)), 500),
			RetryAfter: retryAfter(resp.Header.Get("Retry-After")),
		}
	}
	lps, err := parseLogProbs(raw)
	if err != nil {
		return nil, Usage{}, err
	}
	return lps, parseUsage(raw), nil
}

// (debug output reuses backend.go's debugMu)
