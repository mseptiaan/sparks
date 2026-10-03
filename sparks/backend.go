package sparks

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Backend returns log-probabilities of the most likely next tokens after the
// chat (system, user) pair. Keys are raw token strings as the server reports
// them. The server owns the chat template; the caller supplies plain texts.
type Backend interface {
	TopLogProbs(ctx context.Context, system, user string) (map[string]float64, Usage, error)
}

// Usage counts the tokens one backend call consumed. The zero value means
// the backend reported nothing (or nothing was consumed, e.g. a cache hit):
// a real call always carries prompt_tokens > 0. CachedTokens is a subset of
// PromptTokens (prefix cache hits), never add them together.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int
}

// debugMu serializes debug dumps so concurrent rotations/workers stay readable.
var debugMu sync.Mutex

// parseUsage reads the OpenAI usage block when present. Absent (some local
// servers) yields zero Usage, which callers treat as unreported.
func parseUsage(raw []byte) Usage {
	var meta struct {
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			Details          *struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if json.Unmarshal(raw, &meta) != nil || meta.Usage == nil {
		return Usage{}
	}
	var cached int
	if meta.Usage.Details != nil {
		cached = meta.Usage.Details.CachedTokens
	}
	return Usage{PromptTokens: meta.Usage.PromptTokens, CompletionTokens: meta.Usage.CompletionTokens, CachedTokens: cached}
}

// BackendError is a non-2xx response from the model server. It lets callers
// separate retryable outages (429/5xx, the ones that killed overnight evals)
// from fatal client errors (4xx) without string matching.
type BackendError struct {
	StatusCode int
	Status     string
	Body       string
	RetryAfter time.Duration
}

func (e *BackendError) Error() string {
	if e.Body == "" {
		return fmt.Sprintf("backend %s", e.Status)
	}
	return fmt.Sprintf("backend %s: %s", e.Status, e.Body)
}

// Retryable reports whether the request may succeed on retry.
func (e *BackendError) Retryable() bool {
	return e.StatusCode == http.StatusTooManyRequests || e.StatusCode >= 500
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func retryAfter(h string) time.Duration {
	if h == "" {
		return 0
	}
	if secs, err := strconv.Atoi(strings.TrimSpace(h)); err == nil && secs >= 0 {
		return time.Duration(secs) * time.Second
	}
	return 0
}

// RetryingBackend wraps a Backend with exponential-backoff retries on
// retryable failures (429/5xx per BackendError, or network timeouts).
// Non-retryable errors and context cancellation return immediately.
type RetryingBackend struct {
	Inner      Backend
	MaxRetries int           // retries after the first attempt; 0 disables
	BaseWait   time.Duration // first backoff step; default 200ms
	MaxWait    time.Duration // backoff cap; default 5s
}

func (r *RetryingBackend) TopLogProbs(ctx context.Context, system, user string) (map[string]float64, Usage, error) {
	return doRetry(ctx, r.BaseWait, r.MaxWait, r.MaxRetries, func() (map[string]float64, Usage, error) {
		return r.Inner.TopLogProbs(ctx, system, user)
	})
}

func doRetry(ctx context.Context, baseWait, maxWait time.Duration, maxRetries int, fn func() (map[string]float64, Usage, error)) (map[string]float64, Usage, error) {
	base := baseWait
	if base <= 0 {
		base = 200 * time.Millisecond
	}
	max := maxWait
	if max <= 0 {
		max = 5 * time.Second
	}
	var err error
	var res map[string]float64
	var usage Usage
	for attempt := 0; ; attempt++ {
		res, usage, err = fn()
		if err == nil || !retryableErr(err) || attempt >= maxRetries {
			return res, usage, err
		}
		wait := base << attempt
		if wait > max || wait <= 0 {
			wait = max
		}
		if be := new(BackendError); errors.As(err, &be) && be.RetryAfter > 0 {
			wait = min(wait+be.RetryAfter, 30*time.Second)
		}
		// Small jitter so parallel rotations don't retry in lockstep.
		wait += time.Duration(rand.Int64N(int64(100 * time.Millisecond)))
		select {
		case <-ctx.Done():
			return nil, Usage{}, ctx.Err()
		case <-time.After(wait):
		}
	}
}

func retryableErr(err error) bool {
	var be *BackendError
	if errors.As(err, &be) {
		return be.Retryable()
	}
	var ne net.Error
	if errors.As(err, &ne) {
		return ne.Timeout()
	}
	return false
}

// parseLogProbs reads the chat logprobs shape
// (choices[0].logprobs.content[0].top_logprobs).
func parseLogProbs(raw []byte) (map[string]float64, error) {
	var out struct {
		Choices []struct {
			Logprobs struct {
				Content []struct {
					TopLogprobs []struct {
						Token   string  `json:"token"`
						Logprob float64 `json:"logprob"`
					} `json:"top_logprobs"`
				} `json:"content"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("decode backend response: %w", err)
	}
	if len(out.Choices) > 0 {
		lp := out.Choices[0].Logprobs
		if len(lp.Content) > 0 && len(lp.Content[0].TopLogprobs) > 0 {
			m := make(map[string]float64, len(lp.Content[0].TopLogprobs))
			for _, t := range lp.Content[0].TopLogprobs {
				m[t.Token] = t.Logprob
			}
			return m, nil
		}
	}
	return nil, fmt.Errorf("no logprobs in backend response (does the server support `logprobs`?)")
}

// CachedBackend memoizes by (system, user). Useful for evaluation: temperature
// search re-scores the same prompts without calling the model again.
// A cache hit reports zero Usage: no new tokens were consumed.
type CachedBackend struct {
	Inner Backend
	mu    sync.Mutex
	m     map[string]cachedProbs
}

type cachedProbs struct {
	probs map[string]float64
	usage Usage
}

func NewCachedBackend(inner Backend) *CachedBackend {
	return &CachedBackend{Inner: inner, m: map[string]cachedProbs{}}
}

func (c *CachedBackend) TopLogProbs(ctx context.Context, system, user string) (map[string]float64, Usage, error) {
	key := system + "\x00" + user
	c.mu.Lock()
	if v, ok := c.m[key]; ok {
		c.mu.Unlock()
		return v.probs, Usage{}, nil
	}
	c.mu.Unlock()
	v, usage, err := c.Inner.TopLogProbs(ctx, system, user)
	if err != nil {
		return nil, Usage{}, err
	}
	c.mu.Lock()
	c.m[key] = cachedProbs{probs: v, usage: usage}
	c.mu.Unlock()
	return v, usage, nil
}
