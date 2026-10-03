# sparks

Sparks is a backend that transforms any LLM into typed decisions, featuring typesafe request compatibility while cutting output tokens by 99%.

It exposes `POST /v1/decide`: you send a `state` plus named typed questions
(`noul`, `choice`, `score`), and it returns one calibrated answer per question —
a yes probability, a picked option, or an expected score — each with a full
probability distribution. No text is generated and nothing needs parsing: a
single forward pass reads the answer-label token (`A`/`B`/`C`, …) log-probs
from any OpenAI-style chat backend. The request/response wire format follows
[Typesafe](https://docs.typesafe.ai/primitives) (`instructions` + `criteria`).

## How it works

| Type | Behavior |
|---|---|
| `noul` | Yes/No scored in both orders and averaged; answer = P(yes), 0–1 |
| `choice` | Options (criteria names) labeled A..Z; option order is cyclically rotated and averaged to cancel position bias |
| `score` | Ordered rubric (legend); levels mapped to A..Z in one forward pass; answer = expected 0-based level index + legend |

Every distribution is temperature-scaled (`T`, calibrated per question via
`-temps`). `confidence` is the best-label probability.

## Requirements

An OpenAI-style server exposing `POST /v1/chat/completions` with `logprobs`.
Sparks sends raw system+user text; your server wraps it in its own chat
template — only the prompt content per backend is calibrated, there is no
template flag. New backends must pass the 30-second filter: one `logprobs`
call must return a distribution. Reasoning models that hide `logprobs`
(or think before emitting the label) cannot be used.

## Run

### With Go

```sh
go build -o sparks ./cmd/sparks
./sparks -backend-url https://api.openai.com -model gpt-6-sol \
  -temps 'default=1.0' \
  -auth-token "$SPARKS_TOKEN"
```

Or without building:

```sh
go run ./cmd/sparks -backend-url https://api.openai.com -model gpt-6-sol
```

`-reasoning` (default `false`) disables thinking so the first token is the
answer label: requests carry `"reasoning": {"enabled": false}` and
`"reasoning_effort": "none"`. Pass `-reasoning=true` for high-effort thinking
(`"reasoning": {"effort": "high"}`, `"reasoning_effort": "high"`).

### With Docker

```sh
docker build -t sparks .
docker run --rm -p 8088:8088 --read-only sparks \
  -backend-url https://api.openai.com -model gpt-6-sol \
  -temps 'default=1.0'
```

Or via compose:

```sh
SPARKS_BACKEND_URL=https://api.openai.com SPARKS_API_KEY=... \
  SPARKS_MODEL=gpt-6-sol SPARKS_REASONING=false \
  SPARKS_TEMPS='default=1.0' \
  docker compose up --build
```

Variables: `SPARKS_BACKEND_URL` (required), `SPARKS_API_KEY`, `SPARKS_MODEL`,
`SPARKS_REASONING` (default `false`), `SPARKS_TEMPS` (default `default=1.0`).
The image (~16MB, static Go binary on distroless, multi-arch amd64/arm64) has
no shell, runs as nonroot (65532), passes `--read-only`, and has no
HEALTHCHECK — probe `GET /healthz` and `/readyz` from your orchestrator.

## Request

```sh
curl localhost:8088/v1/decide -d '{
  "state": "Customer: item arrived broken, I want my money back.",
  "questions": {
    "refund":  {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "queue":   {"type": "choice", "instructions": "Which queue fits?",
                "criteria": {"billing": "Billing issue", "bug": "Bug report", "other": "Anything else"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["Calm", "Normal", "Urgent", "Critical"]}
  }}'
```

`state` accepts a string or any JSON object/array. `choice` criteria is an
object mapping option names to descriptions; `score` criteria is an ordered
list, one entry per level from zero.

## Example response

```json
{
  "model": "sparks",
  "usage": {"input_tokens": 4208, "output_tokens": 10, "cached_tokens": 3328},
  "answers": {
    "refund": {"type": "noul", "noul": 0.97},
    "queue": {
      "type": "choice", "choice": "other", "confidence": 0.81,
      "probabilities": {"billing": 0.06, "bug": 0.13, "other": 0.81}
    },
    "urgency": {
      "type": "score", "score": 2.1, "confidence": 0.55,
      "legend": {"0": "Calm", "1": "Normal", "2": "Urgent", "3": "Critical"},
      "probabilities": {"0": 0.02, "1": 0.18, "2": 0.55, "3": 0.25}
    }
  }
}
```

Shape per type: `choice` → `choice` + `probabilities` + `confidence`;
`score` → `score` (expected 0-based level index) + `legend` + `probabilities` +
`confidence`; `noul` → `noul` (P(yes), no confidence). `usage` sums backend
tokens per request (`input_tokens`/`output_tokens` are `null` when the backend does not report); `cached_tokens`
is a billed-at-cache subset of input — do not double-count. Rotation fan-out
within one transaction shares the system+state prefix, which is where the
cache hits come from.

## Operations

`GET /healthz` / `/readyz`, automatic retries on `429`/`5xx`
(`-backend-retries`), `400` for invalid questions vs `502`/`504` for backend
failures, `429` above `-max-inflight`, `401` when `-auth-token` is set,
graceful shutdown on SIGTERM/SIGINT.

## Calibration and evaluation

```sh
go run ./cmd/sparks-eval -backend-url https://api.openai.com -model gpt-6-sol -api-key sk-xxxxxx -cases eval/cases.v2.jsonl
```

Prints accuracy, NLL, Brier, ECE, and searches the best temperature per
question (feed it to `sparks -temps`). `eval/cases.v2.jsonl` holds 12 starter
cases — grow it to 100+ cases from your own data before trusting the numbers.
To compare models, run `sparks-eval` once per model and compare NLL and ECE,
not just accuracy. Calibration is only valid for the exact
(backend, rotations, question text) triple evaluated — reword a question and you must
refit its `T`.

Rotation gate (`-agree-margin`, default `0.3`): when rotations 0 and 1
pick the same option with an average top-2 margin above it, the remaining
rotations are skipped (3→2 calls on decisive choices; `noul` always costs 2).
Measure before enabling: run `sparks-eval` with and without it on your cases
and compare accuracy/NLL — enable only if unchanged. The printout reports
`rotations skipped by gates` so the saving is visible.

Code-computed questions don't belong in model scoring: `sparks-eval -skip
review_priority,signals_support_alert` scores only the atomic model judgments.
Derive `review_priority` in code from the model's pattern answer
(`eval/compose.py`, verified 200/200 against labels). `signals_support_alert`
has neither model signal (accuracy 0.505, always "no") nor a code rule
separable in the data — route it to an analyst instead of asking.

Pattern: compose in code. Deterministic rules over thresholds (amounts,
counts, chargebacks) do not belong in the model — eval showed a 4B model
cannot apply them in one step (`acc 0.335`, flat distribution). Ask the model
for atomic judgments, compute the composition in code; `eval/compose.py` is
the reference example (state-text parser → threshold predicates).

## Honest notes

- Tested: core logic (unit tests, incl. position-bias cancellation), chat
  `logprobs` parsing, `sparks`/`sparks-eval` flows against a stub backend,
  and live on Netra (~0.7–0.9s/transaction).
- **Not tested** against other chat backends. The 30-second filter applies:
  one `logprobs` call to `/v1/chat/completions` must return a distribution,
  or the backend is unusable.
- Only `top_logprobs` is available; labels outside top-K get a log-prob
  floor (-30).
- Cost: `choice` uses `-rotations` model calls, `noul` 2 calls. The server
  runs them concurrently, but throughput still follows the backend.
- Speed and quality follow the underlying base model.

## Structure

```
sparks/           library: types, backend (chat), prompt builders, decider (+ tests)
cmd/sparks/       HTTP server  POST /v1/decide
cmd/sparks-eval/  evaluation + temperature search
eval/             cases.v2.jsonl, compose.py (code-composition reference),
                  bench_order.py (prompt-order A/B harness)
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
