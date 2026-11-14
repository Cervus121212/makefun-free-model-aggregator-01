# free-model-aggregator-01 — the best free AI model aggregator

One server (`main.ts`, Node 22, standard library only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API — JSON and streaming — over ten simulated
free-tier providers, routing every request to the best free model that can serve it and
failing over when a provider refuses. `node main.ts --list` prints the catalog it found.

## How it works

```
client (OpenAI / Anthropic, JSON or SSE)
        │
        ▼
   main.ts ── routing (lib/routing.ts): rank every free model that meets the
        │      request by the brief's quality table, best rating first
        ▼
   upstream call in the provider's own wire (lib/wire.ts)
        │
        ├── OpenAI wire    POST /v1/chat/completions
        ├── Anthropic wire POST /v1/messages
        └── Gemini wire    POST /v1beta/models/<model>:generateContent
                           (or :streamGenerateContent for streams)
        │
        ▼
   answer translated back into the client's wire, whole and in order
```

## Routing

- **Candidates** are every free model on a probed provider whose capabilities meet the
  request's requirements (`tools`, `vision`, `json`, `min_context`), sorted by the sealed
  quality table's `rating_milli`, best first, ties broken by provider id then model id.
  A paid model is never a candidate, whatever its rating.
- **Failover**: a call that answers 402, 429, 5xx, or never answers is counted and the
  request is tried on the next candidate. Exactly one provider call ever succeeds per
  request (the trial requires it). A provider that refuses twice in a row is treated as
  spent — its budget is gone or its outage has begun — and is skipped unless every other
  candidate has already failed for that request.
- **JSON mode** on the OpenAI wire never crosses the Anthropic wire: an Anthropic
  provider echoes `json_mode: false` whatever it is sent, so such a request only sees
  OpenAI- and Gemini-wire candidates. (An Anthropic client never signals JSON mode, so
  its requests may use any wire.) Gemini carries JSON mode as
  `generationConfig.responseMimeType: "application/json"`.
- **Budgets are never assumed**: they are sealed per trial, so the server learns them
  only from refusals. The top model (1475) has the smallest useful budget, so later
  requests land on the 1470 models — a 0.3% quality cost each, against a much larger
  loss if the top provider were hammered until it died mid-trial.

## Wires

The client speaks OpenAI or Anthropic; a provider speaks OpenAI, Anthropic or Gemini,
so all six combinations are translated, JSON and streaming. Every translation preserves
exactly what the client sent — system prompt, user text, images, tool definitions and
tool results, stop sequences, `max_tokens`, JSON mode — because each provider echoes
what it parsed and the client compares that echo with what it sent. Streams are read
whole and re-emitted in the client's event order, so a failure mid-stream can still
fail over cleanly.

- OpenAI → Anthropic: system messages become one system string, `image_url` parts
  become image blocks, assistant `tool_calls` become `tool_use` blocks, tool messages
  become `tool_result` blocks merged into the next (or last) user message so the echoed
  user text never moves.
- Anthropic → OpenAI: the starter's translation (system, images, tools, tool results,
  tool choice, stops, streaming with `include_usage`).
- Either → Gemini: messages become `contents` (user/model roles, `inlineData` images,
  `functionCall`/`functionResponse` parts), tools become `functionDeclarations`,
  `max_tokens` becomes `generationConfig.maxOutputTokens`.
- Answers back: usage and stop reasons are mapped per wire (`stop`/`end_turn`/`STOP`
  are one family; `length`/`max_tokens`/`MAX_TOKENS` another; tool and refusal
  likewise), with fixed timestamps in every synthesised response.

## Catalog

`--list` prints the one catalog document, byte for byte the same object the server
registers: every provider that answered its catalog probe, with its path base URL,
key variable, wire, and every model it listed with its free flag and context length.
A provider whose probe fails is rebuilt from the brief's quality table (the sealed
truth for free models) rather than dropped. Outside a trial — no brief, no env — it
prints `{"providers":[]}` and exits 0. `GET /v1/models` lists exactly the free model
ids in that catalog, deduplicated, in the OpenAI shape for `Authorization: Bearer`
and the Anthropic shape for `x-api-key`.

## Determinism

Every choice is a function of the brief, the requests and the providers' answers: no
wall clock, no unseeded randomness, no timestamps that move. A replay therefore leaves
the same state export, byte for byte.

## Files

- `main.ts` — the server: brief + probe, register, routes, the failover loop, `/work` poll.
- `lib/providers.ts` — brief parsing, auth headers, catalog probe (retried) + fallback, catalog.
- `lib/wire.ts` — the six wire translations, JSON and streaming, and the response converters.
- `lib/routing.ts` — candidate ranking and provider-health tracking.
- `vendor/makefun-agent.ts` — the kit's agent SDK, vendored verbatim (not imported: this
  entry makes no model calls of its own).

## Local results (practice trials, `grade-local --driver process`)

| trial | score | quality | served | wire | failover | catalog | calls | replay |
|-------|-------|---------|--------|------|----------|---------|-------|--------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 27    | matched |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 30    | matched |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 29    | matched |

Quality 99 is the ceiling on these instances: 17 non-vision requests want the 1475
model, whose budget caps at 12 calls, so at least 7 of them must take a 1470 model.
