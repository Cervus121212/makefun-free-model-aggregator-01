// free-model-aggregator-01: one server, two client wires, ten simulated free-tier providers.
//
// It reads the brief (provider directory, quality table, workload manifest), probes each provider's
// catalog route in its own wire, registers one catalog with the env, then serves the env client's 20
// requests: each answered by the best free model that meets its requirements, failing over down the
// rating ranking when a provider refuses, and translated back into the wire the client used.
//
// Routing (lib/routing.ts): candidates are every free model meeting the request, best rating first.
// A provider that refuses twice in a row is treated as spent and skipped unless nothing else is left.
// JSON mode on the OpenAI wire never crosses the Anthropic wire, whose providers echo json_mode false.
//
// Wires (lib/wire.ts): the client speaks OpenAI or Anthropic; a provider speaks OpenAI, Anthropic or
// Gemini. All six combinations are translated, JSON and streaming, preserving exactly what the client
// sent — system prompt, user text, images, tools, stop sequences, max_tokens, JSON mode — because the
// provider echoes what it parsed and the client compares that echo with what it sent.
//
// Determinism: every choice is a function of the brief, the requests and the providers' answers. No
// wall clock and no unseeded randomness anywhere; fixed timestamps in every synthesised response.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  fallbackProbe,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicStreamEvents,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIJsonMode,
  openAIModelList,
  openAIStreamEvents,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  parseAnthropicStream,
  parseGeminiStream,
  parseOpenAIStream,
  type ServedModel,
} from './lib/wire.ts';
import { attemptOrder, candidatesFor, ProviderHealth, type Candidate } from './lib/routing.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The Anthropic version the Anthropic wire's providers and clients send. */
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** What the trial has shown about each provider: consecutive refusals mark one spent. */
const health = new ProviderHealth();
/** Whether the request now being served streams; set before each upstream call (Gemini's path differs). */
let currentStream = false;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (let i = 0; i < probes.length; i++) {
    const probe = probes[i];
    if (!probe.ok) {
      log(`${probe.provider.id}: ${probe.error ?? 'probe failed'}; falling back to the brief's quality table`);
      const recovered = fallbackProbe(probe.provider, brief.ratings);
      if (recovered.ok) probes[i] = recovered;
      else log(`${probe.provider.id}: no models in the brief either; the provider stays out`);
    }
  }
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, each id once, sorted: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      const row = brief.ratings?.ratings.find((r) => r.provider === probe.provider.id && r.model === model.id) ?? null;
      const free = row !== null ? row.free : (model.free ?? false);
      if (!free) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  out.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request body for one candidate, in the provider's own wire. */
function upstreamBody(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'anthropic') {
    if (clientApi === 'anthropic') return { ...body, model };
    return openAIToAnthropicRequest(body, model);
  }
  // gemini
  return clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
}

/** The upstream URL for one candidate: the model rides in the path on the Gemini wire. */
function upstreamUrl(candidate: Candidate): string {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  if (candidate.probe.provider.wire === 'openai') return `${base}/v1/chat/completions`;
  if (candidate.probe.provider.wire === 'anthropic') return `${base}/v1/messages`;
  const model = encodeURIComponent(candidate.model.id);
  return currentStream
    ? `${base}/v1beta/models/${model}:streamGenerateContent?alt=sse`
    : `${base}/v1beta/models/${model}:generateContent`;
}

/** One inference call to one candidate provider. A thrown error is a refusal like any other. */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const provider = candidate.probe.provider;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(upstreamUrl(candidate), {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamBody(candidate, clientApi, body)),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
}

/**
 * A provider's answer translated into the client's wire. `text` is the provider's whole answer body
 * (JSON or SSE); a stream is parsed whole and re-emitted, so a failure mid-stream can still fail over.
 */
function translateAnswer(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  text: string,
): { status: number; contentType: string; body: string } {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  const stream = body.stream === true;
  if (!stream) {
    let answer: unknown;
    try {
      answer = JSON.parse(text);
    } catch {
      throw new Error(`the ${candidate.probe.provider.id} provider answered a body that is not JSON`);
    }
    if (!isRecord(answer)) throw new Error(`the ${candidate.probe.provider.id} provider answered a JSON value that is not one object`);
    if (wire === 'openai' && clientApi === 'openai') return { status: 200, contentType: 'application/json', body: text };
    if (wire === 'anthropic' && clientApi === 'anthropic') return { status: 200, contentType: 'application/json', body: text };
    if (wire === 'openai' && clientApi === 'anthropic') {
      return { status: 200, contentType: 'application/json', body: JSON.stringify(openAIToAnthropic(answer, model)) };
    }
    if (wire === 'anthropic' && clientApi === 'openai') {
      return { status: 200, contentType: 'application/json', body: JSON.stringify(anthropicToOpenAIResponse(answer, model)) };
    }
    if (wire === 'gemini' && clientApi === 'openai') {
      return { status: 200, contentType: 'application/json', body: JSON.stringify(geminiToOpenAIResponse(answer, model)) };
    }
    return { status: 200, contentType: 'application/json', body: JSON.stringify(geminiToAnthropicResponse(answer, model)) };
  }
  // Streaming: the same wire passes its events through; any other wire is parsed and re-emitted.
  if (wire === 'openai' && clientApi === 'openai') return { status: 200, contentType: 'text/event-stream', body: text };
  if (wire === 'anthropic' && clientApi === 'anthropic') return { status: 200, contentType: 'text/event-stream', body: text };
  const parsed = wire === 'openai' ? parseOpenAIStream(text) : wire === 'anthropic' ? parseAnthropicStream(text) : parseGeminiStream(text);
  const events = clientApi === 'openai' ? openAIStreamEvents(parsed, model) : anthropicStreamEvents(parsed, model);
  return { status: 200, contentType: 'text/event-stream', body: events };
}

/**
 * Serve one client request: try its candidates in rating order until one answers, then translate the
 * answer into the client's wire. Every failed attempt is logged; a request with no candidate left is
 * answered 503, honestly, rather than with a model that cannot serve it.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  // JSON mode on the OpenAI wire cannot cross the Anthropic wire: its providers echo json_mode false.
  const jsonOverAnthropic = !(clientApi === 'openai' && openAIJsonMode(body));
  const candidates = candidatesFor(probes, brief.ratings, requires, jsonOverAnthropic);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  currentStream = body.stream === true;
  let failures = 0;
  for (const candidate of attemptOrder(candidates, health)) {
    const providerId = candidate.probe.provider.id;
    let upstream: Response;
    try {
      upstream = await callProvider(candidate, clientApi, body);
    } catch (err) {
      health.failure(providerId);
      failures += 1;
      log(`${providerId}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok) {
      health.failure(providerId);
      failures += 1;
      const detail = (await upstream.text().catch(() => '')).slice(0, 200);
      log(`${providerId}/${candidate.model.id} refused HTTP ${upstream.status}: ${detail}`);
      continue;
    }
    let text: string;
    try {
      text = await upstream.text();
    } catch (err) {
      health.failure(providerId);
      failures += 1;
      log(`${providerId}/${candidate.model.id} answer unreadable: ${(err as Error).message}`);
      continue;
    }
    health.success(providerId);
    let translated;
    try {
      translated = translateAnswer(candidate, clientApi, body, text);
    } catch (err) {
      // A provider that answered 2xx with an unusable body is treated as a refusal: try the next.
      health.failure(providerId);
      failures += 1;
      log(`${providerId}/${candidate.model.id} answer unusable: ${(err as Error).message}`);
      continue;
    }
    if (failures > 0) log(`served after ${failures} failed attempt(s) via ${providerId}/${candidate.model.id}`);
    sendText(res, translated.status, translated.contentType, translated.body);
    return;
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 50; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
