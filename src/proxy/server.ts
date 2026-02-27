import { Hono } from "hono"
import { cors } from "hono/cors"
import { query } from "@anthropic-ai/claude-agent-sdk"
import type { Context } from "hono"
import type { ProxyConfig } from "./types"
import { DEFAULT_PROXY_CONFIG } from "./types"
import { logInfo, logWarn, logError, logDebug, LOG_DIR, dumpSessionContext } from "../logger"
import { traceStore } from "../trace"
import { sessionStore } from "../session-store"
import { createToolMcpServer } from "../mcpTools"
import { classifyError } from "../errors"
import {
  mapModelToClaudeModel, serializeContent, contentHasImages,
  buildNativeContent, createSDKUserMessage, stripMcpPrefix,
  roughTokens, buildUsage, extractConversationLabel,
} from "../content"
import {
  openaiToAnthropicMessages, openaiToAnthropicTools,
  anthropicToOpenaiResponse,
} from "../openai-compat"
import { execSync } from "child_process"
import { existsSync, readFileSync, readdirSync } from "fs"
import { randomBytes } from "crypto"
import { fileURLToPath } from "url"
import { join, dirname } from "path"

// ── ID generation & version ──────────────────────────────────────────────────

const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
function generateId(prefix: string, length = 24): string {
  const bytes = randomBytes(length)
  let id = prefix
  for (let i = 0; i < length; i++) id += BASE62[bytes[i]! % 62]
  return id
}

const PROXY_VERSION: string = (() => {
  try {
    const pkg = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), "../../package.json"), "utf-8"))
    return pkg.version ?? "unknown"
  } catch { return "unknown" }
})()

function resolveClaudeExecutable(): string {
  try {
    const sdkPath = fileURLToPath(import.meta.resolve("@anthropic-ai/claude-agent-sdk"))
    const sdkCliJs = join(dirname(sdkPath), "cli.js")
    if (existsSync(sdkCliJs)) return sdkCliJs
  } catch {}
  try {
    const claudePath = execSync("which claude", { encoding: "utf-8" }).trim()
    if (claudePath && existsSync(claudePath)) return claudePath
  } catch {}
  throw new Error("Could not find Claude Code executable. Install: npm install -g @anthropic-ai/claude-code")
}

const claudeExecutable = resolveClaudeExecutable()

// ── Concurrency control ──────────────────────────────────────────────────────

const MAX_CONCURRENT = parseInt(process.env.CLAUDE_PROXY_MAX_CONCURRENT ?? "5", 10)
const QUEUE_TIMEOUT_MS = parseInt(process.env.CLAUDE_PROXY_QUEUE_TIMEOUT_MS ?? "30000", 10)

class RequestQueue {
  private active = 0
  private waiting: Array<{ resolve: () => void; reject: (err: Error) => void }> = []

  get activeCount() { return this.active }
  get waitingCount() { return this.waiting.length }

  async acquire(): Promise<void> {
    if (this.active < MAX_CONCURRENT) {
      this.active++
      return
    }
    return new Promise<void>((resolve, reject) => {
      const entry = { resolve: () => { this.active++; resolve() }, reject }
      this.waiting.push(entry)
      const timer = setTimeout(() => {
        const idx = this.waiting.indexOf(entry)
        if (idx !== -1) {
          this.waiting.splice(idx, 1)
          reject(new Error("Queue timeout — all slots busy"))
        }
      }, QUEUE_TIMEOUT_MS)
      const origResolve = entry.resolve
      entry.resolve = () => { clearTimeout(timer); origResolve() }
    })
  }

  release(): void {
    this.active--
    const next = this.waiting.shift()
    if (next) next.resolve()
  }
}

const requestQueue = new RequestQueue()

// ── Last-context cache ───────────────────────────────────────────────────────

interface LastContext {
  systemPrompt: string
  messages: Array<{ role: string; content: string | Array<any> }>
  model: string
  ts: number
}
const lastContextCache = new Map<string, LastContext>()
const MAX_CONTEXT_CACHE = 100

// ── Query options builder ────────────────────────────────────────────────────

function buildQueryOptions(
  model: "sonnet" | "opus" | "haiku",
  opts: {
    partial?: boolean
    systemPrompt?: string
    abortController?: AbortController
    thinking?: { type: "adaptive" } | { type: "enabled"; budgetTokens?: number } | { type: "disabled" }
    resume?: string
    mcpServers?: Record<string, any>
  } = {}
) {
  return {
    model,
    pathToClaudeCodeExecutable: claudeExecutable,
    permissionMode: "bypassPermissions" as const,
    allowDangerouslySkipPermissions: true,
    persistSession: true,
    settingSources: [],
    tools: [] as string[],
    maxTurns: 1,
    ...(opts.partial ? { includePartialMessages: true } : {}),
    ...(opts.abortController ? { abortController: opts.abortController } : {}),
    ...(opts.thinking ? { thinking: opts.thinking } : {}),
    ...(opts.systemPrompt ? { systemPrompt: opts.systemPrompt } : {}),
    ...(opts.resume ? { resume: opts.resume } : {}),
    ...(opts.mcpServers ? { mcpServers: opts.mcpServers } : {}),
  }
}

// ── Shared SDK helpers ───────────────────────────────────────────────────────
// These reduce duplication across the 6 SDK event loops (main/fallback × non-streaming/streaming-tools/streaming-notools).

interface SdkUsage {
  inputTokens: number
  outputTokens: number
  cacheReadTokens: number
  cacheCreationTokens: number
}

function newUsage(): SdkUsage {
  return { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheCreationTokens: 0 }
}

function makeUsageObj(u: SdkUsage) {
  return {
    input_tokens: u.inputTokens,
    output_tokens: u.outputTokens,
    ...(u.cacheReadTokens > 0 ? { cache_read_input_tokens: u.cacheReadTokens } : {}),
    ...(u.cacheCreationTokens > 0 ? { cache_creation_input_tokens: u.cacheCreationTokens } : {}),
  }
}

/** Capture session_id and usage from common SDK message types. Returns session_id if found. */
function processSdkMessage(message: any, usage: SdkUsage): string | undefined {
  let sessionId: string | undefined
  if (message.type === "system" && (message as any).subtype === "init") {
    sessionId = (message as any).session_id
  }
  if (message.type === "result") {
    const r = message as any
    if (r.session_id) sessionId = r.session_id
    if (r.usage) {
      usage.inputTokens = r.usage.input_tokens ?? usage.inputTokens
      usage.outputTokens = r.usage.output_tokens ?? usage.outputTokens
      usage.cacheReadTokens = r.usage.cache_read_input_tokens ?? usage.cacheReadTokens
      usage.cacheCreationTokens = r.usage.cache_creation_input_tokens ?? usage.cacheCreationTokens
    }
  }
  return sessionId
}

/** Capture usage from stream_event message_start. */
function captureMessageStartUsage(ev: any, usage: SdkUsage): void {
  if (ev.type === "message_start" && ev.message?.usage) {
    if (ev.message.usage.input_tokens) usage.inputTokens = ev.message.usage.input_tokens
    if (ev.message.usage.cache_read_input_tokens) usage.cacheReadTokens = ev.message.usage.cache_read_input_tokens
    if (ev.message.usage.cache_creation_input_tokens) usage.cacheCreationTokens = ev.message.usage.cache_creation_input_tokens
  }
}

interface NonStreamToolState {
  contentBlocks: any[]
  currentToolBlock: { id: string; name: string; jsonAccum: string } | null
  fullText: string
  capturedStopReason: string | null
}

function newNonStreamToolState(): NonStreamToolState {
  return { contentBlocks: [], currentToolBlock: null, fullText: "", capturedStopReason: null }
}

/** Process a stream_event for non-streaming tool accumulation. */
function processNonStreamToolEvent(ev: any, state: NonStreamToolState, usage: SdkUsage): void {
  if (ev.type === "content_block_start") {
    const cb = ev.content_block
    if (cb?.type === "tool_use") {
      state.currentToolBlock = { id: cb.id, name: stripMcpPrefix(cb.name), jsonAccum: "" }
    } else if (cb?.type === "text") {
      state.contentBlocks.push({ type: "text", text: "" })
    }
  } else if (ev.type === "content_block_delta") {
    if (ev.delta?.type === "text_delta") {
      const lastText = state.contentBlocks[state.contentBlocks.length - 1]
      if (lastText?.type === "text") {
        lastText.text += ev.delta.text ?? ""
      } else {
        state.contentBlocks.push({ type: "text", text: ev.delta.text ?? "" })
      }
      state.fullText += ev.delta.text ?? ""
    } else if (ev.delta?.type === "input_json_delta" && state.currentToolBlock) {
      state.currentToolBlock.jsonAccum += ev.delta.partial_json ?? ""
    }
  } else if (ev.type === "content_block_stop") {
    if (state.currentToolBlock) {
      let input: any = {}
      try { input = JSON.parse(state.currentToolBlock.jsonAccum) } catch {}
      state.contentBlocks.push({
        type: "tool_use",
        id: state.currentToolBlock.id,
        name: state.currentToolBlock.name,
        input,
      })
      state.currentToolBlock = null
    }
  } else if (ev.type === "message_delta") {
    if (ev.delta?.stop_reason) state.capturedStopReason = ev.delta.stop_reason
    if (ev.usage?.output_tokens) usage.outputTokens = ev.usage.output_tokens
  } else {
    captureMessageStartUsage(ev, usage)
  }
}

function hasToolBlocks(state: NonStreamToolState): boolean {
  return state.contentBlocks.some((b: any) => b.type === "tool_use")
}

/** Save session mapping after successful SDK query. */
function saveSession(
  conversationId: string | null,
  capturedSessionId: string | undefined,
  isResuming: boolean,
  isCompacted: boolean,
  msgCount: number,
  model: string,
  reqId: string,
  logEvent?: string,
): void {
  if (!conversationId || !capturedSessionId) return
  if (isResuming) {
    sessionStore.recordResume(conversationId, msgCount, isCompacted)
  } else {
    sessionStore.set(conversationId, capturedSessionId, model, msgCount)
  }
  if (logEvent) logInfo(logEvent, { reqId, conversationId, sdkSessionId: capturedSessionId })
}

/** Run the non-streaming SDK event loop. Handles tool events, usage, session capture, and tool abort. */
async function runNonStreamSdkLoop(
  promptInput: string | AsyncIterable<any>,
  queryOpts: any,
  hasTools: boolean,
  abortController: AbortController,
  reqId: string,
  usage: SdkUsage,
  resetStallTimer: () => void,
): Promise<{ sessionId?: string; toolState: NonStreamToolState; fullText: string; sdkEventCount: number; aborted: boolean }> {
  const toolState = newNonStreamToolState()
  let fullText = ""
  let sessionId: string | undefined
  let sdkEventCount = 0

  try {
    for await (const message of query({ prompt: promptInput, options: queryOpts })) {
      sdkEventCount++
      resetStallTimer()
      traceStore.sdkEvent(reqId, sdkEventCount, message.type, (message as any).event?.type ?? (message as any).message?.type)
      const sid = processSdkMessage(message, usage)
      if (sid) sessionId = sid

      if (hasTools && message.type === "stream_event") {
        const ev = (message as any).event as any
        processNonStreamToolEvent(ev, toolState, usage)
        // Abort after all content blocks are complete (message_delta comes after all content_block_stop per API spec)
        if (toolState.capturedStopReason && hasToolBlocks(toolState) && !abortController.signal.aborted) {
          logInfo("sdk.abort_after_complete_response", { reqId, contentBlockCount: toolState.contentBlocks.length, stopReason: toolState.capturedStopReason })
          abortController.abort()
        }
      } else if (!hasTools && message.type === "assistant") {
        let turnText = ""
        for (const block of (message as any).message.content) {
          if (block.type === "text") turnText += block.text
        }
        fullText = turnText
        const msgUsage = (message as any).message?.usage
        if (msgUsage?.input_tokens) usage.inputTokens = msgUsage.input_tokens
        if (msgUsage?.output_tokens) usage.outputTokens = msgUsage.output_tokens
      }
    }
  } catch (err) {
    // Abort after message_delta with tool blocks is expected — return accumulated results
    if (abortController.signal.aborted && hasToolBlocks(toolState)) {
      logInfo("sdk.aborted_after_tool_call_nonstream", { reqId, sdkEventCount, contentBlocks: toolState.contentBlocks.length })
      return { sessionId, toolState, fullText: hasTools ? toolState.fullText : fullText, sdkEventCount, aborted: true }
    }
    throw err
  }

  return { sessionId, toolState, fullText: hasTools ? toolState.fullText : fullText, sdkEventCount, aborted: false }
}

// ── Route handler ────────────────────────────────────────────────────────────

export function createProxyServer(config: Partial<ProxyConfig> = {}) {
  const finalConfig = { ...DEFAULT_PROXY_CONFIG, ...config }
  const app = new Hono()

  app.use("*", cors())

  // Optional API key validation
  const requiredApiKey = process.env.CLAUDE_PROXY_API_KEY
  if (requiredApiKey) {
    app.use("*", async (c, next) => {
      if (c.req.path === "/" || c.req.path.startsWith("/debug") || c.req.method === "OPTIONS") return next()
      const key = c.req.header("x-api-key")
        ?? c.req.header("authorization")?.replace(/^Bearer\s+/i, "")
      if (key !== requiredApiKey) {
        return c.json({
          type: "error",
          error: { type: "authentication_error", message: "Invalid API key" },
          request_id: c.res.headers.get("request-id") ?? generateId("req_")
        }, 401)
      }
      return next()
    })
  }

  // Anthropic-compatible headers + HTTP request logging
  app.use("*", async (c, next) => {
    const start = Date.now()
    const requestId = c.req.header("x-request-id") ?? generateId("req_")
    c.header("x-request-id", requestId)
    c.header("request-id", requestId)
    c.header("anthropic-version", "2023-06-01")
    const betaHeader = c.req.header("anthropic-beta")
    if (betaHeader) c.header("anthropic-beta", betaHeader)
    await next()
    const ms = Date.now() - start
    if (c.req.path.startsWith("/debug")) {
      logDebug("http.request", { method: c.req.method, path: c.req.path, status: c.res.status, ms, reqId: requestId })
    } else {
      logInfo("http.request", { method: c.req.method, path: c.req.path, status: c.res.status, ms, reqId: requestId })
    }
  })

  // ── Health / Info ────────────────────────────────────────────────────────

  app.get("/", (c) => c.json({
    status: "ok",
    service: "claude-sdk-proxy",
    version: PROXY_VERSION,
    format: "anthropic",
    endpoints: ["/v1/messages", "/v1/models", "/v1/chat/completions", "/debug/stats", "/debug/traces", "/debug/errors", "/debug/active", "/debug/health", "/sessions", "/sessions/cleanup"],
    queue: { active: requestQueue.activeCount, waiting: requestQueue.waitingCount, max: MAX_CONCURRENT },
    logDir: LOG_DIR,
  }))

  // ── Debug / Observability endpoints ──────────────────────────────────────

  app.get("/debug/stats", (c) => {
    const stats = traceStore.getStats()
    const sessionStats = sessionStore.getStats()
    return c.json({
      version: PROXY_VERSION,
      config: {
        stallTimeoutMs: finalConfig.stallTimeoutMs,
        maxDurationMs: finalConfig.maxDurationMs,
        maxOutputChars: finalConfig.maxOutputChars,
        maxConcurrent: MAX_CONCURRENT,
        queueTimeoutMs: QUEUE_TIMEOUT_MS,
        claudeExecutable,
        logDir: LOG_DIR,
        debug: finalConfig.debug,
      },
      queue: { active: requestQueue.activeCount, waiting: requestQueue.waitingCount, max: MAX_CONCURRENT },
      sessions: sessionStats,
      ...stats,
    })
  })

  app.get("/sessions", (c) => c.json({ sessions: sessionStore.list(), stats: sessionStore.getStats() }))
  app.get("/sessions/cleanup", (c) => c.json(sessionStore.cleanup()))

  app.delete("/sessions/:id", (c) => {
    const id = decodeURIComponent(c.req.param("id"))
    const stored = sessionStore.get(id)
    if (stored) {
      sessionStore.invalidate(id)
      logInfo("session.api_reset", { conversationId: id, sdkSessionId: stored.sdkSessionId })
      return c.json({ ok: true, invalidated: id })
    }
    return c.json({ ok: false, error: "session not found" }, 404)
  })

  app.get("/debug/traces", (c) => c.json(traceStore.getRecentTraces(parseInt(c.req.query("limit") ?? "20", 10))))
  app.get("/debug/traces/:id", (c) => {
    const trace = traceStore.getTrace(c.req.param("id"))
    return trace ? c.json(trace) : c.json({ error: "Trace not found", reqId: c.req.param("id") }, 404)
  })
  app.get("/debug/errors", (c) => c.json(traceStore.getRecentErrors(parseInt(c.req.query("limit") ?? "10", 10))))

  app.get("/debug/logs", (c) => {
    try {
      const files = readdirSync(LOG_DIR).filter(f => f.startsWith("proxy-") && f.endsWith(".log")).sort().reverse()
      return c.json({ logDir: LOG_DIR, files })
    } catch { return c.json({ logDir: LOG_DIR, files: [], error: "Cannot read log directory" }) }
  })

  app.get("/debug/logs/:filename", (c) => {
    const filename = c.req.param("filename")
    if (!filename.match(/^proxy-\d{4}-\d{2}-\d{2}\.log$/)) return c.json({ error: "Invalid log filename" }, 400)
    const tail = parseInt(c.req.query("tail") ?? "100", 10)
    try {
      const content = readFileSync(join(LOG_DIR, filename), "utf-8")
      const lines = content.trim().split("\n")
      const sliced = lines.slice(-tail)
      const parsed = sliced.map(line => { try { return JSON.parse(line) } catch { return { raw: line } } })
      return c.json({ file: filename, total: lines.length, returned: sliced.length, lines: parsed })
    } catch { return c.json({ error: "Log file not found" }, 404) }
  })

  app.get("/debug/errors/:id", (c) => {
    const id = c.req.param("id")
    if (!id.match(/^req_/)) return c.json({ error: "Invalid request ID format" }, 400)
    try {
      return c.json(JSON.parse(readFileSync(join(LOG_DIR, "errors", `${id}.json`), "utf-8")))
    } catch { return c.json({ error: "Error dump not found", reqId: id }, 404) }
  })

  app.get("/debug/active", (c) => {
    const stats = traceStore.getStats()
    return c.json({ queue: { active: requestQueue.activeCount, waiting: requestQueue.waitingCount, max: MAX_CONCURRENT }, activeRequests: stats.activeRequests })
  })

  app.get("/debug/health", (c) => {
    const mem = process.memoryUsage()
    const stats = traceStore.getStats()
    return c.json({
      version: PROXY_VERSION, pid: process.pid, uptimeMs: stats.uptimeMs, uptimeHuman: stats.uptimeHuman,
      memory: {
        rss: `${(mem.rss / 1024 / 1024).toFixed(1)}MB`, heapUsed: `${(mem.heapUsed / 1024 / 1024).toFixed(1)}MB`,
        heapTotal: `${(mem.heapTotal / 1024 / 1024).toFixed(1)}MB`, external: `${(mem.external / 1024 / 1024).toFixed(1)}MB`,
        rssBytes: mem.rss, heapUsedBytes: mem.heapUsed,
      },
      queue: { active: requestQueue.activeCount, waiting: requestQueue.waitingCount, max: MAX_CONCURRENT },
      requests: stats.requests,
      config: { stallTimeoutMs: finalConfig.stallTimeoutMs, maxConcurrent: MAX_CONCURRENT, queueTimeoutMs: QUEUE_TIMEOUT_MS, debug: finalConfig.debug },
    })
  })

  // ── Model endpoints ──────────────────────────────────────────────────────

  const MODELS = [
    { type: "model", id: "claude-opus-4-6",              display_name: "Claude Opus 4.6",    created_at: "2025-08-01T00:00:00Z" },
    { type: "model", id: "claude-opus-4-6-20250801",     display_name: "Claude Opus 4.6",    created_at: "2025-08-01T00:00:00Z" },
    { type: "model", id: "claude-sonnet-4-6",            display_name: "Claude Sonnet 4.6",  created_at: "2025-08-01T00:00:00Z" },
    { type: "model", id: "claude-sonnet-4-6-20250801",   display_name: "Claude Sonnet 4.6",  created_at: "2025-08-01T00:00:00Z" },
    { type: "model", id: "claude-sonnet-4-5-20250929",   display_name: "Claude Sonnet 4.5",  created_at: "2025-09-29T00:00:00Z" },
    { type: "model", id: "claude-haiku-4-5",             display_name: "Claude Haiku 4.5",   created_at: "2025-10-01T00:00:00Z" },
    { type: "model", id: "claude-haiku-4-5-20251001",    display_name: "Claude Haiku 4.5",   created_at: "2025-10-01T00:00:00Z" },
  ]

  const MODELS_DUAL = MODELS.map(m => ({
    ...m, object: "model" as const,
    created: Math.floor(new Date(m.created_at).getTime() / 1000),
    owned_by: "anthropic" as const
  }))

  const handleModels = (c: Context) => c.json({ object: "list", data: MODELS_DUAL })
  app.get("/v1/models", handleModels)
  app.get("/models", handleModels)

  const handleModel = (c: Context) => {
    const id = c.req.param("id")
    const model = MODELS_DUAL.find(m => m.id === id)
    if (!model) return c.json({ type: "error", error: { type: "not_found_error", message: `Model \`${id}\` not found` } }, 404)
    return c.json(model)
  }
  app.get("/v1/models/:id", handleModel)
  app.get("/models/:id", handleModel)

  const handleCountTokens = async (c: Context) => {
    try {
      const body = await c.req.json()
      const sysText = Array.isArray(body.system)
        ? body.system.filter((b: any) => b.type === "text").map((b: any) => b.text).join("\n")
        : String(body.system ?? "")
      const msgText = (body.messages ?? []).map((m: any) => typeof m.content === "string" ? m.content : JSON.stringify(m.content)).join("\n")
      return c.json({ input_tokens: roughTokens(sysText + msgText) })
    } catch { return c.json({ input_tokens: 0 }) }
  }
  app.post("/v1/messages/count_tokens", handleCountTokens)
  app.post("/messages/count_tokens", handleCountTokens)

  // ── Messages handler ─────────────────────────────────────────────────────

  const handleMessages = async (c: Context) => {
    const reqId = generateId("req_")
    let trace: ReturnType<typeof traceStore.create> | undefined
    let requestStarted = Date.now()
    let clientDisconnected = false
    let abortReason: "stall" | "max_duration" | "max_output" | null = null

    try {
      let body: any
      try {
        body = await c.req.json()
      } catch {
        logWarn("request.invalid_json", { reqId })
        return c.json({ type: "error", error: { type: "invalid_request_error", message: "Request body must be valid JSON" }, request_id: reqId }, 400)
      }

      if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
        logWarn("request.missing_messages", { reqId })
        return c.json({ type: "error", error: { type: "invalid_request_error", message: "messages is required and must be a non-empty array" }, request_id: reqId }, 400)
      }

      const model = mapModelToClaudeModel(body.model || "sonnet")
      const stream = body.stream ?? false
      const hasTools = body.tools?.length > 0
      const abortController = new AbortController()

      // ── Timers ───────────────────────────────────────────────────────────
      let stallTimer: ReturnType<typeof setTimeout> | null = null
      const resetStallTimer = () => {
        if (stallTimer) clearTimeout(stallTimer)
        stallTimer = setTimeout(() => {
          abortReason = "stall"
          logWarn("request.stall_timeout", { reqId, stallTimeoutMs: finalConfig.stallTimeoutMs, phase: trace?.phase, sdkEventCount: trace?.sdkEventCount, outputLen: trace?.outputLen, lastEventType: trace?.lastEventType })
          abortController.abort()
        }, finalConfig.stallTimeoutMs)
      }
      const clearStallTimer = () => { if (stallTimer) { clearTimeout(stallTimer); stallTimer = null } }

      let hardTimer: ReturnType<typeof setTimeout> | null = null
      const startHardTimer = () => {
        hardTimer = setTimeout(() => {
          abortReason = "max_duration"
          logError("request.max_duration", { reqId, maxDurationMs: finalConfig.maxDurationMs, phase: trace?.phase, sdkEventCount: trace?.sdkEventCount, outputLen: trace?.outputLen, model: trace?.model, lastEventType: trace?.lastEventType })
          abortController.abort()
        }, finalConfig.maxDurationMs)
      }
      const clearHardTimer = () => { if (hardTimer) { clearTimeout(hardTimer); hardTimer = null } }

      const checkOutputSize = (outputLen: number) => {
        if (outputLen > finalConfig.maxOutputChars && !abortReason) {
          abortReason = "max_output"
          logError("request.max_output", { reqId, outputLen, maxOutputChars: finalConfig.maxOutputChars, phase: trace?.phase, sdkEventCount: trace?.sdkEventCount, model: trace?.model, elapsedMs: trace ? Date.now() - trace.startedAt : undefined })
          abortController.abort()
        }
      }

      const thinking: { type: "adaptive" } | { type: "enabled"; budgetTokens?: number } | { type: "disabled" } | undefined =
        body.thinking?.type === "enabled" ? { type: "enabled", budgetTokens: body.thinking.budget_tokens }
        : body.thinking?.type === "disabled" ? { type: "disabled" }
        : body.thinking?.type === "adaptive" ? { type: "adaptive" }
        : undefined

      let systemContext = ""
      if (body.system) {
        if (typeof body.system === "string") systemContext = body.system
        else if (Array.isArray(body.system)) systemContext = body.system.filter((b: any) => b.type === "text" && b.text).map((b: any) => b.text).join("\n")
      }

      const messages = body.messages as Array<{ role: string; content: string | Array<any> }>
      const mcpServer = hasTools ? createToolMcpServer(body.tools) : null
      const mcpServers = mcpServer ? { "proxy-tools": mcpServer } : undefined

      // ── Session resumption ─────────────────────────────────────────────
      const conversationId = c.req.header("x-conversation-id")
        ?? c.req.header("x-session-id")
        ?? extractConversationLabel(messages)
        ?? null

      if (conversationId) {
        logDebug("session.conversation_id_found", {
          reqId, conversationId,
          source: c.req.header("x-conversation-id") ? "header:x-conversation-id"
            : c.req.header("x-session-id") ? "header:x-session-id"
            : "message_extraction",
        })
      } else {
        logDebug("session.no_conversation_id", { reqId, msgCount: messages.length })
      }

      let resumeSessionId: string | undefined
      let isResuming = false
      let isCompacted = false

      if (conversationId) {
        const stored = sessionStore.get(conversationId)
        if (stored) {
          const countDelta = messages.length - stored.messageCount
          if (countDelta < 0) {
            const event = messages.length <= 2 ? "reset" : "compaction"
            if (event === "reset") {
              logInfo("session.likely_reset", { reqId, conversationId, model, sdkSessionId: stored.sdkSessionId, storedMsgCount: stored.messageCount, currentMsgCount: messages.length })
            } else {
              isCompacted = true
              logInfo("session.compaction_detected", { reqId, conversationId, model, sdkSessionId: stored.sdkSessionId, storedMsgCount: stored.messageCount, currentMsgCount: messages.length, dropped: -countDelta })
            }
            const prevContext = lastContextCache.get(conversationId)
            const dumpPath = dumpSessionContext(reqId, {
              event: `session.${event}`, conversationId, model, sdkSessionId: stored.sdkSessionId,
              storedMsgCount: stored.messageCount, currentMsgCount: messages.length,
              before: prevContext ? { systemPrompt: prevContext.systemPrompt, messages: prevContext.messages, model: prevContext.model, msgCount: prevContext.messages.length, capturedAt: new Date(prevContext.ts).toISOString() } : null,
              after: { systemPrompt: systemContext, messages, model, msgCount: messages.length },
            })
            logInfo("session.context_dumped", { reqId, event, path: dumpPath, msgCount: messages.length, hasBefore: !!prevContext })
            sessionStore.invalidate(conversationId)
          } else {
            resumeSessionId = stored.sdkSessionId
            isResuming = true
            logInfo("session.resuming", { reqId, conversationId, model, storedModel: stored.model, sdkSessionId: resumeSessionId, storedMsgCount: stored.messageCount, currentMsgCount: messages.length, resumeCount: stored.resumeCount, countDelta })
          }
        }
      }

      if (conversationId) {
        lastContextCache.set(conversationId, { systemPrompt: systemContext, messages, model, ts: Date.now() })
        if (lastContextCache.size > MAX_CONTEXT_CACHE) {
          const oldest = [...lastContextCache.entries()].sort((a, b) => a[1].ts - b[1].ts)[0]
          if (oldest) lastContextCache.delete(oldest[0])
        }
      }

      // ── Build prompt input ─────────────────────────────────────────────
      const lastMsg = messages[messages.length - 1]!
      const lastMsgHasImages = contentHasImages(lastMsg.content)
      const promptText = serializeContent(lastMsg.content)
      const systemPrompt = (systemContext || "").trim() || undefined
      const usage = newUsage()

      let promptInput: string | AsyncIterable<any>
      if (isResuming && resumeSessionId) {
        promptInput = lastMsgHasImages
          ? createSDKUserMessage(buildNativeContent(lastMsg.content), resumeSessionId)
          : promptText
        if (lastMsgHasImages) logInfo("session.resume_with_images", { reqId, conversationId })
      } else {
        promptInput = lastMsgHasImages ? createSDKUserMessage(buildNativeContent(lastMsg.content)) : promptText
        if (lastMsgHasImages) logInfo("request.native_images", { reqId })
      }

      requestStarted = Date.now()
      const clientIp = c.req.header("x-forwarded-for") ?? c.req.header("x-real-ip") ?? c.req.header("cf-connecting-ip") ?? "unknown"
      const userAgent = c.req.header("user-agent") ?? "unknown"
      const bodyBytes = JSON.stringify(body).length

      trace = traceStore.create({ reqId, model, requestedModel: body.model || "sonnet", stream, hasTools, thinking: thinking?.type, promptLen: promptText.length, systemLen: systemPrompt?.length ?? 0, msgCount: messages.length, bodyBytes, clientIp, userAgent })
      if (conversationId) {
        traceStore.setSession(reqId, { conversationId, sdkSessionId: resumeSessionId, isResuming, resumeCount: isResuming ? sessionStore.get(conversationId!)?.resumeCount : undefined })
      }

      // ── Queue ─────────────────────────────────────────────────────────────
      traceStore.phase(reqId, "queued", { queueActive: requestQueue.activeCount, queueWaiting: requestQueue.waitingCount })
      if (requestQueue.activeCount >= MAX_CONCURRENT) {
        logInfo("queue.waiting", { reqId, model, queueActive: requestQueue.activeCount, queueWaiting: requestQueue.waitingCount, queueTimeoutMs: QUEUE_TIMEOUT_MS })
      }
      await requestQueue.acquire()
      const queueWaitMs = Date.now() - requestStarted
      traceStore.phase(reqId, "acquired", { queueWaitMs })
      logInfo("queue.acquired", { reqId, queueWaitMs, queueActive: requestQueue.activeCount, queueWaiting: requestQueue.waitingCount })
      resetStallTimer()
      startHardTimer()

      // ── Non-streaming ──────────────────────────────────────────────────
      if (!stream) {
        let result: Awaited<ReturnType<typeof runNonStreamSdkLoop>>
        let capturedSessionId: string | undefined
        try {
          traceStore.phase(reqId, "sdk_starting")
          result = await runNonStreamSdkLoop(promptInput, buildQueryOptions(model, { partial: hasTools, systemPrompt, abortController, thinking, resume: resumeSessionId, mcpServers }), hasTools, abortController, reqId, usage, resetStallTimer)
          capturedSessionId = result.sessionId
          traceStore.phase(reqId, "sdk_done")
          logInfo("request.content_summary", { reqId, hasTools, contentBlockCount: result.toolState.contentBlocks.length, textLength: result.fullText.length, stopReason: result.toolState.capturedStopReason, sdkEventCount: result.sdkEventCount, aborted: result.aborted })
          saveSession(conversationId, capturedSessionId, isResuming, isCompacted, messages.length, model, reqId, isResuming ? "session.resumed_ok" : "session.created")
        } catch (err) {
          if (isResuming && resumeSessionId) {
            logWarn("session.resume_failed", { reqId, conversationId, sdkSessionId: resumeSessionId, error: err instanceof Error ? (err as Error).message : String(err) })
            if (conversationId) { sessionStore.recordFailure(conversationId!); sessionStore.invalidate(conversationId!) }
            logInfo("session.fallback_fresh", { reqId, conversationId })
            const fallbackAbort = new AbortController()
            traceStore.phase(reqId, "sdk_starting")
            result = await runNonStreamSdkLoop(promptInput, buildQueryOptions(model, { partial: hasTools, systemPrompt, abortController: fallbackAbort, thinking, mcpServers }), hasTools, fallbackAbort, reqId, usage, resetStallTimer)
            capturedSessionId = result.sessionId
            traceStore.phase(reqId, "sdk_done")
            logInfo("request.content_summary", { reqId, hasTools, contentBlockCount: result.toolState.contentBlocks.length, textLength: result.fullText.length, stopReason: result.toolState.capturedStopReason, sdkEventCount: result.sdkEventCount, aborted: result.aborted })
            saveSession(conversationId, capturedSessionId, false, false, messages.length, model, reqId, "session.created_after_fallback")
          } else {
            throw err
          }
        } finally {
          clearStallTimer(); clearHardTimer()
          requestQueue.release()
          logDebug("queue.released", { reqId, queueActive: requestQueue.activeCount, queueWaiting: requestQueue.waitingCount })
        }

        traceStore.phase(reqId, "responding")

        if (hasTools) {
          const { contentBlocks, capturedStopReason } = result!.toolState
          const toolCallCount = contentBlocks.filter((b: any) => b.type === "tool_use").length
          const content = contentBlocks.filter((b: any) => !(b.type === "text" && !b.text?.trim()))
          if (content.length === 0) content.push({ type: "text", text: result!.fullText || "..." })
          const stopReason = toolCallCount > 0 ? "tool_use" : (capturedStopReason ?? "end_turn")
          traceStore.setUsage(reqId, makeUsageObj(usage))
          traceStore.complete(reqId, { outputLen: result!.fullText.length, toolCallCount })
          return c.json({ id: generateId("msg_"), type: "message", role: "assistant", content, model: body.model, stop_reason: stopReason, stop_sequence: null, usage: buildUsage(usage.inputTokens, usage.outputTokens, usage.cacheReadTokens, usage.cacheCreationTokens) })
        }

        const fullText = result!.fullText || "..."
        traceStore.setUsage(reqId, makeUsageObj(usage))
        traceStore.complete(reqId, { outputLen: fullText.length })
        return c.json({ id: generateId("msg_"), type: "message", role: "assistant", content: [{ type: "text", text: fullText }], model: body.model, stop_reason: "end_turn", stop_sequence: null, usage: buildUsage(usage.inputTokens, usage.outputTokens, usage.cacheReadTokens, usage.cacheCreationTokens) })
      }

      // ── Streaming ──────────────────────────────────────────────────────
      const encoder = new TextEncoder()
      const readable = new ReadableStream({
        cancel() {
          clientDisconnected = true
          logWarn("stream.client_disconnect", { reqId, phase: trace?.phase, sdkEventCount: trace?.sdkEventCount, outputLen: trace?.outputLen, elapsedMs: trace ? Date.now() - trace.startedAt : undefined, model: trace?.model })
          abortController.abort()
        },
        async start(controller) {
          const messageId = generateId("msg_")
          let queueReleased = false
          const releaseQueue = () => {
            if (!queueReleased) {
              queueReleased = true
              requestQueue.release()
              logDebug("queue.released", { reqId, queueActive: requestQueue.activeCount, queueWaiting: requestQueue.waitingCount })
            }
          }

          let sseSendErrors = 0
          const sse = (event: string, data: object) => {
            try {
              controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`))
            } catch (e) {
              sseSendErrors++
              if (sseSendErrors <= 3) logWarn("stream.sse_send_failed", { reqId, event, sseSendErrors, error: e instanceof Error ? e.message : String(e) })
            }
          }

          try {
            const heartbeat = setInterval(() => {
              try { controller.enqueue(encoder.encode(`event: ping\ndata: {"type": "ping"}\n\n`)) }
              catch (e) { logWarn("stream.heartbeat_failed", { reqId, error: e instanceof Error ? e.message : String(e), phase: trace?.phase, elapsedMs: trace ? Date.now() - trace.startedAt : undefined }); clearInterval(heartbeat) }
            }, 15_000)

            let messageStartSent = false
            const emitMessageStart = () => {
              if (!messageStartSent) {
                messageStartSent = true
                sse("message_start", { type: "message_start", message: { id: messageId, type: "message", role: "assistant", content: [], model: body.model, stop_reason: null, stop_sequence: null, usage: buildUsage(usage.inputTokens, 1, usage.cacheReadTokens, usage.cacheCreationTokens) } })
              }
            }

            if (hasTools) {
              // ── Streaming with tools ─────────────────────────────────────
              let fullText = ""
              let blockIdx = 0
              let toolCallCount = 0
              let capturedStopReason: string | null = null
              let hasEmittedAnyBlock = false
              let sdkEventCount = 0
              let lastEventAt = Date.now()
              const stallLog = setInterval(() => { traceStore.stall(reqId, Date.now() - lastEventAt) }, 15_000)
              let capturedSessionId: string | undefined
              let currentBlockEmitted = false
              let blockOpen = false
              let sseEventCount = 0

              const forwardStreamEvent = (ev: any) => {
                sseEventCount++
                captureMessageStartUsage(ev, usage)
                if (ev.type === "message_start") { emitMessageStart(); return }
                emitMessageStart()
                if (ev.type === "content_block_start") {
                  const cb = ev.content_block
                  if (cb?.type === "tool_use") {
                    toolCallCount++
                    currentBlockEmitted = true
                    blockOpen = true
                    sse("content_block_start", { type: "content_block_start", index: blockIdx, content_block: { type: "tool_use", id: cb.id, name: stripMcpPrefix(cb.name), input: {} } })
                  } else if (cb?.type === "text") {
                    currentBlockEmitted = true
                    blockOpen = true
                    sse("content_block_start", { type: "content_block_start", index: blockIdx, content_block: { type: "text", text: "" } })
                  } else {
                    currentBlockEmitted = false
                  }
                  hasEmittedAnyBlock = hasEmittedAnyBlock || currentBlockEmitted
                } else if (ev.type === "content_block_delta") {
                  if (!currentBlockEmitted) return
                  if (ev.delta?.type === "text_delta") {
                    fullText += ev.delta.text ?? ""
                    traceStore.updateOutput(reqId, fullText.length)
                    checkOutputSize(fullText.length)
                    sse("content_block_delta", { type: "content_block_delta", index: blockIdx, delta: { type: "text_delta", text: ev.delta.text ?? "" } })
                  } else if (ev.delta?.type === "input_json_delta") {
                    sse("content_block_delta", { type: "content_block_delta", index: blockIdx, delta: { type: "input_json_delta", partial_json: ev.delta.partial_json ?? "" } })
                  }
                } else if (ev.type === "content_block_stop") {
                  if (!currentBlockEmitted) return
                  sse("content_block_stop", { type: "content_block_stop", index: blockIdx })
                  blockIdx++
                  blockOpen = false
                } else if (ev.type === "message_delta") {
                  if (ev.delta?.stop_reason) capturedStopReason = ev.delta.stop_reason
                  if (ev.usage?.output_tokens) usage.outputTokens = ev.usage.output_tokens
                }
              }

              let activeAbortController = abortController

              const runStreamToolLoop = async (ac: AbortController, opts: any) => {
                for await (const message of query({ prompt: promptInput, options: opts })) {
                  sdkEventCount++
                  lastEventAt = Date.now()
                  resetStallTimer()
                  const subtype = (message as any).event?.type ?? (message as any).message?.type
                  const sid = processSdkMessage(message, usage)
                  if (sid) capturedSessionId = sid
                  if (message.type === "stream_event") {
                    const ev = (message as any).event as any
                    if (!trace!.firstTokenAt && (ev.type === "content_block_delta" || ev.type === "content_block_start")) traceStore.phase(reqId, "sdk_streaming")
                    forwardStreamEvent(ev)
                    if (capturedStopReason && toolCallCount > 0 && !ac.signal.aborted) {
                      logInfo("sdk.abort_after_complete_stream", { reqId, toolCallCount, blockIdx, stopReason: capturedStopReason })
                      ac.abort()
                    }
                  }
                  traceStore.sdkEvent(reqId, sdkEventCount, message.type, subtype)
                }
              }

              try {
                traceStore.phase(reqId, "sdk_starting")
                await runStreamToolLoop(abortController, buildQueryOptions(model, { partial: true, systemPrompt, abortController, thinking, resume: resumeSessionId, mcpServers }))
                traceStore.phase(reqId, "sdk_done")
                logInfo("request.content_summary", { reqId, hasTools: true, blockIdx, toolCallCount, textLength: fullText.length, stopReason: capturedStopReason, sdkEventCount })
                saveSession(conversationId, capturedSessionId, isResuming, isCompacted, messages.length, model, reqId)
              } catch (resumeErr) {
                if (activeAbortController.signal.aborted && toolCallCount > 0) {
                  traceStore.phase(reqId, "sdk_done")
                  logInfo("sdk.aborted_after_tool_call", { reqId, sdkEventCount, toolCallCount, outputLen: fullText.length })
                  logInfo("request.content_summary", { reqId, hasTools: true, blockIdx, toolCallCount, textLength: fullText.length, stopReason: capturedStopReason, sdkEventCount })
                  saveSession(conversationId, capturedSessionId, isResuming, isCompacted, messages.length, model, reqId)
                } else if (isResuming && resumeSessionId) {
                  logWarn("session.resume_failed_stream", { reqId, conversationId, sdkSessionId: resumeSessionId, error: resumeErr instanceof Error ? resumeErr.message : String(resumeErr) })
                  if (conversationId) { sessionStore.recordFailure(conversationId!); sessionStore.invalidate(conversationId!) }
                  logInfo("session.fallback_fresh_stream", { reqId, conversationId })
                  fullText = ""; blockIdx = 0; toolCallCount = 0; capturedStopReason = null; hasEmittedAnyBlock = false; sdkEventCount = 0
                  const fallbackAbort = new AbortController()
                  activeAbortController = fallbackAbort
                  try {
                    await runStreamToolLoop(fallbackAbort, buildQueryOptions(model, { partial: true, systemPrompt, abortController: fallbackAbort, thinking, mcpServers }))
                    traceStore.phase(reqId, "sdk_done")
                    logInfo("request.content_summary", { reqId, hasTools: true, blockIdx, toolCallCount, textLength: fullText.length, stopReason: capturedStopReason, sdkEventCount })
                    saveSession(conversationId, capturedSessionId, false, false, messages.length, model, reqId, "session.created_after_fallback_stream")
                  } catch (fallbackErr) {
                    if (fallbackAbort.signal.aborted && toolCallCount > 0) {
                      traceStore.phase(reqId, "sdk_done")
                      logInfo("sdk.aborted_after_tool_call_fallback_stream", { reqId, sdkEventCount, toolCallCount, outputLen: fullText.length })
                      logInfo("request.content_summary", { reqId, hasTools: true, blockIdx, toolCallCount, textLength: fullText.length, stopReason: capturedStopReason, sdkEventCount })
                      saveSession(conversationId, capturedSessionId, false, false, messages.length, model, reqId)
                    } else { throw fallbackErr }
                  }
                } else { throw resumeErr }
              } finally {
                clearInterval(stallLog); clearInterval(heartbeat); clearStallTimer(); clearHardTimer(); releaseQueue()
              }

              traceStore.phase(reqId, "responding")
              emitMessageStart()
              if (!hasEmittedAnyBlock) {
                sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })
                sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "..." } })
                sse("content_block_stop", { type: "content_block_stop", index: 0 })
              }
              if (blockOpen) {
                logWarn("stream.closing_orphaned_block", { reqId, blockIdx })
                sse("content_block_stop", { type: "content_block_stop", index: blockIdx })
                blockIdx++; blockOpen = false
              }
              const stopReason = toolCallCount > 0 ? "tool_use" : (capturedStopReason ?? "end_turn")
              sse("message_delta", { type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: usage.outputTokens } })
              sse("message_stop", { type: "message_stop" })
              logInfo("sse.summary", { reqId, sseEventCount, blockIdx, toolCallCount })
              controller.close()
              traceStore.setUsage(reqId, makeUsageObj(usage))
              traceStore.complete(reqId, { outputLen: fullText.length, toolCallCount })
              return
            }

            // ── Streaming without tools ─────────────────────────────────────
            let contentBlockStartSent = false
            let fullText = ""
            let hasStreamed = false
            let sdkEventCount = 0
            let lastEventAt = Date.now()
            let capturedSessionId2: string | undefined
            const stallLog = setInterval(() => { traceStore.stall(reqId, Date.now() - lastEventAt) }, 15_000)

            const processStreamNoToolsEvent = (ev: any) => {
              captureMessageStartUsage(ev, usage)
              if (ev.type === "message_start") { emitMessageStart(); return }
              if (!trace!.firstTokenAt && (ev.type === "content_block_delta" || ev.type === "content_block_start")) traceStore.phase(reqId, "sdk_streaming")
              if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") {
                const text = ev.delta.text ?? ""
                if (text) {
                  emitMessageStart()
                  if (!contentBlockStartSent) { contentBlockStartSent = true; sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }) }
                  fullText += text
                  hasStreamed = true
                  traceStore.updateOutput(reqId, fullText.length)
                  checkOutputSize(fullText.length)
                  sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } })
                }
              } else if (ev.type === "message_delta" && ev.usage?.output_tokens) {
                usage.outputTokens = ev.usage.output_tokens
              }
            }

            const runStreamNoToolsLoop = async (opts: any) => {
              for await (const message of query({ prompt: promptInput, options: opts })) {
                sdkEventCount++
                lastEventAt = Date.now()
                resetStallTimer()
                const subtype = (message as any).event?.type ?? (message as any).message?.type
                const sid = processSdkMessage(message, usage)
                if (sid) capturedSessionId2 = sid
                if (message.type === "stream_event") processStreamNoToolsEvent((message as any).event as any)
                traceStore.sdkEvent(reqId, sdkEventCount, message.type, subtype)
              }
            }

            try {
              traceStore.phase(reqId, "sdk_starting")
              await runStreamNoToolsLoop(buildQueryOptions(model, { partial: true, systemPrompt, abortController, thinking, resume: resumeSessionId }))
              traceStore.phase(reqId, "sdk_done")
              saveSession(conversationId, capturedSessionId2, isResuming, isCompacted, messages.length, model, reqId)
            } catch (resumeErr) {
              if (isResuming && resumeSessionId) {
                logWarn("session.resume_failed_stream", { reqId, conversationId, sdkSessionId: resumeSessionId, error: resumeErr instanceof Error ? resumeErr.message : String(resumeErr) })
                if (conversationId) { sessionStore.recordFailure(conversationId!); sessionStore.invalidate(conversationId!) }
                logInfo("session.fallback_fresh_stream", { reqId, conversationId })
                sdkEventCount = 0
                await runStreamNoToolsLoop(buildQueryOptions(model, { partial: true, systemPrompt, abortController, thinking }))
                traceStore.phase(reqId, "sdk_done")
                saveSession(conversationId, capturedSessionId2, false, false, messages.length, model, reqId, "session.created_after_fallback_stream")
              } else { throw resumeErr }
            } finally {
              clearInterval(stallLog); clearInterval(heartbeat); clearStallTimer(); clearHardTimer(); releaseQueue()
            }

            emitMessageStart()
            if (!contentBlockStartSent) { contentBlockStartSent = true; sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }) }
            if (!hasStreamed) sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "..." } })
            sse("content_block_stop", { type: "content_block_stop", index: 0 })
            sse("message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: usage.outputTokens } })
            sse("message_stop", { type: "message_stop" })
            controller.close()
            traceStore.setUsage(reqId, makeUsageObj(usage))
            traceStore.complete(reqId, { outputLen: fullText.length })

          } catch (error) {
            clearStallTimer(); clearHardTimer(); releaseQueue()
            const err = error instanceof Error ? error : new Error(String(error))
            const classified = classifyError(err, { clientDisconnected, abortReason, stallTimeoutMs: finalConfig.stallTimeoutMs, maxDurationMs: finalConfig.maxDurationMs, maxOutputChars: finalConfig.maxOutputChars, outputLen: trace?.outputLen })
            traceStore.fail(reqId, err, "error", { clientDisconnect: clientDisconnected, abortReason, aborted: err.name === "AbortError" || err.message?.includes("abort"), queueTimeout: err.message.includes("Queue timeout"), stallTimeoutMs: finalConfig.stallTimeoutMs, maxDurationMs: finalConfig.maxDurationMs, maxOutputChars: finalConfig.maxOutputChars, sseSendErrors })
            if (!clientDisconnected) {
              try { sse("error", { type: "error", error: { type: classified.type, message: classified.message }, request_id: reqId }); controller.close() } catch {}
            } else {
              try { controller.close() } catch {}
            }
          }
        }
      })

      return new Response(readable, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive" } })

    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error))
      const classified = classifyError(err, { clientDisconnected, abortReason, stallTimeoutMs: finalConfig.stallTimeoutMs, maxDurationMs: finalConfig.maxDurationMs, maxOutputChars: finalConfig.maxOutputChars })
      if (trace) {
        traceStore.fail(reqId, err, "error", { clientDisconnect: clientDisconnected, aborted: err.name === "AbortError" || err.message?.includes("abort"), queueTimeout: err.message.includes("Queue timeout") })
      } else {
        logError("request.error.no_trace", { reqId, error: classified.message, stack: err.stack })
      }
      return new Response(JSON.stringify({ type: "error", error: { type: classified.type, message: classified.message }, request_id: reqId }), {
        status: classified.status, headers: { "Content-Type": "application/json" }
      })
    }
  }

  app.post("/v1/messages", handleMessages)
  app.post("/messages", handleMessages)

  // Stub: batches API not supported
  const handleBatches = (c: Context) => c.json({ type: "error", error: { type: "not_implemented_error", message: "Batches API is not supported by this proxy" } }, 501)
  app.post("/v1/messages/batches", handleBatches)
  app.get("/v1/messages/batches", handleBatches)
  app.get("/v1/messages/batches/:id", handleBatches)

  // ── OpenAI-compatible /v1/chat/completions ─────────────────────────────

  const handleChatCompletions = async (c: Context) => {
    try {
      let body: any
      try { body = await c.req.json() }
      catch { return c.json({ error: { message: "Request body must be valid JSON", type: "invalid_request_error" } }, 400) }

      if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
        return c.json({ error: { message: "messages is required and must be a non-empty array", type: "invalid_request_error" } }, 400)
      }

      const { system, messages } = openaiToAnthropicMessages(body.messages)
      const stream = body.stream ?? false
      const requestedModel = body.model ?? "claude-sonnet-4-6"

      const anthropicBody: any = { model: requestedModel, messages, stream }
      if (system) anthropicBody.system = system
      if (body.max_tokens || body.max_completion_tokens) anthropicBody.max_tokens = body.max_tokens ?? body.max_completion_tokens
      if (body.temperature !== undefined) anthropicBody.temperature = body.temperature
      if (body.top_p !== undefined) anthropicBody.top_p = body.top_p
      if (body.stop) anthropicBody.stop_sequences = Array.isArray(body.stop) ? body.stop : [body.stop]
      if (body.tools?.length) anthropicBody.tools = openaiToAnthropicTools(body.tools)

      const internalHeaders: Record<string, string> = { "Content-Type": "application/json" }
      const authHeader = c.req.header("authorization") ?? c.req.header("x-api-key")
      if (authHeader) {
        if (c.req.header("authorization")) internalHeaders["authorization"] = authHeader
        else internalHeaders["x-api-key"] = authHeader
      }
      const internalRes = await app.fetch(new Request(`http://localhost/v1/messages`, { method: "POST", headers: internalHeaders, body: JSON.stringify(anthropicBody) }))

      if (!stream) {
        const anthropicJson = await internalRes.json() as any
        if (anthropicJson.type === "error") return c.json({ error: anthropicJson.error }, internalRes.status as any)
        return c.json(anthropicToOpenaiResponse(anthropicJson, requestedModel, generateId("chatcmpl-")))
      }

      const includeUsage = body.stream_options?.include_usage === true
      const readable = new ReadableStream({
        async start(controller) {
          try {
            const reader = internalRes.body?.getReader()
            if (!reader) { controller.close(); return }
            const decoder = new TextDecoder()
            let buffer = ""
            const chatId = generateId("chatcmpl-")
            const created = Math.floor(Date.now() / 1000)
            let sentRole = false
            let finishReason: string | null = null
            const activeToolCalls: Map<number, { id: string; name: string }> = new Map()
            let toolCallIndex = 0
            let usageInfo: { input_tokens: number; output_tokens: number } | null = null

            while (true) {
              const { done, value } = await reader.read()
              if (done) break
              buffer += decoder.decode(value, { stream: true })
              const lines = buffer.split("\n")
              buffer = lines.pop() ?? ""

              for (const line of lines) {
                if (!line.startsWith("data: ")) continue
                try {
                  const event = JSON.parse(line.slice(6))

                  if (!sentRole && (event.type === "content_block_start" || event.type === "content_block_delta")) {
                    sentRole = true
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ id: chatId, object: "chat.completion.chunk", created, model: requestedModel, choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] })}\n\n`))
                  }

                  if (event.type === "content_block_start" && event.content_block?.type === "tool_use") {
                    const idx = toolCallIndex++
                    activeToolCalls.set(event.index, { id: event.content_block.id, name: event.content_block.name })
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ id: chatId, object: "chat.completion.chunk", created, model: requestedModel, choices: [{ index: 0, delta: { tool_calls: [{ index: idx, id: event.content_block.id, type: "function", function: { name: event.content_block.name, arguments: "" } }] }, finish_reason: null }] })}\n\n`))
                  } else if (event.type === "content_block_delta" && event.delta?.type === "input_json_delta") {
                    const tc = activeToolCalls.get(event.index)
                    if (tc) {
                      const idx = Array.from(activeToolCalls.keys()).indexOf(event.index)
                      controller.enqueue(encoder.encode(`data: ${JSON.stringify({ id: chatId, object: "chat.completion.chunk", created, model: requestedModel, choices: [{ index: 0, delta: { tool_calls: [{ index: idx, function: { arguments: event.delta.partial_json } }] }, finish_reason: null }] })}\n\n`))
                    }
                  } else if (event.type === "content_block_delta" && event.delta?.type === "text_delta") {
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ id: chatId, object: "chat.completion.chunk", created, model: requestedModel, choices: [{ index: 0, delta: { content: event.delta.text }, finish_reason: null }] })}\n\n`))
                  } else if (event.type === "message_delta") {
                    const sr = event.delta?.stop_reason
                    finishReason = sr === "tool_use" ? "tool_calls" : sr === "max_tokens" ? "length" : "stop"
                    if (event.usage) {
                      usageInfo = { input_tokens: event.usage.input_tokens ?? usageInfo?.input_tokens ?? 0, output_tokens: event.usage.output_tokens ?? usageInfo?.output_tokens ?? 0 }
                    }
                  } else if (event.type === "message_start" && event.message?.usage) {
                    usageInfo = { input_tokens: event.message.usage.input_tokens ?? 0, output_tokens: 0 }
                  } else if (event.type === "message_stop") {
                    const finalChunk: any = { id: chatId, object: "chat.completion.chunk", created, model: requestedModel, choices: [{ index: 0, delta: {}, finish_reason: finishReason ?? "stop" }] }
                    if (includeUsage && usageInfo) {
                      finalChunk.usage = { prompt_tokens: usageInfo.input_tokens, completion_tokens: usageInfo.output_tokens, total_tokens: usageInfo.input_tokens + usageInfo.output_tokens }
                    }
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify(finalChunk)}\n\n`))
                    controller.enqueue(encoder.encode("data: [DONE]\n\n"))
                  }
                } catch {}
              }
            }
            controller.close()
          } catch { controller.close() }
        }
      })

      return new Response(readable, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive" } })
    } catch (error) {
      return c.json({ error: { message: error instanceof Error ? error.message : "Unknown error", type: "server_error" } }, 500)
    }
  }

  app.post("/v1/chat/completions", handleChatCompletions)
  app.post("/chat/completions", handleChatCompletions)
  app.get("/v1/chat/models", (c) => c.json({ object: "list", data: MODELS.map(m => ({ id: m.id, object: "model", created: Math.floor(new Date(m.created_at).getTime() / 1000), owned_by: "anthropic" })) }))

  app.all("*", (c) => c.json({ type: "error", error: { type: "not_found_error", message: `${c.req.method} ${c.req.path} not found` } }, 404))

  return { app, config: finalConfig }
}

export async function startProxyServer(config: Partial<ProxyConfig> = {}) {
  const { app, config: finalConfig } = createProxyServer(config)

  const server = Bun.serve({ port: finalConfig.port, hostname: finalConfig.host, fetch: app.fetch, idleTimeout: 0 })

  logInfo("proxy.started", { version: PROXY_VERSION, host: finalConfig.host, port: finalConfig.port, stallTimeoutMs: finalConfig.stallTimeoutMs, maxDurationMs: finalConfig.maxDurationMs, maxOutputChars: finalConfig.maxOutputChars, maxConcurrent: MAX_CONCURRENT, queueTimeoutMs: QUEUE_TIMEOUT_MS, claudeExecutable, logDir: LOG_DIR, debug: finalConfig.debug, pid: process.pid })
  console.log(`Claude SDK Proxy v${PROXY_VERSION} running at http://${finalConfig.host}:${finalConfig.port}`)
  console.log(`  Logs: ${LOG_DIR}`)
  console.log(`  Debug: http://${finalConfig.host}:${finalConfig.port}/debug/stats`)

  const healthInterval = setInterval(() => {
    const mem = process.memoryUsage()
    const stats = traceStore.getStats()
    logInfo("proxy.health", { pid: process.pid, rssBytes: mem.rss, rssMB: +(mem.rss / 1024 / 1024).toFixed(1), heapUsedMB: +(mem.heapUsed / 1024 / 1024).toFixed(1), heapTotalMB: +(mem.heapTotal / 1024 / 1024).toFixed(1), externalMB: +(mem.external / 1024 / 1024).toFixed(1), uptimeMs: stats.uptimeMs, totalRequests: stats.requests.total, totalErrors: stats.requests.errors, activeRequests: stats.requests.active, queueActive: requestQueue.activeCount, queueWaiting: requestQueue.waitingCount })
  }, 300_000)

  const shutdown = (signal: string) => {
    const stats = traceStore.getStats()
    logInfo("proxy.shutdown", { signal, pid: process.pid, totalRequests: stats.requests.total, totalErrors: stats.requests.errors, activeRequests: stats.requests.active, uptimeMs: stats.uptimeMs })
    clearInterval(healthInterval)
    console.log(`\nReceived ${signal}, shutting down...`)
    server.stop(true)
    process.exit(0)
  }
  process.on("SIGINT", () => shutdown("SIGINT"))
  process.on("SIGTERM", () => shutdown("SIGTERM"))

  return server
}
