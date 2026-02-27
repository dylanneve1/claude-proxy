import { Hono } from "hono"
import { cors } from "hono/cors"
import { query } from "@anthropic-ai/claude-agent-sdk"
import type { Context } from "hono"
import type { ProxyConfig } from "./types"
import { DEFAULT_PROXY_CONFIG } from "./types"
import { logInfo, logWarn, logError, logDebug, LOG_DIR } from "../logger"
import { traceStore } from "../trace"
import { sessionStore } from "../session-store"
import { createToolMcpServer } from "../mcpTools"
import { execSync } from "child_process"
import { existsSync, writeFileSync, readFileSync, readdirSync } from "fs"
import { randomBytes } from "crypto"
import { fileURLToPath } from "url"
import { join, dirname } from "path"

// Base62 ID generator — matches Anthropic's real ID format (e.g. msg_01XFDUDYJgAACzvnptvVoYEL)
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
// Limits simultaneous Claude SDK sessions to prevent resource exhaustion.

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

function mapModelToClaudeModel(model: string): "sonnet" | "opus" | "haiku" {
  if (model.includes("opus")) return "opus"
  if (model.includes("haiku")) return "haiku"
  return "sonnet"
}

// ── Content-block serialization ──────────────────────────────────────────────

function serializeBlock(block: any): string {
  switch (block.type) {
    case "text":
      return block.text || ""
    case "image":
      return "[Image attached]"
    case "tool_use":
      return `<tool_use>\n{"name": "${block.name}", "input": ${JSON.stringify(block.input ?? {})}}\n</tool_use>`
    case "tool_result": {
      const content = Array.isArray(block.content)
        ? block.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("")
        : String(block.content ?? "")
      const truncated = content.length > 4000
        ? content.slice(0, 4000) + `\n...[truncated ${content.length - 4000} chars]`
        : content
      return `[Tool Result (id: ${block.tool_use_id})]\n${truncated}\n[/Tool Result]`
    }
    case "thinking":
      return ""
    default:
      return ""
  }
}

function serializeContent(content: string | Array<any>): string {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return String(content)
  return content.map(b => serializeBlock(b)).filter(Boolean).join("\n")
}

// ── Image handling via SDKUserMessage ────────────────────────────────────────
// The SDK query() accepts AsyncIterable<SDKUserMessage> which supports native
// Anthropic MessageParam content blocks including images. When images are
// detected, we pass them through natively instead of serializing to text.

function contentHasImages(content: string | Array<any>): boolean {
  if (typeof content === "string") return false
  if (!Array.isArray(content)) return false
  return content.some((b: any) => b.type === "image")
}

/** Convert an Anthropic image content block to SDK-compatible format */
function toAnthropicImageBlock(block: any): any {
  if (block.source) return block // already in Anthropic format
  // openclaw may use { type: "image", data: "...", mimeType: "..." }
  if (block.data && block.mimeType) {
    return {
      type: "image",
      source: {
        type: "base64",
        media_type: block.mimeType,
        data: block.data,
      }
    }
  }
  if (block.data && block.media_type) {
    return {
      type: "image",
      source: {
        type: "base64",
        media_type: block.media_type,
        data: block.data,
      }
    }
  }
  return block
}

/** Build Anthropic MessageParam content array, preserving images natively */
function buildNativeContent(content: string | Array<any>): Array<any> {
  if (typeof content === "string") return [{ type: "text", text: content }]
  if (!Array.isArray(content)) return [{ type: "text", text: String(content) }]
  return content.map((block: any) => {
    if (block.type === "image") return toAnthropicImageBlock(block)
    if (block.type === "text") return { type: "text", text: block.text ?? "" }
    // For other types, serialize to text
    const serialized = serializeBlock(block)
    return serialized ? { type: "text", text: serialized } : null
  }).filter(Boolean)
}

/** Create an async iterable yielding a single SDKUserMessage with native content */
function createSDKUserMessage(content: Array<any>, sessionId?: string): AsyncIterable<any> {
  const msg = {
    type: "user" as const,
    message: {
      role: "user" as const,
      content,
    },
    parent_tool_use_id: null,
    session_id: sessionId ?? "",
  }
  return {
    async *[Symbol.asyncIterator]() {
      yield msg
    }
  }
}


// ── Client tool-use support (native MCP) ────────────────────────────────────
// Tool definitions from the client are converted to native MCP tools via
// createToolMcpServer(). The SDK generates proper tool_use content blocks
// instead of text-based XML. canUseTool denies all execution since the
// client handles tool execution.

// The SDK prefixes MCP tool names with "mcp__{server-name}__".
// Strip this prefix so clients see the original tool names they sent.
const MCP_TOOL_PREFIX = "mcp__proxy-tools__"
function stripMcpPrefix(name: string): string {
  return name.startsWith(MCP_TOOL_PREFIX) ? name.slice(MCP_TOOL_PREFIX.length) : name
}

function roughTokens(text: string): number {
  return Math.ceil((text ?? "").length / 4)
}

function buildUsage(input: number, output: number, cacheRead: number, cacheCreation: number) {
  const usage: Record<string, number> = { input_tokens: input, output_tokens: output }
  if (cacheRead > 0) usage.cache_read_input_tokens = cacheRead
  if (cacheCreation > 0) usage.cache_creation_input_tokens = cacheCreation
  return usage
}

// ── Conversation label extraction ────────────────────────────────────────────
// Openclaw embeds "Conversation info (untrusted metadata)" in the last user
// message containing a JSON block with conversation_label. Extract it to use
// as a stable conversation ID for session persistence.

function extractConversationLabel(messages: Array<{ role: string; content: string | Array<any> }>): string | null {
  // Search from the last message backwards for a user message with metadata
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!
    if (msg.role !== "user") continue

    const text = typeof msg.content === "string"
      ? msg.content
      : Array.isArray(msg.content)
        ? msg.content.filter((b: any) => b.type === "text").map((b: any) => b.text ?? "").join("\n")
        : ""

    // Look for the JSON block after "Conversation info"
    const jsonMatch = text.match(/Conversation info[^`]*```json\s*(\{[\s\S]*?\})\s*```/)
    if (!jsonMatch?.[1]) continue

    try {
      const meta = JSON.parse(jsonMatch[1])
      // conversation_label is present for both PMs and groups
      if (meta.conversation_label) return meta.conversation_label
      // Fallback: use sender_id if no label (shouldn't happen but just in case)
      if (meta.sender_id) return `dm:${meta.sender_id}`
    } catch {
      // Regex fallback if JSON parse fails
      const labelMatch = text.match(/"conversation_label"\s*:\s*"([^"]*)"/)
      if (labelMatch?.[1]) return labelMatch[1]
    }
  }
  return null
}

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
    canUseTool?: (toolName: string, input: Record<string, unknown>, options: any) => Promise<any>
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
    ...(opts.canUseTool ? { canUseTool: opts.canUseTool } : {}),
  }
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
    // Only log non-debug HTTP requests at info level; debug endpoints at debug level
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

  // ── Session management endpoints ──────────────────────────────────────

  app.get("/sessions", (c) => {
    return c.json({
      sessions: sessionStore.list(),
      stats: sessionStore.getStats(),
    })
  })

  app.get("/sessions/cleanup", (c) => {
    const result = sessionStore.cleanup()
    return c.json(result)
  })

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

  app.get("/debug/traces", (c) => {
    const limit = parseInt(c.req.query("limit") ?? "20", 10)
    return c.json(traceStore.getRecentTraces(limit))
  })

  app.get("/debug/traces/:id", (c) => {
    const id = c.req.param("id")
    const trace = traceStore.getTrace(id)
    if (!trace) return c.json({ error: "Trace not found", reqId: id }, 404)
    return c.json(trace)
  })

  app.get("/debug/errors", (c) => {
    const limit = parseInt(c.req.query("limit") ?? "10", 10)
    return c.json(traceStore.getRecentErrors(limit))
  })

  app.get("/debug/logs", (c) => {
    // List available log files
    try {
      const files = readdirSync(LOG_DIR)
        .filter(f => f.startsWith("proxy-") && f.endsWith(".log"))
        .sort()
        .reverse()
      return c.json({ logDir: LOG_DIR, files })
    } catch {
      return c.json({ logDir: LOG_DIR, files: [], error: "Cannot read log directory" })
    }
  })

  app.get("/debug/logs/:filename", (c) => {
    // Serve a specific log file (last N lines)
    const filename = c.req.param("filename")
    if (!filename.match(/^proxy-\d{4}-\d{2}-\d{2}\.log$/)) {
      return c.json({ error: "Invalid log filename" }, 400)
    }
    const tail = parseInt(c.req.query("tail") ?? "100", 10)
    try {
      const content = readFileSync(join(LOG_DIR, filename), "utf-8")
      const lines = content.trim().split("\n")
      const sliced = lines.slice(-tail)
      const parsed = sliced.map(line => {
        try { return JSON.parse(line) } catch { return { raw: line } }
      })
      return c.json({ file: filename, total: lines.length, returned: sliced.length, lines: parsed })
    } catch {
      return c.json({ error: "Log file not found" }, 404)
    }
  })

  app.get("/debug/errors/:id", (c) => {
    // Serve a specific error dump file
    const id = c.req.param("id")
    if (!id.match(/^req_/)) return c.json({ error: "Invalid request ID format" }, 400)
    try {
      const content = readFileSync(join(LOG_DIR, "errors", `${id}.json`), "utf-8")
      return c.json(JSON.parse(content))
    } catch {
      return c.json({ error: "Error dump not found", reqId: id }, 404)
    }
  })

  app.get("/debug/active", (c) => {
    // Detailed view of currently active requests
    const stats = traceStore.getStats()
    return c.json({
      queue: { active: requestQueue.activeCount, waiting: requestQueue.waitingCount, max: MAX_CONCURRENT },
      activeRequests: stats.activeRequests,
    })
  })

  app.get("/debug/health", (c) => {
    // Process health: memory, uptime, resource usage
    const mem = process.memoryUsage()
    const stats = traceStore.getStats()
    return c.json({
      version: PROXY_VERSION,
      pid: process.pid,
      uptimeMs: stats.uptimeMs,
      uptimeHuman: stats.uptimeHuman,
      memory: {
        rss: `${(mem.rss / 1024 / 1024).toFixed(1)}MB`,
        heapUsed: `${(mem.heapUsed / 1024 / 1024).toFixed(1)}MB`,
        heapTotal: `${(mem.heapTotal / 1024 / 1024).toFixed(1)}MB`,
        external: `${(mem.external / 1024 / 1024).toFixed(1)}MB`,
        rssBytes: mem.rss,
        heapUsedBytes: mem.heapUsed,
      },
      queue: { active: requestQueue.activeCount, waiting: requestQueue.waitingCount, max: MAX_CONCURRENT },
      requests: stats.requests,
      config: {
        stallTimeoutMs: finalConfig.stallTimeoutMs,
        maxConcurrent: MAX_CONCURRENT,
        queueTimeoutMs: QUEUE_TIMEOUT_MS,
        debug: finalConfig.debug,
      },
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
    ...m,
    object: "model" as const,
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
      const msgText = (body.messages ?? [])
        .map((m: any) => typeof m.content === "string" ? m.content : JSON.stringify(m.content))
        .join("\n")
      return c.json({ input_tokens: roughTokens(sysText + msgText) })
    } catch {
      return c.json({ input_tokens: 0 })
    }
  }
  app.post("/v1/messages/count_tokens", handleCountTokens)
  app.post("/messages/count_tokens", handleCountTokens)

  // ── Messages handler ─────────────────────────────────────────────────────

  const handleMessages = async (c: Context) => {
    const reqId = generateId("req_")
    // Will be set after body parse; needed for outer catch
    let trace: ReturnType<typeof traceStore.create> | undefined
    let requestStarted = Date.now()
    let clientDisconnected = false
    let abortReason: "stall" | "max_duration" | "max_output" | null = null

    try {
      let body: any
      try {
        body = await c.req.json()
      } catch (parseErr) {
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

      // Stall-based timeout: only aborts if no SDK events received for stallTimeoutMs.
      // Resets on every SDK event, so active requests never get killed.
      // NOTE: not started until queue is acquired — queue wait doesn't count.
      let stallTimer: ReturnType<typeof setTimeout> | null = null
      const resetStallTimer = () => {
        if (stallTimer) clearTimeout(stallTimer)
        stallTimer = setTimeout(() => {
          abortReason = "stall"
          logWarn("request.stall_timeout", {
            reqId,
            stallTimeoutMs: finalConfig.stallTimeoutMs,
            phase: trace?.phase,
            sdkEventCount: trace?.sdkEventCount,
            outputLen: trace?.outputLen,
            lastEventType: trace?.lastEventType,
          })
          abortController.abort()
        }, finalConfig.stallTimeoutMs)
      }
      const clearStallTimer = () => {
        if (stallTimer) { clearTimeout(stallTimer); stallTimer = null }
      }

      // Hard max duration: kills request even if actively streaming. Safety valve.
      let hardTimer: ReturnType<typeof setTimeout> | null = null
      const startHardTimer = () => {
        hardTimer = setTimeout(() => {
          abortReason = "max_duration"
          logError("request.max_duration", {
            reqId,
            maxDurationMs: finalConfig.maxDurationMs,
            phase: trace?.phase,
            sdkEventCount: trace?.sdkEventCount,
            outputLen: trace?.outputLen,
            model: trace?.model,
            lastEventType: trace?.lastEventType,
          })
          abortController.abort()
        }, finalConfig.maxDurationMs)
      }
      const clearHardTimer = () => {
        if (hardTimer) { clearTimeout(hardTimer); hardTimer = null }
      }

      // Output size check: kills request if output exceeds maxOutputChars.
      const checkOutputSize = (outputLen: number) => {
        if (outputLen > finalConfig.maxOutputChars && !abortReason) {
          abortReason = "max_output"
          logError("request.max_output", {
            reqId,
            outputLen,
            maxOutputChars: finalConfig.maxOutputChars,
            phase: trace?.phase,
            sdkEventCount: trace?.sdkEventCount,
            model: trace?.model,
            elapsedMs: trace ? Date.now() - trace.startedAt : undefined,
          })
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
        if (typeof body.system === "string") {
          systemContext = body.system
        } else if (Array.isArray(body.system)) {
          systemContext = body.system
            .filter((b: any) => b.type === "text" && b.text)
            .map((b: any) => b.text)
            .join("\n")
        }
      }

      const messages = body.messages as Array<{ role: string; content: string | Array<any> }>

      let promptText: string  // text version for token counting / logging
      let systemPrompt: string | undefined

      // Create MCP server for native tool calling (if client sent tools)
      const mcpServer = hasTools ? createToolMcpServer(body.tools) : null
      const mcpServers = mcpServer ? { "proxy-tools": mcpServer } : undefined
      const canUseTool = hasTools
        ? async () => ({ behavior: "deny" as const, message: "Tool execution handled by client" })
        : undefined

      // ── Session resumption ─────────────────────────────────────────────
      // Derive conversation ID from: headers (explicit) or conversation_label
      // embedded in openclaw message metadata.
      const conversationId = c.req.header("x-conversation-id")
        ?? c.req.header("x-session-id")
        ?? extractConversationLabel(messages)
        ?? null

      let resumeSessionId: string | undefined
      let isResuming = false
      let isCompacted = false

      if (conversationId) {
        const stored = sessionStore.get(conversationId)
        if (stored) {
          const countDelta = messages.length - stored.messageCount

          if (countDelta < 0 && messages.length <= 2) {
            // Likely a /reset — invalidate stale SDK session, start fresh
            logInfo("session.likely_reset", {
              reqId, conversationId, model,
              sdkSessionId: stored.sdkSessionId,
              storedMsgCount: stored.messageCount,
              currentMsgCount: messages.length,
            })
            sessionStore.invalidate(conversationId)
          } else {
            if (countDelta < 0) {
              isCompacted = true
              logInfo("session.compaction_detected", {
                reqId, conversationId, model,
                sdkSessionId: stored.sdkSessionId,
                storedMsgCount: stored.messageCount,
                currentMsgCount: messages.length,
                dropped: -countDelta,
              })
            }

            resumeSessionId = stored.sdkSessionId
            isResuming = true
            logInfo("session.resuming", {
              reqId, conversationId, model,
              storedModel: stored.model,
              sdkSessionId: resumeSessionId,
              storedMsgCount: stored.messageCount,
              currentMsgCount: messages.length,
              resumeCount: stored.resumeCount,
              countDelta,
            })
          }
        }
      }

      // Check if last user message contains images — if so, use native SDK multimodal input
      const lastMsg = messages[messages.length - 1]!
      const lastMsgHasImages = contentHasImages(lastMsg.content)

      // promptInput: either a string (text-only) or AsyncIterable<SDKUserMessage> (multimodal)
      let promptInput: string | AsyncIterable<any>
      // promptText: always the text-only version for token counting and logging
      promptText = serializeContent(lastMsg.content)
      // Track real usage from SDK stream events
      let sdkInputTokens = 0
      let sdkOutputTokens = 0
      let sdkCacheReadTokens = 0
      let sdkCacheCreationTokens = 0

      if (isResuming && resumeSessionId) {
        systemPrompt = (systemContext || "").trim() || undefined
        if (lastMsgHasImages) {
          promptInput = createSDKUserMessage(buildNativeContent(lastMsg.content), resumeSessionId)
          logInfo("session.resume_with_images", { reqId, conversationId })
        } else {
          promptInput = promptText
        }
      } else {
        // Single or multi-turn: always use just the last message as prompt.
        // The client (openclaw) manages conversation context; the proxy is stateless.
        // Prior turns are handled by SDK session resumption when available.
        systemPrompt = (systemContext || "").trim() || undefined
        promptInput = lastMsgHasImages
          ? createSDKUserMessage(buildNativeContent(lastMsg.content))
          : promptText
        if (lastMsgHasImages) logInfo("request.native_images", { reqId })
      }

      requestStarted = Date.now()

      // Capture client info
      const clientIp = c.req.header("x-forwarded-for")
        ?? c.req.header("x-real-ip")
        ?? c.req.header("cf-connecting-ip")
        ?? "unknown"
      const userAgent = c.req.header("user-agent") ?? "unknown"
      const bodyBytes = JSON.stringify(body).length

      // ── Create trace ──────────────────────────────────────────────────────
      trace = traceStore.create({
        reqId,
        model,
        requestedModel: body.model || "sonnet",
        stream,
        hasTools,
        thinking: thinking?.type,
        promptLen: promptText.length,
        systemLen: systemPrompt?.length ?? 0,
        msgCount: messages.length,
        bodyBytes,
        clientIp,
        userAgent,
      })

      // Attach session info to trace
      if (conversationId) {
        traceStore.setSession(reqId, {
          conversationId,
          sdkSessionId: resumeSessionId,
          isResuming,
          resumeCount: isResuming ? sessionStore.get(conversationId!)?.resumeCount : undefined,
        })
      }

      // ── Queue ─────────────────────────────────────────────────────────────
      const queueActive = requestQueue.activeCount
      const queueWaiting = requestQueue.waitingCount
      const needsQueue = queueActive >= MAX_CONCURRENT

      traceStore.phase(reqId, "queued", { queueActive, queueWaiting })

      if (needsQueue) {
        logInfo("queue.waiting", {
          reqId,
          model,
          queueActive,
          queueWaiting,
          queueTimeoutMs: QUEUE_TIMEOUT_MS,
        })
      }

      await requestQueue.acquire()

      const queueWaitMs = Date.now() - requestStarted
      traceStore.phase(reqId, "acquired", { queueWaitMs })

      logInfo("queue.acquired", {
        reqId,
        queueWaitMs,
        queueActive: requestQueue.activeCount,
        queueWaiting: requestQueue.waitingCount,
      })

      // Start timers AFTER queue acquire — queue wait doesn't count
      resetStallTimer()
      startHardTimer()

      // ── Non-streaming ──────────────────────────────────────────────────────
      if (!stream) {
        let fullText = ""
        let capturedSessionId: string | undefined
        const queryOpts = buildQueryOptions(model, { partial: hasTools, systemPrompt, abortController, thinking, resume: resumeSessionId, mcpServers, canUseTool })
        // For tool mode: capture native content blocks from stream events
        const contentBlocks: any[] = []
        let currentToolBlock: { id: string; name: string; jsonAccum: string } | null = null
        let capturedStopReason: string | null = null
        try {
          traceStore.phase(reqId, "sdk_starting")
          let sdkEventCount = 0
          for await (const message of query({ prompt: promptInput, options: queryOpts })) {
            sdkEventCount++
            resetStallTimer()
            traceStore.sdkEvent(reqId, sdkEventCount, message.type, (message as any).event?.type ?? (message as any).message?.type)
            // Capture session_id from init message
            if (message.type === "system" && (message as any).subtype === "init") {
              capturedSessionId = (message as any).session_id
            }
            if (message.type === "result") {
              const r = message as any
              if (r.session_id) capturedSessionId = r.session_id
              if (r.usage) {
                sdkInputTokens = r.usage.input_tokens ?? sdkInputTokens
                sdkOutputTokens = r.usage.output_tokens ?? sdkOutputTokens
                sdkCacheReadTokens = r.usage.cache_read_input_tokens ?? sdkCacheReadTokens
                sdkCacheCreationTokens = r.usage.cache_creation_input_tokens ?? sdkCacheCreationTokens
              }
            }
            if (hasTools && message.type === "stream_event") {
              const ev = (message as any).event as any
              if (ev.type === "content_block_start") {
                const cb = ev.content_block
                if (cb?.type === "tool_use") {
                  currentToolBlock = { id: cb.id, name: stripMcpPrefix(cb.name), jsonAccum: "" }
                } else if (cb?.type === "text") {
                  contentBlocks.push({ type: "text", text: "" })
                }
              } else if (ev.type === "content_block_delta") {
                if (ev.delta?.type === "text_delta") {
                  const lastText = contentBlocks[contentBlocks.length - 1]
                  if (lastText?.type === "text") {
                    lastText.text += ev.delta.text ?? ""
                  } else {
                    contentBlocks.push({ type: "text", text: ev.delta.text ?? "" })
                  }
                  fullText += ev.delta.text ?? ""
                } else if (ev.delta?.type === "input_json_delta" && currentToolBlock) {
                  currentToolBlock.jsonAccum += ev.delta.partial_json ?? ""
                }
              } else if (ev.type === "content_block_stop") {
                if (currentToolBlock) {
                  let input: any = {}
                  try { input = JSON.parse(currentToolBlock.jsonAccum) } catch {}
                  contentBlocks.push({
                    type: "tool_use",
                    id: currentToolBlock.id,
                    name: currentToolBlock.name,
                    input,
                  })
                  currentToolBlock = null
                }
              } else if (ev.type === "message_delta") {
                if (ev.delta?.stop_reason) capturedStopReason = ev.delta.stop_reason
                if (ev.usage?.output_tokens) sdkOutputTokens = ev.usage.output_tokens
              } else if (ev.type === "message_start" && ev.message?.usage) {
                if (ev.message.usage.input_tokens) sdkInputTokens = ev.message.usage.input_tokens
                if (ev.message.usage.cache_read_input_tokens) sdkCacheReadTokens = ev.message.usage.cache_read_input_tokens
                if (ev.message.usage.cache_creation_input_tokens) sdkCacheCreationTokens = ev.message.usage.cache_creation_input_tokens
              }
            } else if (!hasTools && message.type === "assistant") {
              let turnText = ""
              for (const block of message.message.content) {
                if (block.type === "text") turnText += block.text
              }
              fullText = turnText
              // Capture usage from assistant message
              const msgUsage = (message as any).message?.usage
              if (msgUsage?.input_tokens) sdkInputTokens = msgUsage.input_tokens
              if (msgUsage?.output_tokens) sdkOutputTokens = msgUsage.output_tokens
            }
          }
          traceStore.phase(reqId, "sdk_done")

          // Store session mapping for future resumption
          if (conversationId && capturedSessionId) {
            if (isResuming) {
              sessionStore.recordResume(conversationId!, messages.length, isCompacted)
              logInfo("session.resumed_ok", { reqId, conversationId, sdkSessionId: capturedSessionId })
            } else {
              sessionStore.set(conversationId!, capturedSessionId, model, messages.length)
              logInfo("session.created", { reqId, conversationId, sdkSessionId: capturedSessionId })
            }
          }
        } catch (resumeErr) {
          // If resume failed, start a fresh session (no history reconstruction)
          if (isResuming && resumeSessionId) {
            logWarn("session.resume_failed", {
              reqId, conversationId, sdkSessionId: resumeSessionId,
              error: resumeErr instanceof Error ? resumeErr.message : String(resumeErr),
            })
            if (conversationId) {
              sessionStore.recordFailure(conversationId!)
              sessionStore.invalidate(conversationId!)
            }

            logInfo("session.fallback_fresh", { reqId, conversationId })
            // Reset accumulators and retry as a brand new session
            contentBlocks.length = 0
            currentToolBlock = null
            capturedStopReason = null
            fullText = ""
            let sdkEventCount = 0
            for await (const message of query({ prompt: promptInput, options: buildQueryOptions(model, { partial: hasTools, systemPrompt, abortController, thinking, mcpServers, canUseTool }) })) {
              sdkEventCount++
              resetStallTimer()
              traceStore.sdkEvent(reqId, sdkEventCount, message.type, (message as any).event?.type ?? (message as any).message?.type)
              if (message.type === "system" && (message as any).subtype === "init") {
                capturedSessionId = (message as any).session_id
              }
              if (message.type === "result") {
                const r = message as any
                if (r.session_id) capturedSessionId = r.session_id
                if (r.usage) {
                  sdkInputTokens = r.usage.input_tokens ?? sdkInputTokens
                  sdkOutputTokens = r.usage.output_tokens ?? sdkOutputTokens
                  sdkCacheReadTokens = r.usage.cache_read_input_tokens ?? sdkCacheReadTokens
                  sdkCacheCreationTokens = r.usage.cache_creation_input_tokens ?? sdkCacheCreationTokens
                }
              }
              if (hasTools && message.type === "stream_event") {
                const ev = (message as any).event as any
                if (ev.type === "content_block_start") {
                  const cb = ev.content_block
                  if (cb?.type === "tool_use") {
                    currentToolBlock = { id: cb.id, name: stripMcpPrefix(cb.name), jsonAccum: "" }
                  } else if (cb?.type === "text") {
                    contentBlocks.push({ type: "text", text: "" })
                  }
                } else if (ev.type === "content_block_delta") {
                  if (ev.delta?.type === "text_delta") {
                    const lastText = contentBlocks[contentBlocks.length - 1]
                    if (lastText?.type === "text") {
                      lastText.text += ev.delta.text ?? ""
                    } else {
                      contentBlocks.push({ type: "text", text: ev.delta.text ?? "" })
                    }
                    fullText += ev.delta.text ?? ""
                  } else if (ev.delta?.type === "input_json_delta" && currentToolBlock) {
                    currentToolBlock.jsonAccum += ev.delta.partial_json ?? ""
                  }
                } else if (ev.type === "content_block_stop") {
                  if (currentToolBlock) {
                    let input: any = {}
                    try { input = JSON.parse(currentToolBlock.jsonAccum) } catch {}
                    contentBlocks.push({ type: "tool_use", id: currentToolBlock.id, name: currentToolBlock.name, input })
                    currentToolBlock = null
                  }
                } else if (ev.type === "message_delta") {
                  if (ev.delta?.stop_reason) capturedStopReason = ev.delta.stop_reason
                  if (ev.usage?.output_tokens) sdkOutputTokens = ev.usage.output_tokens
                } else if (ev.type === "message_start" && ev.message?.usage) {
                  if (ev.message.usage.input_tokens) sdkInputTokens = ev.message.usage.input_tokens
                  if (ev.message.usage.cache_read_input_tokens) sdkCacheReadTokens = ev.message.usage.cache_read_input_tokens
                  if (ev.message.usage.cache_creation_input_tokens) sdkCacheCreationTokens = ev.message.usage.cache_creation_input_tokens
                }
              } else if (!hasTools && message.type === "assistant") {
                let turnText = ""
                for (const block of message.message.content) {
                  if (block.type === "text") turnText += block.text
                }
                fullText = turnText
                const msgUsage = (message as any).message?.usage
                if (msgUsage?.input_tokens) sdkInputTokens = msgUsage.input_tokens
                if (msgUsage?.output_tokens) sdkOutputTokens = msgUsage.output_tokens
              }
            }
            traceStore.phase(reqId, "sdk_done")
            if (conversationId && capturedSessionId) {
              sessionStore.set(conversationId!, capturedSessionId, model, messages.length)
              logInfo("session.created_after_fallback", { reqId, conversationId, sdkSessionId: capturedSessionId })
            }
          } else {
            throw resumeErr
          }
        } finally {
          clearStallTimer(); clearHardTimer()
          requestQueue.release()
          logDebug("queue.released", {
            reqId,
            queueActive: requestQueue.activeCount,
            queueWaiting: requestQueue.waitingCount,
          })
        }

        traceStore.phase(reqId, "responding")

        if (hasTools) {
          // Use native content blocks captured from stream events
          const toolCallCount = contentBlocks.filter((b: any) => b.type === "tool_use").length
          // Filter out empty text blocks
          const content = contentBlocks.filter((b: any) => {
            if (b.type === "text" && !b.text?.trim()) return false
            return true
          })
          if (content.length === 0) content.push({ type: "text", text: fullText || "..." })

          const stopReason = toolCallCount > 0 ? "tool_use" : (capturedStopReason ?? "end_turn")

          traceStore.setUsage(reqId, { input_tokens: sdkInputTokens, output_tokens: sdkOutputTokens, ...(sdkCacheReadTokens > 0 ? { cache_read_input_tokens: sdkCacheReadTokens } : {}), ...(sdkCacheCreationTokens > 0 ? { cache_creation_input_tokens: sdkCacheCreationTokens } : {}) })
          traceStore.complete(reqId, { outputLen: fullText.length, toolCallCount })

          return c.json({
            id: generateId("msg_"),
            type: "message", role: "assistant", content,
            model: body.model, stop_reason: stopReason, stop_sequence: null,
            usage: buildUsage(sdkInputTokens, sdkOutputTokens, sdkCacheReadTokens, sdkCacheCreationTokens)
          })
        }

        logDebug("usage.tokens", { reqId, sdkInput: sdkInputTokens, sdkOutput: sdkOutputTokens, cacheRead: sdkCacheReadTokens, cacheCreation: sdkCacheCreationTokens })

        if (!fullText || !fullText.trim()) fullText = "..."
        traceStore.setUsage(reqId, { input_tokens: sdkInputTokens, output_tokens: sdkOutputTokens, ...(sdkCacheReadTokens > 0 ? { cache_read_input_tokens: sdkCacheReadTokens } : {}), ...(sdkCacheCreationTokens > 0 ? { cache_creation_input_tokens: sdkCacheCreationTokens } : {}) })
        traceStore.complete(reqId, { outputLen: fullText.length })

        return c.json({
          id: generateId("msg_"),
          type: "message", role: "assistant",
          content: [{ type: "text", text: fullText }],
          model: body.model, stop_reason: "end_turn", stop_sequence: null,
          usage: buildUsage(sdkInputTokens, sdkOutputTokens, sdkCacheReadTokens, sdkCacheCreationTokens)
        })
      }

      // ── Streaming ──────────────────────────────────────────────────────────
      const encoder = new TextEncoder()
      const readable = new ReadableStream({
        cancel() {
          clientDisconnected = true
          logWarn("stream.client_disconnect", {
            reqId,
            phase: trace?.phase,
            sdkEventCount: trace?.sdkEventCount,
            outputLen: trace?.outputLen,
            elapsedMs: trace ? Date.now() - trace.startedAt : undefined,
            model: trace?.model,
          })
          abortController.abort()
        },
        async start(controller) {
          const messageId = generateId("msg_")
          let queueReleased = false
          const releaseQueue = () => {
            if (!queueReleased) {
              queueReleased = true
              requestQueue.release()
              logDebug("queue.released", {
                reqId,
                queueActive: requestQueue.activeCount,
                queueWaiting: requestQueue.waitingCount,
              })
            }
          }

          let sseSendErrors = 0
          const sse = (event: string, data: object) => {
            try {
              controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`))
            } catch (e) {
              sseSendErrors++
              if (sseSendErrors <= 3) {
                logWarn("stream.sse_send_failed", {
                  reqId,
                  event,
                  sseSendErrors,
                  error: e instanceof Error ? e.message : String(e),
                })
              }
            }
          }

          try {
            const heartbeat = setInterval(() => {
              try {
                controller.enqueue(encoder.encode(`event: ping\ndata: {"type": "ping"}\n\n`))
              } catch (e) {
                logWarn("stream.heartbeat_failed", {
                  reqId,
                  error: e instanceof Error ? e.message : String(e),
                  phase: trace?.phase,
                  elapsedMs: trace ? Date.now() - trace.startedAt : undefined,
                })
                clearInterval(heartbeat)
              }
            }, 15_000)

            // Defer message_start until SDK provides real input_tokens
            let messageStartSent = false
            const emitMessageStart = () => {
              if (!messageStartSent) {
                messageStartSent = true
                sse("message_start", {
                  type: "message_start",
                  message: {
                    id: messageId, type: "message", role: "assistant", content: [],
                    model: body.model, stop_reason: null, stop_sequence: null,
                    usage: buildUsage(sdkInputTokens, 1, sdkCacheReadTokens, sdkCacheCreationTokens)
                  }
                })
              }
            }

            if (hasTools) {
              // ── With tools: forward native SDK stream events directly as SSE ──
              let fullText = ""
              let blockIdx = 0
              let toolCallCount = 0
              let capturedStopReason: string | null = null
              let hasEmittedAnyBlock = false
              let sdkEventCount = 0
              let lastEventAt = Date.now()
              const stallLog = setInterval(() => {
                const stallMs = Date.now() - lastEventAt
                traceStore.stall(reqId, stallMs)
              }, 15_000)
              let capturedSessionId: string | undefined

              // Helper to forward a stream event directly as SSE
              // Track whether the current SDK content block was emitted (thinking blocks are skipped)
              let currentBlockEmitted = false
              const forwardStreamEvent = (ev: any) => {
                if (ev.type === "message_start" && ev.message?.usage) {
                  if (ev.message.usage.input_tokens) sdkInputTokens = ev.message.usage.input_tokens
                  if (ev.message.usage.cache_read_input_tokens) sdkCacheReadTokens = ev.message.usage.cache_read_input_tokens
                  if (ev.message.usage.cache_creation_input_tokens) sdkCacheCreationTokens = ev.message.usage.cache_creation_input_tokens
                  emitMessageStart()
                  return
                }
                // Ensure message_start is sent before any content blocks
                emitMessageStart()
                if (ev.type === "content_block_start") {
                  const cb = ev.content_block
                  if (cb?.type === "tool_use") {
                    toolCallCount++
                    currentBlockEmitted = true
                    sse("content_block_start", { type: "content_block_start", index: blockIdx, content_block: { type: "tool_use", id: cb.id, name: stripMcpPrefix(cb.name), input: {} } })
                  } else if (cb?.type === "text") {
                    currentBlockEmitted = true
                    sse("content_block_start", { type: "content_block_start", index: blockIdx, content_block: { type: "text", text: "" } })
                  } else {
                    // Thinking or other block types — skip (don't emit SSE)
                    currentBlockEmitted = false
                  }
                  hasEmittedAnyBlock = hasEmittedAnyBlock || currentBlockEmitted
                } else if (ev.type === "content_block_delta") {
                  if (!currentBlockEmitted) return // Skip deltas for non-emitted blocks (thinking)
                  if (ev.delta?.type === "text_delta") {
                    fullText += ev.delta.text ?? ""
                    traceStore.updateOutput(reqId, fullText.length)
                    checkOutputSize(fullText.length)
                    sse("content_block_delta", { type: "content_block_delta", index: blockIdx, delta: { type: "text_delta", text: ev.delta.text ?? "" } })
                  } else if (ev.delta?.type === "input_json_delta") {
                    sse("content_block_delta", { type: "content_block_delta", index: blockIdx, delta: { type: "input_json_delta", partial_json: ev.delta.partial_json ?? "" } })
                  }
                } else if (ev.type === "content_block_stop") {
                  if (!currentBlockEmitted) return // Skip stop for non-emitted blocks (thinking)
                  sse("content_block_stop", { type: "content_block_stop", index: blockIdx })
                  blockIdx++
                } else if (ev.type === "message_delta") {
                  if (ev.delta?.stop_reason) capturedStopReason = ev.delta.stop_reason
                  if (ev.usage?.output_tokens) sdkOutputTokens = ev.usage.output_tokens
                }
              }

              try {
                traceStore.phase(reqId, "sdk_starting")
                for await (const message of query({ prompt: promptInput, options: buildQueryOptions(model, { partial: true, systemPrompt, abortController, thinking, resume: resumeSessionId, mcpServers, canUseTool }) })) {
                  sdkEventCount++
                  lastEventAt = Date.now()
                  resetStallTimer()
                  const subtype = (message as any).event?.type ?? (message as any).message?.type
                  if (message.type === "system" && (message as any).subtype === "init") {
                    capturedSessionId = (message as any).session_id
                  }
                  if (message.type === "result") {
                    const r = message as any
                    if (r.session_id) capturedSessionId = r.session_id
                    if (r.usage) {
                      sdkInputTokens = r.usage.input_tokens ?? sdkInputTokens
                      sdkOutputTokens = r.usage.output_tokens ?? sdkOutputTokens
                      sdkCacheReadTokens = r.usage.cache_read_input_tokens ?? sdkCacheReadTokens
                      sdkCacheCreationTokens = r.usage.cache_creation_input_tokens ?? sdkCacheCreationTokens
                    }
                  }
                  if (message.type === "stream_event") {
                    const ev = (message as any).event as any
                    if (!trace!.firstTokenAt && (ev.type === "content_block_delta" || ev.type === "content_block_start")) {
                      traceStore.phase(reqId, "sdk_streaming")
                    }
                    forwardStreamEvent(ev)
                  }
                  traceStore.sdkEvent(reqId, sdkEventCount, message.type, subtype)
                }
                traceStore.phase(reqId, "sdk_done")

                // Store session mapping
                if (conversationId && capturedSessionId) {
                  if (isResuming) {
                    sessionStore.recordResume(conversationId!, messages.length, isCompacted)
                  } else {
                    sessionStore.set(conversationId!, capturedSessionId, model, messages.length)
                  }
                }
              } catch (resumeErr) {
                // Resume failed — start a fresh session (no history reconstruction)
                if (isResuming && resumeSessionId) {
                  logWarn("session.resume_failed_stream", {
                    reqId, conversationId, sdkSessionId: resumeSessionId,
                    error: resumeErr instanceof Error ? resumeErr.message : String(resumeErr),
                  })
                  if (conversationId) {
                    sessionStore.recordFailure(conversationId!)
                    sessionStore.invalidate(conversationId!)
                  }

                  logInfo("session.fallback_fresh_stream", { reqId, conversationId })
                  fullText = ""
                  blockIdx = 0
                  toolCallCount = 0
                  capturedStopReason = null
                  hasEmittedAnyBlock = false
                  sdkEventCount = 0
                  for await (const message of query({ prompt: promptInput, options: buildQueryOptions(model, { partial: true, systemPrompt, abortController, thinking, mcpServers, canUseTool }) })) {
                    sdkEventCount++
                    lastEventAt = Date.now()
                    resetStallTimer()
                    const subtype = (message as any).event?.type ?? (message as any).message?.type
                    if (message.type === "system" && (message as any).subtype === "init") {
                      capturedSessionId = (message as any).session_id
                    }
                    if (message.type === "result") {
                      const r = message as any
                      if (r.session_id) capturedSessionId = r.session_id
                      if (r.usage) {
                        sdkInputTokens = r.usage.input_tokens ?? sdkInputTokens
                        sdkOutputTokens = r.usage.output_tokens ?? sdkOutputTokens
                        sdkCacheReadTokens = r.usage.cache_read_input_tokens ?? sdkCacheReadTokens
                        sdkCacheCreationTokens = r.usage.cache_creation_input_tokens ?? sdkCacheCreationTokens
                      }
                    }
                    if (message.type === "stream_event") {
                      const ev = (message as any).event as any
                      if (!trace!.firstTokenAt && (ev.type === "content_block_delta" || ev.type === "content_block_start")) {
                        traceStore.phase(reqId, "sdk_streaming")
                      }
                      forwardStreamEvent(ev)
                    }
                    traceStore.sdkEvent(reqId, sdkEventCount, message.type, subtype)
                  }
                  traceStore.phase(reqId, "sdk_done")
                  if (conversationId && capturedSessionId) {
                    sessionStore.set(conversationId!, capturedSessionId, model, messages.length)
                    logInfo("session.created_after_fallback_stream", { reqId, conversationId, sdkSessionId: capturedSessionId })
                  }
                } else {
                  throw resumeErr
                }
              } finally {
                clearInterval(stallLog)
                clearInterval(heartbeat)
                clearStallTimer(); clearHardTimer()
                releaseQueue()
              }

              traceStore.phase(reqId, "responding")

              // Ensure message_start is sent even if no SDK events came
              emitMessageStart()

              // If no blocks were emitted at all, emit a placeholder text block
              if (!hasEmittedAnyBlock) {
                sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })
                sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "..." } })
                sse("content_block_stop", { type: "content_block_stop", index: 0 })
              }

              const stopReason = toolCallCount > 0 ? "tool_use" : (capturedStopReason ?? "end_turn")
              sse("message_delta", { type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: sdkOutputTokens } })
              sse("message_stop", { type: "message_stop" })
              controller.close()

              traceStore.setUsage(reqId, { input_tokens: sdkInputTokens, output_tokens: sdkOutputTokens, ...(sdkCacheReadTokens > 0 ? { cache_read_input_tokens: sdkCacheReadTokens } : {}), ...(sdkCacheCreationTokens > 0 ? { cache_creation_input_tokens: sdkCacheCreationTokens } : {}) })
              traceStore.complete(reqId, { outputLen: fullText.length, toolCallCount })
              return
            }

            // ── No tools: stream text deltas directly ─────────────────────
            let contentBlockStartSent = false

            let fullText = ""
            let hasStreamed = false
            let sdkEventCount = 0
            let lastEventAt = Date.now()
            let capturedSessionId2: string | undefined
            const stallLog = setInterval(() => {
              const stallMs = Date.now() - lastEventAt
              traceStore.stall(reqId, stallMs)
            }, 15_000)
            try {
              traceStore.phase(reqId, "sdk_starting")
              for await (const message of query({ prompt: promptInput, options: buildQueryOptions(model, { partial: true, systemPrompt, abortController, thinking, resume: resumeSessionId }) })) {
                sdkEventCount++
                lastEventAt = Date.now()
                resetStallTimer()
                const subtype = (message as any).event?.type ?? (message as any).message?.type
                // Capture session_id from init message
                if (message.type === "system" && (message as any).subtype === "init") {
                  capturedSessionId2 = (message as any).session_id
                }
                if (message.type === "result") {
                  const r = message as any
                  if (r.session_id) capturedSessionId2 = r.session_id
                  if (r.usage) {
                    sdkInputTokens = r.usage.input_tokens ?? sdkInputTokens
                    sdkOutputTokens = r.usage.output_tokens ?? sdkOutputTokens
                    sdkCacheReadTokens = r.usage.cache_read_input_tokens ?? sdkCacheReadTokens
                    sdkCacheCreationTokens = r.usage.cache_creation_input_tokens ?? sdkCacheCreationTokens
                  }
                }
                if (message.type === "stream_event") {
                  const ev = message.event as any
                  // Capture usage from SDK message_start (real input tokens)
                  if (ev.type === "message_start" && ev.message?.usage) {
                    if (ev.message.usage.input_tokens) sdkInputTokens = ev.message.usage.input_tokens
                    if (ev.message.usage.cache_read_input_tokens) sdkCacheReadTokens = ev.message.usage.cache_read_input_tokens
                    if (ev.message.usage.cache_creation_input_tokens) sdkCacheCreationTokens = ev.message.usage.cache_creation_input_tokens
                    emitMessageStart()
                  }
                  // Detect first content event BEFORE sdkEvent records it
                  if (!trace!.firstTokenAt && (ev.type === "content_block_delta" || ev.type === "content_block_start")) {
                    traceStore.phase(reqId, "sdk_streaming")
                  }
                  if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") {
                    const text = ev.delta.text ?? ""
                    if (text) {
                      // Ensure message_start and content_block_start are sent before first delta
                      emitMessageStart()
                      if (!contentBlockStartSent) {
                        contentBlockStartSent = true
                        sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })
                      }
                      fullText += text
                      hasStreamed = true
                      traceStore.updateOutput(reqId, fullText.length)
                      checkOutputSize(fullText.length)
                      sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } })
                    }
                  } else if (ev.type === "message_delta" && ev.usage?.output_tokens) {
                    sdkOutputTokens = ev.usage.output_tokens
                  }
                }
                traceStore.sdkEvent(reqId, sdkEventCount, message.type, subtype)
              }
              traceStore.phase(reqId, "sdk_done")

              // Store session mapping
              if (conversationId && capturedSessionId2) {
                if (isResuming) {
                  sessionStore.recordResume(conversationId!, messages.length, isCompacted)
                } else {
                  sessionStore.set(conversationId!, capturedSessionId2, model, messages.length)
                }
              }
            } catch (resumeErr) {
              // Resume failed — start a fresh session (no history reconstruction)
              if (isResuming && resumeSessionId) {
                logWarn("session.resume_failed_stream", {
                  reqId, conversationId, sdkSessionId: resumeSessionId,
                  error: resumeErr instanceof Error ? resumeErr.message : String(resumeErr),
                })
                if (conversationId) {
                  sessionStore.recordFailure(conversationId!)
                  sessionStore.invalidate(conversationId!)
                }

                logInfo("session.fallback_fresh_stream", { reqId, conversationId })
                sdkEventCount = 0
                for await (const message of query({ prompt: promptInput, options: buildQueryOptions(model, { partial: true, systemPrompt, abortController, thinking }) })) {
                  sdkEventCount++
                  lastEventAt = Date.now()
                  resetStallTimer()
                  const subtype = (message as any).event?.type ?? (message as any).message?.type
                  if (message.type === "system" && (message as any).subtype === "init") {
                    capturedSessionId2 = (message as any).session_id
                  }
                  if (message.type === "result") {
                    const r = message as any
                    if (r.session_id) capturedSessionId2 = r.session_id
                    if (r.usage) {
                      sdkInputTokens = r.usage.input_tokens ?? sdkInputTokens
                      sdkOutputTokens = r.usage.output_tokens ?? sdkOutputTokens
                      sdkCacheReadTokens = r.usage.cache_read_input_tokens ?? sdkCacheReadTokens
                      sdkCacheCreationTokens = r.usage.cache_creation_input_tokens ?? sdkCacheCreationTokens
                    }
                  }
                  if (message.type === "stream_event") {
                    const ev = message.event as any
                    if (ev.type === "message_start" && ev.message?.usage) {
                      if (ev.message.usage.input_tokens) sdkInputTokens = ev.message.usage.input_tokens
                      if (ev.message.usage.cache_read_input_tokens) sdkCacheReadTokens = ev.message.usage.cache_read_input_tokens
                      if (ev.message.usage.cache_creation_input_tokens) sdkCacheCreationTokens = ev.message.usage.cache_creation_input_tokens
                      emitMessageStart()
                    }
                    if (!trace!.firstTokenAt && (ev.type === "content_block_delta" || ev.type === "content_block_start")) {
                      traceStore.phase(reqId, "sdk_streaming")
                    }
                    if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") {
                      const text = ev.delta.text ?? ""
                      if (text) {
                        emitMessageStart()
                        if (!contentBlockStartSent) {
                          contentBlockStartSent = true
                          sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })
                        }
                        fullText += text
                        hasStreamed = true
                        traceStore.updateOutput(reqId, fullText.length)
                        checkOutputSize(fullText.length)
                        sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } })
                      }
                    } else if (ev.type === "message_delta" && ev.usage?.output_tokens) {
                      sdkOutputTokens = ev.usage.output_tokens
                    }
                  }
                  traceStore.sdkEvent(reqId, sdkEventCount, message.type, subtype)
                }
                traceStore.phase(reqId, "sdk_done")
                if (conversationId && capturedSessionId2) {
                  sessionStore.set(conversationId!, capturedSessionId2, model, messages.length)
                  logInfo("session.created_after_fallback_stream", { reqId, conversationId, sdkSessionId: capturedSessionId2 })
                }
              } else {
                throw resumeErr
              }
            } finally {
              clearInterval(stallLog)
              clearInterval(heartbeat)
              clearStallTimer(); clearHardTimer()
              // (temp files no longer used — images passed natively)
              releaseQueue()
            }

            // Ensure message_start and content_block_start are emitted even if no stream events came
            emitMessageStart()
            if (!contentBlockStartSent) {
              contentBlockStartSent = true
              sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })
            }

            if (!hasStreamed) {
              sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "..." } })
            }

            sse("content_block_stop", { type: "content_block_stop", index: 0 })
            sse("message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: sdkOutputTokens } })
            sse("message_stop", { type: "message_stop" })
            controller.close()

            traceStore.setUsage(reqId, { input_tokens: sdkInputTokens, output_tokens: sdkOutputTokens, ...(sdkCacheReadTokens > 0 ? { cache_read_input_tokens: sdkCacheReadTokens } : {}), ...(sdkCacheCreationTokens > 0 ? { cache_creation_input_tokens: sdkCacheCreationTokens } : {}) })
            traceStore.complete(reqId, { outputLen: fullText.length })

          } catch (error) {
            clearStallTimer(); clearHardTimer()
            releaseQueue()
            const err = error instanceof Error ? error : new Error(String(error))
            const isAbort = err.name === "AbortError" || err.message?.includes("abort")
            const isQueueTimeout = err.message.includes("Queue timeout")

            let errMsg: string
            let errType: string
            if (clientDisconnected) {
              errMsg = "Client disconnected during streaming."
              errType = "api_error"
            } else if (abortReason === "max_duration") {
              errMsg = `Request exceeded max duration of ${finalConfig.maxDurationMs / 1000}s. Output: ${trace?.outputLen ?? 0} chars.`
              errType = "api_error"
            } else if (abortReason === "max_output") {
              errMsg = `Request exceeded max output size of ${finalConfig.maxOutputChars} chars.`
              errType = "api_error"
            } else if (isAbort) {
              errMsg = `Request stalled — no SDK activity for ${finalConfig.stallTimeoutMs / 1000}s. Please retry.`
              errType = "api_error"
            } else if (isQueueTimeout) {
              errMsg = "Server busy — all request slots are occupied. Please retry shortly."
              errType = "overloaded_error"
            } else {
              errMsg = err.message
              errType = "api_error"
            }

            // Trace the failure with full context
            traceStore.fail(reqId, err, "error", {
              clientDisconnect: clientDisconnected,
              abortReason,
              aborted: isAbort,
              queueTimeout: isQueueTimeout,
              stallTimeoutMs: finalConfig.stallTimeoutMs,
              maxDurationMs: finalConfig.maxDurationMs,
              maxOutputChars: finalConfig.maxOutputChars,
              sseSendErrors,
            })

            // (temp files no longer used — images passed natively)
            if (!clientDisconnected) {
              try {
                sse("error", { type: "error", error: { type: errType, message: errMsg }, request_id: reqId })
                controller.close()
              } catch {}
            } else {
              try { controller.close() } catch {}
            }
          }
        }
      })

      return new Response(readable, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive"
        }
      })

    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error))
      const isAbort = err.name === "AbortError" || err.message?.includes("abort")
      const isQueueTimeout = err.message.includes("Queue timeout")

      let errMsg: string
      let errType: string
      if (clientDisconnected) {
        errMsg = "Client disconnected."
        errType = "api_error"
      } else if (abortReason === "max_duration") {
        errMsg = `Request exceeded max duration of ${finalConfig.maxDurationMs / 1000}s.`
        errType = "api_error"
      } else if (abortReason === "max_output") {
        errMsg = `Request exceeded max output size of ${finalConfig.maxOutputChars} chars.`
        errType = "api_error"
      } else if (isAbort) {
        errMsg = `Request stalled — no SDK activity for ${finalConfig.stallTimeoutMs / 1000}s. Please retry.`
        errType = "api_error"
      } else if (isQueueTimeout) {
        errMsg = "Server busy — all request slots are occupied. Please retry shortly."
        errType = "overloaded_error"
      } else {
        errMsg = err.message
        errType = "api_error"
      }

      // Trace the failure
      if (trace) {
        traceStore.fail(reqId, err, "error", {
          clientDisconnect: clientDisconnected,
          aborted: isAbort,
          queueTimeout: isQueueTimeout,
        })
      } else {
        logError("request.error.no_trace", { reqId, error: errMsg, stack: err.stack })
      }

      if (isQueueTimeout) {
        return new Response(JSON.stringify({ type: "error", error: { type: errType, message: errMsg }, request_id: reqId }), {
          status: 529, headers: { "Content-Type": "application/json" }
        })
      }
      if (isAbort) {
        return new Response(JSON.stringify({ type: "error", error: { type: errType, message: errMsg }, request_id: reqId }), {
          status: 504, headers: { "Content-Type": "application/json" }
        })
      }
      return c.json({ type: "error", error: { type: errType, message: errMsg }, request_id: reqId }, 500)
    }
  }

  app.post("/v1/messages", handleMessages)
  app.post("/messages", handleMessages)

  // Stub: batches API not supported
  const handleBatches = (c: Context) => c.json({
    type: "error",
    error: { type: "not_implemented_error", message: "Batches API is not supported by this proxy" }
  }, 501)
  app.post("/v1/messages/batches", handleBatches)
  app.get("/v1/messages/batches", handleBatches)
  app.get("/v1/messages/batches/:id", handleBatches)

  // ── OpenAI-compatible /v1/chat/completions ─────────────────────────────

  function convertOpenaiContent(content: any): any {
    if (typeof content === "string") return content
    if (!Array.isArray(content)) return String(content ?? "")

    return content.map((part: any) => {
      if (part.type === "text") return { type: "text", text: part.text ?? "" }
      if (part.type === "image_url" && part.image_url?.url) {
        const url = part.image_url.url as string
        const dataMatch = url.match(/^data:(image\/\w+);base64,(.+)$/)
        if (dataMatch) {
          return {
            type: "image",
            source: {
              type: "base64",
              media_type: dataMatch[1]!,
              data: dataMatch[2]!
            }
          }
        }
        return {
          type: "image",
          source: { type: "url", url }
        }
      }
      return part
    })
  }

  function openaiToAnthropicMessages(messages: any[]): { system?: string; messages: any[] } {
    let system: string | undefined
    const converted: any[] = []

    for (const msg of messages) {
      if (msg.role === "system") {
        const sysText = typeof msg.content === "string" ? msg.content
          : Array.isArray(msg.content) ? msg.content.filter((p: any) => p.type === "text").map((p: any) => p.text ?? "").join("")
          : String(msg.content ?? "")
        system = (system ? system + "\n" : "") + sysText
      } else if (msg.role === "user") {
        converted.push({ role: "user", content: convertOpenaiContent(msg.content) })
      } else if (msg.role === "assistant") {
        if (msg.tool_calls?.length) {
          const content: any[] = []
          if (msg.content) content.push({ type: "text", text: msg.content })
          for (const tc of msg.tool_calls) {
            content.push({
              type: "tool_use",
              id: tc.id,
              name: tc.function?.name ?? "",
              input: tc.function?.arguments ? JSON.parse(tc.function.arguments) : {}
            })
          }
          converted.push({ role: "assistant", content })
        } else {
          converted.push({ role: "assistant", content: msg.content ?? "" })
        }
      } else if (msg.role === "tool") {
        converted.push({
          role: "user",
          content: [{
            type: "tool_result",
            tool_use_id: msg.tool_call_id,
            content: msg.content ?? ""
          }]
        })
      }
    }
    return { system, messages: converted }
  }

  function openaiToAnthropicTools(tools: any[]): any[] {
    return tools
      .filter((t: any) => t.type === "function" && t.function)
      .map((t: any) => ({
        name: t.function.name,
        description: t.function.description ?? "",
        input_schema: t.function.parameters ?? { type: "object", properties: {} }
      }))
  }

  function anthropicToOpenaiResponse(anthropicBody: any, model: string): any {
    const textBlocks = (anthropicBody.content ?? []).filter((b: any) => b.type === "text")
    const toolBlocks = (anthropicBody.content ?? []).filter((b: any) => b.type === "tool_use")

    const text = textBlocks.map((b: any) => b.text).join("") || (toolBlocks.length > 0 ? null : "")

    const message: any = { role: "assistant", content: text }

    if (toolBlocks.length > 0) {
      message.tool_calls = toolBlocks.map((b: any, i: number) => ({
        id: b.id,
        type: "function",
        function: {
          name: b.name,
          arguments: JSON.stringify(b.input ?? {})
        }
      }))
    }

    const finishReason = anthropicBody.stop_reason === "tool_use" ? "tool_calls"
      : anthropicBody.stop_reason === "max_tokens" ? "length"
      : "stop"

    return {
      id: generateId("chatcmpl-"),
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        message,
        finish_reason: finishReason
      }],
      usage: {
        prompt_tokens: anthropicBody.usage?.input_tokens ?? 0,
        completion_tokens: anthropicBody.usage?.output_tokens ?? 0,
        total_tokens: (anthropicBody.usage?.input_tokens ?? 0) + (anthropicBody.usage?.output_tokens ?? 0)
      }
    }
  }

  const handleChatCompletions = async (c: Context) => {
    try {
      let body: any
      try {
        body = await c.req.json()
      } catch {
        return c.json({ error: { message: "Request body must be valid JSON", type: "invalid_request_error" } }, 400)
      }

      if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
        return c.json({ error: { message: "messages is required and must be a non-empty array", type: "invalid_request_error" } }, 400)
      }

      const { system, messages } = openaiToAnthropicMessages(body.messages)
      const stream = body.stream ?? false
      const requestedModel = body.model ?? "claude-sonnet-4-6"

      const anthropicBody: any = {
        model: requestedModel,
        messages,
        stream,
      }
      if (system) anthropicBody.system = system
      if (body.max_tokens || body.max_completion_tokens) {
        anthropicBody.max_tokens = body.max_tokens ?? body.max_completion_tokens
      }
      if (body.temperature !== undefined) anthropicBody.temperature = body.temperature
      if (body.top_p !== undefined) anthropicBody.top_p = body.top_p
      if (body.stop) anthropicBody.stop_sequences = Array.isArray(body.stop) ? body.stop : [body.stop]
      if (body.tools?.length) {
        anthropicBody.tools = openaiToAnthropicTools(body.tools)
      }

      const internalHeaders: Record<string, string> = { "Content-Type": "application/json" }
      const authHeader = c.req.header("authorization") ?? c.req.header("x-api-key")
      if (authHeader) {
        if (c.req.header("authorization")) internalHeaders["authorization"] = authHeader
        else internalHeaders["x-api-key"] = authHeader
      }
      const internalRes = await app.fetch(new Request(`http://localhost/v1/messages`, {
        method: "POST",
        headers: internalHeaders,
        body: JSON.stringify(anthropicBody)
      }))

      if (!stream) {
        const anthropicJson = await internalRes.json() as any
        if (anthropicJson.type === "error") {
          return c.json({ error: anthropicJson.error }, internalRes.status as any)
        }
        return c.json(anthropicToOpenaiResponse(anthropicJson, requestedModel))
      }

      const includeUsage = body.stream_options?.include_usage === true
      const encoder = new TextEncoder()
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
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                      id: chatId, object: "chat.completion.chunk", created, model: requestedModel,
                      choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }]
                    })}\n\n`))
                  }

                  if (event.type === "content_block_start" && event.content_block?.type === "tool_use") {
                    const idx = toolCallIndex++
                    activeToolCalls.set(event.index, { id: event.content_block.id, name: event.content_block.name })
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                      id: chatId, object: "chat.completion.chunk", created, model: requestedModel,
                      choices: [{ index: 0, delta: {
                        tool_calls: [{ index: idx, id: event.content_block.id, type: "function", function: { name: event.content_block.name, arguments: "" } }]
                      }, finish_reason: null }]
                    })}\n\n`))
                  } else if (event.type === "content_block_delta" && event.delta?.type === "input_json_delta") {
                    const tc = activeToolCalls.get(event.index)
                    if (tc) {
                      const idx = Array.from(activeToolCalls.keys()).indexOf(event.index)
                      controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                        id: chatId, object: "chat.completion.chunk", created, model: requestedModel,
                        choices: [{ index: 0, delta: {
                          tool_calls: [{ index: idx, function: { arguments: event.delta.partial_json } }]
                        }, finish_reason: null }]
                      })}\n\n`))
                    }
                  } else if (event.type === "content_block_delta" && event.delta?.type === "text_delta") {
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                      id: chatId, object: "chat.completion.chunk", created, model: requestedModel,
                      choices: [{ index: 0, delta: { content: event.delta.text }, finish_reason: null }]
                    })}\n\n`))
                  } else if (event.type === "message_delta") {
                    const sr = event.delta?.stop_reason
                    finishReason = sr === "tool_use" ? "tool_calls" : sr === "max_tokens" ? "length" : "stop"
                    if (event.usage) {
                      const prevInput: number = usageInfo?.input_tokens ?? 0
                      const prevOutput: number = usageInfo?.output_tokens ?? 0
                      usageInfo = {
                        input_tokens: event.usage.input_tokens ?? prevInput,
                        output_tokens: event.usage.output_tokens ?? prevOutput
                      }
                    }
                  } else if (event.type === "message_start" && event.message?.usage) {
                    usageInfo = { input_tokens: event.message.usage.input_tokens ?? 0, output_tokens: 0 }
                  } else if (event.type === "message_stop") {
                    const finalChunk: any = {
                      id: chatId, object: "chat.completion.chunk", created, model: requestedModel,
                      choices: [{ index: 0, delta: {}, finish_reason: finishReason ?? "stop" }]
                    }
                    if (includeUsage && usageInfo) {
                      finalChunk.usage = {
                        prompt_tokens: usageInfo.input_tokens,
                        completion_tokens: usageInfo.output_tokens,
                        total_tokens: usageInfo.input_tokens + usageInfo.output_tokens
                      }
                    }
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify(finalChunk)}\n\n`))
                    controller.enqueue(encoder.encode("data: [DONE]\n\n"))
                  }
                } catch {}
              }
            }
            controller.close()
          } catch {
            controller.close()
          }
        }
      })

      return new Response(readable, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive"
        }
      })
    } catch (error) {
      return c.json({
        error: { message: error instanceof Error ? error.message : "Unknown error", type: "server_error" }
      }, 500)
    }
  }

  app.post("/v1/chat/completions", handleChatCompletions)
  app.post("/chat/completions", handleChatCompletions)

  // OpenAI-format model listing
  const handleOpenaiModels = (c: Context) => c.json({
    object: "list",
    data: MODELS.map(m => ({
      id: m.id,
      object: "model",
      created: Math.floor(new Date(m.created_at).getTime() / 1000),
      owned_by: "anthropic"
    }))
  })
  app.get("/v1/chat/models", handleOpenaiModels)

  // 404 catch-all
  app.all("*", (c) => c.json({
    type: "error",
    error: { type: "not_found_error", message: `${c.req.method} ${c.req.path} not found` }
  }, 404))

  return { app, config: finalConfig }
}

export async function startProxyServer(config: Partial<ProxyConfig> = {}) {
  const { app, config: finalConfig } = createProxyServer(config)

  const server = Bun.serve({
    port: finalConfig.port,
    hostname: finalConfig.host,
    fetch: app.fetch,
    idleTimeout: 0
  })

  // Startup log with full configuration
  logInfo("proxy.started", {
    version: PROXY_VERSION,
    host: finalConfig.host,
    port: finalConfig.port,
    stallTimeoutMs: finalConfig.stallTimeoutMs,
    maxDurationMs: finalConfig.maxDurationMs,
    maxOutputChars: finalConfig.maxOutputChars,
    maxConcurrent: MAX_CONCURRENT,
    queueTimeoutMs: QUEUE_TIMEOUT_MS,
    claudeExecutable,
    logDir: LOG_DIR,
    debug: finalConfig.debug,
    pid: process.pid,
  })

  console.log(`Claude SDK Proxy v${PROXY_VERSION} running at http://${finalConfig.host}:${finalConfig.port}`)
  console.log(`  Logs: ${LOG_DIR}`)
  console.log(`  Debug: http://${finalConfig.host}:${finalConfig.port}/debug/stats`)

  // Periodic health logging (every 5 minutes)
  const healthInterval = setInterval(() => {
    const mem = process.memoryUsage()
    const stats = traceStore.getStats()
    logInfo("proxy.health", {
      pid: process.pid,
      rssBytes: mem.rss,
      rssMB: +(mem.rss / 1024 / 1024).toFixed(1),
      heapUsedMB: +(mem.heapUsed / 1024 / 1024).toFixed(1),
      heapTotalMB: +(mem.heapTotal / 1024 / 1024).toFixed(1),
      externalMB: +(mem.external / 1024 / 1024).toFixed(1),
      uptimeMs: stats.uptimeMs,
      totalRequests: stats.requests.total,
      totalErrors: stats.requests.errors,
      activeRequests: stats.requests.active,
      queueActive: requestQueue.activeCount,
      queueWaiting: requestQueue.waitingCount,
    })
  }, 300_000) // 5 minutes

  // Graceful shutdown
  const shutdown = (signal: string) => {
    const stats = traceStore.getStats()
    logInfo("proxy.shutdown", {
      signal,
      pid: process.pid,
      totalRequests: stats.requests.total,
      totalErrors: stats.requests.errors,
      activeRequests: stats.requests.active,
      uptimeMs: stats.uptimeMs,
    })
    clearInterval(healthInterval)
    console.log(`\nReceived ${signal}, shutting down...`)
    server.stop(true)
    process.exit(0)
  }
  process.on("SIGINT", () => shutdown("SIGINT"))
  process.on("SIGTERM", () => shutdown("SIGTERM"))

  return server
}
