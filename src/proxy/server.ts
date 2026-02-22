import { Hono } from "hono"
import { cors } from "hono/cors"
import { query } from "@anthropic-ai/claude-agent-sdk"
import type { Context } from "hono"
import type { ProxyConfig } from "./types"
import { DEFAULT_PROXY_CONFIG } from "./types"
import { claudeLog } from "../logger"
import { execSync } from "child_process"
import { existsSync, writeFileSync, unlinkSync, readFileSync } from "fs"
import { tmpdir } from "os"
import { randomBytes } from "crypto"
import { fileURLToPath } from "url"
import { join, dirname } from "path"
import { createMcpServer, type McpServerState } from "../mcpTools"

const PROXY_VERSION: string = (() => {
  try {
    const pkg = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), "../../package.json"), "utf-8"))
    return pkg.version ?? "unknown"
  } catch { return "unknown" }
})()

// Only block tools that add noise — everything else (Read, Write, Edit, Bash,
// Glob, Grep, WebFetch, WebSearch) uses Claude Code's robust built-in implementations.
const BLOCKED_BUILTIN_TOOLS = ["TodoWrite", "NotebookEdit"]

const MCP_SERVER_NAME = "opencode"

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

class RequestQueue {
  private active = 0
  private waiting: Array<() => void> = []

  get activeCount() { return this.active }
  get waitingCount() { return this.waiting.length }

  async acquire(): Promise<void> {
    if (this.active < MAX_CONCURRENT) {
      this.active++
      return
    }
    return new Promise<void>((resolve) => {
      this.waiting.push(() => { this.active++; resolve() })
    })
  }

  release(): void {
    this.active--
    const next = this.waiting.shift()
    if (next) next()
  }
}

const requestQueue = new RequestQueue()

function mapModelToClaudeModel(model: string): "sonnet" | "opus" | "haiku" {
  if (model.includes("opus")) return "opus"
  if (model.includes("haiku")) return "haiku"
  return "sonnet"
}

// ── Content-block serialization ──────────────────────────────────────────────

function saveImageToTemp(block: any, tempFiles: string[]): string | null {
  try {
    let data: string | undefined
    let mediaType = "image/jpeg"

    if (typeof block.data === "string") {
      data = block.data
      mediaType = block.media_type || mediaType
    } else if (block.source) {
      if (block.source.type === "base64" && block.source.data) {
        data = block.source.data
        mediaType = block.source.media_type || mediaType
      } else if (block.source.url) {
        return block.source.url
      }
    }

    if (!data) return null

    const ext = mediaType.split("/")[1]?.replace("jpeg", "jpg") || "jpg"
    const tmpPath = join(tmpdir(), `proxy-img-${randomBytes(8).toString("hex")}.${ext}`)
    writeFileSync(tmpPath, Buffer.from(data, "base64"))
    tempFiles.push(tmpPath)
    return tmpPath
  } catch {
    return null
  }
}

function serializeBlock(block: any, tempFiles: string[]): string {
  switch (block.type) {
    case "text":
      return block.text || ""
    case "image": {
      const imgPath = saveImageToTemp(block, tempFiles)
      return imgPath ? `[Image: ${imgPath}]` : "[Image: (unable to save)]"
    }
    case "tool_use":
      // Use <tool_use> XML format so the model continues using parseable blocks
      return `<tool_use>\n{"name": "${block.name}", "input": ${JSON.stringify(block.input ?? {})}}\n</tool_use>`
    case "tool_result": {
      const content = Array.isArray(block.content)
        ? block.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("")
        : String(block.content ?? "")
      const truncated = content.length > 4000
        ? content.slice(0, 4000) + `\n...[truncated ${content.length - 4000} chars]`
        : content
      return `<tool_result tool_use_id="${block.tool_use_id}">\n${truncated}\n</tool_result>`
    }
    case "thinking":
      return ""
    default:
      return ""
  }
}

function serializeContent(content: string | Array<any>, tempFiles: string[]): string {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return String(content)
  return content.map(b => serializeBlock(b, tempFiles)).filter(Boolean).join("\n")
}

function cleanupTempFiles(tempFiles: string[]) {
  for (const f of tempFiles) {
    try { unlinkSync(f) } catch {}
  }
}

// ── Client tool-use support ──────────────────────────────────────────────────
// When the caller provides tool definitions (e.g. Claude Code, LangChain, etc.)
// we switch to single-turn mode: inject tool defs into the system prompt, run
// one LLM turn, parse <tool_use> blocks from the output, and return them as
// proper Anthropic tool_use content blocks.
//
// We stay in agent mode (multi-turn, built-in + MCP tools) when:
//   - No tools in the request, OR
//   - The request has markers indicating the agent manages its own tool loop

function isClientToolMode(body: any): boolean {
  if (!body.tools?.length) return false
  if (body.messages?.some((m: any) =>
    Array.isArray(m.content) && m.content.some((b: any) => b.type === "tool_result")
  )) return true
  const sysText = Array.isArray(body.system)
    ? body.system.filter((b: any) => b.type === "text").map((b: any) => b.text).join(" ")
    : String(body.system ?? "")
  if (sysText.includes("conversation_label") || sysText.includes("chat id:")) return false
  return true
}

function buildClientToolsPrompt(tools: any[]): string {
  const defs = tools.map((t: any) => {
    const schema = t.input_schema ? `\nInput schema:\n${JSON.stringify(t.input_schema, null, 2)}` : ""
    return `### ${t.name}\n${t.description ?? ""}${schema}`
  }).join("\n\n")
  return `\n\n## Available Tools\n\nTo call a tool, output a <tool_use> block:\n\n` +
    `<tool_use>\n{"name": "TOOL_NAME", "input": {ARGUMENTS}}\n</tool_use>\n\n` +
    `- You may write reasoning text before the block\n` +
    `- Call multiple tools by including multiple <tool_use> blocks\n` +
    `- Each block must be valid JSON with "name" and "input" keys\n\n` +
    defs
}

interface ToolCall { id: string; name: string; input: unknown }

function parseToolUse(text: string): { toolCalls: ToolCall[]; textBefore: string } {
  const calls: ToolCall[] = []
  let firstIdx = -1

  // Parse <tool_use> XML blocks (primary format)
  const xmlRegex = /<tool_use>([\s\S]*?)<\/tool_use>/g
  let m: RegExpExecArray | null
  while ((m = xmlRegex.exec(text)) !== null) {
    if (firstIdx < 0) firstIdx = m.index
    try {
      const p = JSON.parse(m[1]!.trim())
      calls.push({
        id: `toolu_${randomBytes(16).toString("hex")}`,
        name: String(p.name ?? ""),
        input: p.input ?? {}
      })
    } catch { /* skip malformed block */ }
  }

  // Fallback: parse [Tool call: name\nInput: {...}] format
  if (calls.length === 0) {
    const bracketRegex = /\[Tool call:\s*(\w+)\s*\nInput:\s*([\s\S]*?)\]/g
    while ((m = bracketRegex.exec(text)) !== null) {
      if (firstIdx < 0) firstIdx = m.index
      try {
        const input = JSON.parse(m[2]!.trim())
        calls.push({
          id: `toolu_${randomBytes(16).toString("hex")}`,
          name: m[1]!.trim(),
          input
        })
      } catch { /* skip malformed block */ }
    }
  }

  return { toolCalls: calls, textBefore: firstIdx > 0 ? text.slice(0, firstIdx).trim() : "" }
}

function roughTokens(text: string): number {
  return Math.ceil((text ?? "").length / 4)
}

// ── Query options builder ────────────────────────────────────────────────────

function buildQueryOptions(
  model: "sonnet" | "opus" | "haiku",
  opts: {
    partial?: boolean
    clientToolMode?: boolean
    systemPrompt?: string
    mcpState?: McpServerState
    abortController?: AbortController
    maxThinkingTokens?: number
  } = {}
) {
  const base = {
    model,
    pathToClaudeCodeExecutable: claudeExecutable,
    permissionMode: "bypassPermissions" as const,
    allowDangerouslySkipPermissions: true,
    persistSession: false,
    settingSources: [],
    ...(opts.partial ? { includePartialMessages: true } : {}),
    ...(opts.abortController ? { abortController: opts.abortController } : {}),
    ...(opts.maxThinkingTokens ? { maxThinkingTokens: opts.maxThinkingTokens } : {}),
    ...(opts.systemPrompt ? { systemPrompt: opts.systemPrompt } : {}),
    disallowedTools: [...BLOCKED_BUILTIN_TOOLS],
  }

  if (opts.clientToolMode) {
    // Disable ALL built-in tools — the caller manages its own tool loop.
    // Tool definitions are already baked into the systemPrompt.
    return {
      ...base,
      maxTurns: 1,
      tools: [] as string[],
    }
  }

  return {
    ...base,
    maxTurns: 50,
    mcpServers: { [MCP_SERVER_NAME]: createMcpServer(opts.mcpState) }
  }
}

// ── Route handler ────────────────────────────────────────────────────────────

export function createProxyServer(config: Partial<ProxyConfig> = {}) {
  const finalConfig = { ...DEFAULT_PROXY_CONFIG, ...config }
  const app = new Hono()

  app.use("*", cors())

  // Optional API key validation — when CLAUDE_PROXY_API_KEY is set,
  // require a matching x-api-key or Authorization: Bearer header.
  const requiredApiKey = process.env.CLAUDE_PROXY_API_KEY
  if (requiredApiKey) {
    app.use("*", async (c, next) => {
      // Skip auth for health check and OPTIONS
      if (c.req.path === "/" || c.req.method === "OPTIONS") return next()
      const key = c.req.header("x-api-key")
        ?? c.req.header("authorization")?.replace(/^Bearer\s+/i, "")
      if (key !== requiredApiKey) {
        return c.json({
          type: "error",
          error: { type: "authentication_error", message: "Invalid API key" }
        }, 401)
      }
      return next()
    })
  }

  // Anthropic-compatible headers + request logging
  app.use("*", async (c, next) => {
    const start = Date.now()
    const requestId = c.req.header("x-request-id") ?? `req_${randomBytes(12).toString("hex")}`
    c.header("x-request-id", requestId)
    c.header("request-id", requestId)
    // Echo back Anthropic-standard headers
    c.header("anthropic-version", "2024-10-22")
    const betaHeader = c.req.header("anthropic-beta")
    if (betaHeader) c.header("anthropic-beta", betaHeader)
    await next()
    const ms = Date.now() - start
    claudeLog("proxy.http", { method: c.req.method, path: c.req.path, status: c.res.status, ms, requestId })
  })

  app.get("/", (c) => c.json({
    status: "ok",
    service: "claude-max-proxy",
    version: PROXY_VERSION,
    format: "anthropic",
    endpoints: ["/v1/messages", "/v1/models", "/v1/chat/completions"],
    queue: { active: requestQueue.activeCount, waiting: requestQueue.waitingCount, max: MAX_CONCURRENT }
  }))

  const MODELS = [
    { type: "model", id: "claude-opus-4-6",              display_name: "Claude Opus 4.6",    created_at: "2025-08-01T00:00:00Z" },
    { type: "model", id: "claude-opus-4-6-20250801",     display_name: "Claude Opus 4.6",    created_at: "2025-08-01T00:00:00Z" },
    { type: "model", id: "claude-sonnet-4-6",            display_name: "Claude Sonnet 4.6",  created_at: "2025-08-01T00:00:00Z" },
    { type: "model", id: "claude-sonnet-4-6-20250801",   display_name: "Claude Sonnet 4.6",  created_at: "2025-08-01T00:00:00Z" },
    { type: "model", id: "claude-sonnet-4-5-20250929",   display_name: "Claude Sonnet 4.5",  created_at: "2025-09-29T00:00:00Z" },
    { type: "model", id: "claude-haiku-4-5",             display_name: "Claude Haiku 4.5",   created_at: "2025-10-01T00:00:00Z" },
    { type: "model", id: "claude-haiku-4-5-20251001",    display_name: "Claude Haiku 4.5",   created_at: "2025-10-01T00:00:00Z" },
  ]

  // Dual-format model data: includes fields for both Anthropic and OpenAI SDKs
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

  const handleMessages = async (c: Context) => {
    const reqId = randomBytes(4).toString("hex")
    try {
      let body: any
      try {
        body = await c.req.json()
      } catch {
        return c.json({ type: "error", error: { type: "invalid_request_error", message: "Request body must be valid JSON" } }, 400)
      }

      if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
        return c.json({ type: "error", error: { type: "invalid_request_error", message: "messages is required and must be a non-empty array" } }, 400)
      }

      const model = mapModelToClaudeModel(body.model || "sonnet")
      const stream = body.stream ?? false
      const clientToolMode = isClientToolMode(body)
      const mcpState: McpServerState = { messageSent: false }
      const abortController = new AbortController()
      const timeout = setTimeout(() => abortController.abort(), finalConfig.requestTimeoutMs)

      // Extended thinking: extract budget_tokens from thinking parameter
      const maxThinkingTokens = body.thinking?.type === "enabled" ? body.thinking.budget_tokens : undefined

      claudeLog("proxy.request", { reqId, model, stream, msgs: body.messages?.length, clientToolMode, ...(maxThinkingTokens ? { maxThinkingTokens } : {}), queueActive: requestQueue.activeCount, queueWaiting: requestQueue.waitingCount })

      // Acquire a slot in the concurrency queue
      await requestQueue.acquire()

      const tempFiles: string[] = []

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

      // Build the prompt from messages. The SDK's query() takes a single prompt
      // string. To avoid the model continuing a "Human:/Assistant:" format in its
      // response, we use neutral delimiters and only the last user message as the
      // primary prompt when there's minimal context.
      const messages = body.messages as Array<{ role: string; content: string | Array<any> }>

      let prompt: string
      let systemPrompt: string | undefined

      if (clientToolMode) {
        // Client tool mode: serialize all messages as context, inject tools
        const conversationParts = messages
          .map((m) => {
            const label = m.role === "assistant" ? "[assistant]" : "[user]"
            return `${label}\n${serializeContent(m.content, tempFiles)}`
          })
          .join("\n\n")
        const toolsSection = buildClientToolsPrompt(body.tools)
        systemPrompt = systemContext
          ? `${systemContext}${toolsSection}`
          : toolsSection
        prompt = conversationParts
      } else if (messages.length === 1) {
        // Single message: pass directly as prompt (most common case)
        systemPrompt = systemContext || undefined
        prompt = serializeContent(messages[0]!.content, tempFiles)
      } else {
        // Multi-turn: build conversation context with neutral delimiters.
        // Put prior turns in system prompt as context, last user message as prompt.
        const lastMsg = messages[messages.length - 1]!
        const priorMsgs = messages.slice(0, -1)

        const contextParts = priorMsgs
          .map((m) => {
            const label = m.role === "assistant" ? "[assistant]" : "[user]"
            return `${label}\n${serializeContent(m.content, tempFiles)}`
          })
          .join("\n\n")

        const baseSystem = systemContext || ""
        const contextSection = contextParts
          ? `\n\n<conversation_history>\n${contextParts}\n</conversation_history>`
          : ""
        systemPrompt = (baseSystem + contextSection).trim() || undefined
        prompt = serializeContent(lastMsg.content, tempFiles)
      }

      // ── Non-streaming ──────────────────────────────────────────────────────
      if (!stream) {
        let fullText = ""
        let lastCleanText = ""
        try {
          for await (const message of query({ prompt, options: buildQueryOptions(model, { partial: false, clientToolMode, systemPrompt, mcpState, abortController, maxThinkingTokens }) })) {
            if (message.type === "assistant") {
              let turnText = ""
              let hasToolUse = false
              for (const block of message.message.content) {
                if (block.type === "text") turnText += block.text
                if (block.type === "tool_use") hasToolUse = true
              }
              if (!hasToolUse && turnText) {
                lastCleanText = turnText
              }
              fullText = turnText
            }
          }
        } finally {
          clearTimeout(timeout)
          cleanupTempFiles(tempFiles)
          requestQueue.release()
        }
        // In agent mode, prefer the last turn that had no tool_use
        if (!clientToolMode && lastCleanText) fullText = lastCleanText

        if (clientToolMode) {
          const { toolCalls, textBefore } = parseToolUse(fullText)
          const content: any[] = []
          if (textBefore) content.push({ type: "text", text: textBefore })
          for (const tc of toolCalls) content.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.input })
          if (content.length === 0) content.push({ type: "text", text: fullText || "..." })
          const stopReason = toolCalls.length > 0 ? "tool_use" : "end_turn"
          claudeLog("proxy.response", { reqId, len: fullText.length, toolCalls: toolCalls.length })
          return c.json({
            id: `msg_${Date.now()}`,
            type: "message", role: "assistant", content,
            model: body.model, stop_reason: stopReason, stop_sequence: null,
            usage: { input_tokens: roughTokens(prompt), output_tokens: roughTokens(fullText) }
          })
        }

        // If the MCP message tool delivered anything, suppress the proxy's
        // own text response so the client doesn't double-deliver.
        if (mcpState.messageSent) fullText = "NO_REPLY"
        if (!fullText || !fullText.trim()) fullText = "..."
        claudeLog("proxy.response", { reqId, len: fullText.length, messageSent: mcpState.messageSent })
        return c.json({
          id: `msg_${Date.now()}`,
          type: "message", role: "assistant",
          content: [{ type: "text", text: fullText }],
          model: body.model, stop_reason: "end_turn", stop_sequence: null,
          usage: { input_tokens: roughTokens(prompt), output_tokens: roughTokens(fullText) }
        })
      }

      // ── Streaming ──────────────────────────────────────────────────────────
      const encoder = new TextEncoder()
      const readable = new ReadableStream({
        async start(controller) {
          const messageId = `msg_${Date.now()}`
          let queueReleased = false
          const releaseQueue = () => { if (!queueReleased) { queueReleased = true; requestQueue.release() } }

          const sse = (event: string, data: object) => {
            try {
              controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`))
            } catch {}
          }

          try {
            const heartbeat = setInterval(() => {
              try { controller.enqueue(encoder.encode(": ping\n\n")) } catch { clearInterval(heartbeat) }
            }, 15_000)

            sse("message_start", {
              type: "message_start",
              message: {
                id: messageId, type: "message", role: "assistant", content: [],
                model: body.model, stop_reason: null, stop_sequence: null,
                usage: { input_tokens: roughTokens(prompt), output_tokens: 0 }
              }
            })

            // ── Client tool mode: buffer → emit blocks at end ─────────────
            if (clientToolMode) {
              let fullText = ""
              try {
                for await (const message of query({ prompt, options: buildQueryOptions(model, { partial: true, clientToolMode: true, systemPrompt, abortController, maxThinkingTokens }) })) {
                  if (message.type === "stream_event") {
                    const ev = message.event as any
                    if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") {
                      fullText += ev.delta.text ?? ""
                    }
                  }
                }
              } finally {
                clearInterval(heartbeat)
                clearTimeout(timeout)
                cleanupTempFiles(tempFiles)
                releaseQueue()
              }

              const { toolCalls, textBefore } = parseToolUse(fullText)
              claudeLog("proxy.stream.done", { reqId, len: fullText.length, toolCalls: toolCalls.length })

              let blockIdx = 0
              const textContent = toolCalls.length === 0 ? (fullText || "...") : textBefore
              if (textContent) {
                sse("content_block_start", { type: "content_block_start", index: blockIdx, content_block: { type: "text", text: "" } })
                sse("content_block_delta", { type: "content_block_delta", index: blockIdx, delta: { type: "text_delta", text: textContent } })
                sse("content_block_stop", { type: "content_block_stop", index: blockIdx })
                blockIdx++
              } else if (toolCalls.length === 0) {
                sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })
                sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "..." } })
                sse("content_block_stop", { type: "content_block_stop", index: 0 })
                blockIdx = 1
              }
              for (const tc of toolCalls) {
                sse("content_block_start", { type: "content_block_start", index: blockIdx, content_block: { type: "tool_use", id: tc.id, name: tc.name, input: "" } })
                sse("content_block_delta", { type: "content_block_delta", index: blockIdx, delta: { type: "input_json_delta", partial_json: JSON.stringify(tc.input) } })
                sse("content_block_stop", { type: "content_block_stop", index: blockIdx })
                blockIdx++
              }

              const stopReason = toolCalls.length > 0 ? "tool_use" : "end_turn"
              sse("message_delta", { type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: roughTokens(fullText) } })
              sse("message_stop", { type: "message_stop" })
              controller.close()
              return
            }

            // ── Agent mode: buffer turns, emit only the final turn ────────
            // The SDK runs multi-turn with tools (maxTurns=50). Intermediate
            // turns contain tool calls and reasoning that should NOT be sent
            // to the client. We buffer each turn's text and only emit the
            // final turn (the one without tool_use blocks).
            sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })

            let lastCleanTurnText = ""
            let currentTurnText = ""
            let currentTurnHasToolUse = false
            try {
              for await (const message of query({ prompt, options: buildQueryOptions(model, { partial: true, systemPrompt, mcpState, abortController, maxThinkingTokens }) })) {
                if (message.type === "stream_event") {
                  const ev = message.event as any
                  if (ev.type === "message_start") {
                    // New turn starting — save previous if it had no tool use
                    if (currentTurnText && !currentTurnHasToolUse) {
                      lastCleanTurnText = currentTurnText
                    }
                    currentTurnText = ""
                    currentTurnHasToolUse = false
                  } else if (ev.type === "content_block_start" && ev.content_block?.type === "tool_use") {
                    currentTurnHasToolUse = true
                  } else if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") {
                    currentTurnText += ev.delta.text ?? ""
                  }
                }
              }
            } finally {
              clearInterval(heartbeat)
              clearTimeout(timeout)
              cleanupTempFiles(tempFiles)
              releaseQueue()
            }

            // Use the last turn if it had no tool use, otherwise use the last clean turn
            const finalText = (!currentTurnHasToolUse && currentTurnText)
              ? currentTurnText
              : lastCleanTurnText

            claudeLog("proxy.stream.done", { reqId, len: finalText.length, messageSent: mcpState.messageSent, turns: lastCleanTurnText ? "multi" : "single" })
            const fullText = finalText

            // Emit the buffered final text as a single delta
            if (mcpState.messageSent) {
              // Messages were delivered via the MCP message tool — suppress
              sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "NO_REPLY" } })
            } else if (fullText && fullText.trim()) {
              sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: fullText } })
            } else {
              sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "..." } })
            }

            sse("content_block_stop", { type: "content_block_stop", index: 0 })
            sse("message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: roughTokens(fullText) } })
            sse("message_stop", { type: "message_stop" })
            controller.close()

          } catch (error) {
            clearTimeout(timeout)
            releaseQueue()
            const isAbort = error instanceof Error && error.name === "AbortError"
            const errMsg = isAbort ? "Request timeout" : (error instanceof Error ? error.message : "Unknown error")
            const errType = isAbort ? "timeout_error" : "api_error"
            claudeLog("proxy.stream.error", { reqId, error: errMsg })
            cleanupTempFiles(tempFiles)
            try {
              sse("error", { type: "error", error: { type: errType, message: errMsg } })
              controller.close()
            } catch {}
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
      const isAbort = error instanceof Error && error.name === "AbortError"
      const errMsg = isAbort ? "Request timeout" : (error instanceof Error ? error.message : "Unknown error")
      const errType = isAbort ? "timeout_error" : "api_error"
      const status = isAbort ? 408 : 500
      claudeLog("proxy.error", { reqId, error: errMsg })
      return c.json({ type: "error", error: { type: errType, message: errMsg } }, status)
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
  // Translates OpenAI ChatCompletion format to/from Anthropic Messages API
  // so tools expecting OpenAI endpoints (LangChain, LiteLLM, etc.) just work.

  function convertOpenaiContent(content: any): any {
    // String content → pass through
    if (typeof content === "string") return content
    if (!Array.isArray(content)) return String(content ?? "")

    // Array content → convert image_url parts to Anthropic image blocks
    return content.map((part: any) => {
      if (part.type === "text") return { type: "text", text: part.text ?? "" }
      if (part.type === "image_url" && part.image_url?.url) {
        const url = part.image_url.url as string
        // Data URL: data:image/jpeg;base64,...
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
        // HTTP URL — pass as URL source
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
        system = (system ? system + "\n" : "") + (typeof msg.content === "string" ? msg.content : "")
      } else if (msg.role === "user") {
        converted.push({ role: "user", content: convertOpenaiContent(msg.content) })
      } else if (msg.role === "assistant") {
        // Handle assistant messages with tool_calls (OpenAI format)
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
        // OpenAI tool result → Anthropic tool_result
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
      id: `chatcmpl-${Date.now()}`,
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

      // Build Anthropic-format request body
      const anthropicBody: any = {
        model: requestedModel,
        messages,
        stream,
      }
      if (system) anthropicBody.system = system
      if (body.max_tokens) anthropicBody.max_tokens = body.max_tokens
      if (body.temperature !== undefined) anthropicBody.temperature = body.temperature
      // Convert OpenAI tools format to Anthropic tools format
      if (body.tools?.length) {
        anthropicBody.tools = openaiToAnthropicTools(body.tools)
      }

      // Forward to our own /v1/messages handler by making an internal request
      const internalRes = await app.fetch(new Request(`http://localhost/v1/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(anthropicBody)
      }))

      if (!stream) {
        const anthropicJson = await internalRes.json() as any
        if (anthropicJson.type === "error") {
          return c.json({ error: anthropicJson.error }, internalRes.status as any)
        }
        return c.json(anthropicToOpenaiResponse(anthropicJson, requestedModel))
      }

      // Streaming: translate SSE events from Anthropic format to OpenAI format
      const includeUsage = body.stream_options?.include_usage === true
      const encoder = new TextEncoder()
      const readable = new ReadableStream({
        async start(controller) {
          try {
            const reader = internalRes.body?.getReader()
            if (!reader) { controller.close(); return }

            const decoder = new TextDecoder()
            let buffer = ""
            const chatId = `chatcmpl-${Date.now()}`
            const created = Math.floor(Date.now() / 1000)
            let sentRole = false
            let finishReason: string | null = null
            // Track active tool calls for streaming
            const activeToolCalls: Map<number, { id: string; name: string; argBuffer: string }> = new Map()
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

                  // Emit role delta on first event
                  if (!sentRole && (event.type === "content_block_start" || event.type === "content_block_delta")) {
                    sentRole = true
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                      id: chatId, object: "chat.completion.chunk", created, model: requestedModel,
                      choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }]
                    })}\n\n`))
                  }

                  if (event.type === "content_block_start" && event.content_block?.type === "tool_use") {
                    // Start of a tool_use block → emit tool_call function header
                    const idx = toolCallIndex++
                    activeToolCalls.set(event.index, { id: event.content_block.id, name: event.content_block.name, argBuffer: "" })
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                      id: chatId, object: "chat.completion.chunk", created, model: requestedModel,
                      choices: [{ index: 0, delta: {
                        tool_calls: [{ index: idx, id: event.content_block.id, type: "function", function: { name: event.content_block.name, arguments: "" } }]
                      }, finish_reason: null }]
                    })}\n\n`))
                  } else if (event.type === "content_block_delta" && event.delta?.type === "input_json_delta") {
                    // Tool call argument streaming
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
                    // Capture finish reason and usage for final chunk
                    const sr = event.delta?.stop_reason
                    finishReason = sr === "tool_use" ? "tool_calls" : sr === "max_tokens" ? "length" : "stop"
                    if (event.usage) {
                      usageInfo = { input_tokens: event.usage.input_tokens ?? 0, output_tokens: event.usage.output_tokens ?? 0 }
                    }
                  } else if (event.type === "message_start" && event.message?.usage) {
                    // Capture input token count from message_start
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

  console.log(`Claude Max Proxy v${PROXY_VERSION} running at http://${finalConfig.host}:${finalConfig.port}`)

  // Graceful shutdown
  const shutdown = (signal: string) => {
    console.log(`\nReceived ${signal}, shutting down...`)
    server.stop(true) // true = wait for in-flight requests
    process.exit(0)
  }
  process.on("SIGINT", () => shutdown("SIGINT"))
  process.on("SIGTERM", () => shutdown("SIGTERM"))

  return server
}
