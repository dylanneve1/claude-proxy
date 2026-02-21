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
      return `[Tool call: ${block.name}\nInput: ${JSON.stringify(block.input ?? {}, null, 2)}]`
    case "tool_result": {
      const content = Array.isArray(block.content)
        ? block.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("")
        : String(block.content ?? "")
      const truncated = content.length > 4000
        ? content.slice(0, 4000) + `\n...[truncated ${content.length - 4000} chars]`
        : content
      return `[Tool result (id=${block.tool_use_id}): ${truncated}]`
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
  const regex = /<tool_use>([\s\S]*?)<\/tool_use>/g
  const calls: ToolCall[] = []
  let firstIdx = -1
  let m: RegExpExecArray | null
  while ((m = regex.exec(text)) !== null) {
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
    clientSystemPrompt?: string
    mcpState?: McpServerState
    abortController?: AbortController
  } = {}
) {
  const base = {
    model,
    pathToClaudeCodeExecutable: claudeExecutable,
    permissionMode: "bypassPermissions" as const,
    allowDangerouslySkipPermissions: true,
    persistSession: false,
    ...(opts.partial ? { includePartialMessages: true } : {}),
    ...(opts.abortController ? { abortController: opts.abortController } : {}),
    disallowedTools: [...BLOCKED_BUILTIN_TOOLS],
  }

  if (opts.clientToolMode) {
    // Disable ALL built-in tools — the caller manages its own tool loop.
    // Inject tool definitions via systemPrompt so Claude respects them.
    return {
      ...base,
      maxTurns: 1,
      tools: [] as string[],
      ...(opts.clientSystemPrompt ? { systemPrompt: opts.clientSystemPrompt } : {}),
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

  // Request logging
  app.use("*", async (c, next) => {
    const start = Date.now()
    await next()
    const ms = Date.now() - start
    claudeLog("proxy.http", { method: c.req.method, path: c.req.path, status: c.res.status, ms })
  })

  app.get("/", (c) => c.json({
    status: "ok",
    service: "claude-max-proxy",
    version: PROXY_VERSION,
    format: "anthropic",
    endpoints: ["/v1/messages", "/messages", "/v1/models", "/models"]
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

  const handleModels = (c: Context) => c.json({ data: MODELS })
  app.get("/v1/models", handleModels)
  app.get("/models", handleModels)

  const handleModel = (c: Context) => {
    const id = c.req.param("id")
    const model = MODELS.find(m => m.id === id)
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
      const body = await c.req.json()

      if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
        return c.json({ type: "error", error: { type: "invalid_request_error", message: "messages is required and must be a non-empty array" } }, 400)
      }

      const model = mapModelToClaudeModel(body.model || "sonnet")
      const stream = body.stream ?? true
      const clientToolMode = isClientToolMode(body)
      const mcpState: McpServerState = { messageSent: false }
      const abortController = new AbortController()
      const timeout = setTimeout(() => abortController.abort(), finalConfig.requestTimeoutMs)

      claudeLog("proxy.request", { reqId, model, stream, msgs: body.messages?.length, clientToolMode })

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

      const conversationParts = body.messages
        ?.map((m: { role: string; content: string | Array<any> }) => {
          const role = m.role === "assistant" ? "Assistant" : "Human"
          return `${role}: ${serializeContent(m.content, tempFiles)}`
        })
        .join("\n\n") || ""

      // Client tool mode: put system+tools into SDK systemPrompt, conversation as prompt.
      // Agent mode: pass system + conversation as prompt text, no injected notes.
      let prompt: string
      let clientSystemPrompt: string | undefined
      if (clientToolMode) {
        const toolsSection = buildClientToolsPrompt(body.tools)
        clientSystemPrompt = systemContext
          ? `${systemContext}${toolsSection}`
          : toolsSection
        prompt = conversationParts
      } else {
        prompt = systemContext
          ? `${systemContext}\n\n${conversationParts}`
          : conversationParts
      }

      // ── Non-streaming ──────────────────────────────────────────────────────
      if (!stream) {
        let fullText = ""
        try {
          for await (const message of query({ prompt, options: buildQueryOptions(model, { partial: false, clientToolMode, clientSystemPrompt, mcpState, abortController }) })) {
            if (message.type === "assistant") {
              fullText = ""
              for (const block of message.message.content) {
                if (block.type === "text") fullText += block.text
              }
            }
          }
        } finally {
          clearTimeout(timeout)
          cleanupTempFiles(tempFiles)
        }

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
                for await (const message of query({ prompt, options: buildQueryOptions(model, { partial: true, clientToolMode: true, clientSystemPrompt, abortController }) })) {
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

            // ── Agent mode: stream text deltas in real-time ───────────────
            sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })

            let fullText = ""
            try {
              for await (const message of query({ prompt, options: buildQueryOptions(model, { partial: true, mcpState, abortController }) })) {
                if (message.type === "stream_event") {
                  const ev = message.event as any
                  if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") {
                    const chunk = ev.delta.text ?? ""
                    if (chunk) {
                      fullText += chunk
                      sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: chunk } })
                    }
                  }
                }
              }
            } finally {
              clearInterval(heartbeat)
              clearTimeout(timeout)
              cleanupTempFiles(tempFiles)
            }

            claudeLog("proxy.stream.done", { reqId, len: fullText.length, messageSent: mcpState.messageSent })

            // Auto-suppress: if messages were delivered via the MCP message tool,
            // append NO_REPLY so the client knows not to double-deliver.
            if (mcpState.messageSent && !fullText.trimEnd().endsWith("NO_REPLY")) {
              sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "\nNO_REPLY" } })
            } else if (!fullText || !fullText.trim()) {
              sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "..." } })
            }

            sse("content_block_stop", { type: "content_block_stop", index: 0 })
            sse("message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: roughTokens(fullText) } })
            sse("message_stop", { type: "message_stop" })
            controller.close()

          } catch (error) {
            clearTimeout(timeout)
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

  console.log(`Claude Max Proxy running at http://${finalConfig.host}:${finalConfig.port}`)
  return server
}
