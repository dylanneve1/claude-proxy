import { Hono } from "hono"
import { cors } from "hono/cors"
import { query } from "@anthropic-ai/claude-agent-sdk"
import type { Context } from "hono"
import type { ProxyConfig } from "./types"
import { DEFAULT_PROXY_CONFIG } from "./types"
import { claudeLog } from "../logger"
import { execSync } from "child_process"
import { existsSync, writeFileSync, unlinkSync } from "fs"
import { tmpdir } from "os"
import { randomBytes } from "crypto"
import { fileURLToPath } from "url"
import { join, dirname } from "path"
import { createMcpServer } from "../mcpTools"

// Built-in Claude Code tools to block. "Read" is intentionally kept so Claude
// can open image temp-files saved from inbound media blocks.
const BLOCKED_BUILTIN_TOOLS = [
  "Write", "Edit", "MultiEdit",
  "Bash", "Glob", "Grep", "NotebookEdit",
  "WebFetch", "WebSearch", "TodoWrite"
]

const MCP_SERVER_NAME = "opencode"

const ALLOWED_MCP_TOOLS = [
  `mcp__${MCP_SERVER_NAME}__read`,
  `mcp__${MCP_SERVER_NAME}__write`,
  `mcp__${MCP_SERVER_NAME}__edit`,
  `mcp__${MCP_SERVER_NAME}__bash`,
  `mcp__${MCP_SERVER_NAME}__glob`,
  `mcp__${MCP_SERVER_NAME}__grep`,
  `mcp__${MCP_SERVER_NAME}__message`,
]

// Injected after the system prompt. Bridges openclaw's "message" tool
// description to the actual MCP tool name, and clarifies file paths.
const SEND_MESSAGE_NOTE = `
## Tool note (proxy context)
The \`message\` tool described in your system prompt is available as \`mcp__opencode__message\`.
- Use it exactly as described: action=send, with \`to\` (chat ID from conversation_label), \`message\` (text), and/or \`filePath\` (absolute path like /tmp/file.png).
- Always write files to /tmp/ before sending — use bash or write tool with an absolute /tmp/ path.
- After all message tool calls, output ONLY: NO_REPLY`

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
      // openclaw format: { type: "image", data: "<base64>", media_type?: "…" }
      data = block.data
      mediaType = block.media_type || mediaType
    } else if (block.source) {
      // Anthropic API format: { type: "image", source: { type: "base64", data, media_type } }
      if (block.source.type === "base64" && block.source.data) {
        data = block.source.data
        mediaType = block.source.media_type || mediaType
      } else if (block.source.url) {
        return block.source.url
      }
    }

    if (!data) return null

    const ext = mediaType.split("/")[1]?.replace("jpeg", "jpg") || "jpg"
    const tmpPath = join(tmpdir(), `openclaw-img-${randomBytes(8).toString("hex")}.${ext}`)
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
      // Truncate large tool results — cap at 4000 chars to prevent context explosion.
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

// ── Response normalization ───────────────────────────────────────────────────

// If the agent used the message tool it may output verbose text before NO_REPLY.
// openclaw's isSilentReplyText handles "ends with NO_REPLY", but we normalize to
// just "NO_REPLY" for a clean session history.
// If the agent produced no text (maxTurns exhausted or empty run), return "..."
// so openclaw shows something rather than silence.
function normalizeResponse(text: string): string {
  if (!text || !text.trim()) return "..."
  const trimmed = text.trimEnd()
  if (trimmed.endsWith("NO_REPLY")) return "NO_REPLY"
  return text
}

// ── Query options builder ────────────────────────────────────────────────────

function buildQueryOptions(model: "sonnet" | "opus" | "haiku", partial?: boolean) {
  return {
    maxTurns: 50,
    model,
    pathToClaudeCodeExecutable: claudeExecutable,
    ...(partial ? { includePartialMessages: true } : {}),
    disallowedTools: [...BLOCKED_BUILTIN_TOOLS],
    allowedTools: [...ALLOWED_MCP_TOOLS],
    mcpServers: {
      // Fresh instance per request — avoids concurrency issues when multiple
      // requests share the same SDK MCP server object.
      [MCP_SERVER_NAME]: createMcpServer()
    }
  }
}

// ── Route handler ────────────────────────────────────────────────────────────

export function createProxyServer(config: Partial<ProxyConfig> = {}) {
  const finalConfig = { ...DEFAULT_PROXY_CONFIG, ...config }
  const app = new Hono()

  app.use("*", cors())

  app.get("/", (c) => c.json({
    status: "ok",
    service: "claude-max-proxy",
    version: "1.0.0",
    format: "anthropic",
    endpoints: ["/v1/messages", "/messages"]
  }))

  const handleMessages = async (c: Context) => {
    const reqId = randomBytes(4).toString("hex")
    try {
      const body = await c.req.json()
      const model = mapModelToClaudeModel(body.model || "sonnet")
      const stream = body.stream ?? true

      claudeLog("proxy.request", { reqId, model, stream, msgs: body.messages?.length })

      const tempFiles: string[] = []

      // System prompt
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

      // Conversation history
      const conversationParts = body.messages
        ?.map((m: { role: string; content: string | Array<any> }) => {
          const role = m.role === "assistant" ? "Assistant" : "Human"
          return `${role}: ${serializeContent(m.content, tempFiles)}`
        })
        .join("\n\n") || ""

      // Build prompt: system context + send_message guidance + conversation.
      // SEND_MESSAGE_NOTE is always appended so the tool is explained even
      // when there is no system prompt (e.g. plain API calls).
      const prompt = systemContext
        ? `${systemContext}${SEND_MESSAGE_NOTE}\n\n${conversationParts}`
        : `${SEND_MESSAGE_NOTE}\n\n${conversationParts}`

      // ── Non-streaming ──────────────────────────────────────────────────────
      if (!stream) {
        let fullContent = ""
        try {
          for await (const message of query({ prompt, options: buildQueryOptions(model) })) {
            if (message.type === "assistant") {
              for (const block of message.message.content) {
                if (block.type === "text") fullContent += block.text
              }
            }
          }
        } finally {
          cleanupTempFiles(tempFiles)
        }

        fullContent = normalizeResponse(fullContent)
        claudeLog("proxy.response", { reqId, len: fullContent.length })
        return c.json({
          id: `msg_${Date.now()}`,
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: fullContent }],
          model: body.model,
          stop_reason: "end_turn",
          usage: { input_tokens: 0, output_tokens: 0 }
        })
      }

      // ── Streaming ──────────────────────────────────────────────────────────
      // Emit message_start immediately before the agent run so the client sees
      // an active stream from the first byte — prevents HTTP timeouts during
      // long multi-turn tool work. Text deltas stream in real-time.
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
                id: messageId,
                type: "message",
                role: "assistant",
                content: [],
                model: body.model,
                stop_reason: null,
                stop_sequence: null,
                usage: { input_tokens: 0, output_tokens: 0 }
              }
            })
            sse("content_block_start", {
              type: "content_block_start",
              index: 0,
              content_block: { type: "text", text: "" }
            })

            // Stream text deltas in real-time as the agent produces them.
            // Accumulate into fullText for normalization at the end.
            let fullText = ""

            try {
              for await (const message of query({ prompt, options: buildQueryOptions(model, true) })) {
                if (message.type === "stream_event") {
                  const ev = message.event as any
                  if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") {
                    const chunk = ev.delta.text ?? ""
                    if (chunk) {
                      fullText += chunk
                      sse("content_block_delta", {
                        type: "content_block_delta",
                        index: 0,
                        delta: { type: "text_delta", text: chunk }
                      })
                    }
                  }
                }
              }
            } finally {
              clearInterval(heartbeat)
              cleanupTempFiles(tempFiles)
            }

            // Normalize after the full run: if text ends with NO_REPLY, collapse
            // it; if empty (maxTurns exhausted), emit the fallback "...".
            const normalized = normalizeResponse(fullText)
            claudeLog("proxy.stream.done", { reqId, len: fullText.length, normalized: normalized !== fullText })

            if (normalized !== fullText) {
              // The accumulated deltas were already sent raw. We need to send
              // a correction delta: clear what was sent and resend the normalized
              // form. The cleanest approach is to send one more delta with the
              // difference, but SSE text streams are append-only. Instead, send
              // a terminal delta that replaces — openclaw accumulates all deltas,
              // so we emit a negative-length correction by sending a stop and a
              // fresh block with only the normalized text.
              //
              // Practical note: "NO_REPLY" is the main case. openclaw checks
              // trimEnd().endsWith("NO_REPLY") on the full accumulated text, so
              // even if we streamed "...\nNO_REPLY" it will still be treated as
              // silent. We just need to make sure we don't emit an extra delta
              // for the empty-text case since we already streamed nothing.
              if (!fullText || !fullText.trim()) {
                // Agent produced no text at all — send the "..." fallback now.
                sse("content_block_delta", {
                  type: "content_block_delta",
                  index: 0,
                  delta: { type: "text_delta", text: normalized }
                })
              }
              // If normalized === "NO_REPLY" and we already streamed the raw text,
              // openclaw will still handle it correctly via its endsWith check.
            }

            sse("content_block_stop", { type: "content_block_stop", index: 0 })
            sse("message_delta", {
              type: "message_delta",
              delta: { stop_reason: "end_turn", stop_sequence: null },
              usage: { output_tokens: fullText.length }
            })
            sse("message_stop", { type: "message_stop" })
            controller.close()

          } catch (error) {
            claudeLog("proxy.stream.error", { reqId, error: error instanceof Error ? error.message : String(error) })
            cleanupTempFiles(tempFiles)
            try {
              sse("error", {
                type: "error",
                error: { type: "api_error", message: error instanceof Error ? error.message : "Unknown error" }
              })
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
      claudeLog("proxy.error", { reqId, error: error instanceof Error ? error.message : String(error) })
      return c.json({
        type: "error",
        error: { type: "api_error", message: error instanceof Error ? error.message : "Unknown error" }
      }, 500)
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
