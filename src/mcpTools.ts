import { createSdkMcpServer, tool } from "@anthropic-ai/claude-agent-sdk"
import { z } from "zod"
import { createPrivateKey, createPublicKey, sign, randomBytes } from "node:crypto"
import { readFileSync } from "node:fs"
import { homedir } from "node:os"

// ── Gateway helpers ──────────────────────────────────────────────────────────

function b64urlEncode(buf: Buffer): string {
  return buf.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "")
}

function signPayload(privateKeyPem: string, payload: string): string {
  const key = createPrivateKey(privateKeyPem)
  return b64urlEncode(sign(null, Buffer.from(payload, "utf8"), key))
}

function pubKeyRawB64url(publicKeyPem: string): string {
  const pubKey = createPublicKey(publicKeyPem)
  const der = pubKey.export({ type: "spki", format: "der" }) as Buffer
  return b64urlEncode(der.slice(12)) // strip 12-byte ED25519 SPKI prefix
}

let _identity: { deviceId: string; privateKeyPem: string; publicKeyPem: string } | null = null
let _gatewayToken: string | null = null

function loadGatewayConfig(): { identity: typeof _identity; token: string } {
  if (!_identity || !_gatewayToken) {
    const identity = JSON.parse(readFileSync(`${homedir()}/.openclaw/identity/device.json`, "utf8"))
    const cfg = JSON.parse(readFileSync(`${homedir()}/.openclaw/openclaw.json`, "utf8"))
    const token: string = cfg?.gateway?.auth?.token
    if (!token) throw new Error("gateway token not found in openclaw.json")
    _identity = identity
    _gatewayToken = token
  }
  return { identity: _identity!, token: _gatewayToken! }
}

function invalidateGatewayConfig() {
  _identity = null
  _gatewayToken = null
}

async function sendViaGateway(
  to: string,
  message?: string,
  mediaUrl?: string
): Promise<{ ok: boolean; error?: string }> {
  let identity: ReturnType<typeof loadGatewayConfig>["identity"]
  let token: string
  try {
    const cfg = loadGatewayConfig()
    identity = cfg.identity
    token = cfg.token
  } catch (e) {
    invalidateGatewayConfig()
    return { ok: false, error: `config error: ${e instanceof Error ? e.message : String(e)}` }
  }

  return new Promise((resolve) => {
    const ws = new WebSocket("ws://127.0.0.1:18789")
    let settled = false
    let connected = false

    const finish = (result: { ok: boolean; error?: string }) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      try { ws.close() } catch {}
      resolve(result)
    }

    const timer = setTimeout(() => finish({ ok: false, error: "timeout waiting for gateway" }), 10_000)

    ws.onerror = () => finish({ ok: false, error: "gateway websocket error" })

    ws.onclose = (event: CloseEvent) => {
      if (!settled) finish({ ok: false, error: `gateway closed unexpectedly (code=${event.code})` })
    }

    ws.onmessage = (event: MessageEvent) => {
      try {
        const frame = JSON.parse(event.data as string)

        if (!connected && frame.type === "event" && frame.event === "connect.challenge") {
          const nonce: string = frame.payload.nonce
          const signedAtMs = Date.now()
          const SCOPES = ["operator.admin", "operator.write"]
          const authPayload = ["v2", identity!.deviceId, "cli", "cli", "operator",
            SCOPES.join(","), String(signedAtMs), token, nonce].join("|")
          ws.send(JSON.stringify({
            type: "req", id: "conn1", method: "connect",
            params: {
              minProtocol: 3, maxProtocol: 3,
              client: { id: "cli", version: "1.0.0", platform: "linux", mode: "cli" },
              caps: [],
              scopes: SCOPES,
              auth: { token },
              device: {
                id: identity!.deviceId,
                publicKey: pubKeyRawB64url(identity!.publicKeyPem),
                signature: signPayload(identity!.privateKeyPem, authPayload),
                signedAt: signedAtMs,
                nonce
              }
            }
          }))

        } else if (!connected && frame.type === "res" && frame.id === "conn1") {
          if (!frame.ok) {
            if (frame.error?.message?.includes("unauthorized") ||
                frame.error?.message?.includes("pairing")) {
              invalidateGatewayConfig()
            }
            finish({ ok: false, error: `gateway connect failed: ${frame.error?.message || "unknown"}` })
            return
          }
          connected = true
          const sendParams: Record<string, unknown> = {
            to,
            channel: "telegram",
            idempotencyKey: randomBytes(16).toString("hex")
          }
          if (message) sendParams.message = message
          if (mediaUrl) sendParams.mediaUrl = mediaUrl
          ws.send(JSON.stringify({
            type: "req", id: "send1", method: "send",
            params: sendParams
          }))

        } else if (connected && frame.type === "res" && frame.id === "send1") {
          if (frame.ok) {
            finish({ ok: true })
          } else {
            finish({ ok: false, error: frame.error?.message || "send failed" })
          }
        }
      } catch (e) {
        finish({ ok: false, error: `parse error: ${e instanceof Error ? e.message : String(e)}` })
      }
    }
  })
}

// ── MCP server factory ───────────────────────────────────────────────────────
// Provides only the gateway message tool. File operations, bash, etc. use
// Claude Code's built-in tools (which are more robust and don't need MCP).
//
// state.messageSent is set on successful delivery. The proxy uses this to
// auto-suppress text responses when messages were sent via tool (prevents
// double-delivery without requiring Claude to know about any sentinel value).

export interface McpServerState { messageSent: boolean }

export function createMcpServer(state: McpServerState = { messageSent: false }) {
  return createSdkMcpServer({
    name: "opencode",
    version: "1.0.0",
    tools: [
      tool(
        "message",
        "Send a message or file to a chat. Provide `to` (chat ID from conversation_label, e.g. '-1001426819337'), and either `message` (text) or `filePath`/`path`/`media` (absolute path to a file). Write files to /tmp/ before sending.",
        {
          action: z.string().optional().describe("Action to perform. Default: 'send'."),
          to: z.string().describe("Chat ID, extracted from conversation_label."),
          message: z.string().optional().describe("Text message to send."),
          filePath: z.string().optional().describe("Absolute path to a file to send as attachment."),
          path: z.string().optional().describe("Alias for filePath."),
          media: z.string().optional().describe("Alias for filePath."),
          caption: z.string().optional().describe("Caption for a media attachment."),
        },
        async (args) => {
          try {
            const rawMedia = args.media ?? args.path ?? args.filePath
            let mediaUrl: string | undefined
            if (rawMedia) {
              if (rawMedia.startsWith("http://") || rawMedia.startsWith("https://") || rawMedia.startsWith("file://")) {
                mediaUrl = rawMedia
              } else {
                const absPath = rawMedia.startsWith("/") ? rawMedia : `/tmp/${rawMedia}`
                mediaUrl = `file://${absPath}`
              }
            }
            const textMessage = args.message ?? args.caption
            if (!textMessage && !mediaUrl) {
              return { content: [{ type: "text", text: "Error: provide message or filePath/path/media" }], isError: true }
            }
            const result = await sendViaGateway(args.to, textMessage, mediaUrl)
            if (result.ok) {
              state.messageSent = true
              return { content: [{ type: "text", text: `Sent to ${args.to}` }] }
            }
            return {
              content: [{ type: "text", text: `Failed: ${result.error}` }],
              isError: true
            }
          } catch (error) {
            return {
              content: [{ type: "text", text: `Error: ${error instanceof Error ? error.message : String(error)}` }],
              isError: true
            }
          }
        }
      )
    ]
  })
}
