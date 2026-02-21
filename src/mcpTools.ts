import { createSdkMcpServer, tool } from "@anthropic-ai/claude-agent-sdk"
import { z } from "zod"
import * as fs from "node:fs/promises"
import * as path from "node:path"
import { exec } from "node:child_process"
import { promisify } from "node:util"
import { glob as globLib } from "glob"
import { createPrivateKey, createPublicKey, sign, randomBytes } from "node:crypto"
import { readFileSync } from "node:fs"
import { homedir } from "node:os"

const execAsync = promisify(exec)
const getCwd = () => process.cwd()

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

// Cache identity + gateway token — read once, invalidate on error
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

/**
 * Send a message (text and/or media) to a Telegram chat via the openclaw gateway WebSocket.
 * Uses Ed25519 device identity with operator.admin scope (bypasses all method
 * scope checks server-side). Opens a fresh WS connection per call.
 */
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
          const SCOPES = ["operator.admin"]
          // Build the auth payload that gets signed with the device Ed25519 key.
          // Format: v2|deviceId|clientId|clientMode|role|scopes|signedAtMs|token|nonce
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
            // If connect fails with auth error, invalidate cached creds
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
// Returns a fresh MCP server instance. Called per-request to avoid concurrency
// issues when multiple simultaneous queries share the same server object.

export function createMcpServer() {
  return createSdkMcpServer({
    name: "opencode",
    version: "1.0.0",
    tools: [

      tool(
        "read",
        "Read the contents of a file at the specified path",
        {
          path: z.string().describe("Absolute or relative path to the file"),
          encoding: z.string().optional().describe("File encoding, defaults to utf-8")
        },
        async (args) => {
          try {
            const filePath = path.isAbsolute(args.path)
              ? args.path
              : path.resolve(getCwd(), args.path)
            const content = await fs.readFile(filePath, (args.encoding || "utf-8") as BufferEncoding)
            return { content: [{ type: "text", text: content }] }
          } catch (error) {
            return {
              content: [{ type: "text", text: `Error reading file: ${error instanceof Error ? error.message : String(error)}` }],
              isError: true
            }
          }
        }
      ),

      tool(
        "write",
        "Write content to a file, creating directories if needed",
        {
          path: z.string().describe("Path to write to"),
          content: z.string().describe("Content to write")
        },
        async (args) => {
          try {
            const filePath = path.isAbsolute(args.path)
              ? args.path
              : path.resolve(getCwd(), args.path)
            await fs.mkdir(path.dirname(filePath), { recursive: true })
            await fs.writeFile(filePath, args.content, "utf-8")
            return { content: [{ type: "text", text: `Successfully wrote to ${filePath}` }] }
          } catch (error) {
            return {
              content: [{ type: "text", text: `Error writing file: ${error instanceof Error ? error.message : String(error)}` }],
              isError: true
            }
          }
        }
      ),

      tool(
        "edit",
        "Edit a file by replacing oldString with newString",
        {
          path: z.string().describe("Path to the file to edit"),
          oldString: z.string().describe("The text to replace"),
          newString: z.string().describe("The replacement text")
        },
        async (args) => {
          try {
            const filePath = path.isAbsolute(args.path)
              ? args.path
              : path.resolve(getCwd(), args.path)
            const content = await fs.readFile(filePath, "utf-8")
            if (!content.includes(args.oldString)) {
              return { content: [{ type: "text", text: "Error: oldString not found in file" }], isError: true }
            }
            await fs.writeFile(filePath, content.replace(args.oldString, args.newString), "utf-8")
            return { content: [{ type: "text", text: `Successfully edited ${args.path}` }] }
          } catch (error) {
            return {
              content: [{ type: "text", text: `Error editing file: ${error instanceof Error ? error.message : String(error)}` }],
              isError: true
            }
          }
        }
      ),

      tool(
        "bash",
        "Execute a bash command and return the output",
        {
          command: z.string().describe("The command to execute"),
          cwd: z.string().optional().describe("Working directory for the command")
        },
        async (args) => {
          try {
            const { stdout, stderr } = await execAsync(args.command, {
              cwd: args.cwd || getCwd(),
              timeout: 120_000
            })
            return { content: [{ type: "text", text: stdout || stderr || "(no output)" }] }
          } catch (error: unknown) {
            const e = error as { stdout?: string; stderr?: string; message?: string }
            return {
              content: [{ type: "text", text: e.stdout || e.stderr || e.message || String(error) }],
              isError: true
            }
          }
        }
      ),

      tool(
        "glob",
        "Find files matching a glob pattern",
        {
          pattern: z.string().describe("Glob pattern like **/*.ts"),
          cwd: z.string().optional().describe("Base directory for the search")
        },
        async (args) => {
          try {
            const files = await globLib(args.pattern, {
              cwd: args.cwd || getCwd(),
              nodir: true,
              ignore: ["**/node_modules/**", "**/.git/**"]
            })
            return { content: [{ type: "text", text: files.join("\n") || "(no matches)" }] }
          } catch (error) {
            return {
              content: [{ type: "text", text: `Error: ${error instanceof Error ? error.message : String(error)}` }],
              isError: true
            }
          }
        }
      ),

      tool(
        "grep",
        "Search for a pattern in files",
        {
          pattern: z.string().describe("Regex pattern to search for"),
          path: z.string().optional().describe("Directory or file to search in"),
          include: z.string().optional().describe("File pattern to include, e.g., *.ts")
        },
        async (args) => {
          try {
            const searchPath = args.path || getCwd()
            const includePattern = args.include || "*"
            const cmd = `grep -rn --include="${includePattern}" "${args.pattern}" "${searchPath}" 2>/dev/null || true`
            const { stdout } = await execAsync(cmd, { maxBuffer: 10 * 1024 * 1024 })
            return { content: [{ type: "text", text: stdout || "(no matches)" }] }
          } catch (error) {
            return {
              content: [{ type: "text", text: `Error: ${error instanceof Error ? error.message : String(error)}` }],
              isError: true
            }
          }
        }
      ),

      tool(
        "message",
        "Send a Telegram message or file via the openclaw gateway. Use action=send (default). Provide `to` (chat ID from conversation_label, e.g. '-1001426819337'), and either `message` (text) or `filePath`/`path`/`media` (absolute path to file in /tmp/, e.g. '/tmp/image.png'). Always write files to /tmp/ first. After all calls, output only: NO_REPLY",
        {
          action: z.string().optional().describe("Action to perform. Use 'send' (default)."),
          to: z.string().describe(
            "Telegram chat ID. Extract from conversation_label: e.g. 'chat id:-1001426819337' → '-1001426819337'."
          ),
          message: z.string().optional().describe("Text message to send."),
          filePath: z.string().optional().describe("Absolute path to a file in /tmp/ to send as attachment, e.g. '/tmp/image.png'."),
          path: z.string().optional().describe("Alias for filePath."),
          media: z.string().optional().describe("Alias for filePath — file path or URL to send as media attachment."),
          caption: z.string().optional().describe("Caption text to accompany a media attachment."),
        },
        async (args) => {
          try {
            // Resolve media path from whichever alias Claude provides
            const rawMedia = args.media ?? args.path ?? args.filePath
            // Normalize to a file:// URL if it's a plain path (gateway requires this for local files)
            let mediaUrl: string | undefined
            if (rawMedia) {
              if (rawMedia.startsWith("http://") || rawMedia.startsWith("https://") || rawMedia.startsWith("file://")) {
                mediaUrl = rawMedia
              } else {
                // Plain absolute path → file:// URL
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
