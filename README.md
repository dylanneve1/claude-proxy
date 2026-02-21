# claude-proxy

A proxy server that bridges **openclaw** (and any Anthropic Messages API client) to the **Claude Agent SDK**, letting Claude Max subscribers use openclaw at zero API cost with full Telegram multi-message and image support.

```
openclaw → claude-proxy (:3456) → Claude Agent SDK → Claude Max subscription
```

## Features

- **Claude Max** — zero API cost, uses your existing subscription via the official Agent SDK
- **Streaming SSE** — `message_start` emitted immediately so long agent runs never hit HTTP timeouts
- **Multi-message sends** — Claude calls the `message` MCP tool multiple times to send separate Telegram messages, then outputs `NO_REPLY` to suppress duplicate delivery
- **Image / file sends** — Claude writes to `/tmp/`, calls `message` with `filePath`, proxy converts to `file://` URL for gateway delivery
- **Inbound images** — base64 image blocks saved to temp files so Claude can read them with the `read` tool
- **Tool serialization** — `tool_use` / `tool_result` / `thinking` blocks serialized cleanly for the Agent SDK
- **Large tool result protection** — results truncated at 4 000 chars to prevent context explosion
- **Per-request MCP servers** — no shared state between concurrent requests
- **Empty response fallback** — if maxTurns is exhausted with no text output, returns `"..."` instead of silence

## Architecture

```
POST /v1/messages  (Anthropic Messages API)
  │
  ├─ Deserialize messages → text, images (→ /tmp/), tool_use, tool_result
  ├─ Build prompt: system + SEND_MESSAGE_NOTE + conversation history
  │
  └─ Claude Agent SDK  query()  (maxTurns=50)
       │
       ├─ MCP tools:
       │    • message  → openclaw gateway WS (Ed25519 auth, operator.admin scope)
       │    • read / write / edit / bash / glob / grep
       │
       └─ Normalize response → Anthropic SSE stream or JSON
```

## Quick Start

### Prerequisites

1. **Claude Max subscription** + Claude CLI authenticated:
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude login
   ```

2. **Bun** runtime:
   ```bash
   curl -fsSL https://bun.sh/install | bash
   ```

3. **openclaw** with gateway enabled (provides the `message` tool's WebSocket backend):
   ```bash
   openclaw gateway
   ```

### Install & Run

```bash
git clone https://github.com/dylanneve1/claude-proxy
cd claude-proxy
bun install
bun run proxy
# Proxy listening at http://127.0.0.1:3456
```

### Configure openclaw

In `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "claude-proxy": {
        "baseUrl": "http://127.0.0.1:3456",
        "apiKey": "dummy",
        "api": "anthropic-messages",
        "models": [
          { "id": "claude-sonnet-4-6" },
          { "id": "claude-opus-4-6" }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "claude-proxy/claude-sonnet-4-6"
      }
    }
  }
}
```

### systemd (Linux auto-start)

```ini
[Unit]
Description=Claude Max API Proxy
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/claude-proxy
ExecStart=/home/user/.bun/bin/bun run proxy
Restart=always
RestartSec=3
Environment=CLAUDE_PROXY_PORT=3456

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now claude-max-api.service
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_PROXY_PORT` | `3456` | Proxy listen port |
| `CLAUDE_PROXY_HOST` | `127.0.0.1` | Proxy bind address |
| `OPENCODE_CLAUDE_PROVIDER_DEBUG` | unset | Enable debug logging |

## Model Mapping

| Request model string | Claude SDK tier |
|---------------------|-----------------|
| `*opus*` | opus |
| `*haiku*` | haiku |
| anything else | sonnet |

## MCP `message` Tool

The proxy exposes a `message` MCP tool (available as `mcp__opencode__message`) that connects to the openclaw gateway WebSocket with Ed25519 device authentication at `operator.admin` scope.

**Parameters:**

| Param | Required | Description |
|-------|----------|-------------|
| `to` | ✓ | Telegram chat ID, e.g. `"-1001426819337"` |
| `message` | one of | Text message to send |
| `filePath` / `path` / `media` | one of | Absolute path to a file in `/tmp/` — proxy converts to `file://` URL |
| `caption` | | Caption accompanying a media file |
| `action` | | `"send"` (default, can be omitted) |

**Usage pattern:**

When openclaw delivers a message to the agent, the system prompt includes the conversation's `chat id:XXXXXXXXX`. Claude extracts this and calls `mcp__opencode__message` once per outbound message. After all sends, Claude outputs only `NO_REPLY` — openclaw suppresses duplicate text delivery on its end.

## Blocked Built-in Tools

The following Claude Code built-in tools are blocked so Claude uses only the MCP tools above:

`Write, Edit, MultiEdit, Bash, Glob, Grep, NotebookEdit, WebFetch, WebSearch, TodoWrite`

`Read` is intentionally kept so Claude can open inbound image temp files.

## License

MIT
