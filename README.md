# claude-proxy

A proxy server that bridges **openclaw** (and any Anthropic Messages API client) to the **Claude Agent SDK**, letting Claude Max subscribers use openclaw at zero API cost with full Telegram multi-message and image support.

```
openclaw → claude-proxy (:3456) → Claude Agent SDK → Claude Max subscription
```

## Features

- **Drop-in Anthropic API replacement** — set `ANTHROPIC_BASE_URL=http://127.0.0.1:3456` in any SDK and it just works
- **Claude Max** — zero API cost, uses your existing subscription via the official Agent SDK
- **Full tool use support** — returns proper `tool_use` content blocks with `stop_reason: "tool_use"`, handles multi-turn tool loops, `input_json_delta` streaming
- **`/v1/models` endpoint** — returns all supported model IDs
- **Streaming SSE** — `message_start` emitted immediately so long agent runs never hit HTTP timeouts; 15 s heartbeat keeps connections alive
- **Multi-message sends** — Claude calls the `message` MCP tool multiple times to send separate Telegram messages in openclaw context
- **Image / file sends** — Claude writes to `/tmp/`, calls `message` with `filePath`, proxy converts to `file://` URL for gateway delivery
- **Inbound images** — base64 image blocks saved to temp files so Claude can read them with the `read` tool
- **Large tool result protection** — results truncated at 4 000 chars to prevent context explosion
- **Per-request MCP servers** — no shared state between concurrent requests
- **Non-zero usage stats** — rough token estimates returned in every response

## Architecture

The proxy has two operating modes selected automatically per request:

### Agent mode (openclaw / no caller tools)
```
POST /v1/messages
  │
  ├─ Deserialize messages → text, images (→ /tmp/), tool_use, tool_result
  ├─ Build prompt: system + MCP tool note + conversation history
  │
  └─ Claude Agent SDK  query()  (maxTurns=50)
       │
       ├─ MCP tools: message, read, write, edit, bash, glob, grep
       └─ Normalize → stream text or JSON
```

### Client tool mode (any app using Anthropic tool use API)
```
POST /v1/messages  with  "tools": [...]
  │
  ├─ Inject tool definitions into system prompt
  │
  └─ Claude Agent SDK  query()  (maxTurns=1)
       │
       └─ Parse <tool_use> blocks from output
            └─ Emit tool_use content blocks + stop_reason="tool_use"
```

Client tool mode is detected automatically: the request has `tools` and doesn't look like an openclaw session (no `conversation_label` / `chat id:` in the system prompt).

## Drop-in Usage

```bash
# Any Anthropic SDK — set the base URL, keep any API key value
ANTHROPIC_BASE_URL=http://127.0.0.1:3456 ANTHROPIC_API_KEY=dummy your-app

# Or in Python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:3456", api_key="dummy")
```

Full tool use, streaming, and `/v1/models` work out of the box.

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
