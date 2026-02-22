# claude-proxy

A drop-in Anthropic Messages API proxy backed by the **Claude Agent SDK**. Use your Claude Max subscription with any Anthropic API client — zero API cost.

Also supports **OpenAI-compatible** endpoints so tools like LangChain, LiteLLM, and the OpenAI SDK work out of the box.

```
Any Anthropic/OpenAI client → claude-proxy (:3456) → Claude Agent SDK → Claude Max
```

## Features

- **Drop-in replacement** — set `ANTHROPIC_BASE_URL=http://127.0.0.1:3456` and go
- **OpenAI compatibility** — `/v1/chat/completions` with streaming + non-streaming
- **Zero cost** — routes through your Claude Max subscription via the Agent SDK
- **Full tool use** — proper `tool_use` content blocks, `stop_reason: "tool_use"`, `input_json_delta` streaming
- **Built-in agent tools** — Claude has access to Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch
- **API key protection** — optional `CLAUDE_PROXY_API_KEY` to secure network-exposed instances
- **Streaming SSE** — `message_start` emitted immediately; 15s heartbeat keeps connections alive
- **Request timeout** — configurable per-request timeout (default 5 minutes)
- **Graceful shutdown** — SIGINT/SIGTERM handlers wait for in-flight requests
- **Docker ready** — Dockerfile and docker-compose.yml included

## Quick Start

### Prerequisites

1. **Claude Max subscription** + Claude CLI:
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude login
   ```

2. **Bun** runtime:
   ```bash
   curl -fsSL https://bun.sh/install | bash
   ```

### Install & Run

```bash
# From npm
bunx claude-proxy

# Or clone and run
git clone https://github.com/dylanneve1/claude-proxy
cd claude-proxy
bun install
bun run proxy
# Proxy listening at http://127.0.0.1:3456
```

### Docker

```bash
docker compose up -d
# or
docker build -t claude-proxy . && docker run -p 3456:3456 -v ~/.claude:/root/.claude:ro claude-proxy
```

## Usage Examples

### Anthropic SDK (Python)

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:3456",
    api_key="dummy"  # any value works unless CLAUDE_PROXY_API_KEY is set
)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:3456/v1/chat",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### curl (Anthropic format)

```bash
curl http://127.0.0.1:3456/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "stream": false,
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

### curl (OpenAI format)

```bash
curl http://127.0.0.1:3456/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "stream": false,
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

### Environment variable approach

```bash
# Works with any Anthropic SDK client
ANTHROPIC_BASE_URL=http://127.0.0.1:3456 ANTHROPIC_API_KEY=dummy your-app

# Works with any OpenAI SDK client
OPENAI_BASE_URL=http://127.0.0.1:3456/v1/chat OPENAI_API_KEY=dummy your-app
```

## Architecture

### Agent mode (no caller tools)
```
POST /v1/messages
  → Serialize messages → prompt
  → Claude Agent SDK query() (maxTurns=50)
       ├─ Built-in tools: Read, Write, Edit, Bash, Glob, Grep, ...
       └─ MCP tools: message (optional gateway delivery)
```

### Client tool mode (caller provides tools)
```
POST /v1/messages  with  "tools": [...]
  → Inject tool definitions into system prompt (all built-in tools disabled)
  → Claude Agent SDK query() (maxTurns=1)
       → Parse <tool_use> blocks → emit tool_use content blocks
```

Client tool mode is auto-detected when the request has a `tools` array and the system prompt doesn't contain agent session markers.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check / service info |
| `GET` | `/v1/models` | List available models (Anthropic format) |
| `GET` | `/v1/models/:id` | Get model details |
| `POST` | `/v1/messages` | Create a message (streaming or non-streaming) |
| `POST` | `/v1/messages/count_tokens` | Estimate token count |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `GET` | `/v1/chat/models` | List models (OpenAI format) |

All Anthropic endpoints are also available without the `/v1` prefix.

## CLI Options

```
claude-proxy [options]

  -p, --port <port>   Listen port (default: 3456, env: CLAUDE_PROXY_PORT)
  -H, --host <host>   Bind address (default: 127.0.0.1, env: CLAUDE_PROXY_HOST)
  -d, --debug         Enable debug logging
  -v, --version       Show version
  -h, --help          Show help
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_PROXY_PORT` | `3456` | Proxy listen port |
| `CLAUDE_PROXY_HOST` | `127.0.0.1` | Proxy bind address |
| `CLAUDE_PROXY_DEBUG` | unset | Enable debug logging (`1` to enable) |
| `CLAUDE_PROXY_API_KEY` | unset | When set, require this key via `x-api-key` or `Authorization: Bearer` header |
| `CLAUDE_PROXY_MAX_CONCURRENT` | `5` | Max simultaneous Claude SDK sessions |
| `CLAUDE_PROXY_TIMEOUT_MS` | `300000` | Per-request timeout in milliseconds |

## Testing

```bash
# Unit tests (no running proxy needed)
bun test

# Integration tests (requires running proxy + Claude CLI auth)
bun run test:integration

# All tests
bun run test:all

# Type checking
bun run typecheck
```

## Model Mapping

| Request model string | Claude SDK tier |
|---------------------|-----------------|
| `*opus*` | opus |
| `*haiku*` | haiku |
| anything else | sonnet |

## Deploying on a Server

Expose the proxy as an HTTPS endpoint so you can use it from chat apps like TypingMind, ChatWise, or any Anthropic-compatible client.

### 1. Install & configure the proxy

```bash
git clone https://github.com/dylanneve1/claude-proxy
cd claude-proxy
bun install
```

### 2. Set up systemd

Create `~/.config/systemd/user/claude-proxy.service`:

```ini
[Unit]
Description=Claude Max API Proxy
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/claude-proxy
ExecStart=/home/user/.bun/bin/bun run proxy
Environment=PATH=/home/user/.bun/bin:/usr/local/bin:/usr/bin:/bin
Environment=CLAUDE_PROXY_HOST=0.0.0.0
Environment=CLAUDE_PROXY_API_KEY=your-secret-api-key
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

- `CLAUDE_PROXY_HOST=0.0.0.0` binds to all interfaces (required for external access)
- `CLAUDE_PROXY_API_KEY` protects the endpoint — clients must send this via `x-api-key` or `Authorization: Bearer` header

Generate a random key:

```bash
openssl rand -hex 32
```

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now claude-proxy
```

### 3. Add HTTPS with Caddy

A reverse proxy with auto TLS is the easiest way to get HTTPS. Get a free domain from [duckdns.org](https://www.duckdns.org) and point it at your server IP.

Install Caddy:

```bash
sudo apt install caddy
```

Edit `/etc/caddy/Caddyfile`:

```
yourdomain.duckdns.org {
    reverse_proxy localhost:3456 {
        flush_interval -1
    }
}
```

`flush_interval -1` ensures SSE streaming responses are forwarded immediately without buffering.

```bash
sudo systemctl reload caddy
```

Caddy automatically provisions a Let's Encrypt TLS certificate.

### 4. Connect a chat app

Use your endpoint in any Anthropic-compatible chat app:

| Setting | Value |
|---------|-------|
| **Base URL** | `https://yourdomain.duckdns.org` |
| **API Key** | Your `CLAUDE_PROXY_API_KEY` value |
| **Model** | `claude-opus-4-6`, `claude-sonnet-4-6`, or `claude-haiku-4-5-20251001` |

## License

MIT
