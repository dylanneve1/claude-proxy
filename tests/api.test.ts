/**
 * API endpoint tests for claude-proxy.
 *
 * These tests verify the HTTP layer (routing, request validation, response
 * format) using Hono's testClient. They do NOT call the Claude Agent SDK —
 * the actual `query()` calls are tested in the integration tests.
 */
import { describe, test, expect } from "bun:test"
import { createProxyServer } from "../src/proxy/server"

const { app } = createProxyServer({ port: 0, host: "127.0.0.1" })

// Helper: make a request against the Hono app directly (no network needed)
async function req(path: string, init?: RequestInit) {
  return app.fetch(new Request(`http://localhost${path}`, init))
}

async function json(path: string, init?: RequestInit): Promise<{ status: number; body: any }> {
  const res = await req(path, init)
  return { status: res.status, body: await res.json() }
}

// ── Health / Root ────────────────────────────────────────────────────────────

describe("GET /", () => {
  test("returns service info", async () => {
    const { status, body } = await json("/")
    expect(status).toBe(200)
    expect(body.status).toBe("ok")
    expect(body.service).toBe("claude-max-proxy")
    expect(body.version).toBeDefined()
    expect(body.format).toBe("anthropic")
    expect(body.endpoints).toBeArray()
    expect(body.queue).toBeDefined()
    expect(typeof body.queue.active).toBe("number")
    expect(typeof body.queue.max).toBe("number")
  })
})

// ── Models ───────────────────────────────────────────────────────────────────

describe("GET /v1/models", () => {
  test("returns model list", async () => {
    const { status, body } = await json("/v1/models")
    expect(status).toBe(200)
    expect(body.data).toBeArray()
    expect(body.data.length).toBeGreaterThan(0)
    for (const model of body.data) {
      expect(model.type).toBe("model")
      expect(model.id).toBeDefined()
      expect(model.display_name).toBeDefined()
      expect(model.created_at).toBeDefined()
    }
  })

  test("includes opus, sonnet, and haiku", async () => {
    const { body } = await json("/v1/models")
    const ids = body.data.map((m: any) => m.id)
    expect(ids).toContain("claude-opus-4-6")
    expect(ids).toContain("claude-sonnet-4-6")
    expect(ids.some((id: string) => id.includes("haiku"))).toBe(true)
  })

  test("includes dated model variants", async () => {
    const { body } = await json("/v1/models")
    const ids = body.data.map((m: any) => m.id)
    expect(ids).toContain("claude-opus-4-6-20250801")
    expect(ids).toContain("claude-sonnet-4-6-20250801")
    expect(ids).toContain("claude-haiku-4-5-20251001")
  })
})

describe("GET /models (alias)", () => {
  test("same response as /v1/models", async () => {
    const a = await json("/v1/models")
    const b = await json("/models")
    expect(a.body).toEqual(b.body)
  })
})

describe("GET /v1/models/:id", () => {
  test("returns specific model", async () => {
    const { status, body } = await json("/v1/models/claude-sonnet-4-6")
    expect(status).toBe(200)
    expect(body.id).toBe("claude-sonnet-4-6")
    expect(body.type).toBe("model")
    expect(body.display_name).toBe("Claude Sonnet 4.6")
  })

  test("404 for unknown model", async () => {
    const { status, body } = await json("/v1/models/nonexistent-model")
    expect(status).toBe(404)
    expect(body.type).toBe("error")
    expect(body.error.type).toBe("not_found_error")
  })
})

describe("GET /models/:id (alias)", () => {
  test("works the same as /v1/models/:id", async () => {
    const { status, body } = await json("/models/claude-opus-4-6")
    expect(status).toBe(200)
    expect(body.id).toBe("claude-opus-4-6")
  })
})

// ── Count Tokens ─────────────────────────────────────────────────────────────

describe("POST /v1/messages/count_tokens", () => {
  test("returns token count for messages", async () => {
    const { status, body } = await json("/v1/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-6",
        system: "You are helpful.",
        messages: [{ role: "user", content: "Hello, how are you?" }]
      })
    })
    expect(status).toBe(200)
    expect(body.input_tokens).toBeGreaterThan(0)
    expect(typeof body.input_tokens).toBe("number")
  })

  test("handles system as array", async () => {
    const { status, body } = await json("/v1/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-6",
        system: [{ type: "text", text: "You are helpful." }],
        messages: [{ role: "user", content: "Hi" }]
      })
    })
    expect(status).toBe(200)
    expect(body.input_tokens).toBeGreaterThan(0)
  })

  test("returns 0 for empty body", async () => {
    const { status, body } = await json("/v1/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    })
    expect(status).toBe(200)
    expect(body.input_tokens).toBe(0)
  })
})

describe("POST /messages/count_tokens (alias)", () => {
  test("works the same", async () => {
    const { status, body } = await json("/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: "Test" }]
      })
    })
    expect(status).toBe(200)
    expect(typeof body.input_tokens).toBe("number")
  })
})

// ── Messages validation ──────────────────────────────────────────────────────

describe("POST /v1/messages (validation)", () => {
  test("400 when messages is missing", async () => {
    const { status, body } = await json("/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "claude-sonnet-4-6" })
    })
    expect(status).toBe(400)
    expect(body.type).toBe("error")
    expect(body.error.type).toBe("invalid_request_error")
  })

  test("400 when messages is empty array", async () => {
    const { status, body } = await json("/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "claude-sonnet-4-6", messages: [] })
    })
    expect(status).toBe(400)
    expect(body.error.type).toBe("invalid_request_error")
  })
})

// ── Headers ──────────────────────────────────────────────────────────────────

describe("CORS", () => {
  test("includes CORS headers", async () => {
    const res = await req("/", {
      method: "OPTIONS",
      headers: { Origin: "http://example.com" }
    })
    expect(res.headers.get("access-control-allow-origin")).toBeDefined()
  })
})

describe("Request ID", () => {
  test("generates request-id when not provided", async () => {
    const res = await req("/")
    const requestId = res.headers.get("x-request-id")
    expect(requestId).toBeDefined()
    expect(requestId!.startsWith("req_")).toBe(true)
  })

  test("echoes back provided x-request-id", async () => {
    const res = await req("/", {
      headers: { "x-request-id": "test-id-123" }
    })
    expect(res.headers.get("x-request-id")).toBe("test-id-123")
    expect(res.headers.get("request-id")).toBe("test-id-123")
  })
})

// ── Anthropic headers ────────────────────────────────────────────────────────

describe("Anthropic headers", () => {
  test("includes anthropic-version header", async () => {
    const res = await req("/")
    expect(res.headers.get("anthropic-version")).toBe("2024-10-22")
  })

  test("echoes back anthropic-beta header", async () => {
    const res = await req("/", {
      headers: { "anthropic-beta": "extended-thinking-2025-04-11" }
    })
    expect(res.headers.get("anthropic-beta")).toBe("extended-thinking-2025-04-11")
  })
})

// ── Batches stub ─────────────────────────────────────────────────────────────

describe("POST /v1/messages/batches", () => {
  test("returns 501 Not Implemented", async () => {
    const { status, body } = await json("/v1/messages/batches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    })
    expect(status).toBe(501)
    expect(body.error.type).toBe("not_implemented_error")
  })
})

// ── Malformed JSON ──────────────────────────────────────────────────────────

describe("POST /v1/messages (malformed JSON)", () => {
  test("400 for invalid JSON body", async () => {
    const res = await req("/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "not json at all {"
    })
    const body = await res.json() as any
    expect(res.status).toBe(400)
    expect(body.error.type).toBe("invalid_request_error")
    expect(body.error.message).toContain("valid JSON")
  })
})

describe("POST /v1/chat/completions (malformed JSON)", () => {
  test("400 for invalid JSON body", async () => {
    const res = await req("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{broken"
    })
    const body = await res.json() as any
    expect(res.status).toBe(400)
    expect(body.error.type).toBe("invalid_request_error")
  })

  test("400 for empty messages", async () => {
    const { status, body } = await json("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "claude-sonnet-4-6", messages: [] })
    })
    expect(status).toBe(400)
    expect(body.error.type).toBe("invalid_request_error")
  })
})

// ── API key validation ──────────────────────────────────────────────────────

describe("API key validation", () => {
  test("when CLAUDE_PROXY_API_KEY is set, unauthenticated requests to protected endpoints are rejected", async () => {
    // Create a separate server instance with API key set
    const originalKey = process.env.CLAUDE_PROXY_API_KEY
    process.env.CLAUDE_PROXY_API_KEY = "test-secret-key"
    try {
      const { app: authedApp } = createProxyServer({ port: 0, host: "127.0.0.1" })
      const res = await authedApp.fetch(new Request("http://localhost/v1/models"))
      const body = await res.json() as any
      expect(res.status).toBe(401)
      expect(body.error.type).toBe("authentication_error")
    } finally {
      if (originalKey) process.env.CLAUDE_PROXY_API_KEY = originalKey
      else delete process.env.CLAUDE_PROXY_API_KEY
    }
  })

  test("health endpoint bypasses auth", async () => {
    const originalKey = process.env.CLAUDE_PROXY_API_KEY
    process.env.CLAUDE_PROXY_API_KEY = "test-secret-key"
    try {
      const { app: authedApp } = createProxyServer({ port: 0, host: "127.0.0.1" })
      const res = await authedApp.fetch(new Request("http://localhost/"))
      expect(res.status).toBe(200)
    } finally {
      if (originalKey) process.env.CLAUDE_PROXY_API_KEY = originalKey
      else delete process.env.CLAUDE_PROXY_API_KEY
    }
  })

  test("valid x-api-key header passes auth", async () => {
    const originalKey = process.env.CLAUDE_PROXY_API_KEY
    process.env.CLAUDE_PROXY_API_KEY = "test-secret-key"
    try {
      const { app: authedApp } = createProxyServer({ port: 0, host: "127.0.0.1" })
      const res = await authedApp.fetch(new Request("http://localhost/v1/models", {
        headers: { "x-api-key": "test-secret-key" }
      }))
      expect(res.status).toBe(200)
    } finally {
      if (originalKey) process.env.CLAUDE_PROXY_API_KEY = originalKey
      else delete process.env.CLAUDE_PROXY_API_KEY
    }
  })

  test("valid Authorization Bearer header passes auth", async () => {
    const originalKey = process.env.CLAUDE_PROXY_API_KEY
    process.env.CLAUDE_PROXY_API_KEY = "test-secret-key"
    try {
      const { app: authedApp } = createProxyServer({ port: 0, host: "127.0.0.1" })
      const res = await authedApp.fetch(new Request("http://localhost/v1/models", {
        headers: { "Authorization": "Bearer test-secret-key" }
      }))
      expect(res.status).toBe(200)
    } finally {
      if (originalKey) process.env.CLAUDE_PROXY_API_KEY = originalKey
      else delete process.env.CLAUDE_PROXY_API_KEY
    }
  })
})

// ── 404 ──────────────────────────────────────────────────────────────────────

describe("Unknown routes", () => {
  test("returns 404 for unknown paths", async () => {
    const res = await req("/v1/nonexistent")
    expect(res.status).toBe(404)
  })
})
