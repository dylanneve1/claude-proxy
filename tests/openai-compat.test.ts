/**
 * Tests for OpenAI-compatible /v1/chat/completions endpoint.
 * These verify the format translation layer works correctly.
 */
import { describe, test, expect } from "bun:test"
import { createProxyServer } from "../src/proxy/server"

const { app } = createProxyServer({ port: 0, host: "127.0.0.1" })

async function req(path: string, init?: RequestInit) {
  return app.fetch(new Request(`http://localhost${path}`, init))
}

async function json(path: string, init?: RequestInit): Promise<{ status: number; body: any }> {
  const res = await req(path, init)
  return { status: res.status, body: await res.json() }
}

// ── OpenAI model listing ────────────────────────────────────────────────────

describe("GET /v1/chat/models", () => {
  test("returns OpenAI-format model list", async () => {
    const { status, body } = await json("/v1/chat/models")
    expect(status).toBe(200)
    expect(body.object).toBe("list")
    expect(body.data).toBeArray()
    expect(body.data.length).toBeGreaterThan(0)
    for (const model of body.data) {
      expect(model.object).toBe("model")
      expect(model.id).toBeDefined()
      expect(model.owned_by).toBe("anthropic")
      expect(typeof model.created).toBe("number")
    }
  })
})

// ── Request validation ──────────────────────────────────────────────────────

describe("POST /v1/chat/completions (validation)", () => {
  test("handles empty messages gracefully", async () => {
    const { status } = await json("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-6",
        messages: []
      })
    })
    // Should forward to Anthropic handler which returns 400
    expect(status).toBe(400)
  })
})

// ── Integration tests (need running proxy) ─────────────────────────────────

const PROXY_URL = process.env.PROXY_URL || "http://127.0.0.1:3456"
const SKIP = !process.env.INTEGRATION

function skipUnlessIntegration() {
  if (SKIP) {
    console.log("  [SKIPPED - set INTEGRATION=1 to run]")
    return true
  }
  return false
}

describe("Integration: OpenAI non-streaming", () => {
  test("simple chat completion", async () => {
    if (skipUnlessIntegration()) return
    const res = await fetch(`${PROXY_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer dummy" },
      body: JSON.stringify({
        model: "claude-sonnet-4-6",
        messages: [{ role: "user", content: "What is 3+3? Reply with ONLY the number." }],
        stream: false
      })
    })
    const body = await res.json() as any
    expect(res.status).toBe(200)
    expect(body.object).toBe("chat.completion")
    expect(body.choices).toBeArray()
    expect(body.choices[0].message.role).toBe("assistant")
    expect(body.choices[0].message.content).toContain("6")
    expect(body.choices[0].finish_reason).toBe("stop")
    expect(body.usage).toBeDefined()
    expect(body.usage.total_tokens).toBeGreaterThan(0)
  }, 120_000)

  test("system message is forwarded", async () => {
    if (skipUnlessIntegration()) return
    const res = await fetch(`${PROXY_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-6",
        messages: [
          { role: "system", content: "Reply with 'PONG' and nothing else." },
          { role: "user", content: "PING" }
        ],
        stream: false
      })
    })
    const body = await res.json() as any
    expect(body.choices[0].message.content.toUpperCase()).toContain("PONG")
  }, 120_000)
})

describe("Integration: OpenAI streaming", () => {
  test("streams chat completion chunks", async () => {
    if (skipUnlessIntegration()) return
    const res = await fetch(`${PROXY_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-6",
        messages: [{ role: "user", content: "What is 2+2? Reply with ONLY the number." }],
        stream: true
      })
    })
    const text = await res.text()
    const lines = text.split("\n").filter(l => l.startsWith("data: "))
    expect(lines.length).toBeGreaterThan(0)

    // Check for content chunks
    const contentChunks = lines.filter(l => {
      if (l === "data: [DONE]") return false
      try {
        const data = JSON.parse(l.slice(6))
        return data.choices?.[0]?.delta?.content
      } catch { return false }
    })
    expect(contentChunks.length).toBeGreaterThan(0)

    // Check for [DONE] terminator
    expect(lines.some(l => l === "data: [DONE]")).toBe(true)

    // Check format of a chunk
    const firstChunk = JSON.parse(contentChunks[0]!.slice(6))
    expect(firstChunk.object).toBe("chat.completion.chunk")
    expect(firstChunk.model).toBe("claude-sonnet-4-6")
  }, 120_000)
})
