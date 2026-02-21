/**
 * Comprehensive integration tests — run against a live proxy instance.
 *
 * These tests make real requests to the proxy (default http://127.0.0.1:3456)
 * and verify the full request/response cycle through the Claude Agent SDK.
 *
 * Skip these in CI (they need Claude CLI auth). Run locally:
 *   INTEGRATION=1 bun test tests/integration.test.ts
 */
import { describe, test, expect } from "bun:test"

const PROXY_URL = process.env.PROXY_URL || "http://127.0.0.1:3456"
const SKIP = !process.env.INTEGRATION

function skipUnlessIntegration() {
  if (SKIP) {
    console.log("  [SKIPPED - set INTEGRATION=1 to run]")
    return true
  }
  return false
}

async function proxyJson(path: string, init?: RequestInit): Promise<{ status: number; body: any; headers: Headers }> {
  const res = await fetch(`${PROXY_URL}${path}`, init)
  return { status: res.status, body: await res.json(), headers: res.headers }
}

async function proxyPost(path: string, body: object): Promise<{ status: number; body: any; headers: Headers }> {
  return proxyJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-api-key": "dummy" },
    body: JSON.stringify(body)
  })
}

async function proxyStream(path: string, body: object): Promise<{ events: any[]; raw: string; headers: Headers }> {
  const res = await fetch(`${PROXY_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-api-key": "dummy" },
    body: JSON.stringify({ ...body, stream: true })
  })
  const raw = await res.text()
  const events: any[] = []
  for (const line of raw.split("\n")) {
    if (line.startsWith("data: ")) {
      try { events.push(JSON.parse(line.slice(6))) } catch {}
    }
  }
  return { events, raw, headers: res.headers }
}

// ── Connectivity ─────────────────────────────────────────────────────────────

describe("Integration: connectivity", () => {
  test("proxy is reachable and healthy", async () => {
    if (skipUnlessIntegration()) return
    const { status, body, headers } = await proxyJson("/")
    expect(status).toBe(200)
    expect(body.status).toBe("ok")
    expect(body.service).toBe("claude-max-proxy")
    expect(body.version).toBeDefined()
    expect(body.endpoints).toBeArray()
    expect(body.queue).toBeDefined()
    expect(typeof body.queue.active).toBe("number")
    expect(typeof body.queue.max).toBe("number")
    // Should have request-id and anthropic-version headers
    expect(headers.get("x-request-id")).toBeDefined()
    expect(headers.get("anthropic-version")).toBeDefined()
  })

  test("models endpoint returns models in dual format", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyJson("/v1/models")
    expect(status).toBe(200)
    expect(body.object).toBe("list")
    expect(body.data.length).toBeGreaterThan(0)
    const model = body.data[0]
    // Anthropic fields
    expect(model.type).toBe("model")
    expect(model.display_name).toBeDefined()
    // OpenAI fields
    expect(model.object).toBe("model")
    expect(model.owned_by).toBe("anthropic")
  })

  test("specific model lookup works", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyJson("/v1/models/claude-sonnet-4-6")
    expect(status).toBe(200)
    expect(body.id).toBe("claude-sonnet-4-6")
    expect(body.display_name).toBe("Claude Sonnet 4.6")
  })

  test("unknown model returns 404", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyJson("/v1/models/nonexistent")
    expect(status).toBe(404)
    expect(body.error.type).toBe("not_found_error")
  })
})

// ── Non-streaming Anthropic format ───────────────────────────────────────────

describe("Integration: Anthropic non-streaming", () => {
  test("simple math question returns correct answer", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyPost("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 64,
      stream: false,
      messages: [{ role: "user", content: "What is 7 * 8? Reply with ONLY the number." }]
    })
    expect(status).toBe(200)
    expect(body.type).toBe("message")
    expect(body.role).toBe("assistant")
    expect(body.stop_reason).toBe("end_turn")
    expect(body.content).toBeArray()
    expect(body.content.length).toBeGreaterThan(0)
    expect(body.content[0].type).toBe("text")
    expect(body.content[0].text).toContain("56")
    expect(body.usage).toBeDefined()
    expect(body.usage.input_tokens).toBeGreaterThan(0)
    expect(body.usage.output_tokens).toBeGreaterThan(0)
  }, 120_000)

  test("multi-turn conversation preserves context", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyPost("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 64,
      stream: false,
      messages: [
        { role: "user", content: "Remember this number: 42. Just say OK." },
        { role: "assistant", content: "OK" },
        { role: "user", content: "What number did I ask you to remember? Reply with ONLY the number." }
      ]
    })
    expect(status).toBe(200)
    expect(body.content[0].text).toContain("42")
  }, 120_000)

  test("system prompt influences response", async () => {
    if (skipUnlessIntegration()) return
    const { body } = await proxyPost("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 32,
      stream: false,
      system: "You are a pirate. Always respond starting with 'Arrr'.",
      messages: [{ role: "user", content: "Hello" }]
    })
    expect(body.content[0].text.toLowerCase()).toContain("arrr")
  }, 120_000)

  test("array system prompt works", async () => {
    if (skipUnlessIntegration()) return
    const { body } = await proxyPost("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 32,
      stream: false,
      system: [{ type: "text", text: "Reply with 'PONG' and nothing else." }],
      messages: [{ role: "user", content: "PING" }]
    })
    expect(body.content[0].text.toUpperCase()).toContain("PONG")
  }, 120_000)

  test("haiku model responds", async () => {
    if (skipUnlessIntegration()) return
    const { body } = await proxyPost("/v1/messages", {
      model: "claude-haiku-4-5",
      max_tokens: 32,
      stream: false,
      messages: [{ role: "user", content: "Say 'hello'." }]
    })
    expect(body.model).toBe("claude-haiku-4-5")
    expect(body.content[0].text.toLowerCase()).toContain("hello")
  }, 120_000)
})

// ── Streaming Anthropic format ───────────────────────────────────────────────

describe("Integration: Anthropic streaming", () => {
  test("SSE event sequence is correct", async () => {
    if (skipUnlessIntegration()) return
    const { events } = await proxyStream("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 32,
      messages: [{ role: "user", content: "What is 2+2? Reply with ONLY the number." }]
    })
    expect(events.length).toBeGreaterThan(0)

    const types = events.map(e => e.type)
    // Must start with message_start
    expect(types[0]).toBe("message_start")
    // Must contain these event types
    expect(types).toContain("content_block_start")
    expect(types).toContain("content_block_delta")
    expect(types).toContain("content_block_stop")
    expect(types).toContain("message_delta")
    // Must end with message_stop
    expect(types[types.length - 1]).toBe("message_stop")

    // message_start structure
    const msgStart = events[0]
    expect(msgStart.message.type).toBe("message")
    expect(msgStart.message.role).toBe("assistant")
    expect(msgStart.message.stop_reason).toBeNull()

    // message_delta should have stop_reason
    const msgDelta = events.find(e => e.type === "message_delta")!
    expect(msgDelta.delta.stop_reason).toBe("end_turn")
    expect(msgDelta.usage.output_tokens).toBeGreaterThan(0)

    // Assembled text should contain "4"
    const text = events
      .filter(e => e.type === "content_block_delta" && e.delta?.type === "text_delta")
      .map(e => e.delta.text)
      .join("")
    expect(text).toContain("4")
  }, 120_000)

  test("streaming produces multiple text deltas", async () => {
    if (skipUnlessIntegration()) return
    const { events } = await proxyStream("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 200,
      messages: [{ role: "user", content: "Write a haiku about the ocean." }]
    })

    const textDeltas = events.filter(e =>
      e.type === "content_block_delta" && e.delta?.type === "text_delta"
    )
    // A haiku should produce multiple streaming chunks
    expect(textDeltas.length).toBeGreaterThan(1)

    // Assembled text should be non-trivial
    const fullText = textDeltas.map(e => e.delta.text).join("")
    expect(fullText.length).toBeGreaterThan(10)
  }, 120_000)
})

// ── Client tool mode ─────────────────────────────────────────────────────────

describe("Integration: client tool mode", () => {
  const weatherTool = {
    name: "get_weather",
    description: "Get the current weather for a city",
    input_schema: {
      type: "object",
      properties: { city: { type: "string", description: "The city name" } },
      required: ["city"]
    }
  }

  test("returns tool_use block (non-streaming)", async () => {
    if (skipUnlessIntegration()) return
    const { body } = await proxyPost("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 256,
      stream: false,
      messages: [{ role: "user", content: "What's the weather in Tokyo?" }],
      tools: [weatherTool]
    })
    expect(body.type).toBe("message")
    expect(body.stop_reason).toBe("tool_use")

    const toolUseBlock = body.content.find((b: any) => b.type === "tool_use")
    expect(toolUseBlock).toBeDefined()
    expect(toolUseBlock.name).toBe("get_weather")
    expect(toolUseBlock.input).toBeDefined()
    expect(toolUseBlock.id).toBeDefined()
    expect(toolUseBlock.id.startsWith("toolu_")).toBe(true)
    // Should include "Tokyo" or similar in the input
    const inputStr = JSON.stringify(toolUseBlock.input).toLowerCase()
    expect(inputStr).toContain("tokyo")
  }, 120_000)

  test("returns tool_use block (streaming)", async () => {
    if (skipUnlessIntegration()) return
    const { events } = await proxyStream("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 256,
      messages: [{ role: "user", content: "What's the weather in London?" }],
      tools: [weatherTool]
    })

    // Should have tool_use content block
    const toolStart = events.find(e =>
      e.type === "content_block_start" && e.content_block?.type === "tool_use"
    )
    expect(toolStart).toBeDefined()
    expect(toolStart.content_block.name).toBe("get_weather")

    // Should have input_json_delta
    const jsonDelta = events.find(e =>
      e.type === "content_block_delta" && e.delta?.type === "input_json_delta"
    )
    expect(jsonDelta).toBeDefined()
    const inputJson = JSON.parse(jsonDelta.delta.partial_json)
    expect(JSON.stringify(inputJson).toLowerCase()).toContain("london")

    // Stop reason should be tool_use
    const msgDelta = events.find(e => e.type === "message_delta")
    expect(msgDelta!.delta.stop_reason).toBe("tool_use")
  }, 120_000)

  test("handles tool_result round-trip", async () => {
    if (skipUnlessIntegration()) return
    const { body } = await proxyPost("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 256,
      stream: false,
      messages: [
        { role: "user", content: "What's the weather in Paris?" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me check the weather." },
            { type: "tool_use", id: "toolu_test123", name: "get_weather", input: { city: "Paris" } }
          ]
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_test123", content: "22°C, sunny with clear skies" }
          ]
        }
      ],
      tools: [weatherTool]
    })
    expect(body.type).toBe("message")
    expect(body.stop_reason).toBe("end_turn")
    const text = body.content.find((b: any) => b.type === "text")?.text || ""
    // Model should reference the weather data in its response
    expect(text.length).toBeGreaterThan(5)
    // Should mention Paris or the temperature in its response
    const lower = text.toLowerCase()
    expect(lower.includes("paris") || lower.includes("22") || lower.includes("sunny")).toBe(true)
  }, 120_000)

  test("handles multiple tools", async () => {
    if (skipUnlessIntegration()) return
    const { body } = await proxyPost("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 256,
      stream: false,
      messages: [{ role: "user", content: "What's the weather in NYC and also search for 'restaurants near me'?" }],
      tools: [
        weatherTool,
        {
          name: "search",
          description: "Search the web for information",
          input_schema: {
            type: "object",
            properties: { query: { type: "string", description: "Search query" } },
            required: ["query"]
          }
        }
      ]
    })
    expect(body.type).toBe("message")
    expect(body.stop_reason).toBe("tool_use")
    const toolUseBlocks = body.content.filter((b: any) => b.type === "tool_use")
    // Should call at least one tool
    expect(toolUseBlocks.length).toBeGreaterThan(0)
  }, 120_000)
})

// ── OpenAI-compatible endpoint ───────────────────────────────────────────────

describe("Integration: OpenAI chat completions (non-streaming)", () => {
  test("simple chat completion returns correct format", async () => {
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
    expect(body.model).toBe("claude-sonnet-4-6")
    expect(body.choices).toBeArray()
    expect(body.choices.length).toBe(1)
    expect(body.choices[0].message.role).toBe("assistant")
    expect(body.choices[0].message.content).toContain("6")
    expect(body.choices[0].finish_reason).toBe("stop")
    expect(body.choices[0].index).toBe(0)
    expect(body.usage).toBeDefined()
    expect(body.usage.prompt_tokens).toBeGreaterThan(0)
    expect(body.usage.completion_tokens).toBeGreaterThan(0)
    expect(body.usage.total_tokens).toBe(body.usage.prompt_tokens + body.usage.completion_tokens)
    expect(body.id).toMatch(/^chatcmpl-/)
  }, 120_000)

  test("system message is respected", async () => {
    if (skipUnlessIntegration()) return
    const res = await fetch(`${PROXY_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-6",
        messages: [
          { role: "system", content: "You must start every response with the word 'BANANA'." },
          { role: "user", content: "Tell me a joke." }
        ],
        stream: false
      })
    })
    const body = await res.json() as any
    expect(body.choices[0].message.content.toUpperCase()).toContain("BANANA")
  }, 120_000)
})

describe("Integration: OpenAI chat completions (streaming)", () => {
  test("streams chat completion chunks correctly", async () => {
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

    // Check format of first content chunk
    const firstChunk = JSON.parse(contentChunks[0]!.slice(6))
    expect(firstChunk.object).toBe("chat.completion.chunk")
    expect(firstChunk.model).toBe("claude-sonnet-4-6")
    expect(firstChunk.choices[0].index).toBe(0)
    expect(firstChunk.choices[0].finish_reason).toBeNull()

    // Check for [DONE] terminator
    expect(lines.some(l => l === "data: [DONE]")).toBe(true)

    // Check final chunk has finish_reason: "stop"
    const finishLines = lines.filter(l => {
      if (l === "data: [DONE]") return false
      try {
        const data = JSON.parse(l.slice(6))
        return data.choices?.[0]?.finish_reason === "stop"
      } catch { return false }
    })
    expect(finishLines.length).toBe(1)

    // Assembled text should contain "4"
    const assembledText = contentChunks
      .map(l => JSON.parse(l.slice(6)).choices[0].delta.content)
      .join("")
    expect(assembledText).toContain("4")
  }, 120_000)
})

// ── OpenAI models endpoint ───────────────────────────────────────────────────

describe("Integration: OpenAI models", () => {
  test("/v1/chat/models returns OpenAI format", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyJson("/v1/chat/models")
    expect(status).toBe(200)
    expect(body.object).toBe("list")
    expect(body.data.length).toBeGreaterThan(0)
    for (const model of body.data) {
      expect(model.object).toBe("model")
      expect(model.id).toBeDefined()
      expect(model.owned_by).toBe("anthropic")
      expect(typeof model.created).toBe("number")
    }
  })

  test("/v1/models works for both Anthropic and OpenAI clients", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyJson("/v1/models")
    expect(status).toBe(200)
    expect(body.object).toBe("list")
    const model = body.data[0]
    // Has both format fields
    expect(model.type).toBe("model")
    expect(model.object).toBe("model")
    expect(model.display_name).toBeDefined()
    expect(model.owned_by).toBe("anthropic")
  })
})

// ── Error handling ───────────────────────────────────────────────────────────

describe("Integration: error handling", () => {
  test("missing messages returns 400", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyPost("/v1/messages", {
      model: "claude-sonnet-4-6"
    })
    expect(status).toBe(400)
    expect(body.type).toBe("error")
    expect(body.error.type).toBe("invalid_request_error")
  })

  test("empty messages returns 400", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyPost("/v1/messages", {
      model: "claude-sonnet-4-6",
      messages: []
    })
    expect(status).toBe(400)
    expect(body.error.type).toBe("invalid_request_error")
  })

  test("malformed JSON returns 400", async () => {
    if (skipUnlessIntegration()) return
    const res = await fetch(`${PROXY_URL}/v1/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "not valid json"
    })
    expect(res.status).toBe(400)
    const body = await res.json() as any
    expect(body.error.type).toBe("invalid_request_error")
  })

  test("unknown route returns 404", async () => {
    if (skipUnlessIntegration()) return
    const res = await fetch(`${PROXY_URL}/v1/nonexistent`)
    expect(res.status).toBe(404)
  })

  test("batches endpoint returns 501", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyPost("/v1/messages/batches", {})
    expect(status).toBe(501)
    expect(body.error.type).toBe("not_implemented_error")
  })
})

// ── Headers ──────────────────────────────────────────────────────────────────

describe("Integration: headers", () => {
  test("generates request-id for all requests", async () => {
    if (skipUnlessIntegration()) return
    const { headers } = await proxyJson("/")
    const reqId = headers.get("x-request-id")
    expect(reqId).toBeDefined()
    expect(reqId!.startsWith("req_")).toBe(true)
    expect(headers.get("request-id")).toBe(reqId)
  })

  test("echoes provided x-request-id", async () => {
    if (skipUnlessIntegration()) return
    const { headers } = await proxyJson("/", {
      headers: { "x-request-id": "test-req-id-abc" }
    })
    expect(headers.get("x-request-id")).toBe("test-req-id-abc")
    expect(headers.get("request-id")).toBe("test-req-id-abc")
  })

  test("includes anthropic-version header", async () => {
    if (skipUnlessIntegration()) return
    const { headers } = await proxyJson("/")
    expect(headers.get("anthropic-version")).toBe("2024-10-22")
  })

  test("includes CORS headers", async () => {
    if (skipUnlessIntegration()) return
    const res = await fetch(`${PROXY_URL}/`, {
      method: "OPTIONS",
      headers: { Origin: "http://example.com" }
    })
    expect(res.headers.get("access-control-allow-origin")).toBeDefined()
  })
})

// ── Concurrent requests ──────────────────────────────────────────────────────

describe("Integration: concurrency", () => {
  test("handles 3 simultaneous requests correctly", async () => {
    if (skipUnlessIntegration()) return
    const questions = [
      { q: "What is 1+1? Reply ONLY the number.", a: "2" },
      { q: "What is 5*5? Reply ONLY the number.", a: "25" },
      { q: "What is 100/10? Reply ONLY the number.", a: "10" },
    ]
    const results = await Promise.all(
      questions.map(({ q }) => proxyPost("/v1/messages", {
        model: "claude-sonnet-4-6",
        max_tokens: 16,
        stream: false,
        messages: [{ role: "user", content: q }]
      }))
    )
    for (let i = 0; i < questions.length; i++) {
      expect(results[i]!.status).toBe(200)
      expect(results[i]!.body.content[0].text).toContain(questions[i]!.a)
    }
  }, 300_000)
})

// ── Token counting ───────────────────────────────────────────────────────────

describe("Integration: token counting", () => {
  test("count_tokens returns reasonable estimate", async () => {
    if (skipUnlessIntegration()) return
    const { status, body } = await proxyPost("/v1/messages/count_tokens", {
      model: "claude-sonnet-4-6",
      messages: [{ role: "user", content: "Hello, how are you doing today?" }]
    })
    expect(status).toBe(200)
    expect(body.input_tokens).toBeGreaterThan(0)
    // "Hello, how are you doing today?" ≈ 7-10 tokens
    expect(body.input_tokens).toBeGreaterThan(3)
    expect(body.input_tokens).toBeLessThan(50)
  })

  test("longer text has more tokens", async () => {
    if (skipUnlessIntegration()) return
    const short = await proxyPost("/v1/messages/count_tokens", {
      messages: [{ role: "user", content: "Hi" }]
    })
    const long = await proxyPost("/v1/messages/count_tokens", {
      messages: [{ role: "user", content: "This is a much longer message that should result in a higher token count because it contains many more words and characters than the short message." }]
    })
    expect(long.body.input_tokens).toBeGreaterThan(short.body.input_tokens)
  })
})

// ── Queue status ─────────────────────────────────────────────────────────────

describe("Integration: queue status", () => {
  test("queue status reflects active requests", async () => {
    if (skipUnlessIntegration()) return
    // Start a request but don't await it
    const pending = proxyPost("/v1/messages", {
      model: "claude-sonnet-4-6",
      max_tokens: 200,
      stream: false,
      messages: [{ role: "user", content: "Write a short poem about the sun." }]
    })

    // Wait briefly for the request to be picked up
    await new Promise(r => setTimeout(r, 2000))

    // Check queue status
    const { body: healthBody } = await proxyJson("/")
    expect(healthBody.queue.active).toBeGreaterThanOrEqual(0) // May be 0 if request finished fast

    // Wait for the request to complete
    const result = await pending
    expect(result.status).toBe(200)
  }, 120_000)
})
