/**
 * Unit tests for internal helper functions.
 * These test pure logic without any network or SDK calls.
 */
import { describe, test, expect } from "bun:test"
import { z } from "zod"
import { jsonSchemaToZod, createToolMcpServer } from "../src/mcpTools"

// ── Model mapping ────────────────────────────────────────────────────────────

describe("model mapping logic", () => {
  function mapModel(model: string): "sonnet" | "opus" | "haiku" {
    if (model.includes("opus")) return "opus"
    if (model.includes("haiku")) return "haiku"
    return "sonnet"
  }

  test("maps opus models", () => {
    expect(mapModel("claude-opus-4-6")).toBe("opus")
    expect(mapModel("claude-opus-4-6-20250801")).toBe("opus")
  })

  test("maps haiku models", () => {
    expect(mapModel("claude-haiku-4-5")).toBe("haiku")
    expect(mapModel("claude-haiku-4-5-20251001")).toBe("haiku")
  })

  test("maps sonnet models", () => {
    expect(mapModel("claude-sonnet-4-6")).toBe("sonnet")
    expect(mapModel("claude-sonnet-4-6-20250801")).toBe("sonnet")
  })

  test("defaults to sonnet for unknown models", () => {
    expect(mapModel("unknown-model")).toBe("sonnet")
    expect(mapModel("gpt-4")).toBe("sonnet")
    expect(mapModel("")).toBe("sonnet")
  })
})

// ── JSON Schema → Zod conversion ────────────────────────────────────────────

describe("jsonSchemaToZod", () => {
  test("converts string type", () => {
    const schema = jsonSchemaToZod({ type: "string" })
    expect(schema instanceof z.ZodString).toBe(true)
    expect(schema.parse("hello")).toBe("hello")
  })

  test("converts number type", () => {
    const schema = jsonSchemaToZod({ type: "number" })
    expect(schema instanceof z.ZodNumber).toBe(true)
    expect(schema.parse(42)).toBe(42)
  })

  test("converts integer type as number", () => {
    const schema = jsonSchemaToZod({ type: "integer" })
    expect(schema instanceof z.ZodNumber).toBe(true)
    expect(schema.parse(7)).toBe(7)
  })

  test("converts boolean type", () => {
    const schema = jsonSchemaToZod({ type: "boolean" })
    expect(schema instanceof z.ZodBoolean).toBe(true)
    expect(schema.parse(true)).toBe(true)
  })

  test("converts null type", () => {
    const schema = jsonSchemaToZod({ type: "null" })
    expect(schema.parse(null)).toBe(null)
  })

  test("converts array type", () => {
    const schema = jsonSchemaToZod({ type: "array", items: { type: "string" } })
    expect(schema instanceof z.ZodArray).toBe(true)
    expect(schema.parse(["a", "b"])).toEqual(["a", "b"])
  })

  test("converts array with no items to array of any", () => {
    const schema = jsonSchemaToZod({ type: "array" })
    expect(schema instanceof z.ZodArray).toBe(true)
    expect(schema.parse([1, "two", true])).toEqual([1, "two", true])
  })

  test("converts simple object with required properties", () => {
    const schema = jsonSchemaToZod({
      type: "object",
      properties: {
        city: { type: "string" },
        temp: { type: "number" }
      },
      required: ["city"]
    })
    expect(schema instanceof z.ZodObject).toBe(true)
    // Required field present
    expect(schema.parse({ city: "Tokyo", temp: 22 })).toEqual({ city: "Tokyo", temp: 22 })
    // Optional field missing is ok
    expect(schema.parse({ city: "Tokyo" })).toEqual({ city: "Tokyo" })
  })

  test("makes non-required properties optional", () => {
    const schema = jsonSchemaToZod({
      type: "object",
      properties: {
        name: { type: "string" },
        age: { type: "number" }
      },
      required: ["name"]
    })
    // age is optional, should pass without it
    const result = schema.parse({ name: "Alice" })
    expect(result.name).toBe("Alice")
  })

  test("converts string enum", () => {
    const schema = jsonSchemaToZod({ enum: ["red", "green", "blue"] })
    expect(schema.parse("red")).toBe("red")
    expect(() => schema.parse("purple")).toThrow()
  })

  test("converts nested objects", () => {
    const schema = jsonSchemaToZod({
      type: "object",
      properties: {
        location: {
          type: "object",
          properties: {
            lat: { type: "number" },
            lng: { type: "number" }
          },
          required: ["lat", "lng"]
        }
      },
      required: ["location"]
    })
    expect(schema.parse({ location: { lat: 35.6, lng: 139.7 } }))
      .toEqual({ location: { lat: 35.6, lng: 139.7 } })
  })

  test("converts anyOf with null → nullable", () => {
    const schema = jsonSchemaToZod({
      anyOf: [{ type: "string" }, { type: "null" }]
    })
    expect(schema.parse("hello")).toBe("hello")
    expect(schema.parse(null)).toBe(null)
  })

  test("converts oneOf as union", () => {
    const schema = jsonSchemaToZod({
      oneOf: [{ type: "string" }, { type: "number" }]
    })
    expect(schema.parse("hello")).toBe("hello")
    expect(schema.parse(42)).toBe(42)
  })

  test("converts array type notation like [\"string\", \"null\"]", () => {
    const schema = jsonSchemaToZod({ type: ["string", "null"] })
    expect(schema.parse("test")).toBe("test")
    expect(schema.parse(null)).toBe(null)
  })

  test("handles const values", () => {
    const schema = jsonSchemaToZod({ const: "fixed_value" })
    expect(schema.parse("fixed_value")).toBe("fixed_value")
  })

  test("falls back to z.any() for null/undefined schema", () => {
    expect(jsonSchemaToZod(null).parse("anything")).toBe("anything")
    expect(jsonSchemaToZod(undefined).parse(42)).toBe(42)
  })

  test("falls back to z.any() for unknown type", () => {
    const schema = jsonSchemaToZod({ type: "unknown_type" })
    expect(schema.parse("anything")).toBe("anything")
  })

  test("handles object with no properties (passthrough)", () => {
    const schema = jsonSchemaToZod({ type: "object" })
    expect(schema.parse({ any: "value", works: true })).toEqual({ any: "value", works: true })
  })

  test("handles schema with properties but no type", () => {
    const schema = jsonSchemaToZod({
      properties: { name: { type: "string" } },
      required: ["name"]
    })
    expect(schema.parse({ name: "test" })).toEqual({ name: "test" })
  })

  test("handles mixed literal enum", () => {
    const schema = jsonSchemaToZod({ enum: [1, 2, 3] })
    expect(schema.parse(1)).toBe(1)
  })

  test("converts real-world Anthropic tool schema", () => {
    const schema = jsonSchemaToZod({
      type: "object",
      properties: {
        command: {
          type: "string",
          description: "The bash command to execute"
        },
        timeout: {
          type: "integer",
          description: "Timeout in seconds"
        }
      },
      required: ["command"]
    })
    expect(schema.parse({ command: "ls -la" })).toEqual({ command: "ls -la" })
    expect(schema.parse({ command: "pwd", timeout: 30 })).toEqual({ command: "pwd", timeout: 30 })
  })
})

// ── createToolMcpServer ─────────────────────────────────────────────────────

describe("createToolMcpServer", () => {
  test("creates server with single tool", () => {
    const server = createToolMcpServer([{
      name: "get_weather",
      description: "Get weather for a city",
      input_schema: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"]
      }
    }])
    expect(server.type).toBe("sdk")
    expect(server.name).toBe("proxy-tools")
    expect(server.instance).toBeDefined()
  })

  test("creates server with multiple tools", () => {
    const server = createToolMcpServer([
      {
        name: "search",
        description: "Search the web",
        input_schema: {
          type: "object",
          properties: { query: { type: "string" } },
          required: ["query"]
        }
      },
      {
        name: "calculate",
        description: "Do math",
        input_schema: {
          type: "object",
          properties: {
            expression: { type: "string" },
            precision: { type: "integer" }
          },
          required: ["expression"]
        }
      }
    ])
    expect(server.type).toBe("sdk")
    expect(server.name).toBe("proxy-tools")
    expect(server.instance).toBeDefined()
  })

  test("handles tool with no input_schema", () => {
    const server = createToolMcpServer([{
      name: "noop",
      description: "Does nothing"
    }])
    expect(server.type).toBe("sdk")
    expect(server.instance).toBeDefined()
  })

  test("handles tool with empty schema", () => {
    const server = createToolMcpServer([{
      name: "ping",
      description: "Ping",
      input_schema: {}
    }])
    expect(server.type).toBe("sdk")
    expect(server.instance).toBeDefined()
  })
})

// ── Client tool mode detection ───────────────────────────────────────────────

describe("client tool mode detection", () => {
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

  test("false when no tools", () => {
    expect(isClientToolMode({ messages: [{ role: "user", content: "hi" }] })).toBe(false)
  })

  test("false when tools is empty", () => {
    expect(isClientToolMode({ tools: [], messages: [] })).toBe(false)
  })

  test("true when tools provided with normal system prompt", () => {
    expect(isClientToolMode({
      tools: [{ name: "t1" }],
      system: "You are helpful.",
      messages: [{ role: "user", content: "hi" }]
    })).toBe(true)
  })

  test("true when messages contain tool_result", () => {
    expect(isClientToolMode({
      tools: [{ name: "t1" }],
      messages: [
        { role: "user", content: [{ type: "tool_result", tool_use_id: "x", content: "ok" }] }
      ]
    })).toBe(true)
  })

  test("false when system contains conversation_label (agent mode)", () => {
    expect(isClientToolMode({
      tools: [{ name: "t1" }],
      system: "conversation_label: -123456 | Some Chat",
      messages: [{ role: "user", content: "hi" }]
    })).toBe(false)
  })

  test("false when system contains chat id:", () => {
    expect(isClientToolMode({
      tools: [{ name: "t1" }],
      system: "chat id: 12345",
      messages: [{ role: "user", content: "hi" }]
    })).toBe(false)
  })

  test("false when system is array with conversation_label", () => {
    expect(isClientToolMode({
      tools: [{ name: "t1" }],
      system: [{ type: "text", text: "conversation_label: -123" }],
      messages: [{ role: "user", content: "hi" }]
    })).toBe(false)
  })
})

// ── Rough token estimation ───────────────────────────────────────────────────

describe("rough token estimation", () => {
  function roughTokens(text: string): number {
    return Math.ceil((text ?? "").length / 4)
  }

  test("estimates tokens", () => {
    expect(roughTokens("hello world")).toBe(3) // 11 chars / 4 = 2.75 → 3
  })

  test("handles empty string", () => {
    expect(roughTokens("")).toBe(0)
  })

  test("handles long text", () => {
    const text = "a".repeat(1000)
    expect(roughTokens(text)).toBe(250)
  })
})
