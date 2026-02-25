// mcpTools.ts — Converts Anthropic API tool definitions to native MCP tools
// via the Claude Agent SDK's in-process MCP server support.

import { z } from "zod"
import { tool, createSdkMcpServer } from "@anthropic-ai/claude-agent-sdk"

// ── JSON Schema → Zod converter ─────────────────────────────────────────────

/**
 * Convert a JSON Schema object (as used in Anthropic tool input_schema)
 * to a Zod schema. Handles: string, number, integer, boolean, array,
 * object, enum, nullable, required/optional, nested objects, anyOf/oneOf.
 * Falls back to z.any() for unrecognized types.
 */
export function jsonSchemaToZod(schema: any): z.ZodTypeAny {
  if (!schema || typeof schema !== "object") return z.any()

  // anyOf / oneOf → z.union
  if (schema.anyOf || schema.oneOf) {
    const variants = (schema.anyOf ?? schema.oneOf) as any[]
    // Common pattern: { anyOf: [{ type: "string" }, { type: "null" }] } → nullable
    const nonNull = variants.filter((v: any) => v.type !== "null")
    const hasNull = variants.some((v: any) => v.type === "null")
    if (hasNull && nonNull.length === 1) {
      return jsonSchemaToZod(nonNull[0]).nullable()
    }
    if (variants.length === 0) return z.any()
    if (variants.length === 1) return jsonSchemaToZod(variants[0])
    const zodVariants = variants.map((v: any) => jsonSchemaToZod(v))
    // z.union needs at least 2 elements
    return z.union(zodVariants as [z.ZodTypeAny, z.ZodTypeAny, ...z.ZodTypeAny[]])
  }

  // enum → z.enum or z.union of literals
  if (schema.enum) {
    const values = schema.enum as any[]
    if (values.length === 0) return z.any()
    // String enums
    if (values.every((v: any) => typeof v === "string")) {
      return z.enum(values as [string, ...string[]])
    }
    // Mixed literal types
    const literals = values.map((v: any) => z.literal(v))
    if (literals.length === 1) return literals[0]!
    return z.union(literals as [z.ZodTypeAny, z.ZodTypeAny, ...z.ZodTypeAny[]])
  }

  // const → z.literal
  if ("const" in schema) {
    return z.literal(schema.const)
  }

  switch (schema.type) {
    case "string":
      return z.string()

    case "number":
    case "integer":
      return z.number()

    case "boolean":
      return z.boolean()

    case "null":
      return z.null()

    case "array": {
      const itemSchema = schema.items ? jsonSchemaToZod(schema.items) : z.any()
      return z.array(itemSchema)
    }

    case "object": {
      return jsonSchemaObjectToZod(schema)
    }

    default: {
      // Handle arrays of types like { type: ["string", "null"] }
      if (Array.isArray(schema.type)) {
        const types = schema.type as string[]
        const hasNull = types.includes("null")
        const nonNullTypes = types.filter((t: string) => t !== "null")
        if (nonNullTypes.length === 0) return z.null()
        if (nonNullTypes.length === 1) {
          const base = jsonSchemaToZod({ ...schema, type: nonNullTypes[0] })
          return hasNull ? base.nullable() : base
        }
        const variants = nonNullTypes.map((t: string) => jsonSchemaToZod({ ...schema, type: t }))
        if (hasNull) variants.push(z.null())
        if (variants.length === 1) return variants[0]!
        return z.union(variants as [z.ZodTypeAny, z.ZodTypeAny, ...z.ZodTypeAny[]])
      }

      // No type specified but has properties → treat as object
      if (schema.properties) {
        return jsonSchemaObjectToZod(schema)
      }

      return z.any()
    }
  }
}

function jsonSchemaObjectToZod(schema: any): z.ZodTypeAny {
  const properties = schema.properties ?? {}
  const required = new Set<string>(schema.required ?? [])
  const shape: Record<string, z.ZodTypeAny> = {}

  for (const [key, propSchema] of Object.entries(properties)) {
    let zodProp = jsonSchemaToZod(propSchema as any)
    if (!required.has(key)) {
      zodProp = zodProp.optional()
    }
    shape[key] = zodProp
  }

  if (Object.keys(shape).length === 0) {
    return z.object({}).passthrough()
  }

  // Use passthrough to allow additional properties (common in tool schemas)
  return z.object(shape).passthrough()
}

// ── Zod shape extraction ────────────────────────────────────────────────────

/**
 * Extract the "raw shape" from a Zod object schema for use with the SDK's
 * tool() helper which expects a ZodRawShape (Record<string, ZodTypeAny>).
 * If the schema isn't a ZodObject, wrap it as { input: schema }.
 */
function extractZodShape(zodSchema: z.ZodTypeAny): Record<string, z.ZodTypeAny> {
  // ZodObject stores its shape
  if (zodSchema instanceof z.ZodObject) {
    return zodSchema.shape
  }
  // Fallback: wrap non-object schemas
  return { input: zodSchema }
}

// ── MCP Server factory ──────────────────────────────────────────────────────

interface AnthropicToolDef {
  name: string
  description?: string
  input_schema?: any
}

/**
 * Create an in-process MCP server from an array of Anthropic tool definitions.
 * Each tool is registered with the SDK's tool() helper. The handlers are
 * no-ops since canUseTool will deny execution (client handles it).
 */
export function createToolMcpServer(tools: AnthropicToolDef[]) {
  const toolDefs = tools.map((t) => {
    let zodSchema: z.ZodTypeAny
    try {
      zodSchema = jsonSchemaToZod(t.input_schema ?? { type: "object", properties: {} })
    } catch (e) {
      // If conversion fails, use a passthrough object as fallback
      zodSchema = z.object({}).passthrough()
    }

    const shape = extractZodShape(zodSchema)

    return tool(
      t.name,
      t.description ?? "",
      shape,
      async () => {
        // Should never be called — canUseTool denies all execution
        return {
          content: [{ type: "text" as const, text: "Error: tool execution denied by proxy" }],
          isError: true,
        }
      }
    )
  })

  return createSdkMcpServer({
    name: "proxy-tools",
    tools: toolDefs,
  })
}
