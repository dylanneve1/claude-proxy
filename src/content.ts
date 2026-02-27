// Pure functions for content serialization, model mapping, and message construction.

export const MCP_TOOL_PREFIX = "mcp__proxy-tools__"

export function mapModelToClaudeModel(model: string): "sonnet" | "opus" | "haiku" {
  if (model.includes("opus")) return "opus"
  if (model.includes("haiku")) return "haiku"
  return "sonnet"
}

export function serializeBlock(block: any): string {
  switch (block.type) {
    case "text":
      return block.text || ""
    case "image":
      return "[Image attached]"
    case "tool_use":
      return `<tool_use>\n{"name": "${block.name}", "input": ${JSON.stringify(block.input ?? {})}}\n</tool_use>`
    case "tool_result": {
      const content = Array.isArray(block.content)
        ? block.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("")
        : String(block.content ?? "")
      const truncated = content.length > 4000
        ? content.slice(0, 4000) + `\n...[truncated ${content.length - 4000} chars]`
        : content
      return `[Tool Result (id: ${block.tool_use_id})]\n${truncated}\n[/Tool Result]`
    }
    case "thinking":
      return ""
    default:
      return ""
  }
}

export function serializeContent(content: string | Array<any>): string {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return String(content)
  return content.map(b => serializeBlock(b)).filter(Boolean).join("\n")
}

export function contentHasImages(content: string | Array<any>): boolean {
  if (typeof content === "string") return false
  if (!Array.isArray(content)) return false
  return content.some((b: any) => b.type === "image")
}

export function toAnthropicImageBlock(block: any): any {
  if (block.source) return block
  if (block.data && block.mimeType) {
    return {
      type: "image",
      source: {
        type: "base64",
        media_type: block.mimeType,
        data: block.data,
      }
    }
  }
  if (block.data && block.media_type) {
    return {
      type: "image",
      source: {
        type: "base64",
        media_type: block.media_type,
        data: block.data,
      }
    }
  }
  return block
}

export function buildNativeContent(content: string | Array<any>): Array<any> {
  if (typeof content === "string") return [{ type: "text", text: content }]
  if (!Array.isArray(content)) return [{ type: "text", text: String(content) }]
  return content.map((block: any) => {
    if (block.type === "image") return toAnthropicImageBlock(block)
    if (block.type === "text") return { type: "text", text: block.text ?? "" }
    const serialized = serializeBlock(block)
    return serialized ? { type: "text", text: serialized } : null
  }).filter(Boolean)
}

export function createSDKUserMessage(content: Array<any>, sessionId?: string): AsyncIterable<any> {
  const msg = {
    type: "user" as const,
    message: {
      role: "user" as const,
      content,
    },
    parent_tool_use_id: null,
    session_id: sessionId ?? "",
  }
  return {
    async *[Symbol.asyncIterator]() {
      yield msg
    }
  }
}

export function stripMcpPrefix(name: string): string {
  return name.startsWith(MCP_TOOL_PREFIX) ? name.slice(MCP_TOOL_PREFIX.length) : name
}

export function roughTokens(text: string): number {
  return Math.ceil((text ?? "").length / 4)
}

export function buildUsage(input: number, output: number, cacheRead: number, cacheCreation: number) {
  const usage: Record<string, number> = { input_tokens: input, output_tokens: output }
  if (cacheRead > 0) usage.cache_read_input_tokens = cacheRead
  if (cacheCreation > 0) usage.cache_creation_input_tokens = cacheCreation
  return usage
}

export function extractConversationLabel(messages: Array<{ role: string; content: string | Array<any> }>): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!
    if (msg.role !== "user") continue
    const text = typeof msg.content === "string"
      ? msg.content
      : Array.isArray(msg.content)
        ? msg.content.filter((b: any) => b.type === "text").map((b: any) => b.text ?? "").join("\n")
        : ""
    const jsonMatch = text.match(/Conversation info[^`]*```json\s*(\{[\s\S]*?\})\s*```/)
    if (!jsonMatch?.[1]) continue
    try {
      const meta = JSON.parse(jsonMatch[1])
      if (meta.conversation_label) return meta.conversation_label
      if (meta.sender_id) return `dm:${meta.sender_id}`
    } catch {
      const labelMatch = text.match(/"conversation_label"\s*:\s*"([^"]*)"/)
      if (labelMatch?.[1]) return labelMatch[1]
    }
  }
  return null
}
