// OpenAI-to-Anthropic format conversion utilities for the /v1/chat/completions compatibility layer.

export function convertOpenaiContent(content: any): any {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return String(content ?? "")

  return content.map((part: any) => {
    if (part.type === "text") return { type: "text", text: part.text ?? "" }
    if (part.type === "image_url" && part.image_url?.url) {
      const url = part.image_url.url as string
      const dataMatch = url.match(/^data:(image\/\w+);base64,(.+)$/)
      if (dataMatch) {
        return {
          type: "image",
          source: {
            type: "base64",
            media_type: dataMatch[1]!,
            data: dataMatch[2]!
          }
        }
      }
      return {
        type: "image",
        source: { type: "url", url }
      }
    }
    return part
  })
}

export function openaiToAnthropicMessages(messages: any[]): { system?: string; messages: any[] } {
  let system: string | undefined
  const converted: any[] = []

  for (const msg of messages) {
    if (msg.role === "system") {
      const sysText = typeof msg.content === "string" ? msg.content
        : Array.isArray(msg.content) ? msg.content.filter((p: any) => p.type === "text").map((p: any) => p.text ?? "").join("")
        : String(msg.content ?? "")
      system = (system ? system + "\n" : "") + sysText
    } else if (msg.role === "user") {
      converted.push({ role: "user", content: convertOpenaiContent(msg.content) })
    } else if (msg.role === "assistant") {
      if (msg.tool_calls?.length) {
        const content: any[] = []
        if (msg.content) content.push({ type: "text", text: msg.content })
        for (const tc of msg.tool_calls) {
          content.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function?.name ?? "",
            input: tc.function?.arguments ? JSON.parse(tc.function.arguments) : {}
          })
        }
        converted.push({ role: "assistant", content })
      } else {
        converted.push({ role: "assistant", content: msg.content ?? "" })
      }
    } else if (msg.role === "tool") {
      converted.push({
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: msg.tool_call_id,
          content: msg.content ?? ""
        }]
      })
    }
  }
  return { system, messages: converted }
}

export function openaiToAnthropicTools(tools: any[]): any[] {
  return tools
    .filter((t: any) => t.type === "function" && t.function)
    .map((t: any) => ({
      name: t.function.name,
      description: t.function.description ?? "",
      input_schema: t.function.parameters ?? { type: "object", properties: {} }
    }))
}

export function anthropicToOpenaiResponse(anthropicBody: any, model: string, chatId: string): any {
  const textBlocks = (anthropicBody.content ?? []).filter((b: any) => b.type === "text")
  const toolBlocks = (anthropicBody.content ?? []).filter((b: any) => b.type === "tool_use")

  const text = textBlocks.map((b: any) => b.text).join("") || (toolBlocks.length > 0 ? null : "")

  const message: any = { role: "assistant", content: text }

  if (toolBlocks.length > 0) {
    message.tool_calls = toolBlocks.map((b: any, i: number) => ({
      id: b.id,
      type: "function",
      function: {
        name: b.name,
        arguments: JSON.stringify(b.input ?? {})
      }
    }))
  }

  const finishReason = anthropicBody.stop_reason === "tool_use" ? "tool_calls"
    : anthropicBody.stop_reason === "max_tokens" ? "length"
    : "stop"

  return {
    id: chatId,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message,
      finish_reason: finishReason
    }],
    usage: {
      prompt_tokens: anthropicBody.usage?.input_tokens ?? 0,
      completion_tokens: anthropicBody.usage?.output_tokens ?? 0,
      total_tokens: (anthropicBody.usage?.input_tokens ?? 0) + (anthropicBody.usage?.output_tokens ?? 0)
    }
  }
}
