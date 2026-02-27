// src/errors.ts — Error classification for proxy responses

export interface ErrorClassification {
  message: string
  type: string  // "api_error" | "overloaded_error"
  status: number
}

export function classifyError(
  error: Error,
  opts: {
    clientDisconnected: boolean
    abortReason: "stall" | "max_duration" | "max_output" | null
    stallTimeoutMs: number
    maxDurationMs: number
    maxOutputChars: number
    outputLen?: number
  }
): ErrorClassification {
  const isAbort = error.name === "AbortError" || error.message?.includes("abort")
  const isQueueTimeout = error.message.includes("Queue timeout")

  if (opts.clientDisconnected) {
    return { message: "Client disconnected.", type: "api_error", status: 500 }
  }
  if (opts.abortReason === "max_duration") {
    return {
      message: `Request exceeded max duration of ${opts.maxDurationMs / 1000}s.${opts.outputLen != null ? ` Output: ${opts.outputLen} chars.` : ""}`,
      type: "api_error",
      status: 504,
    }
  }
  if (opts.abortReason === "max_output") {
    return {
      message: `Request exceeded max output size of ${opts.maxOutputChars} chars.`,
      type: "api_error",
      status: 504,
    }
  }
  if (isAbort) {
    return {
      message: `Request stalled — no SDK activity for ${opts.stallTimeoutMs / 1000}s. Please retry.`,
      type: "api_error",
      status: 504,
    }
  }
  if (isQueueTimeout) {
    return {
      message: "Server busy — all request slots are occupied. Please retry shortly.",
      type: "overloaded_error",
      status: 529,
    }
  }
  return { message: error.message, type: "api_error", status: 500 }
}
