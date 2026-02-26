import { logInfo, logWarn, logError, logDebug, dumpError } from "./logger"

// ── Request Trace ────────────────────────────────────────────────────────────
// Captures the full lifecycle of a single API request with timing milestones,
// so you can see exactly WHERE time was spent and WHERE failures occurred.

export type TracePhase =
  | "received"      // HTTP request received, body parsed
  | "validated"     // Request validated, model resolved
  | "queued"        // Waiting for concurrency slot
  | "acquired"      // Concurrency slot acquired
  | "sdk_starting"  // About to call SDK query()
  | "sdk_streaming" // Receiving events from SDK
  | "sdk_done"      // SDK query() iterator finished
  | "responding"    // Building/sending HTTP response
  | "completed"     // Successfully sent response
  | "error"         // Failed at some point

export type TraceStatus = "active" | "completed" | "error"

export interface TraceError {
  type: string       // "AbortError", "sdk_error", "queue_timeout", "parse_error", etc.
  message: string
  stack?: string
  phase: TracePhase  // Which phase the error occurred in
}

export interface TraceUsage {
  input_tokens: number
  output_tokens: number
  cache_read_input_tokens?: number
  cache_creation_input_tokens?: number
}

export interface TraceSession {
  conversationId?: string
  sdkSessionId?: string
  isResuming: boolean
  resumeCount?: number
  resumeFailed?: boolean
}

export interface RequestTrace {
  reqId: string
  startedAt: number       // Date.now() when request received

  // Request metadata
  model: string           // "haiku" | "sonnet" | "opus"
  requestedModel: string  // Original model string from caller (e.g. "claude-haiku-4-5")
  stream: boolean
  hasTools: boolean
  thinking?: string       // "enabled" | "disabled" | "adaptive"
  promptLen: number       // Character length of serialized prompt
  systemLen: number       // Character length of system prompt
  msgCount: number        // Number of messages in request
  bodyBytes: number       // Raw HTTP body size in bytes

  // Client info
  clientIp?: string
  userAgent?: string

  // Session / resumption info
  session?: TraceSession

  // SDK usage (authoritative, from result message)
  usage?: TraceUsage

  // Timing milestones (all Date.now() values)
  queuedAt?: number       // When we started waiting for a slot
  acquiredAt?: number     // When we got a concurrency slot
  sdkStartedAt?: number   // When query() was called
  firstTokenAt?: number   // When first content event arrived
  sdkEndedAt?: number     // When query() iterator finished
  completedAt?: number    // When HTTP response was fully sent

  // Phase tracking
  phase: TracePhase
  status: TraceStatus

  // Output metrics
  sdkEventCount: number
  outputLen: number       // Character length of generated text
  toolCallCount: number
  stallCount: number      // How many 15s stall intervals where stallMs > 30s occurred

  // SDK event type distribution
  eventTypes: Record<string, number>  // e.g. { "content_block_delta": 500, "content_block_start": 1, ... }

  // Last event tracking (for stall diagnostics)
  lastEventAt: number     // Date.now() of last SDK event
  lastEventType?: string  // Type of last SDK event received

  // Termination reason (more specific than just error/completed)
  endReason?: "completed" | "client_disconnect" | "stall_timeout" | "queue_timeout" | "sdk_error" | "abort" | "unknown"

  // Error info
  error?: TraceError

  // SDK debug log path (from DEBUG_CLAUDE_AGENT_SDK)
  sdkDebugLogPath?: string
}

// ── Per-model stats ──────────────────────────────────────────────────────────

interface ModelStats {
  total: number
  errors: number
  totalDurationMs: number
  totalTimeToFirstTokenMs: number
  firstTokenCount: number  // requests where we got a first token (for avg calc)
  maxDurationMs: number
  lastErrorAt?: number
  lastErrorReqId?: string
}

function newModelStats(): ModelStats {
  return { total: 0, errors: 0, totalDurationMs: 0, totalTimeToFirstTokenMs: 0, firstTokenCount: 0, maxDurationMs: 0 }
}

// ── Trace Store ──────────────────────────────────────────────────────────────
// In-memory ring buffer of recent traces + aggregate stats.

const BUFFER_SIZE = 200
const ERROR_BUFFER_SIZE = 50

class TraceStore {
  private traces: RequestTrace[] = []
  private errorTraces: RequestTrace[] = []
  private activeTraces = new Map<string, RequestTrace>()
  private stats = {
    totalRequests: 0,
    totalErrors: 0,
    totalDurationMs: 0,
    startedAt: Date.now(),
  }
  private modelStats = new Map<string, ModelStats>()

  /** Create a new trace for a request. */
  create(init: {
    reqId: string
    model: string
    requestedModel: string
    stream: boolean
    hasTools: boolean
    thinking?: string
    promptLen: number
    systemLen: number
    msgCount: number
    bodyBytes: number
    clientIp?: string
    userAgent?: string
  }): RequestTrace {
    const now = Date.now()
    const trace: RequestTrace = {
      ...init,
      startedAt: now,
      phase: "received",
      status: "active",
      sdkEventCount: 0,
      outputLen: 0,
      toolCallCount: 0,
      stallCount: 0,
      eventTypes: {},
      lastEventAt: now,
    }
    this.activeTraces.set(init.reqId, trace)
    this.stats.totalRequests++

    logInfo("trace.created", {
      reqId: init.reqId,
      model: init.model,
      requestedModel: init.requestedModel,
      stream: init.stream,
      hasTools: init.hasTools,
      thinking: init.thinking,
      promptLen: init.promptLen,
      systemLen: init.systemLen,
      msgCount: init.msgCount,
      bodyBytes: init.bodyBytes,
      clientIp: init.clientIp,
      userAgent: init.userAgent,
    })

    return trace
  }

  /** Update the phase of an active trace. Logs the transition. */
  phase(reqId: string, phase: TracePhase, extra?: Record<string, unknown>) {
    const trace = this.activeTraces.get(reqId)
    if (!trace) return

    const now = Date.now()
    trace.phase = phase

    switch (phase) {
      case "queued":
        trace.queuedAt = now
        break
      case "acquired":
        trace.acquiredAt = now
        break
      case "sdk_starting":
        trace.sdkStartedAt = now
        break
      case "sdk_streaming":
        if (!trace.firstTokenAt) trace.firstTokenAt = now
        break
      case "sdk_done":
        trace.sdkEndedAt = now
        break
      case "completed":
        trace.completedAt = now
        break
    }

    const elapsed = now - trace.startedAt
    logDebug("trace.phase", { reqId, phase, elapsedMs: elapsed, ...extra })
  }

  /** Record an SDK event. Tracks timing, event type distribution, and first-token detection. */
  sdkEvent(reqId: string, eventNum: number, eventType: string, subtype?: string) {
    const trace = this.activeTraces.get(reqId)
    if (!trace) return

    const now = Date.now()
    trace.sdkEventCount = eventNum
    trace.lastEventAt = now

    // Track event type distribution
    const key = subtype ?? eventType
    trace.lastEventType = key
    trace.eventTypes[key] = (trace.eventTypes[key] ?? 0) + 1

    // Mark first content event
    if (!trace.firstTokenAt && (subtype === "content_block_delta" || subtype === "content_block_start")) {
      trace.firstTokenAt = now
      const ttft = now - trace.startedAt
      const ttftFromSdk = trace.sdkStartedAt ? now - trace.sdkStartedAt : undefined

      logInfo("trace.first_token", {
        reqId,
        ttftMs: ttft,
        ttftFromSdkMs: ttftFromSdk,
        eventNum,
        model: trace.model,
      })
    }

    // Log first 5 events, then every 200th, plus every thinking event
    if (eventNum <= 5 || eventNum % 200 === 0 || subtype === "thinking") {
      logDebug("trace.sdk_event", {
        reqId,
        n: eventNum,
        type: eventType,
        subtype,
        elapsedMs: now - trace.startedAt,
        outputLen: trace.outputLen,
      })
    }

    // Log important lifecycle events at info level
    if (eventType === "result" || subtype === "message_start" || subtype === "message_delta") {
      logInfo("trace.sdk_lifecycle", {
        reqId,
        n: eventNum,
        type: eventType,
        subtype,
        elapsedMs: now - trace.startedAt,
        model: trace.model,
      })
    }
  }

  /** Record a stall check (called every 15s). Only warns if idle > 30s. */
  stall(reqId: string, stallMs: number) {
    const trace = this.activeTraces.get(reqId)
    if (!trace) return

    // Only count meaningful stalls (>30s idle)
    if (stallMs < 30_000) {
      // Short gap — debug log only, not a real stall
      logDebug("trace.stall_check", {
        reqId,
        stallMs,
        sdkEventCount: trace.sdkEventCount,
        phase: trace.phase,
      })
      return
    }

    trace.stallCount++
    const level = stallMs > 60_000 ? "error" : "warn"
    const log = level === "error" ? logError : logWarn

    log("trace.stall", {
      reqId,
      stallMs,
      stallCount: trace.stallCount,
      sdkEventCount: trace.sdkEventCount,
      outputLen: trace.outputLen,
      elapsedMs: Date.now() - trace.startedAt,
      phase: trace.phase,
      model: trace.model,
      lastEventType: trace.lastEventType,
      eventTypes: trace.eventTypes,
    })
  }

  /** Mark a trace as successfully completed. */
  complete(reqId: string, extra?: { outputLen?: number; toolCallCount?: number }) {
    const trace = this.activeTraces.get(reqId)
    if (!trace) return

    const now = Date.now()
    trace.completedAt = now
    trace.phase = "completed"
    trace.status = "completed"
    trace.endReason = "completed"
    if (extra?.outputLen !== undefined) trace.outputLen = extra.outputLen
    if (extra?.toolCallCount !== undefined) trace.toolCallCount = extra.toolCallCount

    const duration = now - trace.startedAt
    const timings = this.computeTimings(trace)

    // Compute throughput (chars/sec) over the streaming period
    const streamDuration = trace.sdkStartedAt ? now - trace.sdkStartedAt : duration
    const charsPerSec = streamDuration > 0 ? Math.round((trace.outputLen / streamDuration) * 1000) : 0
    const eventsPerSec = streamDuration > 0 ? Math.round((trace.sdkEventCount / streamDuration) * 1000) : 0

    logInfo("trace.completed", {
      reqId,
      model: trace.model,
      requestedModel: trace.requestedModel,
      durationMs: duration,
      ...timings,
      sdkEventCount: trace.sdkEventCount,
      outputLen: trace.outputLen,
      toolCallCount: trace.toolCallCount,
      stallCount: trace.stallCount,
      charsPerSec,
      eventsPerSec,
      eventTypes: trace.eventTypes,
      ...(trace.usage ? { usage: trace.usage } : {}),
      ...(trace.session ? { session: trace.session } : {}),
    })

    // Update stats
    this.stats.totalDurationMs += duration
    const ms = this.getModelStats(trace.model)
    ms.total++
    ms.totalDurationMs += duration
    if (ms.maxDurationMs < duration) ms.maxDurationMs = duration
    if (timings.ttftMs !== undefined) {
      ms.totalTimeToFirstTokenMs += timings.ttftMs
      ms.firstTokenCount++
    }

    this.archive(trace)
  }

  /** Mark a trace as failed. Dumps error context to file. */
  fail(reqId: string, error: Error | string, phase?: TracePhase, extra?: Record<string, unknown>) {
    const trace = this.activeTraces.get(reqId)
    if (!trace) {
      // No trace found — log the error anyway
      logError("trace.fail.no_trace", { reqId, error: String(error), phase })
      return
    }

    const now = Date.now()
    trace.completedAt = now
    trace.phase = phase ?? trace.phase
    trace.status = "error"

    const err = error instanceof Error ? error : new Error(String(error))
    const errorType = classifyError(err)

    // Determine specific end reason
    trace.endReason = extra?.clientDisconnect ? "client_disconnect"
      : errorType === "stall_timeout" ? "stall_timeout"
      : errorType === "queue_timeout" ? "queue_timeout"
      : errorType === "timeout" ? "stall_timeout"
      : err.name === "AbortError" ? "abort"
      : "sdk_error"

    trace.error = {
      type: errorType,
      message: err.message,
      stack: err.stack,
      phase: trace.phase,
    }

    const duration = now - trace.startedAt
    const timings = this.computeTimings(trace)
    const timeSinceLastEvent = now - trace.lastEventAt

    // Compute throughput (chars/sec) over the streaming period
    const streamDuration = trace.sdkStartedAt ? now - trace.sdkStartedAt : duration
    const charsPerSec = streamDuration > 0 ? Math.round((trace.outputLen / streamDuration) * 1000) : 0

    logError("trace.failed", {
      reqId,
      model: trace.model,
      requestedModel: trace.requestedModel,
      endReason: trace.endReason,
      errorType,
      error: err.message,
      phase: trace.phase,
      durationMs: duration,
      ...timings,
      sdkEventCount: trace.sdkEventCount,
      outputLen: trace.outputLen,
      stallCount: trace.stallCount,
      charsPerSec,
      timeSinceLastEventMs: timeSinceLastEvent,
      lastEventType: trace.lastEventType,
      eventTypes: trace.eventTypes,
      ...(trace.usage ? { usage: trace.usage } : {}),
      ...(trace.session ? { session: trace.session } : {}),
      ...extra,
    })

    // Update stats
    this.stats.totalErrors++
    this.stats.totalDurationMs += duration
    const ms = this.getModelStats(trace.model)
    ms.total++
    ms.errors++
    ms.totalDurationMs += duration
    if (ms.maxDurationMs < duration) ms.maxDurationMs = duration
    ms.lastErrorAt = now
    ms.lastErrorReqId = reqId
    if (timings.ttftMs !== undefined) {
      ms.totalTimeToFirstTokenMs += timings.ttftMs
      ms.firstTokenCount++
    }

    // Dump full error context to file
    const dumpPath = dumpError(reqId, {
      trace: this.serializeTrace(trace),
      error: { type: errorType, message: err.message, stack: err.stack, phase: trace.phase },
      endReason: trace.endReason,
      timeSinceLastEventMs: timeSinceLastEvent,
      lastEventType: trace.lastEventType,
      eventTypes: trace.eventTypes,
      charsPerSec,
      ...extra,
    })
    logInfo("trace.error_dumped", { reqId, path: dumpPath })

    // Store in error buffer
    this.errorTraces.push({ ...trace })
    if (this.errorTraces.length > ERROR_BUFFER_SIZE) {
      this.errorTraces.shift()
    }

    this.archive(trace)
  }

  /** Update output length on a live trace (during streaming). */
  updateOutput(reqId: string, outputLen: number) {
    const trace = this.activeTraces.get(reqId)
    if (trace) trace.outputLen = outputLen
  }

  /** Set session/resumption info on a trace. */
  setSession(reqId: string, session: TraceSession) {
    const trace = this.activeTraces.get(reqId)
    if (trace) trace.session = session
  }

  /** Set final SDK usage on a trace. */
  setUsage(reqId: string, usage: TraceUsage) {
    const trace = this.activeTraces.get(reqId)
    if (trace) trace.usage = usage
  }

  /** Set the SDK debug log path for a trace. */
  setSdkDebugLog(reqId: string, path: string) {
    const trace = this.activeTraces.get(reqId)
    if (trace) trace.sdkDebugLogPath = path
  }

  // ── Query methods (for debug endpoints) ──────────────────────────────────

  /** Get aggregate stats. */
  getStats() {
    const now = Date.now()
    const uptimeMs = now - this.stats.startedAt
    const avgDurationMs = this.stats.totalRequests > 0
      ? Math.round(this.stats.totalDurationMs / this.stats.totalRequests)
      : 0

    const byModel: Record<string, {
      total: number
      errors: number
      avgDurationMs: number
      avgTtftMs: number
      maxDurationMs: number
      lastErrorAt?: string
      lastErrorReqId?: string
    }> = {}
    for (const [model, ms] of this.modelStats) {
      byModel[model] = {
        total: ms.total,
        errors: ms.errors,
        avgDurationMs: ms.total > 0 ? Math.round(ms.totalDurationMs / ms.total) : 0,
        avgTtftMs: ms.firstTokenCount > 0 ? Math.round(ms.totalTimeToFirstTokenMs / ms.firstTokenCount) : 0,
        maxDurationMs: ms.maxDurationMs,
        ...(ms.lastErrorAt ? { lastErrorAt: new Date(ms.lastErrorAt).toISOString() } : {}),
        ...(ms.lastErrorReqId ? { lastErrorReqId: ms.lastErrorReqId } : {}),
      }
    }

    return {
      uptimeMs,
      uptimeHuman: humanDuration(uptimeMs),
      requests: {
        total: this.stats.totalRequests,
        errors: this.stats.totalErrors,
        active: this.activeTraces.size,
        avgDurationMs,
        errorRate: this.stats.totalRequests > 0
          ? `${((this.stats.totalErrors / this.stats.totalRequests) * 100).toFixed(1)}%`
          : "0%",
      },
      byModel,
      activeRequests: Array.from(this.activeTraces.values()).map(t => ({
        reqId: t.reqId,
        model: t.model,
        requestedModel: t.requestedModel,
        phase: t.phase,
        stream: t.stream,
        hasTools: t.hasTools,
        thinking: t.thinking,
        elapsedMs: now - t.startedAt,
        timeSinceLastEventMs: now - t.lastEventAt,
        lastEventType: t.lastEventType,
        sdkEventCount: t.sdkEventCount,
        outputLen: t.outputLen,
        stallCount: t.stallCount,
        promptLen: t.promptLen,
        systemLen: t.systemLen,
        bodyBytes: t.bodyBytes,
        clientIp: t.clientIp,
      })),
    }
  }

  /** Get recent traces (most recent first). */
  getRecentTraces(limit = 20): ReturnType<typeof this.serializeTrace>[] {
    return this.traces.slice(-limit).reverse().map(t => this.serializeTrace(t))
  }

  /** Get a specific trace by reqId. */
  getTrace(reqId: string): ReturnType<typeof this.serializeTrace> | null {
    // Check active first
    const active = this.activeTraces.get(reqId)
    if (active) return this.serializeTrace(active)
    // Check buffer
    const archived = this.traces.find(t => t.reqId === reqId)
    if (archived) return this.serializeTrace(archived)
    // Check error buffer
    const err = this.errorTraces.find(t => t.reqId === reqId)
    if (err) return this.serializeTrace(err)
    return null
  }

  /** Get recent error traces. */
  getRecentErrors(limit = 10): ReturnType<typeof this.serializeTrace>[] {
    return this.errorTraces.slice(-limit).reverse().map(t => this.serializeTrace(t))
  }

  // ── Internal ─────────────────────────────────────────────────────────────

  private archive(trace: RequestTrace) {
    this.activeTraces.delete(trace.reqId)
    this.traces.push(trace)
    if (this.traces.length > BUFFER_SIZE) {
      this.traces.shift()
    }
  }

  private getModelStats(model: string): ModelStats {
    let ms = this.modelStats.get(model)
    if (!ms) {
      ms = newModelStats()
      this.modelStats.set(model, ms)
    }
    return ms
  }

  private computeTimings(trace: RequestTrace) {
    const result: Record<string, number | undefined> = {}
    if (trace.queuedAt && trace.acquiredAt) {
      result.queueWaitMs = trace.acquiredAt - trace.queuedAt
    }
    if (trace.firstTokenAt) {
      result.ttftMs = trace.firstTokenAt - trace.startedAt
      if (trace.sdkStartedAt) {
        result.ttftFromSdkMs = trace.firstTokenAt - trace.sdkStartedAt
      }
    }
    if (trace.sdkStartedAt && trace.sdkEndedAt) {
      result.sdkDurationMs = trace.sdkEndedAt - trace.sdkStartedAt
    }
    if (trace.completedAt) {
      result.totalDurationMs = trace.completedAt - trace.startedAt
    }
    return result
  }

  private serializeTrace(trace: RequestTrace) {
    const now = Date.now()
    const timings = this.computeTimings(trace)
    const duration = (trace.completedAt ?? now) - trace.startedAt
    const streamDuration = trace.sdkStartedAt ? (trace.completedAt ?? now) - trace.sdkStartedAt : duration
    const charsPerSec = streamDuration > 0 ? Math.round((trace.outputLen / streamDuration) * 1000) : 0
    const timeSinceLastEvent = now - trace.lastEventAt

    return {
      reqId: trace.reqId,
      model: trace.model,
      requestedModel: trace.requestedModel,
      stream: trace.stream,
      hasTools: trace.hasTools,
      thinking: trace.thinking,
      promptLen: trace.promptLen,
      systemLen: trace.systemLen,
      msgCount: trace.msgCount,
      bodyBytes: trace.bodyBytes,
      clientIp: trace.clientIp,
      phase: trace.phase,
      status: trace.status,
      endReason: trace.endReason,
      sdkEventCount: trace.sdkEventCount,
      outputLen: trace.outputLen,
      toolCallCount: trace.toolCallCount,
      stallCount: trace.stallCount,
      charsPerSec,
      eventTypes: trace.eventTypes,
      lastEventType: trace.lastEventType,
      startedAt: new Date(trace.startedAt).toISOString(),
      ...(trace.completedAt
        ? { completedAt: new Date(trace.completedAt).toISOString() }
        : { elapsedMs: now - trace.startedAt, timeSinceLastEventMs: timeSinceLastEvent }),
      ...timings,
      ...(trace.usage ? { usage: trace.usage } : {}),
      ...(trace.session ? { session: trace.session } : {}),
      ...(trace.error ? { error: trace.error } : {}),
      ...(trace.sdkDebugLogPath ? { sdkDebugLogPath: trace.sdkDebugLogPath } : {}),
    }
  }
}

// ── Error classification ─────────────────────────────────────────────────────

export function classifyError(err: Error): string {
  if (err.name === "AbortError" || err.message?.includes("aborted")) return "stall_timeout"
  if (err.message.includes("Queue timeout")) return "queue_timeout"
  if (err.message.includes("client disconnect") || err.message.includes("cancel")) return "client_disconnect"
  if (err.message.includes("process aborted")) return "sdk_aborted"
  if (err.message.includes("SIGTERM") || err.message.includes("SIGKILL")) return "sdk_killed"
  if (err.message.includes("spawn") || err.message.includes("ENOENT")) return "sdk_spawn_error"
  if (err.message.includes("JSON")) return "parse_error"
  if (err.message.includes("ECONNREFUSED") || err.message.includes("ECONNRESET")) return "connection_error"
  if (err.message.includes("EPIPE") || err.message.includes("broken pipe")) return "broken_pipe"
  if (err.message.includes("memory") || err.message.includes("OOM")) return "oom_error"
  if (err.message.includes("rate limit") || err.message.includes("429")) return "rate_limit"
  return "unknown_error"
}

function humanDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`
  const h = Math.floor(ms / 3600000)
  const m = Math.floor((ms % 3600000) / 60000)
  return `${h}h ${m}m`
}

// ── Singleton ────────────────────────────────────────────────────────────────

export const traceStore = new TraceStore()
