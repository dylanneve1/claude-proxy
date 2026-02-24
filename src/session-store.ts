/**
 * Session Store — Maps proxy conversation IDs to Claude SDK session IDs.
 * Enables session resumption to avoid resending full conversation history.
 */

import { readFileSync, writeFileSync, mkdirSync } from "fs"
import { join, dirname } from "path"
import { fileURLToPath } from "url"
import { logInfo, logWarn, logDebug } from "./logger"

const __dirname = dirname(fileURLToPath(import.meta.url))
const STORE_PATH = join(__dirname, "..", "session-store.json")

export interface SessionEntry {
  sdkSessionId: string
  createdAt: number
  lastUsed: number
  messageCount: number
  model: string
  /** Number of successful resumes */
  resumeCount: number
  /** Number of resume failures (fell back to full context) */
  failureCount: number
}

export interface SessionStoreStats {
  totalSessions: number
  activeSessions: number
  totalResumes: number
  totalFailures: number
  hitRate: string
}

const DEFAULT_TTL_MS = 24 * 60 * 60 * 1000 // 24 hours
const MAX_SESSIONS = 500

class SessionStore {
  private sessions: Map<string, SessionEntry> = new Map()
  private hits = 0
  private misses = 0
  private ttlMs: number

  constructor(ttlMs = DEFAULT_TTL_MS) {
    this.ttlMs = ttlMs
    this.load()
    // Clear all sessions on startup — SDK sessions don't survive proxy restarts
    if (this.sessions.size > 0) {
      logInfo("session-store.startup_clear", { cleared: this.sessions.size })
      this.sessions.clear()
      this.save()
    }
  }

  /** Get a session by conversation ID */
  get(conversationId: string): SessionEntry | undefined {
    const entry = this.sessions.get(conversationId)
    if (!entry) {
      this.misses++
      return undefined
    }
    // Check TTL
    if (Date.now() - entry.lastUsed > this.ttlMs) {
      logDebug("session-store.expired", { conversationId, lastUsed: entry.lastUsed })
      this.sessions.delete(conversationId)
      this.misses++
      this.save()
      return undefined
    }
    this.hits++
    return entry
  }

  /** Store or update a session mapping */
  set(conversationId: string, sdkSessionId: string, model: string, messageCount: number): void {
    const existing = this.sessions.get(conversationId)
    this.sessions.set(conversationId, {
      sdkSessionId,
      createdAt: existing?.createdAt ?? Date.now(),
      lastUsed: Date.now(),
      messageCount,
      model,
      resumeCount: existing?.resumeCount ?? 0,
      failureCount: existing?.failureCount ?? 0,
    })
    this.enforceMaxSessions()
    this.save()
  }

  /** Record a successful resume */
  recordResume(conversationId: string): void {
    const entry = this.sessions.get(conversationId)
    if (entry) {
      entry.resumeCount++
      entry.lastUsed = Date.now()
      this.save()
    }
  }

  /** Record a resume failure */
  recordFailure(conversationId: string): void {
    const entry = this.sessions.get(conversationId)
    if (entry) {
      entry.failureCount++
      entry.lastUsed = Date.now()
      this.save()
    }
  }

  /** Invalidate a session (e.g., after resume failure) */
  invalidate(conversationId: string): void {
    this.sessions.delete(conversationId)
    this.save()
  }

  /** Evict sessions older than TTL */
  cleanup(): { evicted: number; remaining: number } {
    const now = Date.now()
    let evicted = 0
    for (const [id, entry] of this.sessions) {
      if (now - entry.lastUsed > this.ttlMs) {
        this.sessions.delete(id)
        evicted++
      }
    }
    if (evicted > 0) {
      logInfo("session-store.cleanup", { evicted, remaining: this.sessions.size })
      this.save()
    }
    return { evicted, remaining: this.sessions.size }
  }

  /** Get stats for debug endpoint */
  getStats(): SessionStoreStats & { hits: number; misses: number } {
    let totalResumes = 0
    let totalFailures = 0
    for (const entry of this.sessions.values()) {
      totalResumes += entry.resumeCount
      totalFailures += entry.failureCount
    }
    const total = this.hits + this.misses
    return {
      totalSessions: this.sessions.size,
      activeSessions: this.sessions.size,
      totalResumes,
      totalFailures,
      hitRate: total > 0 ? `${((this.hits / total) * 100).toFixed(1)}%` : "N/A",
      hits: this.hits,
      misses: this.misses,
    }
  }

  /** List all sessions (for debug) */
  list(): Array<{ conversationId: string } & SessionEntry> {
    return Array.from(this.sessions.entries()).map(([id, entry]) => ({
      conversationId: id,
      ...entry,
    }))
  }

  private enforceMaxSessions(): void {
    if (this.sessions.size <= MAX_SESSIONS) return
    // Evict oldest by lastUsed
    const sorted = Array.from(this.sessions.entries())
      .sort((a, b) => a[1].lastUsed - b[1].lastUsed)
    const toEvict = sorted.slice(0, this.sessions.size - MAX_SESSIONS)
    for (const [id] of toEvict) {
      this.sessions.delete(id)
    }
    logInfo("session-store.max_eviction", { evicted: toEvict.length, remaining: this.sessions.size })
  }

  private load(): void {
    try {
      const data = readFileSync(STORE_PATH, "utf-8")
      const parsed = JSON.parse(data)
      if (Array.isArray(parsed)) {
        for (const [id, entry] of parsed) {
          this.sessions.set(id, entry)
        }
      }
      logDebug("session-store.loaded", { count: this.sessions.size })
    } catch {
      // No existing store or parse error — start fresh
      logDebug("session-store.init", { path: STORE_PATH })
    }
  }

  private save(): void {
    try {
      const data = JSON.stringify(Array.from(this.sessions.entries()), null, 2)
      writeFileSync(STORE_PATH, data, "utf-8")
    } catch (err) {
      logWarn("session-store.save_failed", { error: err instanceof Error ? err.message : String(err) })
    }
  }
}

export const sessionStore = new SessionStore()
