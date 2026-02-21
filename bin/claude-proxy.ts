#!/usr/bin/env bun

import { startProxyServer } from "../src/proxy/server"

const args = process.argv.slice(2)

if (args.includes("--help") || args.includes("-h")) {
  console.log(`claude-proxy — Anthropic Messages API proxy backed by Claude Agent SDK

Usage: claude-proxy [options]

Options:
  -p, --port <port>   Listen port (default: 3456, env: CLAUDE_PROXY_PORT)
  -H, --host <host>   Bind address (default: 127.0.0.1, env: CLAUDE_PROXY_HOST)
  -d, --debug         Enable debug logging (env: OPENCODE_CLAUDE_PROVIDER_DEBUG=1)
  -v, --version       Show version
  -h, --help          Show this help

Examples:
  claude-proxy                       # Start on 127.0.0.1:3456
  claude-proxy -p 8080               # Start on port 8080
  claude-proxy -H 0.0.0.0 -p 3456   # Listen on all interfaces`)
  process.exit(0)
}

if (args.includes("--version") || args.includes("-v")) {
  const { readFileSync } = await import("fs")
  const { join, dirname } = await import("path")
  const { fileURLToPath } = await import("url")
  try {
    const pkg = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), "../package.json"), "utf-8"))
    console.log(`claude-proxy v${pkg.version}`)
  } catch {
    console.log("claude-proxy (unknown version)")
  }
  process.exit(0)
}

function getArg(flags: string[]): string | undefined {
  for (const flag of flags) {
    const idx = args.indexOf(flag)
    if (idx !== -1 && idx + 1 < args.length) return args[idx + 1]
  }
  return undefined
}

if (args.includes("--debug") || args.includes("-d")) {
  process.env.OPENCODE_CLAUDE_PROVIDER_DEBUG = "1"
}

const port = parseInt(getArg(["-p", "--port"]) ?? process.env.CLAUDE_PROXY_PORT ?? "3456", 10)
const host = getArg(["-H", "--host"]) ?? process.env.CLAUDE_PROXY_HOST ?? "127.0.0.1"

await startProxyServer({ port, host })
