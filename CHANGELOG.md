# Changelog

## [2.2.0](https://github.com/dylanneve1/claude-proxy/compare/v2.1.0...v2.2.0) (2026-02-22)


### Features

* API key auth, Docker support, extended thinking, error handling ([9e7bd4d](https://github.com/dylanneve1/claude-proxy/commit/9e7bd4d46da7a810d319c26fb9fceaf29a9ef940))
* concurrency control, anthropic-version headers, queue status ([21468c2](https://github.com/dylanneve1/claude-proxy/commit/21468c2a3f1280692995114976ac31b28c9b0e32))
* dual-format /v1/models for Anthropic and OpenAI SDK compat ([b431e58](https://github.com/dylanneve1/claude-proxy/commit/b431e58865bc320ec7f6cd7eccf4f1b6571420ba))
* graceful shutdown, batches stub, test coverage ([8603ef1](https://github.com/dylanneve1/claude-proxy/commit/8603ef1d7c606200c4a6a1ad79df93e00ff7efe0))
* OpenAI tool_call and multi-modal image support ([213cf54](https://github.com/dylanneve1/claude-proxy/commit/213cf54df584789f1e0458187de607b38b2bf045))
* OpenAI-compatible /v1/chat/completions endpoint ([5a73457](https://github.com/dylanneve1/claude-proxy/commit/5a73457d4e34269082453fb42a7d0ee661293e3e))
* real-time token streaming and OpenAI auth forwarding ([37c5809](https://github.com/dylanneve1/claude-proxy/commit/37c580962df3ed2f46fa63a6f527ddc9ab010322))


### Bug Fixes

* agent mode turn buffering, tool_use XML serialization, OpenAI streaming ([7a2d46e](https://github.com/dylanneve1/claude-proxy/commit/7a2d46e0e053a4e7b36dd736538cc6e9628fc0f4))
* default to non-streaming responses for chat app compatibility ([cb13bda](https://github.com/dylanneve1/claude-proxy/commit/cb13bda491c9b840ec894da90594b2951dca68da))

## [2.1.0](https://github.com/dylanneve1/claude-proxy/compare/v2.0.0...v2.1.0) (2026-02-21)


### Features

* add /v1/models/:id, count_tokens endpoint, dynamic version ([e241392](https://github.com/dylanneve1/claude-proxy/commit/e241392c1bccfbb23f568f4e24a99b5b9eea5358))
* CLI flags, request timeout, session isolation, expanded models ([1fa17b3](https://github.com/dylanneve1/claude-proxy/commit/1fa17b3066f98d0e2f03a0dd26519106108558bc))
* full Anthropic API compatibility — tool use, /v1/models, usage stats ([dca51c3](https://github.com/dylanneve1/claude-proxy/commit/dca51c3b67d3fb3fedf3c1e457ffd1425a49aa85))
* request-id headers, settings isolation, more tests ([6d57df0](https://github.com/dylanneve1/claude-proxy/commit/6d57df0e42cc85007d679ed9c395754603d5b979))


### Bug Fixes

* minimal SEND_MESSAGE_NOTE focused on core workflow constraint ([db48ff2](https://github.com/dylanneve1/claude-proxy/commit/db48ff2d94f6b55365a541d73be8a40fb264d2a0))
* rename tool to 'message', add filePath/path/media aliases, align with openclaw ([27931b9](https://github.com/dylanneve1/claude-proxy/commit/27931b9cd03b3fba38349faba9e773bf7fe39126))
* rewrite SEND_MESSAGE_NOTE to enforce agentic tool-first workflow ([2b01c64](https://github.com/dylanneve1/claude-proxy/commit/2b01c64e2212f5e74a75bdb910bf4ac4a5d5f704))
* streaming timeout, maxTurns exhaustion, tool result overflow, image sends ([70892b6](https://github.com/dylanneve1/claude-proxy/commit/70892b644629cadcdb26366ac30a22e5da8111b9))
* suppress tool narration in responses, add input validation ([85c6a98](https://github.com/dylanneve1/claude-proxy/commit/85c6a9898a463ca79151dfb65ec628bd07c8b412))

## [1.0.2](https://github.com/rynfar/opencode-claude-max-proxy/compare/v1.0.1...v1.0.2) (2026-01-26)


### Bug Fixes

* remove bun install from publish job ([966b2ea](https://github.com/rynfar/opencode-claude-max-proxy/commit/966b2ea8a06f4dc12dd4f0f19be94b3539b83dfd))
* remove bun install from publish job ([cd36411](https://github.com/rynfar/opencode-claude-max-proxy/commit/cd36411193af22e779638232427dd8c49f8926e0))

## [1.0.1](https://github.com/rynfar/opencode-claude-max-proxy/compare/v1.0.0...v1.0.1) (2026-01-26)


### Bug Fixes

* move npm publish into release-please workflow ([82db07c](https://github.com/rynfar/opencode-claude-max-proxy/commit/82db07c07bf87bfc69ae08cc8f24c007408ad3ed))
* move npm publish into release-please workflow ([f7c4b2c](https://github.com/rynfar/opencode-claude-max-proxy/commit/f7c4b2c08a6993d20239e63b9fb668017577ab32))

## 1.0.0 (2026-01-26)


### Features

* Claude Max proxy for OpenCode ([b9df612](https://github.com/rynfar/opencode-claude-max-proxy/commit/b9df6121564b90b3dbbf821f981d67851d7a4e1e))


### Bug Fixes

* add SSE heartbeat to prevent connection resets ([194fd51](https://github.com/rynfar/opencode-claude-max-proxy/commit/194fd51e2fdf375cbac06fbfcf634800adab5d72))
* add SSE heartbeat to prevent connection resets ([ec7120d](https://github.com/rynfar/opencode-claude-max-proxy/commit/ec7120d22eef490e146530e5d66c1d90b055d0b5)), closes [#1](https://github.com/rynfar/opencode-claude-max-proxy/issues/1)
