import { describe, expect, test } from "bun:test"
import { SessionCompaction } from "../../src/session/compaction"
import { Token } from "../../src/util/token"
import { Instance } from "../../src/project/instance"
import { Log } from "../../src/util/log"
import { tmpdir } from "../fixture/fixture"
import type { Provider } from "../../src/provider/provider"
import type { MessageV2 } from "../../src/session/message-v2"

Log.init({ print: false })

function createModel(opts: {
  context: number
  output: number
  input?: number
  npm?: string
}): Provider.Model {
  return {
    id: "test-model",
    providerID: "test",
    name: "Test",
    limit: {
      context: opts.context,
      input: opts.input,
      output: opts.output,
    },
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    capabilities: {
      toolcall: true,
      attachment: false,
      reasoning: false,
      temperature: true,
      input: { text: true, image: false, audio: false, video: false },
      output: { text: true, image: false, audio: false, video: false },
    },
    api: { npm: opts.npm ?? "@ai-sdk/anthropic" },
    options: {},
  } as Provider.Model
}

describe("rolling window integration", () => {
  describe("constants", () => {
    test("ROLLING_THRESHOLD has correct value", () => {
      expect(SessionCompaction.ROLLING_THRESHOLD).toBe(0.85)
    })
    test("ROLLING_TARGET has correct value", () => {
      expect(SessionCompaction.ROLLING_TARGET).toBe(0.7)
    })
  })

  describe("estimateTurnTokens", () => {
    function makeMsgWithText(id: string, text: string, role: "user" | "assistant"): MessageV2.WithParts {
      if (role === "user") {
        return {
          info: {
            role: "user",
            id,
            sessionID: "s1",
            time: { created: 1 },
            agent: "test",
            model: { providerID: "t", modelID: "t" },
          } as any,
          parts: [{ type: "text", text, id: "p_" + id, sessionID: "s1", messageID: id } as any],
        }
      }
      return {
        info: {
          role: "assistant",
          id,
          sessionID: "s1",
          parentID: "u0",
          time: { created: 2 },
          mode: "test",
          agent: "test",
          path: { cwd: "/", root: "/" },
          cost: 0,
          tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
          modelID: "t",
          providerID: "t",
        } as any,
        parts: [{ type: "text", text, id: "p_" + id, sessionID: "s1", messageID: id } as any],
      }
    }

    test("estimates tokens proportional to text length", () => {
      const text = "a".repeat(1000)
      const msgs = [makeMsgWithText("u1", text, "user")]
      const result = SessionCompaction.estimateTurnTokens(msgs, 0, 1)
      expect(result).toBe(Token.estimate(text))
      expect(result).toBe(250) // 1000/4
    })

    test("combines multiple messages in range", () => {
      const msgs = [
        makeMsgWithText("u1", "a".repeat(400), "user"),
        makeMsgWithText("a1", "b".repeat(400), "assistant"),
        makeMsgWithText("u2", "c".repeat(400), "user"),
      ]
      const result = SessionCompaction.estimateTurnTokens(msgs, 0, 2)
      // Two texts joined with newline: 400 + 1 + 400 = 801 chars → 201 tokens
      expect(result).toBe(Token.estimate("a".repeat(400) + "\n" + "b".repeat(400)))
    })

    test("returns 0 for empty range", () => {
      expect(SessionCompaction.estimateTurnTokens([], 0, 0)).toBe(0)
    })
  })

  describe("shouldRollingCompact + estimateTurnTokens combined", () => {
    test("large message set triggers compaction", async () => {
      await using tmp = await tmpdir()
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const model = createModel({ context: 200_000, output: 32_000 })
          // usable = 168000, triggerAt = 142800
          // Simulate counting tokens for a large conversation
          const decision = await SessionCompaction.shouldRollingCompact({
            tokenCount: 150_000,
            model,
          })
          expect(decision.needed).toBe(true)
          if (decision.needed) {
            expect(decision.targetTokens).toBeCloseTo(117_600, -1)
          }
        },
      })
    })

    test("small conversation does not trigger compaction", async () => {
      await using tmp = await tmpdir()
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const model = createModel({ context: 200_000, output: 32_000 })
          const decision = await SessionCompaction.shouldRollingCompact({
            tokenCount: 50_000,
            model,
          })
          expect(decision.needed).toBe(false)
        },
      })
    })
  })

  describe("config mode switching", () => {
    test("mode full disables rolling compaction", async () => {
      await using tmp = await tmpdir({ config: { compaction: { mode: "full" } } as any })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const model = createModel({ context: 200_000, output: 32_000 })
          const decision = await SessionCompaction.shouldRollingCompact({
            tokenCount: 999_999,
            model,
          })
          expect(decision.needed).toBe(false)
        },
      })
    })

    test("default mode (rolling) enables rolling compaction", async () => {
      await using tmp = await tmpdir()
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const model = createModel({ context: 200_000, output: 32_000 })
          const decision = await SessionCompaction.shouldRollingCompact({
            tokenCount: 150_000,
            model,
          })
          expect(decision.needed).toBe(true)
        },
      })
    })
  })

  describe("getOverflowStrategy", () => {
    test("no overflow returns none", async () => {
      await using tmp = await tmpdir()
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const model = createModel({ context: 200_000, output: 32_000 })
          const tokens = { input: 50_000, output: 5_000, reasoning: 0, cache: { read: 0, write: 0 } }
          expect(await SessionCompaction.getOverflowStrategy({ tokens, model })).toBe("none")
        },
      })
    })

    test("overflow with rolling mode returns rolling", async () => {
      await using tmp = await tmpdir()
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const model = createModel({ context: 100_000, output: 32_000 })
          const tokens = { input: 60_000, output: 10_000, reasoning: 0, cache: { read: 0, write: 0 } }
          expect(await SessionCompaction.getOverflowStrategy({ tokens, model })).toBe("rolling")
        },
      })
    })

    test("emergency fallback returns full", async () => {
      await using tmp = await tmpdir()
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const model = createModel({ context: 100_000, output: 32_000 })
          const tokens = { input: 60_000, output: 10_000, reasoning: 0, cache: { read: 0, write: 0 } }
          expect(await SessionCompaction.getOverflowStrategy({ tokens, model, rollingAttempted: true })).toBe("full")
        },
      })
    })
  })
})
