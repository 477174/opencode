import { describe, expect, test } from "bun:test"
import path from "path"
import { SessionCompaction } from "../../src/session/compaction"
import { Token } from "../../src/util/token"
import { Instance } from "../../src/project/instance"
import { Log } from "../../src/util/log"
import { tmpdir } from "../fixture/fixture"
import { Session } from "../../src/session"
import { Identifier } from "../../src/id/id"
import type { Provider } from "../../src/provider/provider"
import type { MessageV2 } from "../../src/session/message-v2"

Log.init({ print: false })

function createModel(opts: {
  context: number
  output: number
  input?: number
  cost?: Provider.Model["cost"]
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
    cost: opts.cost ?? { input: 0, output: 0, cache: { read: 0, write: 0 } },
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

describe("session.compaction.isOverflow", () => {
  test("returns true when token count exceeds usable context", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 100_000, output: 32_000 })
        const tokens = { input: 75_000, output: 5_000, reasoning: 0, cache: { read: 0, write: 0 } }
        expect(await SessionCompaction.isOverflow({ tokens, model })).toBe(true)
      },
    })
  })

  test("returns false when token count within usable context", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, output: 32_000 })
        const tokens = { input: 100_000, output: 10_000, reasoning: 0, cache: { read: 0, write: 0 } }
        expect(await SessionCompaction.isOverflow({ tokens, model })).toBe(false)
      },
    })
  })

  test("includes cache.read in token count", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 100_000, output: 32_000 })
        const tokens = { input: 60_000, output: 10_000, reasoning: 0, cache: { read: 10_000, write: 0 } }
        expect(await SessionCompaction.isOverflow({ tokens, model })).toBe(true)
      },
    })
  })

  test("respects input limit for input caps", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 400_000, input: 272_000, output: 128_000 })
        const tokens = { input: 271_000, output: 1_000, reasoning: 0, cache: { read: 2_000, write: 0 } }
        expect(await SessionCompaction.isOverflow({ tokens, model })).toBe(true)
      },
    })
  })

  test("returns false when input/output are within input caps", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 400_000, input: 272_000, output: 128_000 })
        const tokens = { input: 200_000, output: 20_000, reasoning: 0, cache: { read: 10_000, write: 0 } }
        expect(await SessionCompaction.isOverflow({ tokens, model })).toBe(false)
      },
    })
  })

  test("returns false when output within limit with input caps", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, input: 120_000, output: 10_000 })
        const tokens = { input: 50_000, output: 9_999, reasoning: 0, cache: { read: 0, write: 0 } }
        expect(await SessionCompaction.isOverflow({ tokens, model })).toBe(false)
      },
    })
  })

  // ─── Bug reproduction tests ───────────────────────────────────────────
  // These tests demonstrate that when limit.input is set, isOverflow()
  // does not subtract any headroom for the next model response. This means
  // compaction only triggers AFTER we've already consumed the full input
  // budget, leaving zero room for the next API call's output tokens.
  //
  // Compare: without limit.input, usable = context - output (reserves space).
  // With limit.input, usable = limit.input (reserves nothing).
  //
  // Related issues: #10634, #8089, #11086, #12621
  // Open PRs: #6875, #12924

  test("BUG: no headroom when limit.input is set — compaction should trigger near boundary but does not", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        // Simulate Claude with prompt caching: input limit = 200K, output limit = 32K
        const model = createModel({ context: 200_000, input: 200_000, output: 32_000 })

        // We've used 198K tokens total. Only 2K under the input limit.
        // On the next turn, the full conversation (198K) becomes input,
        // plus the model needs room to generate output — this WILL overflow.
        const tokens = { input: 180_000, output: 15_000, reasoning: 0, cache: { read: 3_000, write: 0 } }
        // count = 180K + 3K + 15K = 198K
        // usable = limit.input = 200K (no output subtracted!)
        // 198K > 200K = false → no compaction triggered

        // WITHOUT limit.input: usable = 200K - 32K = 168K, and 198K > 168K = true ✓
        // WITH limit.input: usable = 200K, and 198K > 200K = false ✗

        // With 198K used and only 2K headroom, the next turn will overflow.
        // Compaction MUST trigger here.
        expect(await SessionCompaction.isOverflow({ tokens, model })).toBe(true)
      },
    })
  })

  test("BUG: without limit.input, same token count correctly triggers compaction", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        // Same model but without limit.input — uses context - output instead
        const model = createModel({ context: 200_000, output: 32_000 })

        // Same token usage as above
        const tokens = { input: 180_000, output: 15_000, reasoning: 0, cache: { read: 3_000, write: 0 } }
        // count = 198K
        // usable = context - output = 200K - 32K = 168K
        // 198K > 168K = true → compaction correctly triggered

        const result = await SessionCompaction.isOverflow({ tokens, model })
        expect(result).toBe(true) // ← Correct: headroom is reserved
      },
    })
  })

  test("BUG: asymmetry — limit.input model allows 30K more usage before compaction than equivalent model without it", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        // Two models with identical context/output limits, differing only in limit.input
        const withInputLimit = createModel({ context: 200_000, input: 200_000, output: 32_000 })
        const withoutInputLimit = createModel({ context: 200_000, output: 32_000 })

        // 170K total tokens — well above context-output (168K) but below input limit (200K)
        const tokens = { input: 166_000, output: 10_000, reasoning: 0, cache: { read: 5_000, write: 0 } }

        const withLimit = await SessionCompaction.isOverflow({ tokens, model: withInputLimit })
        const withoutLimit = await SessionCompaction.isOverflow({ tokens, model: withoutInputLimit })

        // Both models have identical real capacity — they should agree:
        expect(withLimit).toBe(true) // should compact (170K leaves no room for 32K output)
        expect(withoutLimit).toBe(true) // correctly compacts (170K > 168K)
      },
    })
  })

  test("returns false when model context limit is 0", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 0, output: 32_000 })
        const tokens = { input: 100_000, output: 10_000, reasoning: 0, cache: { read: 0, write: 0 } }
        expect(await SessionCompaction.isOverflow({ tokens, model })).toBe(false)
      },
    })
  })

  test("returns false when compaction.auto is disabled", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "opencode.json"),
          JSON.stringify({
            compaction: { auto: false },
          }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 100_000, output: 32_000 })
        const tokens = { input: 75_000, output: 5_000, reasoning: 0, cache: { read: 0, write: 0 } }
        expect(await SessionCompaction.isOverflow({ tokens, model })).toBe(false)
      },
    })
  })
})

describe("util.token.estimate", () => {
  test("estimates tokens from text (4 chars per token)", () => {
    const text = "x".repeat(4000)
    expect(Token.estimate(text)).toBe(1000)
  })

  test("estimates tokens from larger text", () => {
    const text = "y".repeat(20_000)
    expect(Token.estimate(text)).toBe(5000)
  })

  test("returns 0 for empty string", () => {
    expect(Token.estimate("")).toBe(0)
  })
})

describe("session.getUsage", () => {
  test("normalizes standard usage to token format", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = Session.getUsage({
      model,
      usage: {
        inputTokens: 1000,
        outputTokens: 500,
        totalTokens: 1500,
      },
    })

    expect(result.tokens.input).toBe(1000)
    expect(result.tokens.output).toBe(500)
    expect(result.tokens.reasoning).toBe(0)
    expect(result.tokens.cache.read).toBe(0)
    expect(result.tokens.cache.write).toBe(0)
  })

  test("extracts cached tokens to cache.read", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = Session.getUsage({
      model,
      usage: {
        inputTokens: 1000,
        outputTokens: 500,
        totalTokens: 1500,
        cachedInputTokens: 200,
      },
    })

    expect(result.tokens.input).toBe(800)
    expect(result.tokens.cache.read).toBe(200)
  })

  test("handles anthropic cache write metadata", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = Session.getUsage({
      model,
      usage: {
        inputTokens: 1000,
        outputTokens: 500,
        totalTokens: 1500,
      },
      metadata: {
        anthropic: {
          cacheCreationInputTokens: 300,
        },
      },
    })

    expect(result.tokens.cache.write).toBe(300)
  })

  test("does not subtract cached tokens for anthropic provider", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = Session.getUsage({
      model,
      usage: {
        inputTokens: 1000,
        outputTokens: 500,
        totalTokens: 1500,
        cachedInputTokens: 200,
      },
      metadata: {
        anthropic: {},
      },
    })

    expect(result.tokens.input).toBe(1000)
    expect(result.tokens.cache.read).toBe(200)
  })

  test("handles reasoning tokens", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = Session.getUsage({
      model,
      usage: {
        inputTokens: 1000,
        outputTokens: 500,
        totalTokens: 1500,
        reasoningTokens: 100,
      },
    })

    expect(result.tokens.reasoning).toBe(100)
  })

  test("handles undefined optional values gracefully", () => {
    const model = createModel({ context: 100_000, output: 32_000 })
    const result = Session.getUsage({
      model,
      usage: {
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
      },
    })

    expect(result.tokens.input).toBe(0)
    expect(result.tokens.output).toBe(0)
    expect(result.tokens.reasoning).toBe(0)
    expect(result.tokens.cache.read).toBe(0)
    expect(result.tokens.cache.write).toBe(0)
    expect(Number.isNaN(result.cost)).toBe(false)
  })

  test("calculates cost correctly", () => {
    const model = createModel({
      context: 100_000,
      output: 32_000,
      cost: {
        input: 3,
        output: 15,
        cache: { read: 0.3, write: 3.75 },
      },
    })
    const result = Session.getUsage({
      model,
      usage: {
        inputTokens: 1_000_000,
        outputTokens: 100_000,
        totalTokens: 1_100_000,
      },
    })

    expect(result.cost).toBe(3 + 1.5)
  })

  test.each(["@ai-sdk/anthropic", "@ai-sdk/amazon-bedrock", "@ai-sdk/google-vertex/anthropic"])(
    "computes total from components for %s models",
    (npm) => {
      const model = createModel({ context: 100_000, output: 32_000, npm })
      const usage = {
        inputTokens: 1000,
        outputTokens: 500,
        // These providers typically report total as input + output only,
        // excluding cache read/write.
        totalTokens: 1500,
        cachedInputTokens: 200,
      }
      if (npm === "@ai-sdk/amazon-bedrock") {
        const result = Session.getUsage({
          model,
          usage,
          metadata: {
            bedrock: {
              usage: {
                cacheWriteInputTokens: 300,
              },
            },
          },
        })

        expect(result.tokens.input).toBe(1000)
        expect(result.tokens.cache.read).toBe(200)
        expect(result.tokens.cache.write).toBe(300)
        expect(result.tokens.total).toBe(2000)
        return
      }

      const result = Session.getUsage({
        model,
        usage,
        metadata: {
          anthropic: {
            cacheCreationInputTokens: 300,
          },
        },
      })

      expect(result.tokens.input).toBe(1000)
      expect(result.tokens.cache.read).toBe(200)
      expect(result.tokens.cache.write).toBe(300)
      expect(result.tokens.total).toBe(2000)
    },
  )
})

// ─── Rolling compaction tests ─────────────────────────────────────────────────

describe("session.compaction.shouldRollingCompact", () => {
  // Model: context=200000, output=32000 (no input limit)
  // maxOutputTokens = min(32000, 32000) = 32000
  // usable = 200000 - 32000 = 168000
  // Default threshold=0.85 → triggerAt = 168000 * 0.85 = 142800
  // Default target=0.70 → targetTokens = 168000 * 0.70 = 117600

  test("returns needed:false when below threshold", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, output: 32_000 })
        const result = await SessionCompaction.shouldRollingCompact({ tokenCount: 140_000, model })
        expect(result).toEqual({ needed: false })
      },
    })
  })

  test("returns needed:true with token details when above threshold", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, output: 32_000 })
        const result = await SessionCompaction.shouldRollingCompact({ tokenCount: 150_000, model })
        expect(result.needed).toBe(true)
        if (result.needed) {
          expect(result.currentTokens).toBe(150_000)
          expect(result.targetTokens).toBeCloseTo(117_600, 0)
        }
      },
    })
  })

  test("returns needed:false when mode is full", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "opencode.json"),
          JSON.stringify({ compaction: { mode: "full" } }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, output: 32_000 })
        const result = await SessionCompaction.shouldRollingCompact({ tokenCount: 150_000, model })
        expect(result).toEqual({ needed: false })
      },
    })
  })

  test("returns needed:false when auto is disabled", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "opencode.json"),
          JSON.stringify({ compaction: { auto: false } }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, output: 32_000 })
        const result = await SessionCompaction.shouldRollingCompact({ tokenCount: 150_000, model })
        expect(result).toEqual({ needed: false })
      },
    })
  })

  test("returns needed:false when context is 0", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 0, output: 32_000 })
        const result = await SessionCompaction.shouldRollingCompact({ tokenCount: 150_000, model })
        expect(result).toEqual({ needed: false })
      },
    })
  })

  test("uses custom threshold and target from config", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "opencode.json"),
          JSON.stringify({ compaction: { threshold: 0.9, target: 0.6 } }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, output: 32_000 })
        // usable = 168000, triggerAt = 168000*0.90 = 151200, target = 168000*0.60 = 100800
        // 150000 < 151200 → not needed
        const result = await SessionCompaction.shouldRollingCompact({ tokenCount: 150_000, model })
        expect(result).toEqual({ needed: false })
      },
    })
  })

  test("falls back to defaults when threshold <= target (invalid config)", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "opencode.json"),
          JSON.stringify({ compaction: { threshold: 0.5, target: 0.8 } }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, output: 32_000 })
        // Invalid → uses defaults: threshold=0.85, target=0.70
        // triggerAt = 142800, targetTokens = 117600
        // 150000 >= 142800 → needed
        const result = await SessionCompaction.shouldRollingCompact({ tokenCount: 150_000, model })
        expect(result.needed).toBe(true)
        if (result.needed) {
          expect(result.currentTokens).toBe(150_000)
          expect(result.targetTokens).toBeCloseTo(117_600, 0)
        }
      },
    })
  })
})

describe("session.compaction.estimateTurnTokens", () => {
  function textMsg(id: string, text: string): MessageV2.WithParts {
    return {
      info: {
        id,
        sessionID: "test",
        role: "user",
        time: { created: 0 },
        agent: "test",
        model: { providerID: "test", modelID: "test" },
      } as MessageV2.User,
      parts: [
        {
          id: `${id}-p`,
          sessionID: "test",
          messageID: id,
          type: "text",
          text,
        } as MessageV2.TextPart,
      ],
    }
  }

  test("estimates tokens for a slice of messages", () => {
    const messages = [textMsg("m1", "x".repeat(400)), textMsg("m2", "y".repeat(400))]
    // serializeTurn joins: "x"*400 + "\n" + "y"*400 = 801 chars
    // Token.estimate uses Math.round(801/4) = Math.round(200.25) = 200
    expect(SessionCompaction.estimateTurnTokens(messages, 0, 2)).toBe(200)
  })

  test("returns 0 for empty slice", () => {
    expect(SessionCompaction.estimateTurnTokens([], 0, 0)).toBe(0)
  })

  test("estimates tokens for a partial slice", () => {
    const messages = [textMsg("m1", "a".repeat(100)), textMsg("m2", "b".repeat(200))]
    // Only second message: "b"*200 = 200 chars → ceil(200/4) = 50
    expect(SessionCompaction.estimateTurnTokens(messages, 1, 2)).toBe(50)
  })
})

describe("session.compaction.getOverflowStrategy", () => {
  // Model: context=100000, output=32000
  // usable = 100000 - 32000 = 68000
  // Overflow tokens: input=60000 + output=10000 = 70000 >= 68000
  // Safe tokens: input=50000 + output=10000 = 60000 < 68000

  test("returns 'none' when no overflow", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 100_000, output: 32_000 })
        const tokens = { input: 50_000, output: 10_000, reasoning: 0, cache: { read: 0, write: 0 } }
        const result = await SessionCompaction.getOverflowStrategy({ tokens, model })
        expect(result).toBe("none")
      },
    })
  })

  test("returns 'rolling' when overflow with default config", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 100_000, output: 32_000 })
        const tokens = { input: 60_000, output: 10_000, reasoning: 0, cache: { read: 0, write: 0 } }
        const result = await SessionCompaction.getOverflowStrategy({ tokens, model })
        expect(result).toBe("rolling")
      },
    })
  })

  test("returns 'full' when overflow with mode=full config", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "opencode.json"),
          JSON.stringify({ compaction: { mode: "full" } }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 100_000, output: 32_000 })
        const tokens = { input: 60_000, output: 10_000, reasoning: 0, cache: { read: 0, write: 0 } }
        const result = await SessionCompaction.getOverflowStrategy({ tokens, model })
        expect(result).toBe("full")
      },
    })
  })

  test("returns 'full' when overflow and rollingAttempted is true (emergency fallback)", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 100_000, output: 32_000 })
        const tokens = { input: 60_000, output: 10_000, reasoning: 0, cache: { read: 0, write: 0 } }
        const result = await SessionCompaction.getOverflowStrategy({ tokens, model, rollingAttempted: true })
        expect(result).toBe("full")
      },
    })
  })
})

describe("session.compaction.constants", () => {
  test("ROLLING_THRESHOLD is 0.85", () => {
    expect(SessionCompaction.ROLLING_THRESHOLD).toBe(0.85)
  })
  test("ROLLING_TARGET is 0.70", () => {
    expect(SessionCompaction.ROLLING_TARGET).toBe(0.70)
  })
})

describe("session.compaction.getRollingSummary", () => {
  test("returns null when no rolling summary exists", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const session = await Session.create({})
        const result = await SessionCompaction.getRollingSummary(session.id)
        expect(result).toBeNull()
        await Session.remove(session.id)
      },
    })
  })

  test("returns summary text when rolling summary exists", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const session = await Session.create({})
        // Create a user message first (needed as parent)
        const userMsg = await Session.updateMessage({
          id: Identifier.ascending("message"),
          role: "user",
          sessionID: session.id,
          time: { created: Date.now() },
          agent: "test",
          model: { providerID: "test", modelID: "test" },
        })
        // Create rolling summary assistant message
        const summaryMsg = await Session.updateMessage({
          id: Identifier.ascending("message"),
          role: "assistant",
          parentID: userMsg.id,
          sessionID: session.id,
          mode: "rolling-compaction",
          agent: "rolling-compaction",
          summary: true,
          rolling: true,
          path: { cwd: tmp.path, root: tmp.path },
          cost: 0,
          tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
          modelID: "test",
          providerID: "test",
          time: { created: Date.now() },
        })
        // Add text part to the summary
        await Session.updatePart({
          id: Identifier.ascending("part"),
          messageID: summaryMsg.id,
          sessionID: session.id,
          type: "text",
          text: "This is the rolling summary content.",
          time: { start: Date.now(), end: Date.now() },
        })

        const result = await SessionCompaction.getRollingSummary(session.id)
        expect(result).toBe("This is the rolling summary content.")

        await Session.remove(session.id)
      },
    })
  })
})

describe("session.compaction.rollingCompact", () => {
  test("returns compacted:false when no eligible turns", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, output: 32_000 })
        const abort = new AbortController().signal
        const result = await SessionCompaction.rollingCompact({
          sessionID: "ses_nonexistent",
          messages: [],
          currentTokens: 150_000,
          targetTokens: 100_000,
          model,
          abort,
        })
        expect(result.compacted).toBe(false)
        if (!result.compacted) {
          expect(result.reason).toContain("no eligible turns")
        }
      },
    })
  })

  test("returns compacted:false when all turns in protection window", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, output: 32_000 })
        const abort = new AbortController().signal
        // Create 2 turn pairs — both will be in the protection window (last 2)
        const msgs: MessageV2.WithParts[] = [
          {
            info: { role: "user", id: "u1", sessionID: "ses_s1", time: { created: 1 }, agent: "test", model: { providerID: "t", modelID: "t" } } as MessageV2.User,
            parts: [{ type: "text", text: "hello", id: "p1", sessionID: "ses_s1", messageID: "u1" } as any],
          },
          {
            info: { role: "assistant", id: "a1", sessionID: "ses_s1", parentID: "u1", time: { created: 2 }, mode: "test", agent: "test", path: { cwd: "/", root: "/" }, cost: 0, tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } }, modelID: "t", providerID: "t", finish: "stop" } as MessageV2.Assistant,
            parts: [{ type: "text", text: "response", id: "p2", sessionID: "ses_s1", messageID: "a1" } as any],
          },
          {
            info: { role: "user", id: "u2", sessionID: "ses_s1", time: { created: 3 }, agent: "test", model: { providerID: "t", modelID: "t" } } as MessageV2.User,
            parts: [{ type: "text", text: "hello2", id: "p3", sessionID: "ses_s1", messageID: "u2" } as any],
          },
          {
            info: { role: "assistant", id: "a2", sessionID: "ses_s1", parentID: "u2", time: { created: 4 }, mode: "test", agent: "test", path: { cwd: "/", root: "/" }, cost: 0, tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } }, modelID: "t", providerID: "t", finish: "stop" } as MessageV2.Assistant,
            parts: [{ type: "text", text: "response2", id: "p4", sessionID: "ses_s1", messageID: "a2" } as any],
          },
        ]
        const result = await SessionCompaction.rollingCompact({
          sessionID: "ses_s1",
          messages: msgs,
          currentTokens: 150_000,
          targetTokens: 100_000,
          model,
          abort,
        })
        expect(result.compacted).toBe(false)
        if (!result.compacted) {
          expect(result.reason).toContain("protection window")
        }
      },
    })
  })

  test("skips turns where assistant has no finish status", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const model = createModel({ context: 200_000, output: 32_000 })
        const abort = new AbortController().signal
        // 3 turn pairs: first has no finish (incomplete), last 2 in protection
        const msgs: MessageV2.WithParts[] = [
          {
            info: { role: "user", id: "u0", sessionID: "ses_s1", time: { created: 0 }, agent: "test", model: { providerID: "t", modelID: "t" } } as MessageV2.User,
            parts: [{ type: "text", text: "hello0", id: "p0", sessionID: "ses_s1", messageID: "u0" } as any],
          },
          {
            info: { role: "assistant", id: "a0", sessionID: "ses_s1", parentID: "u0", time: { created: 1 }, mode: "test", agent: "test", path: { cwd: "/", root: "/" }, cost: 0, tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } }, modelID: "t", providerID: "t" /* NO finish */ } as MessageV2.Assistant,
            parts: [{ type: "text", text: "response0", id: "p1", sessionID: "ses_s1", messageID: "a0" } as any],
          },
          {
            info: { role: "user", id: "u1", sessionID: "ses_s1", time: { created: 2 }, agent: "test", model: { providerID: "t", modelID: "t" } } as MessageV2.User,
            parts: [{ type: "text", text: "hello1", id: "p2", sessionID: "ses_s1", messageID: "u1" } as any],
          },
          {
            info: { role: "assistant", id: "a1", sessionID: "ses_s1", parentID: "u1", time: { created: 3 }, mode: "test", agent: "test", path: { cwd: "/", root: "/" }, cost: 0, tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } }, modelID: "t", providerID: "t", finish: "stop" } as MessageV2.Assistant,
            parts: [{ type: "text", text: "response1", id: "p3", sessionID: "ses_s1", messageID: "a1" } as any],
          },
          {
            info: { role: "user", id: "u2", sessionID: "ses_s1", time: { created: 4 }, agent: "test", model: { providerID: "t", modelID: "t" } } as MessageV2.User,
            parts: [{ type: "text", text: "hello2", id: "p4", sessionID: "ses_s1", messageID: "u2" } as any],
          },
          {
            info: { role: "assistant", id: "a2", sessionID: "ses_s1", parentID: "u2", time: { created: 5 }, mode: "test", agent: "test", path: { cwd: "/", root: "/" }, cost: 0, tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } }, modelID: "t", providerID: "t", finish: "stop" } as MessageV2.Assistant,
            parts: [{ type: "text", text: "response2", id: "p5", sessionID: "ses_s1", messageID: "a2" } as any],
          },
        ]
        const result = await SessionCompaction.rollingCompact({
          sessionID: "ses_s1",
          messages: msgs,
          currentTokens: 150_000,
          targetTokens: 100_000,
          model,
          abort,
        })
        // Turn u0/a0 skipped (no finish), turns u1/a1 and u2/a2 in protection
        expect(result.compacted).toBe(false)
      },
    })
  })
})
