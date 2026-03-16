import { describe, expect, test, beforeEach, afterEach } from "bun:test"
import { TokenCounter } from "../../src/session/token-counter"
import { Token } from "../../src/util/token"
import { Log } from "../../src/util/log"
import type { Provider } from "../../src/provider/provider"
import type { ModelMessage } from "ai"

Log.init({ print: false })

function createModel(opts: {
  npm: string
  context?: number
  id?: string
}): Provider.Model {
  return {
    id: opts.id ?? "test-model",
    providerID: "test",
    name: "Test",
    limit: {
      context: opts.context ?? 200_000,
      input: undefined,
      output: 32_000,
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
    api: { id: opts.id ?? "test-model", npm: opts.npm },
    options: {},
  } as Provider.Model
}

describe("session.token-counter", () => {
  describe("estimation fallback (non-Anthropic)", () => {
    const model = createModel({ npm: "@ai-sdk/openai" })

    test("text message ~1000 chars → ~250 tokens (chars/4)", async () => {
      const text = "a".repeat(1000)
      const result = await TokenCounter.count({
        model,
        system: [],
        messages: [{ role: "user", content: text } as ModelMessage],
      })
      expect(result).toBe(250)
    })

    test("empty messages → 0", async () => {
      const result = await TokenCounter.count({
        model,
        system: [],
        messages: [],
      })
      expect(result).toBe(0)
    })

    test("tool parts → included in estimation", async () => {
      const messages: ModelMessage[] = [
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "tc1",
              toolName: "read_file",
              input: { path: "/test.ts" },
            },
          ],
        } as ModelMessage,
        {
           role: "tool",
           content: [
             {
               type: "tool-result",
               toolCallId: "tc1",
               toolName: "read_file",
               output: "file contents",
             },
           ],
         } as unknown as ModelMessage,
      ]
      const result = await TokenCounter.count({
        model,
        system: [],
        messages,
      })
      expect(result).toBeGreaterThan(0)
      expect(result).toBeGreaterThanOrEqual(Token.estimate("read_file"))
    })

    test("system messages → included in estimation", async () => {
      const sys = "You are a helpful assistant."
      const msg = "hello"
      const result = await TokenCounter.count({
        model,
        system: [{ role: "system", content: sys } as ModelMessage],
        messages: [{ role: "user", content: msg } as ModelMessage],
      })
      expect(result).toBe(Token.estimate(sys) + Token.estimate(msg))
    })

    test("multiple messages → newline-joined then estimated", async () => {
      const result = await TokenCounter.count({
        model,
        system: [],
        messages: [
          { role: "user", content: "hello" } as ModelMessage,
          { role: "assistant", content: "world" } as ModelMessage,
        ],
      })
      expect(result).toBe(Token.estimate("hello\nworld"))
    })
  })

  describe("toAnthropicMessages", () => {
    test("text user + assistant → correct Anthropic format", () => {
      const result = TokenCounter.toAnthropicMessages([
        { role: "user", content: "Hello" } as ModelMessage,
        { role: "assistant", content: "Hi there" } as ModelMessage,
      ])
      expect(result).toEqual([
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there" },
      ])
    })

    test("user array content → text blocks", () => {
      const result = TokenCounter.toAnthropicMessages([
        {
          role: "user",
          content: [{ type: "text", text: "Hello" }],
        } as ModelMessage,
      ])
      expect(result).toEqual([
        { role: "user", content: [{ type: "text", text: "Hello" }] },
      ])
    })

    test("tool-call parts → tool_use content blocks", () => {
      const result = TokenCounter.toAnthropicMessages([
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "tc1",
              toolName: "bash",
              input: { command: "ls" },
            },
          ],
        } as ModelMessage,
      ])
      expect(result).toEqual([
        {
          role: "assistant",
          content: [
            {
              type: "tool_use",
              id: "tc1",
              name: "bash",
              input: { command: "ls" },
            },
          ],
        },
      ])
    })

    test("tool-result parts → user role with tool_result blocks", () => {
       const result = TokenCounter.toAnthropicMessages([
         {
           role: "tool",
           content: [
             {
               type: "tool-result",
               toolCallId: "tc1",
               toolName: "bash",
               output: "file1.ts\nfile2.ts",
             },
           ],
         } as unknown as ModelMessage,
       ])
      expect(result).toEqual([
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "tc1",
              content: "file1.ts\nfile2.ts",
            },
          ],
        },
      ])
    })

    test("system messages → skipped", () => {
      const result = TokenCounter.toAnthropicMessages([
        { role: "system", content: "You are helpful" } as ModelMessage,
        { role: "user", content: "hello" } as ModelMessage,
      ])
      expect(result).toHaveLength(1)
      expect(result[0].role).toBe("user")
    })

    test("empty array content → skipped", () => {
      const result = TokenCounter.toAnthropicMessages([
        { role: "user", content: [] } as ModelMessage,
        { role: "assistant", content: [] } as ModelMessage,
      ])
      expect(result).toEqual([])
    })

    test("mixed text + tool-call in assistant → both converted", () => {
      const result = TokenCounter.toAnthropicMessages([
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me check" },
            {
              type: "tool-call",
              toolCallId: "tc1",
              toolName: "read",
              input: {},
            },
          ],
        } as ModelMessage,
      ])
      expect(result).toHaveLength(1)
      expect(result[0].content).toEqual([
        { type: "text", text: "Let me check" },
        { type: "tool_use", id: "tc1", name: "read", input: {} },
      ])
    })
  })

  describe("toAnthropicSystem", () => {
    test("single system message → returns string", () => {
      const result = TokenCounter.toAnthropicSystem([
        { role: "system", content: "Be helpful" } as ModelMessage,
      ])
      expect(result).toBe("Be helpful")
    })

    test("multiple system messages → returns TextBlockParam array", () => {
      const result = TokenCounter.toAnthropicSystem([
        { role: "system", content: "Be helpful" } as ModelMessage,
        { role: "system", content: "Be concise" } as ModelMessage,
      ])
      expect(result).toEqual([
        { type: "text", text: "Be helpful" },
        { type: "text", text: "Be concise" },
      ])
    })

    test("no system messages → returns empty string", () => {
      expect(TokenCounter.toAnthropicSystem([])).toBe("")
    })

    test("non-system messages → filtered out, returns empty string", () => {
      const result = TokenCounter.toAnthropicSystem([
        { role: "user", content: "hello" } as ModelMessage,
        { role: "assistant", content: "hi" } as ModelMessage,
      ])
      expect(result).toBe("")
    })
  })

  describe("Anthropic provider detection", () => {
    let originalKey: string | undefined

    beforeEach(() => {
      originalKey = process.env.ANTHROPIC_API_KEY
      delete process.env.ANTHROPIC_API_KEY
    })

    afterEach(() => {
      if (originalKey !== undefined) {
        process.env.ANTHROPIC_API_KEY = originalKey
      } else {
        delete process.env.ANTHROPIC_API_KEY
      }
    })

    test("non-Anthropic model uses estimation, never touches API", async () => {
      const model = createModel({ npm: "@ai-sdk/openai" })
      const text = "estimation only path"
      const result = await TokenCounter.count({
        model,
        system: [],
        messages: [{ role: "user", content: text } as ModelMessage],
      })
      expect(result).toBe(Token.estimate(text))
    })

    test("Anthropic model with missing API key falls back to estimation", async () => {
      const model = createModel({
        npm: "@ai-sdk/anthropic",
        id: "claude-sonnet-4-20250514",
      })
      const text = "fallback test message"
      const result = await TokenCounter.count({
        model,
        system: [],
        messages: [{ role: "user", content: text } as ModelMessage],
      })
      expect(result).toBe(Token.estimate(text))
    })

    test("google-vertex/anthropic detected as Anthropic provider", async () => {
      const model = createModel({
        npm: "@ai-sdk/google-vertex/anthropic",
        id: "claude-sonnet-4-20250514",
      })
      const text = "vertex anthropic test"
      const result = await TokenCounter.count({
        model,
        system: [],
        messages: [{ role: "user", content: text } as ModelMessage],
      })
      expect(result).toBe(Token.estimate(text))
    })

    test("error fallback does not throw", async () => {
      const model = createModel({
        npm: "@ai-sdk/anthropic",
        id: "claude-sonnet-4-20250514",
      })
      const result = await TokenCounter.count({
        model,
        system: [
          { role: "system", content: "system prompt" } as ModelMessage,
        ],
        messages: [{ role: "user", content: "hello" } as ModelMessage],
      })
      expect(typeof result).toBe("number")
      expect(result).toBeGreaterThan(0)
    })
  })
})
