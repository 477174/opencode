import type { ModelMessage, AssistantContent, UserContent, ToolContent } from "ai"
import Anthropic from "@anthropic-ai/sdk"
import type { MessageParam, ContentBlockParam, TextBlockParam, ToolUseBlockParam, ToolResultBlockParam } from "@anthropic-ai/sdk/resources/messages/messages"
import { Token } from "@/util/token"
import { Log } from "@/util/log"
import type { Provider } from "@/provider/provider"

export namespace TokenCounter {
  const log = Log.create({ service: "session.token-counter" })

  let client: Anthropic | undefined

  function getClient(): Anthropic {
    if (!client) {
      client = new Anthropic()
    }
    return client
  }

  function isAnthropicProvider(model: Provider.Model): boolean {
    return model.api.npm === "@ai-sdk/anthropic" || model.api.npm === "@ai-sdk/google-vertex/anthropic"
  }

  export function toAnthropicSystem(system: ModelMessage[]): string | Array<TextBlockParam> {
    const parts: TextBlockParam[] = []
    for (const msg of system) {
      if (msg.role !== "system") continue
      if (typeof msg.content === "string") {
        parts.push({ type: "text", text: msg.content })
      }
    }
    if (parts.length === 0) return ""
    if (parts.length === 1) return parts[0].text
    return parts
  }

  export function toAnthropicMessages(messages: ModelMessage[]): MessageParam[] {
    const result: MessageParam[] = []
    for (const msg of messages) {
      if (msg.role === "system") continue

      if (msg.role === "user") {
        const content = convertUserContent(msg.content)
        if (content.length > 0) {
          result.push({ role: "user", content })
        }
        continue
      }

      if (msg.role === "assistant") {
        const content = convertAssistantContent(msg.content)
        if (content.length > 0) {
          result.push({ role: "assistant", content })
        }
        continue
      }

      if (msg.role === "tool") {
        const blocks = convertToolContent(msg.content)
        if (blocks.length > 0) {
          result.push({ role: "user", content: blocks })
        }
        continue
      }
    }
    return result
  }

  function convertUserContent(content: UserContent): string | ContentBlockParam[] {
    if (typeof content === "string") return content
    const blocks: ContentBlockParam[] = []
    for (const part of content) {
      if (part.type === "text") {
        blocks.push({ type: "text", text: part.text })
      }
    }
    return blocks.length > 0 ? blocks : []
  }

  function convertAssistantContent(content: AssistantContent): string | ContentBlockParam[] {
    if (typeof content === "string") return content
    const blocks: ContentBlockParam[] = []
    for (const part of content) {
      if (part.type === "text") {
        blocks.push({ type: "text", text: part.text })
      } else if (part.type === "tool-call") {
        blocks.push({
          type: "tool_use",
          id: part.toolCallId,
          name: part.toolName,
          input: part.input ?? {},
        } satisfies ToolUseBlockParam)
      }
    }
    return blocks.length > 0 ? blocks : []
  }

  function convertToolContent(content: ToolContent): ToolResultBlockParam[] {
    const blocks: ToolResultBlockParam[] = []
    for (const part of content) {
      if (part.type === "tool-result") {
        blocks.push({
          type: "tool_result",
          tool_use_id: part.toolCallId,
          content: serializeToolOutput(part.output),
        } satisfies ToolResultBlockParam)
      }
    }
    return blocks
  }

  function serializeToolOutput(output: unknown): string {
    if (output === undefined || output === null) return ""
    if (typeof output === "string") return output
    if (typeof output === "object") {
      const typed = output as { type?: string; value?: unknown }
      if (typed.type === "text" && typeof typed.value === "string") return typed.value
      if (typed.type === "json") return JSON.stringify(typed.value ?? "")
      if (typed.type === "error-text" && typeof typed.value === "string") return typed.value
      if (typed.type === "error-json") return JSON.stringify(typed.value ?? "")
      if (typed.type === "content" && Array.isArray(typed.value)) {
        return typed.value
          .map((v: { type: string; text?: string }) => (v.type === "text" && v.text ? v.text : ""))
          .join("")
      }
    }
    return JSON.stringify(output)
  }

  function serializeMessages(messages: ModelMessage[]): string {
    const parts: string[] = []
    for (const msg of messages) {
      if (typeof msg.content === "string") {
        parts.push(msg.content)
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if ("text" in part && typeof part.text === "string") {
            parts.push(part.text)
          } else if ("toolName" in part && typeof part.toolName === "string") {
            parts.push(part.toolName)
            if ("input" in part) parts.push(JSON.stringify(part.input))
          } else if ("output" in part) {
            parts.push(serializeToolOutput(part.output))
          }
        }
      }
    }
    return parts.join("\n")
  }

  function estimateTokens(input: {
    system: ModelMessage[]
    messages: ModelMessage[]
  }): number {
    const systemText = serializeMessages(input.system)
    const messagesText = serializeMessages(input.messages)
    return Token.estimate(systemText) + Token.estimate(messagesText)
  }

  export async function count(input: {
    model: Provider.Model
    system: ModelMessage[]
    messages: ModelMessage[]
    tools?: Array<{ name: string; description?: string; parameters?: Record<string, unknown> }>
  }): Promise<number> {
    if (!isAnthropicProvider(input.model)) {
      return estimateTokens(input)
    }

    try {
      const anthropicClient = getClient()
      const system = toAnthropicSystem(input.system)
      const messages = toAnthropicMessages(input.messages)

      if (messages.length === 0) {
        return estimateTokens(input)
      }

      const params: Anthropic.Messages.MessageCountTokensParams = {
        model: input.model.api.id,
        messages,
      }

      if (system) {
        params.system = system
      }

      const result = await anthropicClient.messages.countTokens(params)
      return result.input_tokens
    } catch (err) {
      log.warn("countTokens API failed, falling back to estimation", {
        error: err instanceof Error ? err.message : String(err),
        model: input.model.id,
      })
      return estimateTokens(input)
    }
  }
}
