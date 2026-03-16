import { BusEvent } from "@/bus/bus-event"
import { Bus } from "@/bus"
import { Session } from "."
import { Identifier } from "../id/id"
import { Instance } from "../project/instance"
import { Provider } from "../provider/provider"
import { MessageV2 } from "./message-v2"
import z from "zod"
import { Token } from "../util/token"
import { Log } from "../util/log"
import { SessionProcessor } from "./processor"
import { fn } from "@/util/fn"
import { Agent } from "@/agent/agent"
import { Plugin } from "@/plugin"
import { Config } from "@/config/config"
import { ProviderTransform } from "@/provider/transform"
import { Flag } from "@/flag/flag"
import type { ModelMessage } from "ai"

export namespace SessionCompaction {
  const log = Log.create({ service: "session.compaction" })

  export const Event = {
    Compacted: BusEvent.define(
      "session.compacted",
      z.object({
        sessionID: z.string(),
      }),
    ),
  }

  const COMPACTION_BUFFER = 20_000
  export const ROLLING_THRESHOLD = 0.85
  export const ROLLING_TARGET = 0.70

  export async function isOverflow(input: { tokens: MessageV2.Assistant["tokens"]; model: Provider.Model }) {
    const config = await Config.get()
    if (config.compaction?.auto === false) return false
    const context = input.model.limit.context
    if (context === 0) return false

    const count =
      input.tokens.total ||
      input.tokens.input + input.tokens.output + input.tokens.cache.read + input.tokens.cache.write

    const reserved =
      config.compaction?.reserved ?? Math.min(COMPACTION_BUFFER, ProviderTransform.maxOutputTokens(input.model))
    const usable = input.model.limit.input
      ? input.model.limit.input - reserved
      : context - ProviderTransform.maxOutputTokens(input.model)
    return count >= usable
  }

  export async function getOverflowStrategy(input: {
    tokens: MessageV2.Assistant["tokens"]
    model: Provider.Model
    rollingAttempted?: boolean
  }): Promise<"rolling" | "full" | "none"> {
    const overflow = await isOverflow({ tokens: input.tokens, model: input.model })
    if (!overflow) return "none"

    const config = await Config.get()
    // If rolling was already attempted and still overflowing, emergency fallback to full
    if (input.rollingAttempted) return "full"
    // If mode is explicitly "full", use full compaction
    if (config.compaction?.mode === "full") return "full"
    // Check if rolling compaction is disabled via flag
    if (Flag.OPENCODE_DISABLE_ROLLING_COMPACTION) return "full"

    return "rolling"
  }

  export const PRUNE_MINIMUM = 20_000
  export const PRUNE_PROTECT = 40_000

  const PRUNE_PROTECTED_TOOLS = ["skill"]

  // goes backwards through parts until there are 40_000 tokens worth of tool
  // calls. then erases output of previous tool calls. idea is to throw away old
  // tool calls that are no longer relevant.
  export async function prune(input: { sessionID: string }) {
    const config = await Config.get()
    if (config.compaction?.prune === false) return
    log.info("pruning")
    const msgs = await Session.messages({ sessionID: input.sessionID })
    let total = 0
    let pruned = 0
    const toPrune = []
    let turns = 0

    loop: for (let msgIndex = msgs.length - 1; msgIndex >= 0; msgIndex--) {
      const msg = msgs[msgIndex]
      if (msg.info.role === "user") turns++
      if (turns < 2) continue
      if (msg.info.role === "assistant" && msg.info.summary) break loop
      for (let partIndex = msg.parts.length - 1; partIndex >= 0; partIndex--) {
        const part = msg.parts[partIndex]
        if (part.type === "tool")
          if (part.state.status === "completed") {
            if (PRUNE_PROTECTED_TOOLS.includes(part.tool)) continue

            if (part.state.time.compacted) break loop
            const estimate = Token.estimate(part.state.output)
            total += estimate
            if (total > PRUNE_PROTECT) {
              pruned += estimate
              toPrune.push(part)
            }
          }
      }
    }
    log.info("found", { pruned, total })
    if (pruned > PRUNE_MINIMUM) {
      for (const part of toPrune) {
        if (part.state.status === "completed") {
          part.state.time.compacted = Date.now()
          await Session.updatePart(part)
        }
      }
      log.info("pruned", { count: toPrune.length })
    }
  }

  export async function process(input: {
    parentID: string
    messages: MessageV2.WithParts[]
    sessionID: string
    abort: AbortSignal
    auto: boolean
    overflow?: boolean
  }) {
    const userMessage = input.messages.findLast((m) => m.info.id === input.parentID)!.info as MessageV2.User

    let messages = input.messages
    let replay: MessageV2.WithParts | undefined
    if (input.overflow) {
      const idx = input.messages.findIndex((m) => m.info.id === input.parentID)
      for (let i = idx - 1; i >= 0; i--) {
        const msg = input.messages[i]
        if (msg.info.role === "user" && !msg.parts.some((p) => p.type === "compaction")) {
          replay = msg
          messages = input.messages.slice(0, i)
          break
        }
      }
      const hasContent =
        replay && messages.some((m) => m.info.role === "user" && !m.parts.some((p) => p.type === "compaction"))
      if (!hasContent) {
        replay = undefined
        messages = input.messages
      }
    }

    const agent = await Agent.get("compaction")
    const model = agent.model
      ? await Provider.getModel(agent.model.providerID, agent.model.modelID)
      : await Provider.getModel(userMessage.model.providerID, userMessage.model.modelID)
    const msg = (await Session.updateMessage({
      id: Identifier.ascending("message"),
      role: "assistant",
      parentID: input.parentID,
      sessionID: input.sessionID,
      mode: "compaction",
      agent: "compaction",
      variant: userMessage.variant,
      summary: true,
      path: {
        cwd: Instance.directory,
        root: Instance.worktree,
      },
      cost: 0,
      tokens: {
        output: 0,
        input: 0,
        reasoning: 0,
        cache: { read: 0, write: 0 },
      },
      modelID: model.id,
      providerID: model.providerID,
      time: {
        created: Date.now(),
      },
    })) as MessageV2.Assistant
    const processor = SessionProcessor.create({
      assistantMessage: msg,
      sessionID: input.sessionID,
      model,
      abort: input.abort,
    })
    // Allow plugins to inject context or replace compaction prompt
    const compacting = await Plugin.trigger(
      "experimental.session.compacting",
      { sessionID: input.sessionID },
      { context: [], prompt: undefined },
    )
    const defaultPrompt = `Provide a detailed prompt for continuing our conversation above.
Focus on information that would be helpful for continuing the conversation, including what we did, what we're doing, which files we're working on, and what we're going to do next.
The summary that you construct will be used so that another agent can read it and continue the work.

When constructing the summary, try to stick to this template:
---
## Goal

[What goal(s) is the user trying to accomplish?]

## Instructions

- [What important instructions did the user give you that are relevant]
- [If there is a plan or spec, include information about it so next agent can continue using it]

## Discoveries

[What notable things were learned during this conversation that would be useful for the next agent to know when continuing the work]

## Accomplished

[What work has been completed, what work is still in progress, and what work is left?]

## Relevant files / directories

[Construct a structured list of relevant files that have been read, edited, or created that pertain to the task at hand. If all the files in a directory are relevant, include the path to the directory.]
---`

    const promptText = compacting.prompt ?? [defaultPrompt, ...compacting.context].join("\n\n")
    const result = await processor.process({
      user: userMessage,
      agent,
      abort: input.abort,
      sessionID: input.sessionID,
      tools: {},
      system: [],
      messages: [
        ...MessageV2.toModelMessages(messages, model, { stripMedia: true }),
        {
          role: "user",
          content: [
            {
              type: "text",
              text: promptText,
            },
          ],
        },
      ],
      model,
    })

    if (result === "compact") {
      processor.message.error = new MessageV2.ContextOverflowError({
        message: replay
          ? "Conversation history too large to compact - exceeds model context limit"
          : "Session too large to compact - context exceeds model limit even after stripping media",
      }).toObject()
      processor.message.finish = "error"
      await Session.updateMessage(processor.message)
      return "stop"
    }

    if (result === "continue" && input.auto) {
      if (replay) {
        const original = replay.info as MessageV2.User
        const replayMsg = await Session.updateMessage({
          id: Identifier.ascending("message"),
          role: "user",
          sessionID: input.sessionID,
          time: { created: Date.now() },
          agent: original.agent,
          model: original.model,
          format: original.format,
          tools: original.tools,
          system: original.system,
          variant: original.variant,
        })
        for (const part of replay.parts) {
          if (part.type === "compaction") continue
          const replayPart =
            part.type === "file" && MessageV2.isMedia(part.mime)
              ? { type: "text" as const, text: `[Attached ${part.mime}: ${part.filename ?? "file"}]` }
              : part
          await Session.updatePart({
            ...replayPart,
            id: Identifier.ascending("part"),
            messageID: replayMsg.id,
            sessionID: input.sessionID,
          })
        }
      } else {
        const continueMsg = await Session.updateMessage({
          id: Identifier.ascending("message"),
          role: "user",
          sessionID: input.sessionID,
          time: { created: Date.now() },
          agent: userMessage.agent,
          model: userMessage.model,
        })
        const text =
          (input.overflow
            ? "The previous request exceeded the provider's size limit due to large media attachments. The conversation was compacted and media files were removed from context. If the user was asking about attached images or files, explain that the attachments were too large to process and suggest they try again with smaller or fewer files.\n\n"
            : "") +
          "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
        await Session.updatePart({
          id: Identifier.ascending("part"),
          messageID: continueMsg.id,
          sessionID: input.sessionID,
          type: "text",
          synthetic: true,
          text,
          time: {
            start: Date.now(),
            end: Date.now(),
          },
        })
      }
    }
    if (processor.message.error) return "stop"
    Bus.publish(Event.Compacted, { sessionID: input.sessionID })
    return "continue"
  }

  export const create = fn(
    z.object({
      sessionID: Identifier.schema("session"),
      agent: z.string(),
      model: z.object({
        providerID: z.string(),
        modelID: z.string(),
      }),
      auto: z.boolean(),
      overflow: z.boolean().optional(),
    }),
    async (input) => {
      const msg = await Session.updateMessage({
        id: Identifier.ascending("message"),
        role: "user",
        model: input.model,
        sessionID: input.sessionID,
        agent: input.agent,
        time: {
          created: Date.now(),
        },
      })
      await Session.updatePart({
        id: Identifier.ascending("part"),
        messageID: msg.id,
        sessionID: msg.sessionID,
        type: "compaction",
        auto: input.auto,
        overflow: input.overflow,
      })
    },
  )

  function serializeTurn(msgs: MessageV2.WithParts[]): string {
    const parts: string[] = []
    for (const msg of msgs) {
      for (const part of msg.parts) {
        if (part.type === "text") parts.push(part.text)
        if (part.type === "reasoning" && part.text) parts.push(part.text)
        if (part.type === "tool" && part.state.status === "completed") {
          parts.push(part.tool)
          if (part.state.input) parts.push(JSON.stringify(part.state.input))
          if (part.state.output) parts.push(part.state.output)
        }
      }
    }
    return parts.join("\n")
  }

  export function estimateTurnTokens(messages: MessageV2.WithParts[], startIdx: number, endIdx: number): number {
    const slice = messages.slice(startIdx, endIdx)
    return Token.estimate(serializeTurn(slice))
  }

  export async function getRollingSummary(sessionID: string): Promise<string | null> {
    const msgs = await Session.messages({ sessionID })
    for (let i = msgs.length - 1; i >= 0; i--) {
      const msg = msgs[i]
      if (msg.info.role === "assistant" && msg.info.rolling === true && msg.info.summary === true) {
        const textParts = msg.parts.filter((p): p is MessageV2.TextPart => p.type === "text")
        if (textParts.length > 0) return textParts.map((p) => p.text).join("\n")
      }
    }
    return null
  }

  export async function shouldRollingCompact(input: {
    tokenCount: number
    model: Provider.Model
  }): Promise<{ needed: true; currentTokens: number; targetTokens: number } | { needed: false }> {
    const config = await Config.get()
    if (config.compaction?.mode === "full") return { needed: false }
    if (config.compaction?.auto === false) return { needed: false }
    if (Flag.OPENCODE_DISABLE_ROLLING_COMPACTION) return { needed: false }
    const context = input.model.limit.context
    if (context === 0) return { needed: false }

    let threshold = config.compaction?.threshold ?? ROLLING_THRESHOLD
    let target = config.compaction?.target ?? ROLLING_TARGET
    if (threshold <= target) {
      log.warn("invalid compaction config: threshold must be > target, using defaults", { threshold, target })
      threshold = ROLLING_THRESHOLD
      target = ROLLING_TARGET
    }

    const reserved =
      config.compaction?.reserved ?? Math.min(COMPACTION_BUFFER, ProviderTransform.maxOutputTokens(input.model))
    const usable = input.model.limit.input
      ? input.model.limit.input - reserved
      : context - ProviderTransform.maxOutputTokens(input.model)
    const triggerAt = usable * threshold
    const targetTokens = usable * target

    if (input.tokenCount >= triggerAt) {
      return { needed: true, currentTokens: input.tokenCount, targetTokens }
    }
    return { needed: false }
  }

  export async function processRolling(input: {
    sessionID: string
    existingSummary: string | null
    turnsToCompact: MessageV2.WithParts[]
    model: Provider.Model
    abort: AbortSignal
  }): Promise<MessageV2.Assistant | null> {
    const agent = await Agent.get("rolling-compaction")
    const model = agent.model
      ? await Provider.getModel(agent.model.providerID, agent.model.modelID)
      : input.model

    // Find or create rolling summary message
    const msgs = await Session.messages({ sessionID: input.sessionID })
    let summaryMsg = msgs.find(
      (m) =>
        m.info.role === "assistant" &&
        (m.info as MessageV2.Assistant).rolling === true &&
        (m.info as MessageV2.Assistant).summary === true,
    )

    const firstUser = msgs.find((m) => m.info.role === "user")
    if (!firstUser) return null

    if (!summaryMsg) {
      const created = (await Session.updateMessage({
        id: Identifier.ascending("message"),
        role: "assistant",
        parentID: firstUser.info.id,
        sessionID: input.sessionID,
        mode: "rolling-compaction",
        agent: "rolling-compaction",
        summary: true,
        rolling: true,
        path: {
          cwd: Instance.directory,
          root: Instance.worktree,
        },
        cost: 0,
        tokens: {
          output: 0,
          input: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
        modelID: model.id,
        providerID: model.providerID,
        time: {
          created: Date.now(),
        },
      })) as MessageV2.Assistant
      // Re-fetch with parts
      const refreshed = await Session.messages({ sessionID: input.sessionID })
      summaryMsg = refreshed.find((m) => m.info.id === created.id) ?? { info: created, parts: [] }
    }

    const processor = SessionProcessor.create({
      assistantMessage: summaryMsg.info as MessageV2.Assistant,
      sessionID: input.sessionID,
      model,
      abort: input.abort,
    })

    // Build messages for the agent
    const contextMessages: ModelMessage[] = []
    if (input.existingSummary) {
      contextMessages.push({
        role: "assistant",
        content: [{ type: "text", text: input.existingSummary }],
      })
    }
    contextMessages.push(...MessageV2.toModelMessages(input.turnsToCompact, model, { stripMedia: true }))
    contextMessages.push({
      role: "user",
      content: [
        {
          type: "text",
          text: "Merge the above conversation turns into the existing summary. Return the updated comprehensive summary.",
        },
      ],
    })

    const userInfo = firstUser.info as MessageV2.User
    await processor.process({
      user: userInfo,
      agent,
      abort: input.abort,
      sessionID: input.sessionID,
      tools: {},
      system: [],
      messages: contextMessages,
      model,
    })

    return summaryMsg.info as MessageV2.Assistant
  }

  export async function rollingCompact(input: {
    sessionID: string
    messages: MessageV2.WithParts[]
    currentTokens: number
    targetTokens: number
    model: Provider.Model
    abort: AbortSignal
  }): Promise<{ compacted: true; turnsCompacted: number; newSummaryID: string } | { compacted: false; reason: string }> {
    log.info("rolling compact starting", {
      currentTokens: input.currentTokens,
      targetTokens: input.targetTokens,
    })

    // Find existing rolling summary
    const existingSummaryMsg = input.messages.find(
      (m) => m.info.role === "assistant" && (m.info as MessageV2.Assistant).rolling === true,
    )
    const existingSummary = await getRollingSummary(input.sessionID)

    // Build turn pairs (user + assistant) from oldest to newest
    type TurnPair = { user: MessageV2.WithParts; assistant: MessageV2.WithParts; allMsgs: MessageV2.WithParts[] }
    const turns: TurnPair[] = []
    for (let i = 0; i < input.messages.length; i++) {
      const msg = input.messages[i]
      if (msg.info.role !== "user") continue
      if (msg.info.compacted) continue // already compacted
      if (existingSummaryMsg && msg.info.id === existingSummaryMsg.info.id) continue // skip summary
      // Find matching assistant
      const assistant = input.messages.find(
        (m) => m.info.role === "assistant" && (m.info as MessageV2.Assistant).parentID === msg.info.id,
      )
      if (!assistant) continue
      const assistantInfo = assistant.info as MessageV2.Assistant
      if (assistantInfo.compacted) continue // already compacted
      if (!assistantInfo.finish) continue // incomplete response
      if (assistantInfo.summary) continue // skip summary messages
      // Collect all messages in this turn (including tool result messages)
      const allMsgs = [msg, assistant]
      turns.push({ user: msg, assistant, allMsgs })
    }

    if (turns.length === 0) {
      return { compacted: false, reason: "no eligible turns found" }
    }

    // Protect last 2 turns
    const protectedCount = 2
    const candidates = turns.slice(0, Math.max(0, turns.length - protectedCount))
    if (candidates.length === 0) {
      return { compacted: false, reason: "all turns are within protection window" }
    }

    // Calculate how many turns to compact
    let accumulated = 0
    let compactCount = 0
    for (const turn of candidates) {
      const estimate = Token.estimate(serializeTurn(turn.allMsgs))
      accumulated += estimate
      compactCount++
      if (input.currentTokens - accumulated <= input.targetTokens) break
    }
    // Always compact at least 1
    compactCount = Math.max(1, compactCount)
    const turnsToCompact = candidates.slice(0, compactCount)

    log.info("compacting turns", { count: turnsToCompact.length, accumulated })

    // Collect all messages to compact
    const msgsToCompact: MessageV2.WithParts[] = []
    for (const turn of turnsToCompact) {
      msgsToCompact.push(...turn.allMsgs)
    }

    // Call the rolling compaction agent
    const summaryResult = await processRolling({
      sessionID: input.sessionID,
      existingSummary,
      turnsToCompact: msgsToCompact,
      model: input.model,
      abort: input.abort,
    })

    if (!summaryResult) {
      return { compacted: false, reason: "processRolling failed" }
    }

    // Mark all compacted messages
    for (const turn of turnsToCompact) {
      for (const msg of turn.allMsgs) {
        const updated = {
          ...msg.info,
          compacted: { at: Date.now(), summaryID: summaryResult.id },
        }
        await Session.updateMessage(updated as MessageV2.Info)
      }
    }

    log.info("rolling compact complete", { turnsCompacted: turnsToCompact.length })
    return { compacted: true, turnsCompacted: turnsToCompact.length, newSummaryID: summaryResult.id }
  }
}
