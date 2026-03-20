import { Effect } from "effect"
import z from "zod"
import { runtime } from "@/effect/runtime"
import * as S from "./effect"

export { OAUTH_DUMMY_KEY } from "./effect"

function runPromise<A>(f: (service: S.AuthEffect.Interface) => Effect.Effect<A, S.AuthError>) {
  return runtime.runPromise(S.AuthEffect.Service.use(f))
}

export namespace Auth {
  export const Oauth = z
    .object({
      type: z.literal("oauth"),
      refresh: z.string(),
      access: z.string(),
      expires: z.number(),
      accountId: z.string().optional(),
      enterpriseUrl: z.string().optional(),
    })
    .meta({ ref: "OAuth" })

  export const Api = z
    .object({
      type: z.literal("api"),
      key: z.string(),
    })
    .meta({ ref: "ApiAuth" })

  export const WellKnown = z
    .object({
      type: z.literal("wellknown"),
      key: z.string(),
      token: z.string(),
    })
    .meta({ ref: "WellKnownAuth" })

  export const Info = z.discriminatedUnion("type", [Oauth, Api, WellKnown]).meta({ ref: "Auth" })
  export type Info = z.infer<typeof Info>
  export type InfoWithKey = Info & { _key: string }

  export async function get(providerID: string) {
    return runPromise((service) => service.get(providerID))
  }

  export async function all(): Promise<Record<string, Info>> {
    return runPromise((service) => service.all())
  }

  export async function set(key: string, info: Info) {
    return runPromise((service) => service.set(key, info))
  }

  export async function remove(key: string) {
    return runPromise((service) => service.remove(key))
  }

  export async function list(providerID: string): Promise<Array<{ key: string; info: Info }>> {
    const data = await all()
    const result: Array<{ key: string; info: Info }> = []
    for (const [key, info] of Object.entries(data)) {
      if (key === providerID || key.startsWith(providerID + ":")) {
        result.push({ key, info })
      }
    }
    return result
  }

  export async function nextKey(providerID: string): Promise<string> {
    const entries = await list(providerID)
    if (entries.length === 0) return providerID
    let max = 0
    for (const entry of entries) {
      const sep = entry.key.indexOf(":")
      if (sep !== -1) max = Math.max(max, parseInt(entry.key.slice(sep + 1)) || 0)
    }
    return `${providerID}:${max + 1}`
  }
}
