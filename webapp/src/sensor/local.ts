import type { Feature } from '../engine/types'
import { loadedModel } from './model'
import { parseObservations } from './parse'
import { askMessages, parseMessages, retryMessages } from './prompts'
import type { Parse, Sensor, Tone } from './types'

type Message = { role: 'system' | 'user' | 'assistant'; content: string }

export function createLocalSensor(): Sensor {
  const complete = async (messages: Message[], maxTokens: number) => {
    const wllama = loadedModel()
    if (!wllama) throw new Error('sensor: model not loaded')

    const res = await wllama.createChatCompletion({
      messages,
      temperature: 0,
      max_tokens: maxTokens,
    })
    return res.choices[0]?.message?.content ?? ''
  }

  return {
    async parse(reply: string, candidates: Feature[]): Promise<Parse[]> {
      const first = parseObservations(
        await complete(parseMessages(reply, candidates), 256),
        candidates,
      )
      if (first) return first

      // One retry only. If a terser prompt still yields nothing parseable,
      // the turn contributes no evidence, which the engine handles as a
      // feature simply not having been answered.
      const second = parseObservations(
        await complete(retryMessages(reply, candidates), 256),
        candidates,
      )
      return second ?? []
    },

    async verbalize(feature: Feature, tone: Tone): Promise<string> {
      const text = await complete(askMessages(feature, tone), 80)
      // KB feature names are already phrased as questions, so falling back to
      // the raw name costs tone rather than meaning.
      return text.trim() || feature.name
    },
  }
}
