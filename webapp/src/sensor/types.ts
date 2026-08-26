import type {
  ConfidenceLabel,
  Feature,
  FeatureId,
  Value,
} from '../engine/types'

// Quantisation-aware distilled, so it survives Q4 far better than a
// post-hoc quantisation of the same model. Roughly 230 MB.
export const MODEL = {
  repo: 'LiquidAI/LFM2.5-350M-GGUF',
  file: 'LFM2.5-350M-QAD-Q4_0.gguf',
}

// What the language layer extracts from one patient utterance. This is the
// entire contract between language and reasoning: no text crosses it.
export type Parse = {
  featureId: FeatureId
  value: Value
  label: ConfidenceLabel
  confidence: number
}

// How certain the engine currently is. Shapes the question's tone only; the
// sensor never learns the posterior itself.
export type Tone = 'low' | 'medium' | 'high'

export type Sensor = {
  // Takes the candidate features the reply might address, not just the one
  // that was asked, so volunteered symptoms are not thrown away. Returns one
  // observation per feature the patient actually spoke to, possibly none.
  parse(reply: string, candidates: Feature[]): Promise<Parse[]>
  verbalize(feature: Feature, tone: Tone): Promise<string>
}

// `downloading` and `loading` are distinct because a cached model skips
// straight to loading, and conflating them would look like a stall.
export type SensorState =
  | { kind: 'idle' }
  | { kind: 'downloading'; received: number; total: number }
  | { kind: 'loading' }
  | { kind: 'ready' }
  | { kind: 'error'; message: string }
