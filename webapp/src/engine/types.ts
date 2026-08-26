// Numerical constants, mirrored from MoBayes config.py.
export const LOG_FLOOR = 1e-300
export const ENTROPY_EPSILON = 1e-20

export const CONFIDENCE_WEIGHTS = {
  very_likely: 1.0,
  likely: 0.8,
  uncertain: 0.5,
  unlikely: 0.25,
  very_unlikely: 0.05,
} as const

export type ConfidenceLabel = keyof typeof CONFIDENCE_WEIGHTS

// Unrecognised labels resolve to 1.0, matching config.py::resolve_confidence.
export function resolveConfidence(label: string | number): number {
  if (typeof label === 'number') return Math.min(1, Math.max(0, label))
  const key = label.trim().toLowerCase() as ConfidenceLabel
  return CONFIDENCE_WEIGHTS[key] ?? 1.0
}

export type FeatureId = string
export type Value = string

export type Feature = {
  id: FeatureId
  name: string
  type: 'binary' | 'categorical'
  values: Value[]
}

// `diseases` fixes the index order of prior and every likelihood vector.
export type KB = {
  id: string
  diseases: string[]
  names: string[]
  prior: number[]
  features: Record<FeatureId, Feature>
  likelihood: Record<FeatureId, Record<Value, number[]>>
  parents: Record<string, string[]>
  roots: string[]
}

// What the patient told us about one feature.
export type Observation = {
  value: Value
  confidence: number
}

// Re-answering a feature replaces its observation.
export type Evidence = Record<FeatureId, Observation>

export type StopReason = 'CONFIDENT' | 'MAX_QUESTIONS' | 'FEATURES_EXHAUSTED'

export type StopConfig = {
  tau: number
  budget: number
}

export const DEFAULT_STOP: StopConfig = { tau: 0.9, budget: 20 }
