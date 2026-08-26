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

// A knowledge base, as loaded from public/kb/*.json
export type KB = {
  id: string
  diseases: string[]
  names: string[]
  prior: Float64Array
  features: Map<FeatureId, Feature>
  // feature -> value -> P(value | disease), one entry per disease.
  likelihood: Map<FeatureId, Map<Value, Float64Array>>
  parents: Map<string, string[]>
  roots: string[]
}

// What the patient told us about one feature.
export type Observation = {
  value: Value
  confidence: number
}

// Everything observed so far. Re-answering a feature replaces its observation;
// insertion order is the order the features were asked.
export type Evidence = Map<FeatureId, Observation>

export type StopReason = 'CONFIDENT' | 'MAX_QUESTIONS' | 'FEATURES_EXHAUSTED'

export type StopConfig = {
  // Commit once the top posterior reaches this.
  tau: number
  // Hard cap on distinct features asked.
  budget: number
}

export const DEFAULT_STOP: StopConfig = { tau: 0.9, budget: 20 }
