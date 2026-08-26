import type { Feature, FeatureId, KB, Value } from './types'

// A non-answer: carries no information, so it leaves the belief untouched.
export const UNKNOWN = 'unknown'

// public/kb/*.json, as written by scripts/export_kb.py.
type KbJson = Omit<KB, 'features'> & { features: Feature[] }

export function parseKb(json: unknown): KB {
  const raw = json as KbJson
  const n = raw.diseases.length
  const fail = (msg: string) => {
    throw new Error(`kb "${raw.id}": ${msg}`)
  }

  if (n === 0) fail('no diseases')
  if (raw.names.length !== n)
    fail(`${raw.names.length} names for ${n} diseases`)
  if (raw.prior.length !== n) fail(`prior has ${raw.prior.length} entries`)

  const priorSum = raw.prior.reduce((a, b) => a + b, 0)
  if (Math.abs(priorSum - 1) > 1e-6) fail(`prior sums to ${priorSum}`)

  for (const f of raw.features) {
    for (const v of f.values) {
      const vec = raw.likelihood[f.id]?.[v]
      if (!vec) fail(`no likelihood for "${f.id}"/"${v}"`)
      else if (vec.length !== n)
        fail(`likelihood "${f.id}"/"${v}" has ${vec.length} entries`)
    }
  }

  return {
    ...raw,
    features: Object.fromEntries(raw.features.map((f) => [f.id, f])),
  }
}

// P(value | disease) for every disease.
export function likelihoodFor(
  kb: KB,
  featureId: FeatureId,
  value: Value,
): number[] {
  const vec = value === UNKNOWN ? undefined : kb.likelihood[featureId]?.[value]
  return vec ?? new Array(kb.diseases.length).fill(1)
}
