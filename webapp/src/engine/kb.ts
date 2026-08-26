import type { Feature, FeatureId, KB, Value } from './types'

// Shape of public/kb/*.json, as written by scripts/export_kb.py.
type KbJson = {
  id: string
  diseases: string[]
  names: string[]
  prior: number[]
  features: Feature[]
  likelihood: Record<FeatureId, Record<Value, number[]>>
  parents: Record<string, string[]>
  roots: string[]
}

export function parseKb(json: unknown): KB {
  const raw = json as KbJson
  const n = raw.diseases.length
  const fail = (msg: string) => {
    throw new Error(`kb "${raw.id}": ${msg}`)
  }

  if (n === 0) fail('no diseases')
  if (raw.names.length !== n)
    fail(`${raw.names.length} names for ${n} diseases`)
  if (raw.prior.length !== n)
    fail(`prior has ${raw.prior.length} entries, expected ${n}`)

  const priorSum = raw.prior.reduce((a, b) => a + b, 0)
  if (Math.abs(priorSum - 1) > 1e-6)
    fail(`prior sums to ${priorSum}, expected 1`)

  const features = new Map<FeatureId, Feature>()
  const likelihood = new Map<FeatureId, Map<Value, Float64Array>>()

  for (const f of raw.features) {
    const byValue = raw.likelihood[f.id]
    if (!byValue) fail(`feature "${f.id}" has no likelihoods`)

    const vectors = new Map<Value, Float64Array>()
    for (const v of f.values) {
      const vec = byValue[v]
      if (!vec) fail(`feature "${f.id}" has no likelihood for value "${v}"`)
      if (vec.length !== n)
        fail(
          `likelihood "${f.id}"/"${v}" has ${vec.length} entries, expected ${n}`,
        )
      vectors.set(v, Float64Array.from(vec))
    }

    features.set(f.id, f)
    likelihood.set(f.id, vectors)
  }

  return {
    id: raw.id,
    diseases: raw.diseases,
    names: raw.names,
    prior: Float64Array.from(raw.prior),
    features,
    likelihood,
    parents: new Map(Object.entries(raw.parents)),
    roots: raw.roots,
  }
}
