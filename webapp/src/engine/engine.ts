import { likelihoodFor } from './kb'
import type { Evidence, FeatureId, KB, StopConfig, StopReason } from './types'
import { DEFAULT_STOP, ENTROPY_EPSILON, LOG_FLOOR } from './types'

// Candidate answers rarer than this contribute nothing measurable to EIG.
const MIN_VALUE_PROB = 1e-9

// Rescales log-probabilities to sum to 1 in linear space.
// The max is subtracted before exponentiating so that exp() cannot overflow.
function logNormalize(log: number[]): number[] {
  const max = Math.max(...log)
  if (!Number.isFinite(max)) return log
  const logTotal =
    max + Math.log(log.reduce((sum, x) => sum + Math.exp(x - max), 0))
  return log.map((x) => x - logTotal)
}

// Shannon entropy in bits.
// Probabilities at or below ENTROPY_EPSILON are skipped and contribute zero.
export function entropyBits(p: number[]): number {
  return p.reduce((h, x) => (x > ENTROPY_EPSILON ? h - x * Math.log2(x) : h), 0)
}

// Bayes update in log space, returning a linear posterior.
//
// With Jeffrey's conditioning
// confidence=1 is a standard update, confidence=0 leaves the belief untouched
function update(post: number[], lik: number[], confidence = 1): number[] {
  const log = post.map((p, i) => {
    const l = confidence >= 1 ? lik[i] : confidence * lik[i] + (1 - confidence)
    return Math.log(Math.max(p, LOG_FLOOR)) + Math.log(Math.max(l, LOG_FLOOR))
  })
  return logNormalize(log).map(Math.exp)
}

// Folds every observation into the prior.
export function posterior(kb: KB, evidence: Evidence): number[] {
  return Object.entries(evidence).reduce(
    (post, [featureId, obs]) =>
      update(post, likelihoodFor(kb, featureId, obs.value), obs.confidence),
    kb.prior,
  )
}

// Probability mass per taxonomy root, renormalised
function categoryDistribution(kb: KB, post: number[]): number[] {
  const index = new Map(kb.diseases.map((d, i) => [d, i]))

  const massOf = (node: string): number => {
    const leaf = index.get(node)
    if (leaf !== undefined) return post[leaf]
    return (kb.parents[node] ?? []).reduce((sum, c) => sum + massOf(c), 0)
  }

  const mass = kb.roots.map(massOf)
  const total = mass.reduce((a, b) => a + b, 0)
  return total > 0 ? mass.map((m) => m / total) : mass
}

export function entropy(kb: KB, post: number[]): number {
  return entropyBits(
    kb.roots.length > 0 ? categoryDistribution(kb, post) : post,
  )
}

export function differential(
  kb: KB,
  post: number[],
  k: number,
): { diseaseId: string; name: string; p: number }[] {
  return post
    .map((p, i) => ({ diseaseId: kb.diseases[i], name: kb.names[i], p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, k)
}

export function eig(kb: KB, post: number[], featureId: FeatureId): number {
  const feature = kb.features[featureId]
  if (!feature) return 0

  const expected = feature.values.reduce((acc, value) => {
    const lik = likelihoodFor(kb, featureId, value)
    const pValue = lik.reduce((sum, l, i) => sum + l * post[i], 0)
    if (pValue <= MIN_VALUE_PROB) return acc
    return acc + pValue * entropy(kb, update(post, lik))
  }, 0)

  return entropy(kb, post) - expected
}

// The unasked feature with the highest EIG.
// Ties go to the earliest feature in KB order
export function selectFeature(
  kb: KB,
  post: number[],
  evidence: Evidence,
): FeatureId | null {
  let best: FeatureId | null = null
  let bestEig = -Infinity

  for (const featureId of Object.keys(kb.features)) {
    if (featureId in evidence) continue
    const gain = eig(kb, post, featureId)
    if (gain > bestEig) {
      bestEig = gain
      best = featureId
    }
  }
  return best
}

export function shouldStop(
  post: number[],
  numAsked: number,
  cfg: StopConfig = DEFAULT_STOP,
): StopReason | null {
  if (numAsked >= cfg.budget) return 'MAX_QUESTIONS'
  if (Math.max(...post) >= cfg.tau) return 'CONFIDENT'
  return null
}
