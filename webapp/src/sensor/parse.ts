import { CONFIDENCE_WEIGHTS, resolveConfidence } from '../engine/types'
import type { ConfidenceLabel, Feature } from '../engine/types'
import type { Parse } from './types'

type Raw = { q?: unknown; value?: unknown; confidence?: unknown }

const isLabel = (x: unknown): x is ConfidenceLabel =>
  typeof x === 'string' && x in CONFIDENCE_WEIGHTS

// Models like to wrap JSON in prose or a fenced block. Take the outermost
// array and ignore whatever surrounds it.
function extractArray(text: string): Raw[] | null {
  const start = text.indexOf('[')
  const end = text.lastIndexOf(']')
  if (start === -1 || end <= start) return null
  try {
    const parsed = JSON.parse(text.slice(start, end + 1))
    return Array.isArray(parsed) ? parsed : null
  } catch {
    return null
  }
}

/**
 * Turns a model's reply into observations we are willing to believe.
 *
 * Anything questionable is dropped rather than repaired: a wrong observation
 * corrupts the posterior silently, while a missing one only costs us the
 * information from this turn. Returns null when the reply was not usable at
 * all, which the caller may treat as grounds for a retry; an empty array
 * means the model was understood and reported nothing.
 */
export function parseObservations(
  text: string,
  candidates: Feature[],
): Parse[] | null {
  const raw = extractArray(text)
  if (!raw) return null

  const seen = new Set<string>()
  const out: Parse[] = []

  for (const item of raw) {
    const index = Number(item?.q)
    const feature = candidates[index - 1]
    if (!feature) continue

    const value = String(item?.value)
    if (!feature.values.includes(value)) continue

    // An unrecognised label would silently become full-strength evidence via
    // resolveConfidence, so require one we know.
    if (!isLabel(item?.confidence)) continue

    // One observation per feature; a repeat is the model contradicting itself.
    if (seen.has(feature.id)) continue
    seen.add(feature.id)

    out.push({
      featureId: feature.id,
      value,
      label: item.confidence,
      confidence: resolveConfidence(item.confidence),
    })
  }

  return out
}
