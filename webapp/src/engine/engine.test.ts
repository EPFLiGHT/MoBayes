import { expect, test } from 'vitest'
import {
  differential,
  eig,
  entropyBits,
  entropy,
  posterior,
  selectFeature,
  shouldStop,
} from './engine'
import type { Evidence, KB } from './types'

// TODO: very important: all this is model generated. It MUST not be trusted. When possible, replace with traces dumped from python.

// mock knowledge base
const kb: KB = {
  id: 'toy',
  diseases: ['a', 'b', 'c'],
  names: ['A', 'B', 'C'],
  prior: [0.5, 0.3, 0.2],
  features: {
    cough: {
      id: 'cough',
      name: 'Cough?',
      type: 'binary',
      values: ['yes', 'no'],
    },
    fever: {
      id: 'fever',
      name: 'Fever?',
      type: 'binary',
      values: ['yes', 'no'],
    },
  },
  likelihood: {
    cough: { yes: [0.9, 0.2, 0.5], no: [0.1, 0.8, 0.5] },
    fever: { yes: [0.5, 0.5, 0.5], no: [0.5, 0.5, 0.5] },
  },
  parents: {},
  roots: [],
}

const ev = (value: string, confidence = 1): Evidence => ({
  cough: { value, confidence },
})

const close = (got: number[], want: number[]) =>
  got.forEach((x, i) => expect(x).toBeCloseTo(want[i], 12))

test('no evidence leaves the prior', () => {
  close(posterior(kb, {}), kb.prior)
})

test('full-confidence update is plain Bayes', () => {
  // [0.5, 0.3, 0.2] * [0.9, 0.2, 0.5] = [0.45, 0.06, 0.10], sum 0.61
  close(posterior(kb, ev('yes')), [0.45 / 0.61, 0.06 / 0.61, 0.1 / 0.61])
})

test('half confidence blends towards a flat likelihood', () => {
  // L' = 0.5L + 0.5 = [0.95, 0.6, 0.75] -> [0.475, 0.18, 0.15], sum 0.805
  close(posterior(kb, ev('yes', 0.5)), [
    0.475 / 0.805,
    0.18 / 0.805,
    0.15 / 0.805,
  ])
})

test('zero confidence, unknown, and off-schema values change nothing', () => {
  close(posterior(kb, ev('yes', 0)), kb.prior)
  close(posterior(kb, ev('unknown')), kb.prior)
  close(posterior(kb, ev('banana')), kb.prior)
  close(posterior(kb, { nosuch: { value: 'yes', confidence: 1 } }), kb.prior)
})

test('posterior stays normalised', () => {
  const sum = posterior(kb, ev('yes', 0.7)).reduce((a, b) => a + b, 0)
  expect(sum).toBeCloseTo(1, 12)
})

test('re-answering replaces rather than compounds', () => {
  expect(posterior(kb, ev('yes'))).toEqual(posterior(kb, ev('yes')))
  close(
    posterior(kb, ev('no')),
    posterior(kb, { cough: { value: 'no', confidence: 1 } }),
  )
})

test('entropy of a uniform belief is log2(n) bits', () => {
  expect(entropyBits([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(2, 12)
  expect(entropyBits([1, 0, 0])).toBeCloseTo(0, 12)
})

test('evidence reduces entropy', () => {
  expect(entropy(kb, posterior(kb, ev('yes')))).toBeLessThan(
    entropy(kb, kb.prior),
  )
})

test('a flat feature has zero EIG, a discriminating one has more', () => {
  expect(eig(kb, kb.prior, 'fever')).toBeCloseTo(0, 12)
  expect(eig(kb, kb.prior, 'cough')).toBeGreaterThan(0)
})

test('selection takes the best unasked feature', () => {
  expect(selectFeature(kb, kb.prior, {})).toBe('cough')
  expect(selectFeature(kb, kb.prior, ev('yes'))).toBe('fever')
  const all = { ...ev('yes'), fever: { value: 'no', confidence: 1 } }
  expect(selectFeature(kb, kb.prior, all)).toBeNull()
})

test('stopping prefers budget over confidence', () => {
  const confident = [0.95, 0.03, 0.02]
  expect(shouldStop(confident, 0, { tau: 0.9, budget: 20 })).toBe('CONFIDENT')
  expect(shouldStop(confident, 20, { tau: 0.9, budget: 20 })).toBe(
    'MAX_QUESTIONS',
  )
  expect(shouldStop([0.5, 0.3, 0.2], 0, { tau: 0.9, budget: 20 })).toBeNull()
})

test('differential ranks by probability', () => {
  expect(differential(kb, [0.2, 0.5, 0.3], 2)).toEqual([
    { diseaseId: 'b', name: 'B', p: 0.5 },
    { diseaseId: 'c', name: 'C', p: 0.3 },
  ])
})
