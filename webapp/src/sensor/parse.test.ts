import { expect, test } from 'vitest'
import type { Feature } from '../engine/types'
import { parseObservations } from './parse'

const candidates: Feature[] = [
  { id: 'fever', name: 'Fever?', type: 'binary', values: ['yes', 'no'] },
  { id: 'cough', name: 'Cough?', type: 'binary', values: ['yes', 'no'] },
  {
    id: 'pain',
    name: 'Pain intensity',
    type: 'categorical',
    values: ['0', '1', '2', 'absent'],
  },
]

const parse = (text: string) => parseObservations(text, candidates)

test('reads a well-formed array', () => {
  expect(parse('[{"q":1,"value":"yes","confidence":"likely"}]')).toEqual([
    { featureId: 'fever', value: 'yes', label: 'likely', confidence: 0.8 },
  ])
})

test('keeps several observations, including unasked ones', () => {
  const got = parse(
    '[{"q":1,"value":"yes","confidence":"uncertain"},{"q":2,"value":"yes","confidence":"very_likely"}]',
  )
  expect(got?.map((o) => o.featureId)).toEqual(['fever', 'cough'])
  expect(got?.map((o) => o.confidence)).toEqual([0.5, 1.0])
})

test('digs the array out of surrounding prose or fences', () => {
  const wrapped = '```json\n[{"q":2,"value":"no","confidence":"likely"}]\n```'
  expect(parse(wrapped)?.[0].featureId).toBe('cough')
  expect(
    parse(
      'Sure! [{"q":2,"value":"no","confidence":"likely"}] Hope that helps',
    )?.[0].featureId,
  ).toBe('cough')
})

test('unusable output is null, an empty answer is []', () => {
  expect(parse('I could not tell.')).toBeNull()
  expect(parse('[{"q":1,')).toBeNull()
  expect(parse('{"q":1,"value":"yes","confidence":"likely"}')).toBeNull()
  expect(parse('[]')).toEqual([])
})

test('drops anything questionable rather than repairing it', () => {
  // out of range, off-schema value, unknown label, missing fields
  expect(parse('[{"q":9,"value":"yes","confidence":"likely"}]')).toEqual([])
  expect(parse('[{"q":1,"value":"maybe","confidence":"likely"}]')).toEqual([])
  expect(parse('[{"q":1,"value":"yes","confidence":"pretty_sure"}]')).toEqual(
    [],
  )
  expect(parse('[{"q":1,"value":"yes"}]')).toEqual([])
})

test('a contradicted feature keeps the first observation', () => {
  const got = parse(
    '[{"q":1,"value":"yes","confidence":"likely"},{"q":1,"value":"no","confidence":"likely"}]',
  )
  expect(got).toHaveLength(1)
  expect(got?.[0].value).toBe('yes')
})

test('numeric schema values are matched as strings', () => {
  expect(parse('[{"q":3,"value":"2","confidence":"likely"}]')?.[0].value).toBe(
    '2',
  )
  expect(parse('[{"q":3,"value":2,"confidence":"likely"}]')?.[0].value).toBe(
    '2',
  )
})
