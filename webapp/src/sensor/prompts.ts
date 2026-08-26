import { CONFIDENCE_WEIGHTS } from '../engine/types'
import type { Feature } from '../engine/types'

const LABELS = Object.keys(CONFIDENCE_WEIGHTS).join(' | ')

// Candidates are numbered rather than named: feature ids like `f_ant_e_125`
// are opaque to a small model and invite hallucinated ids, while an index
// maps back exactly.
const numbered = (features: Feature[]) =>
  features
    .map((f, i) => `${i + 1}. ${f.name} [${f.values.join(', ')}]`)
    .join('\n')

const PARSE_SYSTEM = `You extract structured observations from a patient's reply.

Rules:
- Only report what the patient actually said. Never infer or assume.
- Report a question only if the reply addresses it. Silence is not a "no".
- The patient may volunteer things you did not ask about. Report those too.
- Pick confidence from how firmly they said it: ${LABELS}.
  Hedges like "I think", "maybe" or "sort of" lower it.

Reply with a JSON array and nothing else. Use [] if the reply says nothing useful.`

// Demonstrations do the work the schema cannot: they show hedging mapped to
// confidence, negation, volunteered evidence, and the empty case.
const DEMOS: { user: string; assistant: string }[] = [
  {
    user: `Questions:
1. Do you have a cough? [yes, no]

Patient: "Yes, constantly."`,
    assistant: `[{"q":1,"value":"yes","confidence":"very_likely"}]`,
  },
  {
    user: `Questions:
1. Do you have a fever? [yes, no]
2. Do you have a cough that produces sputum? [yes, no]

Patient: "I think I had a fever last night, and I've been coughing up green stuff for days."`,
    assistant: `[{"q":1,"value":"yes","confidence":"uncertain"},{"q":2,"value":"yes","confidence":"very_likely"}]`,
  },
  {
    user: `Questions:
1. Have you travelled recently? [yes, no]
2. Pain intensity [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, absent]

Patient: "No travel at all. Sorry, what do you mean by intensity?"`,
    assistant: `[{"q":1,"value":"no","confidence":"very_likely"}]`,
  },
]

const parseUser = (reply: string, candidates: Feature[]) =>
  `Questions:\n${numbered(candidates)}\n\nPatient: "${reply}"`

export function parseMessages(reply: string, candidates: Feature[]) {
  return [
    { role: 'system' as const, content: PARSE_SYSTEM },
    ...DEMOS.flatMap((d) => [
      { role: 'user' as const, content: d.user },
      { role: 'assistant' as const, content: d.assistant },
    ]),
    { role: 'user' as const, content: parseUser(reply, candidates) },
  ]
}

// Sent after an unusable reply. Dropping the demonstrations shortens the
// prompt and tends to shake a small model out of a bad continuation.
export function retryMessages(reply: string, candidates: Feature[]) {
  return [
    {
      role: 'system' as const,
      content: `${PARSE_SYSTEM}\n\nYour previous reply was not valid JSON. Output only the array.`,
    },
    { role: 'user' as const, content: parseUser(reply, candidates) },
  ]
}

const ASK_SYSTEM = `You are a clinician taking a history. Ask the one question given, in a single short, plain sentence.
Never mention codes, probabilities, or that a system chose the question.`

export function askMessages(feature: Feature, tone: string) {
  return [
    { role: 'system' as const, content: ASK_SYSTEM },
    {
      role: 'user' as const,
      content: `Question: ${feature.name}\nYour current certainty: ${tone}`,
    },
  ]
}
