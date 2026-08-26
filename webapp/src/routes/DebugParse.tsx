import {
  Button,
  Code,
  HStack,
  Input,
  NativeSelect,
  Stack,
  Text,
} from '@chakra-ui/react'
import { useState } from 'react'
import type { Feature } from '../engine/types'
import { createLocalSensor } from '../sensor/local'
import type { Parse } from '../sensor/types'
import { useRuntime } from '../stores/runtime'
import { KB_ID } from './DebugLayout'

const sensor = createLocalSensor()

export default function DebugParse() {
  const kb = useRuntime((s) => s.kbs[KB_ID])
  const ready = useRuntime((s) => s.sensor.kind === 'ready')

  const [featureId, setFeatureId] = useState('')
  const [reply, setReply] = useState(
    'I think I might have had a fever last night',
  )
  const [extra, setExtra] = useState(4)
  const [result, setResult] = useState<{ obs: Parse[]; ms: number } | null>(
    null,
  )
  const [busy, setBusy] = useState(false)

  const feature: Feature | undefined = kb?.features[featureId]

  // A real turn passes the asked feature plus the top few by EIG. There is no
  // session here, so neighbouring features stand in to exercise the same
  // multi-observation path.
  const candidates: Feature[] = feature
    ? [
        feature,
        ...Object.values(kb.features)
          .filter((f) => f.id !== feature.id)
          .slice(0, extra),
      ]
    : []

  const run = async () => {
    setBusy(true)
    const t0 = performance.now()
    try {
      const obs = await sensor.parse(reply, candidates)
      setResult({ obs, ms: Math.round(performance.now() - t0) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Stack gap={3} overflowY="auto">
      <NativeSelect.Root disabled={!kb} size="sm">
        <NativeSelect.Field
          value={featureId}
          onChange={(e) => setFeatureId(e.currentTarget.value)}
        >
          <option value="">select the asked feature…</option>
          {Object.values(kb?.features ?? {}).map((f) => (
            <option key={f.id} value={f.id}>
              {f.name}
            </option>
          ))}
        </NativeSelect.Field>
        <NativeSelect.Indicator />
      </NativeSelect.Root>

      <HStack>
        <Input
          value={reply}
          onChange={(e) => setReply(e.currentTarget.value)}
          placeholder="patient reply"
          size="sm"
        />
        <Input
          type="number"
          width="24"
          size="sm"
          value={extra}
          title="extra candidate features"
          onChange={(e) => setExtra(Number(e.currentTarget.value) || 0)}
        />
        <Button size="sm" onClick={run} disabled={!ready || !feature || busy}>
          Parse
        </Button>
      </HStack>

      {candidates.length > 0 && (
        <Text fontSize="xs" color="fg.muted" whiteSpace="pre-wrap">
          {candidates
            .map((f, i) => `${i + 1}. ${f.name} [${f.values.join(', ')}]`)
            .join('\n')}
        </Text>
      )}

      {result && (
        <Code p={3} whiteSpace="pre-wrap">
          {result.obs.length === 0
            ? 'no observations'
            : result.obs
                .map(
                  (o) =>
                    `${kb.features[o.featureId].name}\n  -> ${o.value}  ${o.label} (${o.confidence})`,
                )
                .join('\n')}
          {`\n\n${result.ms} ms`}
        </Code>
      )}
    </Stack>
  )
}
