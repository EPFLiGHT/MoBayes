import {
  Badge,
  Button,
  Flex,
  Heading,
  HStack,
  Progress,
  Stack,
  Text,
} from '@chakra-ui/react'
import { NavLink, Outlet } from 'react-router-dom'
import { useRuntime } from '../stores/runtime'

export const KB_ID = 'ddxplus'

const mb = (bytes: number) => (bytes / 1e6).toFixed(0)

function SensorStatus() {
  const state = useRuntime((s) => s.sensor)

  if (state.kind === 'downloading') {
    const pct = state.total ? (state.received / state.total) * 100 : 0
    return (
      <Stack gap={1} flex="1">
        <Text fontSize="sm" color="fg.muted">
          downloading {mb(state.received)} / {mb(state.total)} MB
        </Text>
        <Progress.Root value={pct} size="sm">
          <Progress.Track>
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>
      </Stack>
    )
  }

  const tone =
    state.kind === 'ready' ? 'green' : state.kind === 'error' ? 'red' : 'gray'
  return (
    <HStack>
      <Badge colorPalette={tone}>{state.kind}</Badge>
      {state.kind === 'error' && (
        <Text fontSize="sm" color="fg.error">
          {state.message}
        </Text>
      )}
    </HStack>
  )
}

// Model and KB loading are shared by every debug panel, so they live here and
// the panels below assume both are available.
export default function DebugLayout() {
  const { sensor, kbs, kbError, loadModel, loadKb } = useRuntime()
  const kb = kbs[KB_ID]

  return (
    <Flex direction="column" h="100dvh" p={6} gap={5} maxW="4xl" mx="auto">
      <Heading size="lg">Debug</Heading>

      <Stack gap={3} borderWidth="1px" borderRadius="md" p={4}>
        <HStack gap={4} wrap="wrap">
          <Button
            size="sm"
            onClick={loadModel}
            disabled={sensor.kind !== 'idle' && sensor.kind !== 'error'}
          >
            Load model (~230 MB)
          </Button>
          <SensorStatus />
        </HStack>

        <HStack gap={4} wrap="wrap">
          <Button size="sm" onClick={() => loadKb(KB_ID)} disabled={!!kb}>
            Load {KB_ID}
          </Button>
          <Text fontSize="sm" color={kbError ? 'fg.error' : 'fg.muted'}>
            {kb
              ? `${kb.diseases.length} diseases, ${Object.keys(kb.features).length} features, ${kb.roots.length} roots`
              : (kbError ?? 'not loaded')}
          </Text>
        </HStack>
      </Stack>

      <HStack gap={2}>
        {[
          ['parse', 'Parse'],
          ['chat', 'Chat'],
        ].map(([to, label]) => (
          <NavLink key={to} to={`/debug/${to}`}>
            {({ isActive }) => (
              <Button size="sm" variant={isActive ? 'solid' : 'ghost'}>
                {label}
              </Button>
            )}
          </NavLink>
        ))}
      </HStack>

      <Outlet />
    </Flex>
  )
}
