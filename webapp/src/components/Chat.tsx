import {
  Box,
  Button,
  Flex,
  HStack,
  Spinner,
  Stack,
  Text,
  Textarea,
} from '@chakra-ui/react'
import { useEffect, useRef, useState, type ReactNode } from 'react'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  text: string
  // Rendered under the bubble. Lets a caller attach parse badges or timings
  // without this component knowing what they are.
  meta?: ReactNode
}

type Props = {
  messages: ChatMessage[]
  pending?: boolean
  disabled?: boolean
  placeholder?: string
  onSend: (text: string) => void
}

function Bubble({ message }: { message: ChatMessage }) {
  const mine = message.role === 'user'
  return (
    <Stack gap={1} alignSelf={mine ? 'flex-end' : 'flex-start'} maxW="80%">
      <Box
        px={4}
        py={2}
        borderRadius="xl"
        borderTopRightRadius={mine ? 'sm' : undefined}
        borderTopLeftRadius={mine ? undefined : 'sm'}
        bg={mine ? 'colorPalette.solid' : 'bg.muted'}
        color={mine ? 'colorPalette.contrast' : 'fg'}
        colorPalette="blue"
        whiteSpace="pre-wrap"
        wordBreak="break-word"
      >
        {message.text}
      </Box>
      {message.meta && (
        <Box alignSelf={mine ? 'flex-end' : 'flex-start'}>{message.meta}</Box>
      )}
    </Stack>
  )
}

export function Chat({
  messages,
  pending,
  disabled,
  placeholder,
  onSend,
}: Props) {
  const [draft, setDraft] = useState('')
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, pending])

  const send = () => {
    const text = draft.trim()
    if (!text || disabled) return
    setDraft('')
    onSend(text)
  }

  return (
    <Flex direction="column" flex="1" minH="0" gap={4}>
      <Stack flex="1" minH="0" overflowY="auto" gap={4} py={2}>
        {messages.length === 0 && !pending && (
          <Text color="fg.muted" fontSize="sm">
            {placeholder ?? 'No messages yet.'}
          </Text>
        )}
        {messages.map((m) => (
          <Bubble key={m.id} message={m} />
        ))}
        {pending && (
          <HStack alignSelf="flex-start" color="fg.muted" gap={2}>
            <Spinner size="xs" />
            <Text fontSize="sm">thinking…</Text>
          </HStack>
        )}
        <div ref={bottom} />
      </Stack>

      <HStack align="flex-end" gap={2}>
        <Textarea
          value={draft}
          onChange={(e) => setDraft(e.currentTarget.value)}
          // Enter sends; Shift+Enter breaks the line, as in every chat app.
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
          placeholder="Type a reply…"
          disabled={disabled}
          rows={1}
          resize="none"
          maxH="40"
        />
        <Button onClick={send} disabled={disabled || !draft.trim()}>
          Send
        </Button>
      </HStack>
    </Flex>
  )
}
