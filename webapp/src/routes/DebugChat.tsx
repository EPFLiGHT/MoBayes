import { Button, HStack, Text } from '@chakra-ui/react'
import { useState } from 'react'
import { Chat, type ChatMessage } from '../components/Chat'
import { loadedModel } from '../sensor/model'
import { useRuntime } from '../stores/runtime'

// Free-form chat straight against the model, bypassing prompts and parsing.
// This is how the raw model is judged: fluency, speed, and whether it stays
// coherent over several turns.
export default function DebugChat() {
  const ready = useRuntime((s) => s.sensor.kind === 'ready')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [pending, setPending] = useState(false)

  const send = async (text: string) => {
    const asked: ChatMessage = { id: crypto.randomUUID(), role: 'user', text }
    const history = [...messages, asked]
    setMessages(history)
    setPending(true)

    const started = performance.now()
    try {
      const wllama = loadedModel()
      if (!wllama) throw new Error('model not loaded')

      const res = await wllama.createChatCompletion({
        messages: history.map((m) => ({ role: m.role, content: m.text })),
        temperature: 0.7,
        max_tokens: 512,
      })
      const ms = Math.round(performance.now() - started)

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: res.choices[0]?.message?.content?.trim() ?? '(empty)',
          meta: (
            <Text fontSize="xs" color="fg.muted">
              {ms} ms · {res.usage?.completion_tokens ?? '?'} tokens
            </Text>
          ),
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'assistant', text: String(err) },
      ])
    } finally {
      setPending(false)
    }
  }

  return (
    <>
      <HStack>
        <Button
          size="xs"
          variant="ghost"
          onClick={() => setMessages([])}
          disabled={messages.length === 0}
        >
          Clear
        </Button>
      </HStack>
      <Chat
        messages={messages}
        pending={pending}
        disabled={!ready || pending}
        placeholder={
          ready ? 'Say something to the model.' : 'Load the model first.'
        }
        onSend={send}
      />
    </>
  )
}
