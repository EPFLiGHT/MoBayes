import { create } from 'zustand'
import { parseKb } from '../engine/kb'
import type { KB } from '../engine/types'
import { ensureModel } from '../sensor/model'
import type { SensorState } from '../sensor/types'

type RuntimeState = {
  // Mirrors the model singleton's progress. The instance itself stays in
  // sensor/model.ts, since it is neither serialisable nor React's business.
  sensor: SensorState
  kbs: Record<string, KB>
  kbError: string | null

  loadModel: () => Promise<void>
  loadKb: (id: string) => Promise<KB | null>
}

export const useRuntime = create<RuntimeState>((set, get) => ({
  sensor: { kind: 'idle' },
  kbs: {},
  kbError: null,

  // Downloads ~230 MB on a cold cache, so this only ever runs from an
  // explicit user action. Concurrent calls are joined inside ensureModel.
  loadModel: async () => {
    try {
      await ensureModel((sensor) => set({ sensor }))
    } catch {
      // ensureModel has already reported the failure through onState.
    }
  },

  loadKb: async (id) => {
    const cached = get().kbs[id]
    if (cached) return cached

    try {
      const res = await fetch(`/kb/${id}.json`)
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const kb = parseKb(await res.json())
      set((s) => ({ kbs: { ...s.kbs, [id]: kb }, kbError: null }))
      return kb
    } catch (err) {
      set({ kbError: String(err) })
      return null
    }
  },
}))
