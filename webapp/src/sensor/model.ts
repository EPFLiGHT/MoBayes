// Imported from esm/ rather than the package root: wllama's `main` field
// points at an index.js that is not shipped, so the root entry resolves to
// their TypeScript source and gets compiled as part of this project.
import { Wllama } from '@wllama/wllama/esm/index.js'
import wasmUrl from '@wllama/wllama/esm/wasm/wllama.wasm?url'
import { MODEL, type SensorState } from './types'

type OnState = (state: SensorState) => void

// The model is a ~230 MB WASM-backed singleton: one at a time, expensive to
// build, and it has to survive route changes. It deliberately lives outside
// the stores, which hold only its status.
let instance: Wllama | null = null
let inflight: Promise<Wllama> | null = null

export function loadedModel(): Wllama | null {
  return instance
}

// Never called on mount. The download is large enough that it has to be an
// explicit user action.
export function ensureModel(onState: OnState): Promise<Wllama> {
  if (instance) return Promise.resolve(instance)
  // A second caller joins the first rather than starting another download.
  // React StrictMode double-invokes effects in dev, so this is load-bearing.
  if (inflight) return inflight

  onState({ kind: 'downloading', received: 0, total: 0 })

  inflight = (async () => {
    // allowOffline lets a cached model load with no network at all.
    const wllama = new Wllama({ default: wasmUrl }, { allowOffline: true })
    await wllama.loadModelFromHF(
      { repo: MODEL.repo, file: MODEL.file },
      {
        progressCallback: ({ loaded, total }) =>
          onState(
            loaded < total
              ? { kind: 'downloading', received: loaded, total }
              : { kind: 'loading' },
          ),
      },
    )
    instance = wllama
    onState({ kind: 'ready' })
    return wllama
  })()

  inflight.catch((err) => {
    inflight = null
    onState({ kind: 'error', message: String(err) })
  })

  return inflight
}

export async function disposeModel(): Promise<void> {
  const previous = instance
  instance = null
  inflight = null
  await previous?.exit()
}
