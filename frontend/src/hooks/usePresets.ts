import { useCallback, useEffect, useState } from 'react'

import {
  createPreset,
  deletePreset,
  listPresets,
  setDefaultPreset,
  updatePreset,
  type Preset,
} from '../api/sessions'

export function usePresets() {
  const [presets, setPresets] = useState<Preset[]>([])
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setPresets(await listPresets())
      setError(null)
    } catch (cause) {
      setError(String(cause))
    }
  }, [])

  const create = useCallback(async (name: string, spec: Record<string, unknown>) => {
    try {
      const preset = await createPreset({ name, spec })
      await refresh()
      return preset
    } catch (cause) {
      setError(String(cause))
      throw cause
    }
  }, [refresh])

  const update = useCallback(async (
    id: string,
    name: string,
    spec: Record<string, unknown>,
  ) => {
    try {
      const preset = await updatePreset(id, { name, spec })
      await refresh()
      return preset
    } catch (cause) {
      setError(String(cause))
      throw cause
    }
  }, [refresh])

  const makeDefault = useCallback(async (id: string) => {
    try {
      const preset = await setDefaultPreset(id)
      await refresh()
      return preset
    } catch (cause) {
      setError(String(cause))
      throw cause
    }
  }, [refresh])

  const drop = useCallback(async (id: string) => {
    try {
      await deletePreset(id)
      await refresh()
    } catch (cause) {
      setError(String(cause))
      throw cause
    }
  }, [refresh])

  // oxlint-disable react/set-state-in-effect -- presets arrive from the API after mount.
  useEffect(() => { void refresh() }, [refresh])
  // oxlint-enable react/set-state-in-effect

  return { presets, error, refresh, create, update, makeDefault, drop }
}
