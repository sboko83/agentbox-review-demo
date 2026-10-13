import { useEffect, useState } from 'react'

import { getAgentSpecSchema, getModels, type JsonSchema, type ModelInfo } from '../api/sessions'

/**
 * The AgentSpec schema and the model catalog: loaded once, shared by the
 * agent panel and the matrix screen. Both live above the screens because a
 * screen switch must not re-fetch them.
 */
export function useSettings() {
  const [schema, setSchema] = useState<JsonSchema | null>(null)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let disposed = false
    const load = async () => {
      try {
        const loaded = await getAgentSpecSchema()
        if (disposed) return
        // oxlint-disable-next-line react/set-state-in-effect -- settings arrive from the API after mount.
        setSchema(loaded)
      } catch (cause) {
        if (!disposed) setError(String(cause))
        return
      }
      try {
        const catalog = await getModels()
        if (!disposed) setModels(catalog)
      } catch {
        // A missing key must not prevent editing the schema. The model remains
        // a normal schema field until the catalog becomes available.
      }
    }
    void load()
    return () => {
      disposed = true
    }
  }, [])

  return { schema, models, error }
}
