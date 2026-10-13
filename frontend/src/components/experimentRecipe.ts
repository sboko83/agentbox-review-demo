import {
  createUserProfile, deleteSession, deleteUserProfile, forkSession, updateSessionMemory,
  setSessionUserProfile, type AgentSpec, type MemoryItemInput, type SessionDetail, type UserProfile,
} from '../api/sessions'

export interface SavedExperiment {
  id: string
  name: string
  spec: AgentSpec
  profile: UserProfile | null
  source: { sessionId: string; seq: number; name: string } | null
  longterm: Record<string, MemoryItemInput>
}

export function isolatedSpec(spec: AgentSpec): AgentSpec {
  return { ...spec, memory: { ...spec.memory, longterm_scope: `experiment-${crypto.randomUUID()}` } }
}

/** Every replay owns its profile and long-term scope; mutable library pointers are never replayed. */
export async function prepareExperiment(recipe: SavedExperiment): Promise<{
  spec: AgentSpec; profileId: string | null; session: SessionDetail | null
}> {
  const spec = isolatedSpec(recipe.spec)
  let profileId: string | null = null
  let sessionId: string | null = null
  try {
    if (recipe.profile) {
      const profile = await createUserProfile({
        ...recipe.profile, name: `${recipe.profile.name.slice(0, 65)} · ${crypto.randomUUID().slice(0, 8)}`,
      })
      profileId = profile.id
    }
    if (!recipe.source) return { spec, profileId, session: null }
    const branch = await forkSession(recipe.source.sessionId, recipe.source.seq, recipe.name, { ...spec })
    sessionId = branch.id
    if (spec.memory.longterm_enabled) await updateSessionMemory(branch.id, 'longterm', recipe.longterm)
    const session = await setSessionUserProfile(branch.id, profileId)
    return { spec, profileId, session }
  } catch (error) {
    // Only artifacts created by this failed replay are removed.
    if (sessionId) await deleteSession(sessionId).catch(() => undefined)
    if (profileId) await deleteUserProfile(profileId).catch(() => undefined)
    throw error
  }
}
