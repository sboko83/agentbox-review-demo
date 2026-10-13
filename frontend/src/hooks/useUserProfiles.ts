import { useCallback, useEffect, useState } from 'react'

import {
  createUserProfile,
  deleteUserProfile,
  getUserProfileSchema,
  listUserProfiles,
  setDefaultUserProfile,
  updateUserProfile,
  type JsonSchema,
  type UserProfile,
  type UserProfilePointer,
} from '../api/sessions'

/** Profiles and their form schema; writes throw so the form can show why. */
export function useUserProfiles() {
  const [profiles, setProfiles] = useState<UserProfilePointer[]>([])
  const [schema, setSchema] = useState<JsonSchema | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setProfiles(await listUserProfiles())
      setError(null)
    } catch (cause) {
      setError(String(cause))
    }
  }, [])

  const create = useCallback(async (profile: Partial<UserProfile> & { name: string }) => {
    const created = await createUserProfile(profile)
    await refresh()
    return created
  }, [refresh])

  const update = useCallback(async (id: string, profile: UserProfile) => {
    const updated = await updateUserProfile(id, profile)
    await refresh()
    return updated
  }, [refresh])

  const makeDefault = useCallback(async (id: string) => {
    await setDefaultUserProfile(id)
    await refresh()
  }, [refresh])

  const drop = useCallback(async (id: string) => {
    await deleteUserProfile(id)
    await refresh()
  }, [refresh])

  // oxlint-disable react/set-state-in-effect -- profiles arrive from the API after mount.
  useEffect(() => {
    void refresh()
    getUserProfileSchema().then(setSchema, (cause) => setError(String(cause)))
  }, [refresh])
  // oxlint-enable react/set-state-in-effect

  return { profiles, schema, error, refresh, create, update, makeDefault, drop }
}

export type UserProfileStore = ReturnType<typeof useUserProfiles>
