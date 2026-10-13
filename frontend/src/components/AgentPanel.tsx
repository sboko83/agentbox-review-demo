import { useEffect, useRef } from 'react'

import type { ModelInfo, SessionDetail } from '../api/sessions'
import type { UserProfileStore } from '../hooks/useUserProfiles'
import { ContextPanel } from './ContextPanel'
import { MemoryPanel, type MemoryActions } from './MemoryPanel'
import { ProfilePanel } from './ProfilePanel'

export type PanelTab = 'context' | 'memory' | 'profile'

/** An optional inspector for the open session, separate from configuration. */
export function AgentPanel({ session, models, busy, tab, onTab, onClose, memoryActions,
  profileStore, onSessionProfile, onProfileChanged }: {
  session: SessionDetail | null
  models: ModelInfo[]
  busy: boolean
  tab: PanelTab
  onTab: (tab: PanelTab) => void
  onClose: () => void
  memoryActions: MemoryActions
  profileStore: UserProfileStore
  onSessionProfile: (id: string | null) => Promise<void>
  onProfileChanged: () => Promise<void>
}) {
  const close = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    const previous = document.activeElement
    close.current?.focus()
    return () => { if (previous instanceof HTMLElement && previous.isConnected) previous.focus() }
  }, [])
  return (
    <aside className="panel" aria-label="Диагностика сессии" onKeyDown={(event) => { if (event.key === 'Escape') onClose() }}>
      <div className="panel-tabs">
        {(['context', 'memory', 'profile'] as const).map((key) => (
          <button type="button" key={key} className={tab === key ? 'tab current' : 'tab'} onClick={() => onTab(key)}>
            {{ context: 'контекст', memory: 'память', profile: 'профиль' }[key]}
          </button>
        ))}
        <button ref={close} type="button" className="link panel-close" onClick={onClose} aria-label="Закрыть диагностику">×</button>
      </div>
      <div className="panel-body">
        {tab === 'context' && <ContextPanel session={session} models={models} />}
        {tab === 'memory' && <MemoryPanel session={session} busy={busy} actions={memoryActions} />}
        {tab === 'profile' && <ProfilePanel mode="session" session={session} store={profileStore} busy={busy}
          onSessionProfile={onSessionProfile} onProfileChanged={onProfileChanged} onDecide={memoryActions.onDecide} />}
      </div>
    </aside>
  )
}
