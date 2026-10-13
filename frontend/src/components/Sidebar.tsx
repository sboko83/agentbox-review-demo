import type { SessionSummary } from '../api/sessions'
import { STAGE_LABEL } from './task'

export type Screen = 'chat' | 'experiment' | 'configuration' | 'library'

/** Session navigation stays separate from configuration presets. */
export function Sidebar({
  sessions,
  currentId,
  screen,
  busy,
  onOpen,
  onDrop,
  onNew,
  onRefresh,
  onExperiment,
  onConfiguration,
  onLibrary,
}: {
  sessions: SessionSummary[]
  currentId: string | null
  screen: Screen
  busy: boolean
  onOpen: (id: string) => void
  onDrop: (id: string) => void
  onNew: () => void
  onRefresh: () => void
  onExperiment: () => void
  onConfiguration: () => void
  onLibrary: () => void
}) {
  const sessionIds = new Set(sessions.map((session) => session.id))
  const byParent = new Map<string | null, SessionSummary[]>()
  for (const session of sessions) {
    const parent = session.parent_id && sessionIds.has(session.parent_id) ? session.parent_id : null
    byParent.set(parent, [...(byParent.get(parent) ?? []), session])
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="brand-mark">▌</span>
        agentbox
      </div>

      <button type="button" className="sidebar-new" onClick={onNew} disabled={busy}>
        + новая сессия
      </button>

      <nav className="sidebar-navigation" aria-label="Рабочие экраны">
        <button type="button" className={screen === 'configuration' ? 'sidebar-destination current' : 'sidebar-destination'} onClick={onConfiguration}>Конфигурация</button>
        <button type="button" className={screen === 'library' ? 'sidebar-destination current' : 'sidebar-destination'} onClick={onLibrary}>Библиотека</button>
      </nav>

      <div className="sidebar-group sessions-group">
        <div className="sidebar-group-head">
          <span>сессии</span>
          <button type="button" className="link" onClick={onRefresh} aria-label="обновить список">
            ↻
          </button>
        </div>
        {sessions.length === 0 ? (
          <p className="muted sidebar-empty">пока ни одной</p>
        ) : (
          <ul className="session-list">
            {(byParent.get(null) ?? []).map((item) => (
              <SessionNode
                key={item.id}
                item={item}
                childrenByParent={byParent}
                currentId={currentId}
                currentScreen={screen}
                busy={busy}
                onOpen={onOpen}
                onDrop={onDrop}
              />
            ))}
          </ul>
        )}
      </div>

      <button
        type="button"
        className={screen === 'experiment' ? 'sidebar-destination current' : 'sidebar-destination'}
        onClick={onExperiment}
      >
        матричный прогон
      </button>
    </aside>
  )
}

function SessionNode({
  item,
  childrenByParent,
  currentId,
  currentScreen,
  busy,
  onOpen,
  onDrop,
}: {
  item: SessionSummary
  childrenByParent: Map<string | null, SessionSummary[]>
  currentId: string | null
  currentScreen: Screen
  busy: boolean
  onOpen: (id: string) => void
  onDrop: (id: string) => void
}) {
  const children = childrenByParent.get(item.id) ?? []
  return (
    <li className="session-node">
      <div className={item.id === currentId && currentScreen === 'chat' ? 'session-row current' : 'session-row'}>
        <button
          type="button"
          className="session-open"
          disabled={busy}
          onClick={() => onOpen(item.id)}
        >
          <strong>{item.title || 'новая сессия'}</strong>
          <code>{item.spec.model.model}</code>
          <span className="muted">
            <span className="context-badge">{item.spec.context.mode}</span>
            {' · '}${item.cost_usd.toFixed(4)}
            {item.task_stage && ` · ${STAGE_LABEL[item.task_stage]}${item.task_paused ? ' ⏸' : ''}`}
            {item.fork_seq !== null && ` · от #${item.fork_seq}`}
          </span>
          {item.user_profile_name && (
            <span className="muted session-profile">профиль: {item.user_profile_name}</span>
          )}
        </button>
        <button
          type="button"
          className="session-drop"
          aria-label={`удалить сессию ${item.id}`}
          onClick={() => onDrop(item.id)}
        >
          ×
        </button>
      </div>
      {children.length > 0 && (
        <ul className="session-children">
          {children.map((child) => (
            <SessionNode
              key={child.id}
              item={child}
              childrenByParent={childrenByParent}
              currentId={currentId}
              currentScreen={currentScreen}
              busy={busy}
              onOpen={onOpen}
              onDrop={onDrop}
            />
          ))}
        </ul>
      )}
    </li>
  )
}
