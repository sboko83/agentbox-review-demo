import { useEffect, useState } from 'react'

import type {
  JsonSchema,
  MemoryProposal,
  SessionDetail,
  SessionSummary,
  UserProfile,
  UserProfilePointer,
} from '../api/sessions'
import type { UserProfileStore } from '../hooks/useUserProfiles'
import { nonNullVariant } from './agentSpec'
import { cost } from './tokens'

type FormValues = Record<string, string>

// The schema carries the machine values; the form speaks Russian.
const OPTION_LABEL: Record<string, string> = {
  ru: 'русский',
  en: 'английский',
  auto: 'язык вопроса',
  neutral: 'нейтральный',
  friendly: 'дружелюбный',
  strict: 'строгий',
  brief: 'кратко',
  normal: 'обычно',
  detailed: 'подробно',
  prose: 'связный текст',
  bullets: 'список',
  steps: 'шаги',
  none: 'без кода',
  minimal: 'минимум кода',
  full: 'полные примеры',
}

/** The user profile tab: which person the session talks to, and how. */
export function ProfilePanel({
  session,
  store,
  busy,
  onSessionProfile,
  onProfileChanged,
  onDecide,
  mode = 'library',
  sessions = [],
  onDirty,
}: {
  session: SessionDetail | null
  store: UserProfileStore
  busy: boolean
  onSessionProfile: (id: string | null) => Promise<void>
  onProfileChanged: (deletedProfileId?: string) => Promise<void>
  onDecide: (proposalId: number, decision: 'accept' | 'reject') => Promise<void>
  mode?: 'library' | 'session'
  sessions?: SessionSummary[]
  onDirty?: (dirty: boolean) => void
}) {
  const { profiles, schema } = store
  const sessionProfileId = session?.user_profile_id ?? null
  const [picked, setPicked] = useState<string | null>(null)
  const selected = profiles.find((item) => item.id === picked)
    ?? profiles.find((item) => item.id === sessionProfileId)
    ?? profiles.find((item) => item.is_default)
    ?? profiles[0]
    ?? null
  const [draft, setDraft] = useState<{ id: string; values: FormValues } | null>(null)
  const values = selected
    ? draft?.id === selected.id ? draft.values : encode(selected.profile)
    : null
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const editable = session?.editable ?? true
  const block = session?.context?.user_profile ?? null
  const discard = () => !draft || window.confirm('Отменить несохранённые изменения профиля?')
  const clearDraft = () => { setDraft(null); onDirty?.(false) }
  useEffect(() => {
    const leave = (event: BeforeUnloadEvent) => { if (draft) event.preventDefault() }
    window.addEventListener('beforeunload', leave)
    return () => window.removeEventListener('beforeunload', leave)
  }, [draft])

  async function act(action: () => Promise<void>) {
    if (working) return
    setWorking(true)
    setError(null)
    try {
      await action()
    } catch (cause) {
      setError(String(cause).replace(/^Error: /, ''))
    } finally {
      setWorking(false)
    }
  }

  function uniqueName(base: string): string {
    const taken = new Set(profiles.map((item) => item.name))
    if (!taken.has(base)) return base
    let index = 2
    while (taken.has(`${base} ${index}`)) index += 1
    return `${base} ${index}`
  }

  const save = () => act(async () => {
    if (!selected || !schema || !values) return
    await store.update(selected.id, decode(schema, values))
    clearDraft()
    await onProfileChanged()
  })

  const create = () => act(async () => {
    if (!discard()) return
    const created = await store.create({ name: uniqueName('Новый профиль') })
    setPicked(created.id)
    clearDraft()
  })

  // A copy is a POST with a new name: the server owns uniqueness and says 409.
  const duplicate = () => act(async () => {
    if (!selected || !schema || !values) return
    const copy = await store.create({
      ...decode(schema, values),
      name: uniqueName(`${selected.name} — копия`),
    })
    setPicked(copy.id)
    clearDraft()
  })

  const makeDefault = () => act(async () => {
    if (selected) await store.makeDefault(selected.id)
  })

  // Accepting writes a new profile version, so the list must re-read it.
  const decide = (proposal: MemoryProposal, decision: 'accept' | 'reject') => act(async () => {
    await onDecide(proposal.id, decision)
    if (decision === 'accept') {
      clearDraft()
      await store.refresh()
    }
  })

  const drop = () => act(async () => {
    if (!selected || !window.confirm(`Удалить профиль «${selected.name}»?`)) return
    await store.drop(selected.id)
    setPicked(null)
    clearDraft()
    await onProfileChanged(selected.id)
  })

  return (
    <>
      {mode === 'session' && <section className="panel-section preset-editor">
        <h3>профиль сессии</h3>
        {session ? (
          <label className="spec-field" htmlFor="session-profile">
            <span>Кому отвечает эта сессия</span>
            <select
              id="session-profile"
              value={sessionProfileId ?? ''}
              disabled={!editable || busy || working}
              onChange={(event) => void act(() => onSessionProfile(event.target.value || null))}
            >
              <option value="">без профиля</option>
              {profiles.map((item) => (
                <option key={item.id} value={item.id}>{profileLabel(item)}</option>
              ))}
            </select>
          </label>
        ) : (
          <p className="muted">новая сессия получит профиль, выбранный над чатом</p>
        )}
        {session && (
          <p className="muted">
            {block
              ? `блок профиля в следующем вызове · ${cost(block.payload_tokens, session.context?.estimated_tokens ?? 0)}`
              : 'блок профиля не отправляется'}
          </p>
        )}
        {block && (
          <details>
            <summary>текст блока</summary>
            <pre>{block.content}</pre>
          </details>
        )}
      </section>}

      {mode === 'session' && session && (
        <section className="panel-section preset-editor">
          <h3>кандидаты роутера</h3>
          {!session.memory.profile.router ? (
            <p className="muted">
              автообучения нет: роутер памяти выключен. Включите рабочий или
              долговременный слой в конфигурации, и просьбы вида «впредь отвечай
              по-английски» станут кандидатами.
            </p>
          ) : !session.memory.profile.enabled ? (
            <p className="muted">у сессии нет профиля — предлагать изменения некуда</p>
          ) : session.memory.profile.proposals.length === 0 ? (
            <p className="muted">
              кандидатов нет. Роутер предлагает изменение только по явной просьбе, и
              профиль меняется лишь после «принять».
            </p>
          ) : (
            <ul className="profile-list">
              {session.memory.profile.proposals.map((proposal) => (
                <li key={proposal.id} className="profile-proposal">
                  <span>
                    <strong>{schema?.properties?.[proposal.key]?.title ?? proposal.key}</strong>
                    {' → '}{proposalValue(proposal.value)}
                    <span className="muted"> · #{proposal.source_seq}</span>
                  </span>
                  <span className="preset-actions">
                    <button
                      type="button"
                      disabled={busy || working || !editable}
                      onClick={() => void decide(proposal, 'accept')}
                    >
                      принять
                    </button>
                    <button
                      type="button"
                      className="link"
                      disabled={busy || working || !editable}
                      aria-label={`отклонить ${proposal.key}`}
                      onClick={() => void decide(proposal, 'reject')}
                    >
                      отклонить
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {mode === 'library' && <><section className="panel-section preset-editor">
        <h3>профили</h3>
        {profiles.length === 0 ? (
          <p className="muted">профилей пока нет</p>
        ) : (
          <ul className="profile-list">
            {profiles.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  className={item.id === selected?.id ? 'profile-row current' : 'profile-row'}
                  onClick={() => { if (discard()) { clearDraft(); setPicked(item.id); setError(null) } }}
                >
                  <span>{profileLabel(item)}</span>
                  {item.id === sessionProfileId && <span className="muted">эта сессия</span>}
                </button>
              </li>
            ))}
          </ul>
        )}
        <div className="preset-actions">
          <button type="button" onClick={() => void create()} disabled={working}>создать</button>
          {selected && (
            <>
              <button type="button" onClick={() => void duplicate()} disabled={working}>
                дублировать
              </button>
              {!selected.is_default && (
                <button type="button" onClick={() => void makeDefault()} disabled={working}>
                  ★ по умолчанию
                </button>
              )}
              <button type="button" onClick={() => void drop()} disabled={working}>удалить</button>
            </>
          )}
        </div>
        {(error || store.error) && <p className="error">{error || store.error}</p>}
      </section>

      {selected && values && (
        <section className="panel-section preset-editor">
          <h3>{selected.name}</h3>
          <p className="scope-notice">Общий профиль. Сохранение повлияет на следующие ответы во всех использующих его сессиях.</p>
          <p className="muted">Сессии: {sessions.filter((item) => item.user_profile_id === selected.id).map((item) => item.title || item.id.slice(0, 8)).join(', ') || 'нет в списке'}.</p>
          {schema ? (
            <>
              {Object.entries(schema.properties ?? {}).map(([name, field]) => (
                <ProfileField
                  key={name}
                  name={name}
                  field={field}
                  value={values[name] ?? ''}
                  disabled={working}
                  onChange={(value) => { setPicked(selected.id); onDirty?.(true); setDraft({
                    id: selected.id,
                    values: { ...values, [name]: value },
                  }) }}
                />
              ))}
              <div className="preset-actions">
                <button
                  type="button"
                  onClick={() => void save()}
                  disabled={working || draft?.id !== selected.id}
                >
                  сохранить
                </button>
                {draft?.id === selected.id && (
                  <button type="button" onClick={clearDraft} disabled={working}>
                    отменить
                  </button>
                )}
              </div>
              <small className="muted">версия {selected.version_hash.slice(0, 12)}</small>
            </>
          ) : (
            <p className="muted">загружаю схему профиля…</p>
          )}
        </section>
      )}</>}
    </>
  )
}

function ProfileField({
  name,
  field: raw,
  value,
  disabled,
  onChange,
}: {
  name: string
  field: JsonSchema
  value: string
  disabled: boolean
  onChange: (value: string) => void
}) {
  const field = nonNullVariant(raw)
  const id = `profile-${name}`
  const label = (
    <span className="field-label">
      {raw.title ?? name}
      {raw.description && <span className="muted">— {raw.description}</span>}
    </span>
  )
  let control
  if (field.enum) {
    control = (
      <select id={id} value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
        {field.enum.map((option) => (
          <option key={option} value={option}>{OPTION_LABEL[option] ?? option}</option>
        ))}
      </select>
    )
  } else if (field.type === 'array') {
    control = (
      <textarea
        id={id}
        rows={4}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    )
  } else if (field.type === 'integer' || field.type === 'number') {
    control = (
      <input
        id={id}
        type="number"
        min={field.minimum}
        max={field.maximum}
        step={1}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    )
  } else if ((field.maxLength ?? 0) > 200) {
    control = (
      <textarea
        id={id}
        rows={2}
        maxLength={field.maxLength}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    )
  } else {
    control = (
      <input
        id={id}
        maxLength={field.maxLength}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    )
  }
  return (
    <label className="spec-field" htmlFor={id}>
      {label}
      {control}
    </label>
  )
}

function proposalValue(raw: string): string {
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch {
    return raw
  }
  if (value === null) return 'пусто'
  if (Array.isArray(value)) return value.length ? value.join('; ') : 'пустой список'
  return OPTION_LABEL[String(value)] ?? String(value)
}

function profileLabel(item: UserProfilePointer): string {
  return `${item.is_default ? '★ ' : ''}${item.name}`
}

/** Every value becomes text: a list is one rule per line, null is empty. */
function encode(profile: UserProfile): FormValues {
  return Object.fromEntries(
    Object.entries(profile).map(([name, value]) => [
      name,
      Array.isArray(value) ? value.join('\n') : value === null ? '' : String(value),
    ]),
  )
}

/**
 * Text back into the profile. Checks stay on the server: an invalid number is
 * sent as it is typed and comes back as a 422 naming the field.
 */
function decode(schema: JsonSchema, values: FormValues): UserProfile {
  return Object.fromEntries(
    Object.entries(schema.properties ?? {}).map(([name, raw]) => {
      const field = nonNullVariant(raw)
      const value = values[name] ?? ''
      if (field.type === 'array') return [name, value.split('\n')]
      if (field.type === 'integer' || field.type === 'number') {
        if (value.trim() === '') return [name, null]
        const number = Number(value)
        return [name, Number.isNaN(number) ? value : number]
      }
      return [name, value]
    }),
  ) as unknown as UserProfile
}
