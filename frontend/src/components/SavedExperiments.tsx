import { useState } from 'react'

import type { AgentSpec, SessionDetail, UserProfile } from '../api/sessions'
import type { SavedExperiment } from './experimentRecipe'

const STORAGE_KEY = 'agentbox.saved-experiments.v1'

function readSaved(): SavedExperiment[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
    if (!Array.isArray(stored)) return []
    return stored.filter((item): item is SavedExperiment => item && typeof item.id === 'string'
      && typeof item.name === 'string' && item.spec?.model && item.spec?.memory && 'source' in item)
  } catch { return [] }
}

export function SavedExperiments({ spec, profile, session, busy, onStart }: {
  spec: AgentSpec
  profile: UserProfile | null
  session: SessionDetail | null
  busy: boolean
  onStart: (recipe: SavedExperiment) => Promise<void>
}) {
  const [saved, setSaved] = useState(readSaved)
  const [name, setName] = useState('')
  const [checkpoint, setCheckpoint] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [working, setWorking] = useState(false)
  function persist(next: SavedExperiment[]) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); setSaved(next); setError(''); return true }
    catch { setError('Не удалось сохранить: хранилище браузера недоступно или заполнено.'); return false }
  }
  async function start(recipe: SavedExperiment) {
    setWorking(true); setError('')
    try { await onStart(recipe) } catch (cause) { setError(String(cause)) }
    finally { setWorking(false) }
  }
  function save() {
    const point = session?.checkpoints.find((item) => String(item.seq) === checkpoint)
    if (checkpoint && !point) { setError('Контрольная точка недоступна. Выберите условия старта заново.'); return }
    const versions = session?.memory.longterm.versions ?? []
    const latest = versions.reduce<(typeof versions)[number] | undefined>((last, next) => !last || next.version > last.version ? next : last, undefined)
    const recipe: SavedExperiment = {
      id: crypto.randomUUID(), name: name.trim(), spec: structuredClone(spec),
      profile: profile ? structuredClone(profile) : null,
      source: point ? { sessionId: point.session_id, seq: point.seq, name: point.name } : null,
      longterm: point && latest ? Object.fromEntries(Object.entries(latest.items).map(([key, item]) => [key, { value: item.value, kind: item.kind }])) : {},
    }
    if (persist([...saved, recipe])) { setName(''); setNotice('Конфигурация сохранена в этом браузере.') }
  }
  return (
    <section className="saved-experiments">
      <h2>Сохранённые эксперименты</h2>
      <p className="muted">Хранятся в этом браузере. Снимки настроек и профиля независимы от последующих изменений библиотеки. Каждый запуск получает отдельную долговременную память и копию профиля.</p>
      <div className="library-toolbar">
        <label>Название эксперимента<input value={name} maxLength={100} onChange={(event) => setName(event.target.value)} /></label>
        <label>Начальные условия<select value={checkpoint} onChange={(event) => setCheckpoint(event.target.value)}>
          <option value="">Чистая сессия и память</option>
          {session?.checkpoints.map((point) => <option key={point.seq} value={point.seq}>Точка #{point.seq}: {point.name}</option>)}
        </select></label>
        <button type="button" disabled={busy || working || !name.trim()} onClick={save}>Сохранить текущую конфигурацию</button>
      </div>
      <p className="muted">Для старта из точки: диалог, рабочая память и задача берутся на выбранном сообщении; долговременная память фиксируется сейчас. Исходную сессию нужно сохранить.</p>
      <div className="experiment-cards">
        {saved.map((recipe) => <article key={recipe.id} className="experiment-card">
          <h3>{recipe.name}</h3><p>{recipe.spec.model.model} · {recipe.profile?.name ?? 'без профиля'}</p>
          <p className="muted">{recipe.source ? `Точка #${recipe.source.seq}: ${recipe.source.name}` : 'Чистая сессия и память'}</p>
          <button type="button" disabled={busy || working} onClick={() => void start(recipe)}>Открыть эксперимент</button>
          <button type="button" disabled={working} onClick={() => { if (window.confirm(`Удалить эксперимент «${recipe.name}»?`)) persist(saved.filter((item) => item.id !== recipe.id)) }}>Удалить</button>
        </article>)}
      </div>
      <h3>Учебные сценарии</h3>
      <div className="experiment-cards">
        <article className="experiment-card"><h4>Персонализация</h4><p>Выберите профиль, задайте одинаковый вопрос в чистых сессиях. Для таблицы результатов используйте матричный прогон.</p>
          <button type="button" disabled={busy || working} onClick={() => void start({ id: 'profiles', name: 'Персонализация', spec, profile, source: null, longterm: {} })}>Подготовить чистую сессию</button>
        </article>
        <article className="experiment-card"><h4>Пауза и восстановление</h4><p>Окно — два сообщения, трекер включён. После первого сообщения создайте задачу, утвердите план, поставьте паузу и продолжите после посторонних вопросов.</p>
          <button type="button" disabled={busy || working} onClick={() => void start({ id: 'pause', name: 'Пауза и восстановление',
            spec: { ...spec, context: { mode: 'window', keep_last_messages: 2, window_tokens: spec.context.window_tokens, overflow: spec.context.overflow, budget_usd: spec.context.budget_usd }, task: { ...spec.task, tracker_enabled: true } },
            profile, source: null, longterm: {} })}>Подготовить сценарий</button>
        </article>
      </div>
      {notice && <p role="status">{notice}</p>}{error && <p className="error" role="alert">{error}</p>}
    </section>
  )
}
