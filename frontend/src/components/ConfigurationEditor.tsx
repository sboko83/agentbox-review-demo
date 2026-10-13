import { useEffect, useId, useState } from 'react'

import type { JsonSchema, ModelInfo } from '../api/sessions'
import { AgentSpecForm } from './AgentSpecForm'
import { decodeSpec, SECTION_TITLES, type SpecValues } from './agentSpec'

/** Drafts stay mounted while navigating; only Apply changes the active configuration. */
export function ConfigurationEditor({ schema, models, value, scope, disabled, onApply, onDirty, description }: {
  schema: JsonSchema
  models: ModelInfo[]
  value: SpecValues
  scope: string
  disabled: boolean
  onApply: (spec: SpecValues) => Promise<void>
  onDirty?: (dirty: boolean) => void
  description?: string
}) {
  const [draft, setDraft] = useState<SpecValues | null>(null)
  const formId = useId()
  const [section, setSection] = useState('model')
  const [search, setSearch] = useState('')
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const values = draft ?? value
  const changed = Object.keys(schema.properties ?? {}).filter((key) =>
    JSON.stringify(values[key]) !== JSON.stringify(value[key]),
  )
  const dirty = changed.length > 0

  useEffect(() => {
    onDirty?.(dirty)
    const leave = (event: BeforeUnloadEvent) => { if (dirty) event.preventDefault() }
    window.addEventListener('beforeunload', leave)
    return () => window.removeEventListener('beforeunload', leave)
  }, [dirty, onDirty])

  async function apply() {
    setSaving(true)
    setError('')
    try {
      await onApply(decodeSpec(schema, values))
      setDraft(null)
      setNotice('Применено')
    } catch (cause) {
      setError(String(cause).replace(/^Error: /, ''))
    } finally { setSaving(false) }
  }

  return (
    <div className="configuration-editor">
      <div className="configuration-heading">
        <div><h2>{scope}</h2><p className="muted">{description ?? 'Изменения действуют со следующего сообщения. Пресеты сохраняются отдельно.'}</p></div>
        {disabled && <span className="read-only">изменение недоступно</span>}
      </div>
      <label className="settings-search">
        <span>Поиск настроек</span>
        <input type="search" placeholder="Например: бюджет, модель, timeout" value={search} onChange={(event) => setSearch(event.target.value)} />
      </label>
      <div className="configuration-layout">
        <nav className="configuration-nav" aria-label="Разделы конфигурации">
          {Object.keys(schema.properties ?? {}).map((key) => (
            <button type="button" key={key} aria-current={!search && section === key ? 'page' : undefined}
              onClick={() => { setSection(key); setSearch('') }}>
              {SECTION_TITLES[key] ?? key}{changed.includes(key) && <span aria-label="изменено"> •</span>}
            </button>
          ))}
        </nav>
        <div className="configuration-fields">
          {search && <p className="muted">Совпадения во всех разделах, включая дополнительные параметры.</p>}
          <AgentSpecForm idPrefix={formId} schema={schema} values={values} models={models} section={section} search={search.trim()}
            disabled={disabled || saving} onChange={(name, next) => { setDraft({ ...values, [name]: next }); setNotice(''); setError('') }} />
        </div>
      </div>
      <footer className="configuration-actions">
        <div role="status">{dirty ? `Изменены: ${changed.map((key) => SECTION_TITLES[key] ?? key).join(', ')}` : notice || 'Нет неприменённых изменений'}</div>
        <button type="button" onClick={() => { setDraft(null); setError(''); setNotice('Изменения отменены') }} disabled={!dirty || saving}>Отменить</button>
        <button type="button" className="primary" onClick={() => void apply()} disabled={!dirty || disabled || saving}>{saving ? 'Применяю…' : 'Применить'}</button>
        {error && <p className="error" role="alert">{error}</p>}
      </footer>
    </div>
  )
}
