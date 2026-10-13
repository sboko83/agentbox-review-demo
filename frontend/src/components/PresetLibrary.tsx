import { useState } from 'react'

import type { JsonSchema, ModelInfo } from '../api/sessions'
import type { usePresets } from '../hooks/usePresets'
import { ConfigurationEditor } from './ConfigurationEditor'
import type { SpecValues } from './agentSpec'

export function PresetLibrary({ store, schema, models, current, onDirty }: {
  store: ReturnType<typeof usePresets>
  schema: JsonSchema
  models: ModelInfo[]
  current: SpecValues
  onDirty: (dirty: boolean) => void
}) {
  const [id, setId] = useState<string | null>(null)
  const selected = store.presets.find((item) => item.id === id) ?? store.presets[0]
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [error, setError] = useState('')
  const select = (next: string) => {
    if (dirty && !window.confirm('Отменить изменения пресета?')) return
    setId(next); setDirty(false); onDirty(false)
  }
  async function act(action: () => Promise<unknown>) {
    setBusy(true); setError('')
    try { await action() } catch (cause) { setError(String(cause)) } finally { setBusy(false) }
  }
  return (
    <section>
      <h2>Пресеты агента</h2>
      <p className="muted">Редактирование пресета не меняет открытые сессии. Для сессии выберите его над чатом.</p>
      <div className="library-toolbar">
        <label>Пресет<select value={selected?.id ?? ''} disabled={busy} onChange={(event) => select(event.target.value)}>
          {!selected && <option value="">Нет пресетов</option>}
          {store.presets.map((preset) => <option key={preset.id} value={preset.id}>{preset.is_default ? '★ ' : ''}{preset.name}</option>)}
        </select></label>
        <label>Имя<input value={name} placeholder="Название пресета" maxLength={100} onChange={(event) => setName(event.target.value)} /></label>
        <button type="button" disabled={busy || !name.trim() || dirty} onClick={() => void act(async () => {
          const created = await store.create(name.trim(), current); setId(created.id); setName('')
        })}>Создать из конфигурации сессии</button>
        {selected && <>
          <button type="button" disabled={busy || !name.trim() || dirty} onClick={() => void act(async () => { await store.update(selected.id, name.trim(), { ...selected.spec }); setName('') })}>Переименовать</button>
          <button type="button" disabled={busy || !name.trim() || dirty} onClick={() => void act(async () => { const copy = await store.create(name.trim(), { ...selected.spec }); setId(copy.id); setName('') })}>Дублировать</button>
          <button type="button" disabled={busy || selected.is_default} onClick={() => void act(() => store.makeDefault(selected.id))}>★ По умолчанию</button>
          <button type="button" disabled={busy} onClick={() => {
            if (window.confirm(`Удалить пресет «${selected.name}»?`)) void act(async () => { await store.drop(selected.id); setDirty(false); onDirty(false) })
          }}>Удалить</button>
        </>}
      </div>
      {(error || store.error) && <p className="error" role="alert">{error || store.error}</p>}
      {selected && <ConfigurationEditor key={selected.id} schema={schema} models={models} value={{ ...selected.spec }}
        description="Изменения сохраняются только в этот пресет. Настройки существующих сессий останутся прежними."
        scope={`Пресет: ${selected.name}`} disabled={busy} onApply={async (spec) => { await store.update(selected.id, selected.name, spec) }}
        onDirty={(next) => { setDirty(next); onDirty(next) }} />}
    </section>
  )
}
