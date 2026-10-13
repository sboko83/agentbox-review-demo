import { useState } from 'react'

import type {
  InvariantCategory,
  InvariantRule,
  SessionDetail,
} from '../api/sessions'

const CATEGORY: Record<InvariantCategory, string> = {
  architecture: 'архитектура',
  decision: 'техническое решение',
  stack: 'стек',
  business_rule: 'бизнес-правило',
}

/** Session rules are edited as one optimistic, versioned draft. */
export function InvariantEditor({ session, busy, onApply, onDirty }: {
  session: SessionDetail | null
  busy: boolean
  onApply: (expectedHash: string, rules: InvariantRule[], replanTask: boolean) => Promise<void>
  onDirty: (dirty: boolean) => void
}) {
  const initial = session?.invariants.rules ?? []
  const [rules, setRules] = useState<InvariantRule[]>(() => initial.map((rule) => ({ ...rule })))
  const [replan, setReplan] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const dirty = JSON.stringify(rules) !== JSON.stringify(initial)
  const activeTask = session?.task ?? null

  function change(next: InvariantRule[]) {
    setRules(next)
    setSaved(false)
    onDirty(JSON.stringify(next) !== JSON.stringify(initial))
  }

  function addRule() {
    const used = new Set(rules.map((rule) => rule.id))
    let number = rules.length + 1
    while (used.has(`rule-${number}`)) number += 1
    change([...rules, { id: `rule-${number}`, category: 'decision', text: '' }])
  }

  function patchRule(index: number, patch: Partial<InvariantRule>) {
    change(rules.map((rule, position) => position === index ? { ...rule, ...patch } : rule))
  }

  function cancel() {
    change(initial.map((rule) => ({ ...rule })))
    setReplan(false)
    setError('')
  }

  async function apply() {
    if (!session || !dirty || (activeTask && !replan)) return
    setError('')
    try {
      await onApply(session.invariants.version_hash, rules, Boolean(activeTask))
      setSaved(true)
      onDirty(false)
    } catch (cause) {
      setError(String(cause))
    }
  }

  if (!session) {
    return (
      <section className="invariant-editor">
        <h2>инварианты сессии</h2>
        <p className="muted">Сначала отправьте сообщение: набор правил принадлежит сохранённой сессии, а не пресету агента.</p>
      </section>
    )
  }

  return (
    <section className="invariant-editor">
      <header className="configuration-heading">
        <h2>инварианты сессии</h2>
        <p className="muted">
          Область: «{session.title || session.id.slice(0, 8)}» · применена версия <code>{session.invariants.version_hash.slice(0, 10)}</code>.
          Правила хранятся отдельно от профиля и памяти и действуют со следующего запроса.
        </p>
      </header>
      <div className="invariant-list">
        {rules.length === 0 && <p className="muted">Набор пуст: поведение сессии не ограничено инвариантами.</p>}
        {rules.map((rule, index) => (
          <fieldset className="invariant-rule" key={index}>
            <legend>правило {index + 1}</legend>
            <label>ID<input value={rule.id} maxLength={80} onChange={(event) => patchRule(index, { id: event.target.value })} /></label>
            <label>Категория<select value={rule.category} onChange={(event) => patchRule(index, { category: event.target.value as InvariantCategory })}>
              {(Object.keys(CATEGORY) as InvariantCategory[]).map((category) => <option key={category} value={category}>{CATEGORY[category]}</option>)}
            </select></label>
            <label className="invariant-text">Формулировка<textarea value={rule.text} maxLength={1000} rows={3} onChange={(event) => patchRule(index, { text: event.target.value })} /></label>
            <button type="button" className="danger" onClick={() => change(rules.filter((_, position) => position !== index))}>Удалить правило</button>
          </fieldset>
        ))}
      </div>
      <button type="button" onClick={addRule} disabled={busy || rules.length >= 100}>+ добавить правило</button>
      {activeTask && dirty && (
        <label className="invariant-replan">
          <input type="checkbox" checked={replan} onChange={(event) => setReplan(event.target.checked)} />
          Вернуть активную задачу в планирование и сбросить её прежний план и вердикт
        </label>
      )}
      <div className="configuration-actions">
        <div>
          <strong>{dirty ? 'Есть несохранённые изменения' : saved ? 'Применено' : 'Изменений нет'}</strong>
          <small className="muted"> Запись проверит, что применённая версия не изменилась в другой вкладке.</small>
        </div>
        <button type="button" onClick={cancel} disabled={!dirty || busy}>Отменить</button>
        <button type="button" className="primary" onClick={() => void apply()} disabled={!dirty || busy || Boolean(activeTask && !replan)}>Применить</button>
        {error && <p className="error" role="alert">{error}</p>}
      </div>
    </section>
  )
}
