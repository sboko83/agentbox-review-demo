import { useEffect, useState } from 'react'

import {
  estimateRun,
  getRun,
  getRunDashboard,
  startRun,
  type MatrixQuote,
  type InvariantRule,
  type PoolDashboard,
  type ComparisonRow,
  type Preset,
  type RunReport,
  type UserProfilePointer,
} from '../api/sessions'


/** A paid matrix is intentionally a separate, explicit UI from chat. */
export function Experiment({
  presets,
  profiles,
  invariants,
}: {
  presets: Preset[]
  profiles: UserProfilePointer[]
  invariants: InvariantRule[]
}) {
  const [prompt, setPrompt] = useState('')
  const [repeats, setRepeats] = useState(3)
  const [budget, setBudget] = useState('')
  const [selectedPresets, setSelectedPresets] = useState<string[]>([])
  const [selectedProfiles, setSelectedProfiles] = useState<string[]>([])
  const [quote, setQuote] = useState<MatrixQuote | null>(null)
  const [report, setReport] = useState<RunReport | null>(null)
  const [dashboard, setDashboard] = useState<PoolDashboard | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const activePresetIds = presets
    .filter((preset) => selectedPresets.includes(preset.id))
    .map((preset) => preset.id)
  const activeProfileIds = profiles
    .filter((profile) => selectedProfiles.includes(profile.id))
    .map((profile) => profile.id)

  function payload() {
    const parsedBudget = Number(budget)
    return {
      prompt: prompt.trim(),
      repeats,
      budget_usd: parsedBudget > 0 ? parsedBudget : null,
      preset_ids: activePresetIds,
      user_profile_ids: activeProfileIds,
      invariants: { rules: invariants },
    }
  }

  async function calculate() {
    setError(null)
    setReport(null)
    setDashboard(null)
    try {
      setQuote(await estimateRun(payload()))
    } catch (cause) {
      setQuote(null)
      setError(String(cause))
    }
  }

  async function confirm() {
    if (!quote || busy) return
    setBusy(true)
    setError(null)
    try {
      setDashboard(await startRun(quote.quote_id))
      setQuote(null)
    } catch (cause) {
      setError(String(cause))
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    if (!dashboard?.active) return
    let disposed = false
    const refresh = async () => {
      try {
        const next = await getRunDashboard(dashboard.run_id)
        if (disposed) return
        setDashboard(next)
        if (!next.active) setReport(await getRun(next.run_id))
      } catch (cause) {
        if (!disposed) setError(String(cause))
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 500)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [dashboard?.run_id, dashboard?.active])

  function toggle(setSelected: typeof setSelectedPresets, id: string) {
    setQuote(null)
    setSelected((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    )
  }

  return (
    <section className="experiment">
      <h2>матричный прогон с судьёй</h2>
      <p className="muted experiment-note">
        Сначала показывается верхняя оценка по прайсу, затем требуется отдельное подтверждение. Отчёт использует фактическую стоимость провайдера.
      </p>
      <textarea
        value={prompt}
        onChange={(event) => { setPrompt(event.target.value); setQuote(null) }}
        placeholder="Один запрос для всех конфигураций"
        rows={4}
      />
      <label className="run-repeats">
        Повторов на конфигурацию
        <input type="number" min={1} max={20} value={repeats} onChange={(event) => { setRepeats(Number(event.target.value)); setQuote(null) }} />
      </label>
      <label className="run-repeats">
        Общий бюджет, $ (пусто — без лимита)
        <input type="number" min="0.000001" step="any" value={budget} onChange={(event) => { setBudget(event.target.value); setQuote(null) }} />
      </label>
      <section className="matrix-presets">
        <h3>строки матрицы</h3>
        {presets.length === 0 ? (
          <p className="muted">Сначала создайте хотя бы один пресет в библиотеке.</p>
        ) : (
          presets.map((preset) => (
            <label className="matrix-preset" key={preset.id}>
              <input
                type="checkbox"
                checked={selectedPresets.includes(preset.id)}
                onChange={() => toggle(setSelectedPresets, preset.id)}
              />
              <span>
                <strong>{preset.name}</strong>
                <small><code>{preset.spec.model.provider}/{preset.spec.model.model}</code> · {preset.spec.reasoning.strategy}</small>
              </span>
              {preset.is_default && <span className="muted">по умолчанию</span>}
            </label>
          ))
        )}
      </section>
      <p className="muted">Инварианты: снимок текущей сессии, {invariants.length} правил. Он фиксируется вместе с оценкой.</p>
      <section className="matrix-presets">
        <h3>профили пользователя</h3>
        <p className="muted">Каждый отмеченный профиль умножает строки: прогон идёт по парам «пресет · профиль». Без отметок — без профиля.</p>
        {profiles.length === 0 ? (
          <p className="muted">Профилей пока нет.</p>
        ) : (
          profiles.map((profile) => (
            <label className="matrix-preset" key={profile.id}>
              <input
                type="checkbox"
                checked={selectedProfiles.includes(profile.id)}
                onChange={() => toggle(setSelectedProfiles, profile.id)}
              />
              <span>
                <strong>{profile.name}</strong>
                <small>{profile.profile.language} · {profile.profile.verbosity} · {profile.profile.structure} · код {profile.profile.code_style}</small>
              </span>
            </label>
          ))
        )}
      </section>
      <div className="setup-row">
        <button
          type="button"
          onClick={() => void calculate()}
          disabled={!prompt.trim() || activePresetIds.length === 0 || busy}
        >
          оценить стоимость
        </button>
      </div>
      {quote && (
        <section className="quote">
          <strong>верхняя оценка: ${quote.quote.total_usd.toFixed(6)}</strong>
          <span>{quote.quote.agent_calls} вызовов агента + {quote.quote.judge_calls} судьи</span>
          <button type="button" className="primary" onClick={() => void confirm()} disabled={busy}>
            {busy ? 'выполняется…' : 'подтвердить и запустить'}
          </button>
        </section>
      )}
      {dashboard && <Dashboard dashboard={dashboard} />}
      {report && <Report report={report} />}
      {error && <p className="error">{error}</p>}
    </section>
  )
}

function Dashboard({ dashboard }: { dashboard: PoolDashboard }) {
  const counts = dashboard.agents.reduce<Record<string, number>>((result, agent) => {
    result[agent.state] = (result[agent.state] ?? 0) + 1
    return result
  }, {})
  return (
    <section className="dashboard">
      <h2>пул агентов {dashboard.active ? 'работает' : 'завершён'}</h2>
      <p className="muted">
        {Object.entries(counts).map(([state, count]) => `${state}: ${count}`).join(' · ')}
        {dashboard.budget && <> · ${dashboard.budget.cost_usd.toFixed(6)}{dashboard.budget.budget_usd !== null && ` / $${dashboard.budget.budget_usd.toFixed(6)}`}</>}
      </p>
      <p className="muted">
        провайдер {dashboard.provider.active}/{dashboard.provider.limit}, очередь {dashboard.provider.waiting}
        {' · '}executor {dashboard.executors.active}/{dashboard.executors.limit}, очередь {dashboard.executors.waiting}
      </p>
      <div className="dashboard-scroll">
        <table>
          <thead><tr><th>агент</th><th>повтор</th><th>модель</th><th>состояние</th></tr></thead>
          <tbody>{dashboard.agents.map((agent) => (
            <tr key={agent.id}><td>{agent.name}</td><td>{agent.repeat_index + 1}</td><td><code>{agent.provider}/{agent.model}</code></td><td>{agent.state}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  )
}

function Report({ report }: { report: RunReport }) {
  // Rows of one profile sit together so presets compare within a profile.
  const groups = new Map<string, ComparisonRow[]>()
  for (const row of report.comparison) {
    const key = row.user_profile?.profile.name ?? ''
    groups.set(key, [...(groups.get(key) ?? []), row])
  }
  const withProfiles = report.comparison.some((row) => row.user_profile)
  const firstAnswer = (index: number) =>
    report.results.find((item) => item.config_index === index)?.answer ?? ''
  return (
    <section className="report">
      <h2>сравнительный отчёт</h2>
      <p className="muted">
        {report.status} · оценка ${report.estimated_cost_usd.toFixed(6)} · факт ${report.totals.cost_usd.toFixed(6)} · {report.totals.requests} вызовов
      </p>
      <p className="report-actions">
        <a href={`/api/runs/${report.id}/export?format=json`} download>скачать JSON</a>
        <a href={`/api/runs/${report.id}/export?format=csv`} download>скачать CSV</a>
      </p>
      <div className="report-scroll">
        <table>
          <thead><tr><th>конфигурация</th><th>модель</th><th>прошло</th><th>средний балл</th><th>факт</th><th>ответ</th></tr></thead>
          {[...groups].map(([profile, rows]) => (
            <tbody key={profile}>
              {withProfiles && (
                <tr className="report-group"><th colSpan={6}>{profile ? `профиль «${profile}»` : 'без профиля'}</th></tr>
              )}
              {rows.map((row) => (
                <tr key={row.config_index}>
                  <td>{row.name}</td><td><code>{row.provider}/{row.model}</code></td><td>{row.passed}/{row.runs}</td>
                  <td>{row.average_score ?? '—'}</td><td>${row.actual_cost_usd.toFixed(6)}</td>
                  <td><div className="report-answer" title={firstAnswer(row.config_index)}>{firstAnswer(row.config_index)}</div></td>
                </tr>
              ))}
            </tbody>
          ))}
        </table>
      </div>
    </section>
  )
}
