import type {
  ContextMessage,
  MemoryLayer,
  ModelInfo,
  SessionDetail,
  TokenUsage,
} from '../api/sessions'
import { STAGE_LABEL } from './task'
import { cost, formatTokens } from './tokens'

const LAYER_TITLE: Record<MemoryLayer, string> = {
  longterm: 'долговременная память',
  working: 'рабочая память',
}
const CHECK_TARGET = { request: 'запрос', response: 'ответ', tool: 'действие' } as const
const CHECK_OUTCOME = { allow: 'допустимо', conflict: 'конфликт', error: 'ошибка' } as const

/** Observable composition and accounting for the open session.

 * Layers are shown here as cost, not as content: editing them belongs to the
 * memory tab, so one layer never has two owners on screen.
 */
export function ContextPanel({
  session,
  models,
}: {
  session: SessionDetail | null
  models: ModelInfo[]
}) {
  if (!session || !session.spec || !session.context) {
    return <p className="muted panel-section">подключаюсь…</p>
  }
  const { context, totals, usage, spec } = session
  const latest = usage.filter((item) => item.kind === 'chat').at(-1)
  const turns = usage.filter((item) => item.kind === 'chat')
  const modelWindow = models.find((model) => model.id === spec.model.model)?.context_window ?? null
  const limit = context.limit_tokens ?? modelWindow
  const used = context.estimated_tokens
  const fill = limit ? Math.min(100, (used / limit) * 100) : 0
  const over = limit !== null && used > limit

  return (
    <>
      <section className="panel-section context-meter">
        <h3>следующий вызов</h3>
        <div className="meter-label">
          <strong>{formatTokens(used)}</strong>
          <span className="muted"> / {limit ? formatTokens(limit) : 'лимит неизвестен'}</span>
        </div>
        {limit && (
          <div className="meter" role="meter" aria-valuenow={used} aria-valuemin={0} aria-valuemax={limit}>
            <span className={over ? 'over' : ''} style={{ width: `${fill}%` }} />
          </div>
        )}
        <small className="muted">
          режим {context.mode} · {limitSource(context.limit_source, spec.context.window_tokens, limit)}
        </small>
        <ContextWarning session={session} limit={limit} />
      </section>

      <section className="panel-section context-composition">
        <h3>состав контекста</h3>
        <details>
          <summary>роль · {cost(context.role.tokens, used)}</summary>
          <pre>{context.role.content}</pre>
        </details>
        {context.invariants && (
          <details open>
            <summary>
              инварианты · {context.invariants.rules.length} правил · {cost(context.invariants.payload_tokens, used)}
            </summary>
            <pre>{context.invariants.content}</pre>
          </details>
        )}
        {context.user_profile && (
          <details>
            <summary>
              профиль пользователя · {context.user_profile.name} · {cost(context.user_profile.payload_tokens, used)}
            </summary>
            <pre>{context.user_profile.content}</pre>
          </details>
        )}
        {(['longterm', 'working'] as const).map((layer) => {
          const version = context[layer]
          if (!version) return null
          // The composition lists what the call carries, not what the layer
          // stores: an emptied layer has a version but no block, and a layer
          // over its budget sends only the keys that fit.
          const left = version.dropped_keys ?? []
          const sent = layerValues(version.items, left)
          if (Object.keys(sent).length === 0) return null
          return (
            <details key={layer}>
              <summary>
                {LAYER_TITLE[layer]} · версия {version.version} · {cost(version.payload_tokens, used)}
              </summary>
              <pre>{JSON.stringify(sent, null, 2)}</pre>
              {left.length > 0 && (
                <p className="muted">
                  не поместилось в бюджет слоя: {left.join(', ')}
                </p>
              )}
            </details>
          )
        })}
        {context.task && (
          <details>
            <summary>
              состояние задачи · {STAGE_LABEL[context.task.stage]}
              {context.task.paused ? ' · пауза' : ''} · {cost(context.task.payload_tokens, used)}
            </summary>
            <pre>{context.task.content}</pre>
          </details>
        )}
        {context.summary && (
          <details>
            <summary>
              сводка · версия {context.summary.version ?? '—'} · {cost(context.summary.payload_tokens, used)}
            </summary>
            <pre>{context.summary.text ?? ''}</pre>
          </details>
        )}
        <details open>
          <summary>
            хвост · {context.tail.messages.length} сообщений · {cost(context.tail.tokens, used)}
          </summary>
          <ol className="context-tail">
            {context.tail.messages.map((message) => (
              <li key={message.seq}>
                <span>#{message.seq} · {messageLabel(message)}</span>
                <p>{message.content}</p>
              </li>
            ))}
          </ol>
        </details>
        {context.overhead_tokens > 0 && (
          <p className="muted">
            разметка запроса и схемы инструментов · {cost(context.overhead_tokens, used)}
          </p>
        )}
        {context.dropped_seqs.length > 0 && (
          <p className="trim-notice">
            Вне следующего вызова: <strong>{context.dropped_seqs.length}</strong> сообщений
            <span className="muted"> · seq {context.dropped_seqs.join(', ')}</span>
          </p>
        )}
      </section>

      {latest ? <LastCall usage={latest} /> : (
        <p className="muted panel-section">Отправьте сообщение, чтобы сравнить оценку с фактом.</p>
      )}

      {session.invariant_checks.length > 0 && (
        <section className="panel-section">
          <h3>проверки инвариантов</h3>
          <ol className="context-tail">
            {session.invariant_checks.slice(-8).reverse().map((check) => (
              <li key={check.id}>
                <span>
                  #{check.seq} · {CHECK_TARGET[check.target]} · {CHECK_OUTCOME[check.outcome]}
                  {check.rule_ids.length > 0 ? ` · ${check.rule_ids.join(', ')}` : ''}
                </span>
                <p>{check.explanation}</p>
                {check.alternative && <small className="muted">Альтернатива: {check.alternative}</small>}
              </li>
            ))}
          </ol>
        </section>
      )}

      <section className="panel-section">
        <h3>накопительный расход</h3>
        <dl className="facts">
          <dt>запросов</dt><dd>{totals.requests}</dd>
          <dt>вход</dt><dd>{formatTokens(totals.tokens_in)}</dd>
          <dt>выход</dt><dd>{formatTokens(totals.tokens_out)}</dd>
          <dt>reasoning</dt><dd>{formatTokens(totals.reasoning)}</dd>
          <dt>стоимость</dt><dd>${totals.cost_usd.toFixed(6)}</dd>
        </dl>
      </section>

      {turns.length > 0 && <ContextGrowth turns={turns} />}
    </>
  )
}

function ContextWarning({ session, limit }: { session: SessionDetail; limit: number | null }) {
  const context = session.context
  if (!context) return null
  if (limit !== null && context.estimated_tokens > limit) {
    const text = context.overflow === 'send'
      ? 'Окно превышено: запрос уйдёт провайдеру без защиты.'
      : context.overflow === 'fail'
        ? 'Окно превышено: следующий запрос будет остановлен до вызова модели.'
        : 'Последний полный ход не помещается даже после обрезки.'
    return <p className="context-warning context-warning-error">{text}</p>
  }
  if (context.dropped_seqs.length > 0) {
    return <p className="context-warning">Старых сообщений вне окна: {context.dropped_seqs.length}.</p>
  }
  return null
}

function LastCall({ usage }: { usage: TokenUsage }) {
  return (
    <section className="panel-section">
      <h3>последний ответ</h3>
      <dl className="facts token-breakdown">
        <dt>роль · оценка</dt><dd>{formatTokens(usage.role_estimate_tokens)}</dd>
        <dt>история · оценка</dt><dd>{formatTokens(usage.history_estimate_tokens)}</dd>
        <dt>текущий запрос · оценка</dt><dd>{formatTokens(usage.request_estimate_tokens)}</dd>
        <dt>весь вход · оценка</dt><dd>{formatTokens(usage.input_estimate_tokens)}</dd>
        <dt>весь вход · факт</dt>
        <dd>
          {formatTokens(usage.input_tokens)}{' '}
          <small className={usage.estimate_delta_tokens === 0 ? 'muted' : 'estimate-delta'}>
            ({signed(usage.estimate_delta_tokens)})
          </small>
        </dd>
        <dt>ответ · факт</dt><dd>{formatTokens(usage.output_tokens)}</dd>
        <dt>из них reasoning</dt><dd>{formatTokens(usage.reasoning_tokens)}</dd>
        <dt>задержка</dt><dd>{formatLatency(usage.latency_ms)}</dd>
        <dt>стоимость</dt><dd>${usage.cost_usd.toFixed(6)}</dd>
      </dl>
    </section>
  )
}

function ContextGrowth({ turns }: { turns: TokenUsage[] }) {
  const maximum = Math.max(...turns.map((item) => item.input_tokens), 1)
  return (
    <section className="panel-section">
      <h3>рост по ходам</h3>
      <ol className="token-history">
        {turns.map((item, index) => (
          <li key={item.id}>
            <span>#{index + 1}</span>
            <div className="history-bar" title={`${item.input_tokens} входных токенов`}>
              <span style={{ width: `${(item.input_tokens / maximum) * 100}%` }} />
            </div>
            <small>{formatTokens(item.input_tokens)}</small>
            <small>${item.cost_usd.toFixed(6)}</small>
          </li>
        ))}
      </ol>
    </section>
  )
}

function layerValues(
  items: Record<string, { value: string }> | undefined,
  dropped: string[],
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(items ?? {})
      .filter(([key]) => !dropped.includes(key))
      .map(([key, item]) => [key, item.value]),
  )
}

function messageLabel(message: ContextMessage): string {
  if (message.role === 'user') return 'вы'
  if (message.role === 'assistant') return 'агент'
  return `инструмент ${message.tool ?? ''}`.trim()
}

function limitSource(
  source: 'spec' | 'model' | null,
  configured: number | null,
  limit: number | null,
): string {
  if (source === 'spec' || configured !== null) return 'искусственное окно'
  if (source === 'model' || limit !== null) return 'окно модели'
  return 'лимит окна не получен'
}

function formatLatency(value: number): string {
  return `${(value / 1000).toLocaleString('ru', { maximumFractionDigits: 1 })} с`
}

function signed(value: number): string {
  return `${value > 0 ? '+' : ''}${value.toLocaleString('ru')} к оценке`
}
