import { useState } from 'react'

import {
  LAYER_KINDS,
  type MemoryItem,
  type MemoryLayer,
  type MemoryProposal,
  type SessionDetail,
} from '../api/sessions'
import { cost, formatTokens } from './tokens'

const LAYER_TITLE: Record<MemoryLayer, string> = {
  working: 'рабочая память',
  longterm: 'долговременная память',
}
const LAYER_SUBTITLE: Record<MemoryLayer, string> = {
  working: 'данные текущей задачи: цель, ограничения, шаги, артефакты, блокеры',
  longterm: 'то, что переживает задачу: сведения о человеке, решения, знания',
}
// `profile` stays the stored value; on screen it must not read as the user
// profile tab, which is a different thing.
const KIND_TITLE: Record<string, string> = { profile: 'о человеке' }

export function kindLabel(kind: string): string {
  return KIND_TITLE[kind] ?? kind
}

const ORIGIN_TITLE: Record<MemoryItem['origin'], string> = {
  router: 'роутер',
  manual: 'рука',
  promote: 'перенос',
  migration: 'миграция',
}
const WRITE_TITLE: Record<string, string> = {
  auto: 'пишется сразу',
  confirm: 'через подтверждение',
  manual: 'только рукой',
}

export interface MemoryActions {
  onRemember: (
    layer: MemoryLayer,
    item: { key: string; value: string; kind: string; seq?: number },
  ) => Promise<void>
  onForget: (layer: MemoryLayer, key: string) => Promise<void>
  onMove: (layer: MemoryLayer, key: string, kind: string) => Promise<void>
  onClear: (layer: MemoryLayer) => Promise<void>
  onFinish: (promote: Record<string, string>) => Promise<void>
  onDecide: (proposalId: number, decision: 'accept' | 'reject') => Promise<void>
}

/**
 * The three layers side by side: what each one holds, where it is stored and
 * what it costs in the next call. Short-term is shown, not edited — it is the
 * transcript itself; the other two are routed by hand right here.
 */
export function MemoryPanel({
  session,
  busy,
  actions,
}: {
  session: SessionDetail | null
  busy: boolean
  actions: MemoryActions
}) {
  if (!session || !session.spec || !session.context) {
    return <p className="muted panel-section">подключаюсь…</p>
  }
  const total = session.context.estimated_tokens
  return (
    <>
      <ShortTermCard session={session} total={total} />
      {(['working', 'longterm'] as const).map((layer) => (
        <LayerCard
          key={layer}
          layer={layer}
          session={session}
          total={total}
          busy={busy}
          actions={actions}
        />
      ))}
    </>
  )
}

function ShortTermCard({ session, total }: { session: SessionDetail; total: number }) {
  const context = session.context!
  const spec = session.spec!.context
  const window = spec.mode === 'window'
    ? `последние ${spec.keep_last_messages} сообщений`
    : spec.mode === 'tokens'
      ? `до ${formatTokens(spec.memory_limit_tokens ?? 0)}`
      : spec.mode === 'summary'
        ? `${spec.keep_last_messages} сообщений дословно плюс сводка`
        : 'вся стенограмма'
  return (
    <section className="panel-section memory-card">
      <div className="memory-card-head">
        <h3>краткосрочная память</h3>
        <span className="memory-flag on">только чтение</span>
      </div>
      <p className="muted memory-sub">текущий диалог: дословный хвост и сводка выпавшего</p>
      <dl className="facts">
        <dt>хранится</dt><dd>стенограмма сессии</dd>
        <dt>режим</dt><dd>{context.mode} · {window}</dd>
        <dt>в хвосте</dt>
        <dd>
          {context.tail.messages.length} сообщений
          {context.tail.first_seq !== null && <span className="muted"> · с #{context.tail.first_seq}</span>}
        </dd>
        <dt>выпало</dt><dd>{context.dropped_seqs.length} сообщений</dd>
        <dt>в вызове</dt>
        <dd>
          {cost(context.tail.tokens, total)}
          {context.summary && (
            <span className="muted"> · сводка {cost(context.summary.payload_tokens, total)}</span>
          )}
        </dd>
      </dl>
    </section>
  )
}

function LayerCard({
  layer,
  session,
  total,
  busy,
  actions,
}: {
  layer: MemoryLayer
  session: SessionDetail
  total: number
  busy: boolean
  actions: MemoryActions
}) {
  const state = session.memory[layer]
  const version = session.context?.[layer] ?? null
  const items = Object.entries(version?.items ?? {})
  // The budget bounds the block in the call, not the layer: these keys are
  // stored and editable, they just did not fit into the next request.
  const dropped = version?.dropped_keys ?? []
  const budget = session.spec!.memory[`${layer}_max_tokens`]
  const editable = session.editable && state.enabled
  const scope = layer === 'longterm' ? session.memory.longterm.scope : null
  const write = layer === 'longterm' ? session.memory.longterm.write : null
  const shared = session.memory.longterm.sessions

  return (
    <section className={`panel-section memory-card${state.enabled ? '' : ' memory-off'}`}>
      <div className="memory-card-head">
        <h3>{LAYER_TITLE[layer]}</h3>
        <span className={state.enabled ? 'memory-flag on' : 'memory-flag'}>
          {state.enabled ? 'включена' : 'выключена'}
        </span>
      </div>
      <p className="muted memory-sub">{LAYER_SUBTITLE[layer]}</p>
      <dl className="facts">
        <dt>хранится</dt>
        <dd>
          {layer === 'working' ? (
            'сессия'
          ) : (
            <>
              пространство «{scope ?? 'default'}»
              <span className="muted"> · {shared} {sessionWord(shared)}</span>
            </>
          )}
        </dd>
        {write && (<><dt>запись</dt><dd>{WRITE_TITLE[write] ?? write}</dd></>)}
        <dt>версия</dt>
        <dd>{version ? `${version.version} · ${items.length} элементов` : 'ещё пуста'}</dd>
        <dt>в вызове</dt>
        <dd>
          {cost(version?.payload_tokens ?? 0, total)}
          <span className="muted"> · бюджет {formatTokens(budget)}</span>
        </dd>
        {dropped.length > 0 && (
          <>
            <dt>вне вызова</dt>
            <dd>{dropped.length} {itemWord(dropped.length)} · не влезли в бюджет</dd>
          </>
        )}
      </dl>

      {!state.enabled ? (
        <p className="muted fact-empty">Включите слой во вкладке «агент», чтобы он попадал в вызов.</p>
      ) : (
        <>
          {state.proposals.length > 0 && (
            <Proposals
              layer={layer}
              proposals={state.proposals}
              busy={busy || !session.editable}
              onDecide={actions.onDecide}
            />
          )}
          {items.length > 0 ? (
            <ul className="memory-items">
              {items.map(([key, item]) => (
                <MemoryRow
                  key={key}
                  layer={layer}
                  itemKey={key}
                  item={item}
                  dropped={dropped.includes(key)}
                  busy={busy || !editable}
                  otherEnabled={session.memory[other(layer)].enabled}
                  actions={actions}
                />
              ))}
            </ul>
          ) : (
            <p className="muted fact-empty">Пусто: роутер ещё ничего не положил, а рукой не записано.</p>
          )}
          <AddItem layer={layer} busy={busy || !editable} onRemember={actions.onRemember} />
          {items.length > 0 && (layer === 'working' ? (
            <FinishTask
              items={items}
              busy={busy || !editable}
              promotable={session.memory.longterm.enabled}
              onFinish={actions.onFinish}
            />
          ) : (
            <button
              type="button"
              className="link memory-clear"
              disabled={busy || !editable}
              onClick={() => void actions.onClear(layer)}
            >
              очистить слой
            </button>
          ))}
        </>
      )}
      {!session.editable && <p className="muted fact-empty">Матричная сессия доступна только для чтения.</p>}
    </section>
  )
}

function MemoryRow({
  layer,
  itemKey,
  item,
  dropped,
  busy,
  otherEnabled,
  actions,
}: {
  layer: MemoryLayer
  itemKey: string
  item: MemoryItem
  dropped: boolean
  busy: boolean
  otherEnabled: boolean
  actions: MemoryActions
}) {
  const [value, setValue] = useState(item.value)
  const [kind, setKind] = useState(item.kind)
  const [moving, setMoving] = useState(false)
  const [moveKind, setMoveKind] = useState(LAYER_KINDS[other(layer)][0])
  const [error, setError] = useState<string | null>(null)
  const dirty = value.trim() !== item.value || kind !== item.kind

  function run(action: Promise<void>) {
    setError(null)
    void action.catch((cause) => setError(String(cause)))
  }

  return (
    <li className={dropped ? 'memory-item memory-item-dropped' : 'memory-item'}>
      <div className="memory-item-head">
        <strong>{itemKey}</strong>
        {dropped && <span className="memory-flag" title="не влез в бюджет слоя">вне вызова</span>}
        <span className="memory-origin" title={`seq источника #${item.seq}`}>
          {ORIGIN_TITLE[item.origin] ?? item.origin} · #{item.seq}
        </span>
      </div>
      <input
        aria-label={`значение ${itemKey}`}
        value={value}
        disabled={busy}
        onChange={(event) => setValue(event.target.value)}
      />
      <div className="memory-item-actions">
        <select
          aria-label={`вид ${itemKey}`}
          value={kind}
          disabled={busy}
          onChange={(event) => setKind(event.target.value)}
        >
          {LAYER_KINDS[layer].map((name) => (
            <option key={name} value={name}>{kindLabel(name)}</option>
          ))}
          {!LAYER_KINDS[layer].includes(kind) && <option value={kind}>{kindLabel(kind)}</option>}
        </select>
        {dirty && (
          <button
            type="button"
            disabled={busy || !value.trim()}
            onClick={() => run(actions.onRemember(layer, { key: itemKey, value: value.trim(), kind, seq: item.seq }))}
          >
            сохранить
          </button>
        )}
        {otherEnabled && (
          <button type="button" disabled={busy} onClick={() => setMoving((open) => !open)}>
            {layer === 'working' ? '→ в долговременную' : '→ в рабочую'}
          </button>
        )}
        <button
          type="button"
          className="link fact-delete"
          aria-label={`забыть ${itemKey}`}
          disabled={busy}
          onClick={() => run(actions.onForget(layer, itemKey))}
        >
          забыть
        </button>
      </div>
      {moving && (
        <div className="memory-move">
          <span className="muted">вид в {LAYER_TITLE[other(layer)].split(' ')[0]}:</span>
          <select
            aria-label={`вид переноса ${itemKey}`}
            value={moveKind}
            disabled={busy}
            onChange={(event) => setMoveKind(event.target.value)}
          >
            {LAYER_KINDS[other(layer)].map((name) => (
              <option key={name} value={name}>{kindLabel(name)}</option>
            ))}
          </select>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              setMoving(false)
              run(actions.onMove(layer, itemKey, moveKind))
            }}
          >
            перенести
          </button>
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </li>
  )
}

/**
 * The end of a task, not a clear button: what is marked here moves into the
 * long-term layer, everything else leaves with the task it belonged to.
 */
function FinishTask({
  items,
  busy,
  promotable,
  onFinish,
}: {
  items: [string, MemoryItem][]
  busy: boolean
  promotable: boolean
  onFinish: MemoryActions['onFinish']
}) {
  const [open, setOpen] = useState(false)
  const [promote, setPromote] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)

  if (!open) {
    return (
      <button
        type="button"
        className="link memory-clear"
        disabled={busy}
        onClick={() => setOpen(true)}
      >
        завершить задачу
      </button>
    )
  }
  return (
    <div className="memory-finish">
      <p className="muted">
        Отмеченное уедет в долговременную память, остальное задача унесёт с собой.
      </p>
      <ul>
        {items.map(([key, item]) => (
          <li key={key}>
            <label>
              <input
                type="checkbox"
                checked={key in promote}
                disabled={busy || !promotable}
                onChange={(event) =>
                  setPromote((current) => {
                    const next = { ...current }
                    if (event.target.checked) next[key] = LAYER_KINDS.longterm[1]
                    else delete next[key]
                    return next
                  })
                }
              />
              <span>{key}</span>
              <span className="muted">{item.value}</span>
            </label>
            {key in promote && (
              <select
                aria-label={`вид переноса ${key}`}
                value={promote[key]}
                disabled={busy}
                onChange={(event) =>
                  setPromote((current) => ({ ...current, [key]: event.target.value }))
                }
              >
                {LAYER_KINDS.longterm.map((name) => (
                  <option key={name} value={name}>{kindLabel(name)}</option>
                ))}
              </select>
            )}
          </li>
        ))}
      </ul>
      {!promotable && (
        <p className="muted">Долговременный слой выключен — переносить некуда.</p>
      )}
      <div className="memory-form-actions">
        <button
          type="button"
          disabled={busy}
          onClick={() => {
            setError(null)
            void onFinish(promote)
              .then(() => {
                setOpen(false)
                setPromote({})
              })
              .catch((cause) => setError(String(cause)))
          }}
        >
          завершить
        </button>
        <button type="button" className="link" disabled={busy} onClick={() => setOpen(false)}>
          отмена
        </button>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  )
}

function AddItem({
  layer,
  busy,
  onRemember,
}: {
  layer: MemoryLayer
  busy: boolean
  onRemember: MemoryActions['onRemember']
}) {
  const [open, setOpen] = useState(false)
  const [key, setKey] = useState('')
  const [value, setValue] = useState('')
  const [kind, setKind] = useState(LAYER_KINDS[layer][0])
  const [error, setError] = useState<string | null>(null)

  if (!open) {
    return (
      <button type="button" className="memory-add" disabled={busy} onClick={() => setOpen(true)}>
        + запись
      </button>
    )
  }
  return (
    <div className="memory-form">
      <input
        aria-label="ключ записи"
        placeholder="ключ"
        value={key}
        disabled={busy}
        onChange={(event) => setKey(event.target.value)}
      />
      <input
        aria-label="значение записи"
        placeholder="значение"
        value={value}
        disabled={busy}
        onChange={(event) => setValue(event.target.value)}
      />
      <select
        aria-label="вид записи"
        value={kind}
        disabled={busy}
        onChange={(event) => setKind(event.target.value)}
      >
        {LAYER_KINDS[layer].map((name) => (
          <option key={name} value={name}>{kindLabel(name)}</option>
        ))}
      </select>
      <div className="memory-form-actions">
        <button
          type="button"
          disabled={busy || !key.trim() || !value.trim()}
          onClick={() => {
            setError(null)
            void onRemember(layer, { key: key.trim(), value: value.trim(), kind })
              .then(() => {
                setOpen(false)
                setKey('')
                setValue('')
              })
              .catch((cause) => setError(String(cause)))
          }}
        >
          записать
        </button>
        <button type="button" className="link" disabled={busy} onClick={() => setOpen(false)}>
          отмена
        </button>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  )
}

function Proposals({
  layer,
  proposals,
  busy,
  onDecide,
}: {
  layer: MemoryLayer
  proposals: MemoryProposal[]
  busy: boolean
  onDecide: MemoryActions['onDecide']
}) {
  return (
    <div className="memory-proposals">
      <h4>кандидаты роутера · {proposals.length}</h4>
      <ul>
        {proposals.map((proposal) => (
          <li key={proposal.id}>
            <div className="memory-item-head">
              <strong>{proposal.key}</strong>
              <span className="memory-origin">{kindLabel(proposal.kind)} · #{proposal.source_seq}</span>
            </div>
            <p>{proposal.value || <em>предлагает забыть</em>}</p>
            <div className="memory-item-actions">
              <button type="button" disabled={busy} onClick={() => void onDecide(proposal.id, 'accept')}>
                принять
              </button>
              <button
                type="button"
                className="link"
                disabled={busy}
                aria-label={`отклонить ${proposal.key} в слое ${layer}`}
                onClick={() => void onDecide(proposal.id, 'reject')}
              >
                отклонить
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

function itemWord(count: number): string {
  const tail = count % 100
  if (tail >= 11 && tail <= 14) return 'элементов'
  const last = count % 10
  if (last === 1) return 'элемент'
  if (last >= 2 && last <= 4) return 'элемента'
  return 'элементов'
}

function sessionWord(count: number): string {
  const tail = count % 100
  if (tail >= 11 && tail <= 14) return 'сессий'
  const last = count % 10
  if (last === 1) return 'сессия'
  if (last >= 2 && last <= 4) return 'сессии'
  return 'сессий'
}

function other(layer: MemoryLayer): MemoryLayer {
  return layer === 'working' ? 'longterm' : 'working'
}
