import { Fragment, useEffect, useRef, useState } from 'react'

import {
  LAYER_KINDS,
  type ContextView,
  type MemoryLayer,
  type TaskTransition,
  type TokenUsage,
  type ToolMessage,
} from '../api/sessions'
import type { ChatSession } from '../hooks/useChatSession'
import { LoadingIndicator } from './LoadingIndicator'
import { STAGE_LABEL } from './task'
import { TaskStrip } from './TaskStrip'
import { ApprovalCard, ToolCard } from './ToolCall'
import { fromStored } from './tools'

// Steps, plan drafts and verdicts move too often to mark: the transcript keeps
// the entries that change what the agent may do next.
const MARKED_KINDS = new Set(['start', 'stage', 'pause', 'fork'])

/** The middle zone: dialog and, while a task is active, its persistent panel. */
export function ChatPane({
  chat,
  modelWindow,
  onOpenContext,
}: {
  chat: ChatSession
  modelWindow: number | null
  onOpenContext: () => void
}) {
  const { session, messages, streaming, live, pending, busy, error } = chat
  const [draft, setDraft] = useState('')
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming, live, pending])

  function submit(event: React.FormEvent) {
    event.preventDefault()
    const text = draft.trim()
    if (!text || busy) return
    setDraft('')
    void chat.send(text)
  }

  const usageByMessage = new Map(
    (session?.usage ?? [])
      .filter((item) => item.kind === 'chat' && item.message_id !== null)
      .map((item) => [item.message_id, item]),
  )
  const context = session?.context ?? null
  const journal = (session?.task_journal ?? []).filter((item) => MARKED_KINDS.has(item.kind))
  // A transition made before the first message — the start of a task, say —
  // has no row to sit under, so it opens the transcript.
  const firstSeq = messages[0]?.seq ?? Number.POSITIVE_INFINITY
  const leading = journal.filter((item) => item.seq < firstSeq)

  const taskActive = session?.task !== null && session?.task !== undefined

  return (
    <div className={`chat-pane${taskActive ? ' chat-pane-task' : ''}`}>
      <div className="chat-dialog">
        {!taskActive && <TaskStrip chat={chat} />}
        <section className="transcript">
        <div className="transcript-inner">
          {messages.length === 0 && !streaming && (
            <p className="muted empty">Задайте вопрос — агент ответит потоком.</p>
          )}
          {leading.map((item) => <TaskMarker key={item.id} entry={item} />)}
          {messages.map((message) => (
            <Fragment key={message.seq}>
              {context?.dropped_seqs.length && context.tail.first_seq === message.seq ? (
                <ContextMarker>
                  выше — вне окна ({context.dropped_seqs.length} сообщений отброшено)
                </ContextMarker>
              ) : null}
              {message.role === 'tool' ? (
                <MessageEntry chat={chat} seq={message.seq} text={message.content} busy={busy}>
                  <ToolCard tool={fromStored(message as ToolMessage)} />
                </MessageEntry>
              ) : (
                <MessageEntry chat={chat} seq={message.seq} text={message.content} busy={busy}>
                  <Turn
                    role={message.role}
                    text={message.content}
                    usage={message.id === undefined ? undefined : usageByMessage.get(message.id)}
                  />
                </MessageEntry>
              )}
              {session?.checkpoints
                .filter((checkpoint) => checkpoint.seq === message.seq)
                .map((checkpoint) => (
                  <CheckpointMarker key={checkpoint.seq} name={checkpoint.name} />
                ))}
              {journal
                .filter((item) => item.seq === message.seq)
                .map((item) => <TaskMarker key={item.id} entry={item} />)}
              {context?.summary?.covered_seq === message.seq && (
                <ContextMarker>
                  выше свёрнуто в сводку v{context.summary.version ?? '—'} ({context.summary.message_count ?? 0} сообщений)
                </ContextMarker>
              )}
            </Fragment>
          ))}
          {context?.dropped_seqs.length && context.tail.first_seq === null ? (
            <ContextMarker>
              выше — вне окна ({context.dropped_seqs.length} сообщений отброшено)
            </ContextMarker>
          ) : null}
          {live.map((tool) => (
            <ToolCard key={tool.call_id} tool={tool} />
          ))}
          {pending && (
            <ApprovalCard
              request={pending}
              onAnswer={(approved) => void chat.answerApproval(approved)}
            />
          )}
          {busy && !streaming && !pending && (
            <div className="reply-wait" role="status">
              <LoadingIndicator label="агент думает…" />
            </div>
          )}
          {streaming && <Turn role="assistant" text={streaming} pending />}
          {error && <p className="error">{error}</p>}
          <div ref={bottom} />
        </div>
        </section>

        <form className="composer" onSubmit={submit}>
          <div className="composer-box">
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  submit(event)
                }
              }}
              placeholder="Сообщение агенту. Enter — отправить, Shift+Enter — перенос строки."
              rows={2}
            />
            <button type="submit" disabled={busy || !draft.trim()}>
              {busy ? <LoadingIndicator label="отвечает…" /> : 'отправить'}
            </button>
          </div>
          <ContextStatus context={context} modelWindow={modelWindow} onOpen={onOpenContext} />
        </form>
      </div>
      {taskActive && <TaskStrip chat={chat} />}
    </div>
  )
}

function MessageEntry({
  chat,
  seq,
  text,
  busy,
  children,
}: {
  chat: ChatSession
  seq: number
  text: string
  busy: boolean
  children: React.ReactNode
}) {
  const [remembering, setRemembering] = useState(false)
  const layers = (['working', 'longterm'] as const)
    .filter((layer) => chat.session?.memory[layer].enabled)

  function addCheckpoint() {
    const name = window.prompt('Название checkpoint', `checkpoint #${seq}`)
    if (name === null) return
    void chat.checkpoint(seq, name)
  }

  function addBranch() {
    const title = window.prompt('Название ветки', `ветка от #${seq}`)
    if (title === null) return
    void chat.fork(seq, title.trim() || undefined)
  }

  return (
    <div className="message-entry">
      {children}
      <div className="message-actions">
        <button type="button" disabled={busy} onClick={addCheckpoint}>checkpoint</button>
        <button type="button" disabled={busy} onClick={addBranch}>ветка отсюда</button>
        <button
          type="button"
          disabled={busy || layers.length === 0 || !chat.session?.editable}
          title={layers.length === 0 ? 'включите слой памяти в настройках агента' : undefined}
          onClick={() => setRemembering((open) => !open)}
        >
          запомнить
        </button>
      </div>
      {remembering && layers.length > 0 && (
        <RememberForm
          chat={chat}
          seq={seq}
          text={text}
          layers={layers}
          onClose={() => setRemembering(false)}
        />
      )}
    </div>
  )
}

/** «Запомнить» names the layer and the kind: nothing is routed by guessing. */
function RememberForm({
  chat,
  seq,
  text,
  layers,
  onClose,
}: {
  chat: ChatSession
  seq: number
  text: string
  layers: readonly MemoryLayer[]
  onClose: () => void
}) {
  const [layer, setLayer] = useState<MemoryLayer>(layers[0])
  const [kind, setKind] = useState(LAYER_KINDS[layers[0]][0])
  const [key, setKey] = useState('')
  const [value, setValue] = useState(text.slice(0, 2000))
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function pickLayer(next: MemoryLayer) {
    setLayer(next)
    setKind(LAYER_KINDS[next][0])
  }

  function save() {
    setSaving(true)
    setError(null)
    void chat.rememberItem(layer, { key: key.trim(), value: value.trim(), kind, seq })
      .then(onClose)
      .catch((cause) => setError(String(cause)))
      .finally(() => setSaving(false))
  }

  return (
    <div className="remember-form">
      <div className="remember-row">
        <select
          aria-label="слой памяти"
          value={layer}
          disabled={saving}
          onChange={(event) => pickLayer(event.target.value as MemoryLayer)}
        >
          {layers.map((name) => (
            <option key={name} value={name}>
              {name === 'working' ? 'рабочая' : 'долговременная'}
            </option>
          ))}
        </select>
        <select
          aria-label="вид записи"
          value={kind}
          disabled={saving}
          onChange={(event) => setKind(event.target.value)}
        >
          {LAYER_KINDS[layer].map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
        <input
          aria-label="ключ записи"
          placeholder="ключ"
          value={key}
          disabled={saving}
          onChange={(event) => setKey(event.target.value)}
        />
      </div>
      <textarea
        aria-label="значение записи"
        rows={2}
        value={value}
        disabled={saving}
        onChange={(event) => setValue(event.target.value)}
      />
      <div className="remember-actions">
        <button type="button" disabled={saving || !key.trim() || !value.trim()} onClick={save}>
          {saving ? 'пишу…' : 'записать'}
        </button>
        <button type="button" className="link" disabled={saving} onClick={onClose}>отмена</button>
        <small className="muted">из сообщения #{seq}</small>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  )
}

/** A transition as the transcript shows it: where the task changed, and how. */
function TaskMarker({ entry }: { entry: TaskTransition }) {
  const stage = STAGE_LABEL[entry.to_stage]
  const text = entry.kind === 'start'
    ? 'задача начата'
    : entry.kind === 'fork'
      ? 'состояние задачи перенесено в ветку'
      : entry.kind === 'pause'
        ? `${entry.paused ? 'пауза' : 'продолжение'} на этапе «${stage}»`
        : entry.to_stage === 'cancelled'
          ? 'задача отменена'
          : `этап: ${stage}`
  const mark = entry.kind === 'pause' ? (entry.paused ? '⏸' : '▶') : entry.kind === 'start' ? '▶' : '→'
  return (
    <div className="task-marker" role="note">
      <span>
        {mark} {text}
        {entry.actor === 'tracker' && ' · трекер'}
        {entry.reason && ` · ${entry.reason}`}
      </span>
    </div>
  )
}

function CheckpointMarker({ name }: { name: string }) {
  return (
    <div className="checkpoint-marker" role="note">
      <span>◆</span>
      <strong>{name}</strong>
    </div>
  )
}

/**
 * A full-width transcript row instead of a chat bubble: on a wide screen a
 * right-aligned bubble wastes the width the layout just gained.
 */
function Turn({
  role,
  text,
  pending,
  usage,
}: {
  role: 'user' | 'assistant'
  text: string
  pending?: boolean
  usage?: TokenUsage
}) {
  return (
    <article className={`turn turn-${role}${pending ? ' pending' : ''}`}>
      <span className="turn-role">{role === 'user' ? 'вы' : 'агент'}</span>
      <div>
        <div className="turn-text">{text}</div>
        {role === 'assistant' && usage && <ResponseMeta usage={usage} />}
      </div>
    </article>
  )
}

function ContextMarker({ children }: { children: React.ReactNode }) {
  return (
    <div className="context-marker" role="note">
      <span>{children}</span>
    </div>
  )
}

function ResponseMeta({ usage }: { usage: TokenUsage }) {
  return (
    <div className="response-meta">
      {usage.model} · вход {compactTokens(usage.input_tokens)} · выход {compactTokens(usage.output_tokens)} ·{' '}
      reasoning {compactTokens(usage.reasoning_tokens)} · ${usage.cost_usd.toFixed(6)} · {latency(usage.latency_ms)}
    </div>
  )
}

function ContextStatus({
  context,
  modelWindow,
  onOpen,
}: {
  context: ContextView | null
  modelWindow: number | null
  onOpen: () => void
}) {
  if (!context) return null
  const limit = context.limit_tokens ?? modelWindow
  const fill = limit ? Math.min(100, (context.estimated_tokens / limit) * 100) : 0
  const over = limit !== null && context.estimated_tokens > limit
  const warning = overflowWarning(context, limit)
  return (
    <div className="context-status">
      <div className="context-status-meter">
        <div className="meter" aria-hidden="true">
          <span className={over ? 'over' : ''} style={{ width: `${fill}%` }} />
        </div>
        <span>{compactTokens(context.estimated_tokens)} / {limit ? compactTokens(limit) : '—'}</span>
      </div>
      <span className="context-badge">{context.mode}</span>
      {warning && <span className={over ? 'context-status-warning over' : 'context-status-warning'}>{warning}</span>}
      <button type="button" className="link context-open" onClick={onOpen}>
        что уйдёт в модель
      </button>
    </div>
  )
}

function overflowWarning(context: ContextView, limit: number | null): string | null {
  if (limit !== null && context.estimated_tokens > limit) {
    if (context.overflow === 'fail') return 'следующий запрос будет остановлен'
    if (context.overflow === 'send') return 'окно превышено, защита отключена'
    return 'последний ход не помещается'
  }
  if (context.dropped_seqs.length > 0) return `вне окна: ${context.dropped_seqs.length}`
  return null
}

function compactTokens(value: number): string {
  if (value < 1000) return value.toLocaleString('ru')
  return `${(value / 1000).toLocaleString('ru', { maximumFractionDigits: 1 })}k`
}

function latency(value: number): string {
  return `${(value / 1000).toLocaleString('ru', { maximumFractionDigits: 1 })} с`
}
