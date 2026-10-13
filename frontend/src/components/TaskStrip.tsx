import { useState } from 'react'

import type { Task, TaskStage, TaskStep, TaskStepStatus } from '../api/sessions'
import type { ChatSession } from '../hooks/useChatSession'
import { STAGE_LABEL, STAGE_TRACK, STEP_STATUS_LABEL } from './task'

const STEP_STATUSES: TaskStepStatus[] = ['todo', 'in_progress', 'done', 'blocked']

/**
 * The button that reaches a stage is named by where it comes from: one move of
 * the machine reads differently on the way forward and on the way back.
 */
function moveLabel(from: TaskStage, to: TaskStage): string {
  if (to === 'planning') return from === 'planning' ? 'сбросить план' : 'перепланировать'
  if (to === 'execution') return from === 'planning' ? 'утвердить план' : 'вернуть в выполнение'
  if (to === 'validation') return 'на проверку'
  if (to === 'done') return 'завершить'
  return 'отменить'
}

/** A move the human should look at before making it. */
function needsConfirm(task: Task, to: TaskStage): boolean {
  if (to === 'cancelled' || to === 'planning' || to === 'done') return true
  return task.stage === 'validation' && to === 'execution' && task.criteria.length > 0
}

/**
 * The task panel: stage, expected action, steps and the moves the server allows.
 *
 * The buttons come from `task.available`, so the transition table and blocked
 * reasons stay on the server instead of being duplicated in the browser.
 */
export function TaskStrip({ chat }: { chat: ChatSession }) {
  const { session, busy, taskProposal } = chat
  const task = session?.task ?? null
  const [confirm, setConfirm] = useState<{ stage: TaskStage } | null>(null)

  // A task belongs to a session, so a draft shows the strip disabled rather
  // than hiding it: the control is where it will be, and says why it waits.
  if (!task) {
    return (
      <div className="task-strip task-strip-idle">
        <span className="task-label">задача</span>
        <span className="task-hint muted">
          {session ? 'нет активной задачи — чат идёт как обычно' : 'сессия начнётся с первого сообщения'}
        </span>
        <div className="task-actions">
          <button
            type="button"
            disabled={busy || !session}
            title={session ? undefined : 'сначала отправьте сообщение'}
            onClick={() => void chat.openTask()}
          >
            начать задачу
          </button>
        </div>
      </div>
    )
  }

  const done = task.steps.filter((step) => step.status === 'done').length
  const current = task.current_step

  // An arrow declared past the null check keeps the narrowing a hoisted
  // function declaration would lose.
  const move = (stage: TaskStage) => {
    if (needsConfirm(task, stage)) {
      setConfirm({ stage })
      return
    }
    void chat.moveTaskStage(stage)
  }

  return (
    <aside className={`task-strip task-panel${task.paused ? ' task-paused' : ''}`} aria-label="Состояние задачи">
      <header className="task-panel-header">
        <span className="task-label">задача</span>
        {task.steps.length > 0 && (
          <span className="task-progress">шаг {current ?? '—'} из {task.steps.length} · готово {done}</span>
        )}
      </header>
      <div className="task-line">
        <Stepper stage={task.stage} />
        {task.paused && <span className="task-flag">⏸ пауза</span>}
        {task.verdict && (
          <span className={`task-flag task-verdict-${task.verdict}`}>
            вердикт: {task.verdict === 'pass' ? 'пройдено' : 'не пройдено'}
          </span>
        )}
        <span className="task-hint">
          <span className="muted">{task.expected_action.actor === 'user' ? 'от вас:' : 'от агента:'}</span>{' '}
          {task.expected_action.text}
        </span>
        <div className="task-actions">
          <button
            type="button"
            disabled={busy}
            onClick={() => void chat.pauseTask(!task.paused)}
          >
            {task.paused ? 'продолжить' : 'пауза'}
          </button>
          {task.available.map((option) => (
            <button
              key={option.stage}
              type="button"
              disabled={busy || !option.allowed}
              title={option.reason ?? undefined}
              className={option.stage === taskProposal ? 'task-proposed' : undefined}
              onClick={() => move(option.stage)}
            >
              {moveLabel(task.stage, option.stage)}
            </button>
          ))}
        </div>
      </div>

      {task.plan_error && <p className="task-note task-note-bad">план не разобран: {task.plan_error}</p>}
      {task.available.filter((option) => !option.allowed && option.reason).map((option) => (
        <p className="task-note task-note-bad" key={option.stage}>
          {moveLabel(task.stage, option.stage)}: {option.reason}
        </p>
      ))}
      {taskProposal && (
        <p className="task-note">
          трекер предлагает перейти в «{STAGE_LABEL[taskProposal]}» — решение за вами
        </p>
      )}

      {confirm && (
        <ConfirmMove
          task={task}
          stage={confirm.stage}
          busy={busy}
          onCancel={() => setConfirm(null)}
          onApply={(options) => {
            setConfirm(null)
            void chat.moveTaskStage(confirm.stage, options)
          }}
        />
      )}

      {task.steps.length > 0 && (
        <div className="task-steps">
          <ol>
            {task.steps.map((step) => (
              <StepRow key={step.index} step={step} current={step.index === current} chat={chat} />
            ))}
          </ol>
          {task.criteria.length > 0 && (
            <div className="task-criteria">
              <strong>проверка:</strong>
              <ul>
                {task.criteria.map((item, index) => <li key={index}>{item}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </aside>
  )
}

function Stepper({ stage }: { stage: TaskStage }) {
  const position = STAGE_TRACK.indexOf(stage)
  return (
    <ol className="task-stepper" aria-label="этап задачи">
      {STAGE_TRACK.map((item, index) => (
        <li
          key={item}
          className={item === stage ? 'now' : index < position ? 'passed' : ''}
          aria-current={item === stage ? 'step' : undefined}
        >
          {STAGE_LABEL[item]}
        </li>
      ))}
      {stage === 'cancelled' && <li className="now">{STAGE_LABEL.cancelled}</li>}
    </ol>
  )
}

/**
 * Replanning and cancelling throw work away, so each is confirmed. Going back from
 * validation also collects the criteria that failed: they become new steps.
 */
function ConfirmMove({
  task,
  stage,
  busy,
  onCancel,
  onApply,
}: {
  task: Task
  stage: TaskStage
  busy: boolean
  onCancel: () => void
  onApply: (options: { reason?: string; failedCriteria: string[] }) => void
}) {
  const [reason, setReason] = useState('')
  const [failed, setFailed] = useState<string[]>([])
  const collects = task.stage === 'validation' && stage === 'execution' && task.criteria.length > 0

  return (
    <div className="task-confirm">
      <p>
        <strong>{moveLabel(task.stage, stage)}</strong>
        {stage === 'planning' && ' — шаги сбрасываются, выполненное остаётся в журнале'}
        {stage === 'cancelled' && ' — задача закрывается, рабочая память очищается'}
        {stage === 'done' && ' — задача закрывается, рабочая память очищается'}
      </p>
      {collects && (
        <fieldset className="task-failed">
          <legend>какие критерии не прошли — они станут новыми шагами</legend>
          {task.criteria.map((item, index) => (
            <label key={index}>
              <input
                type="checkbox"
                checked={failed.includes(item)}
                onChange={(event) => setFailed((prev) => (
                  event.target.checked ? [...prev, item] : prev.filter((value) => value !== item)
                ))}
              />
              {item}
            </label>
          ))}
        </fieldset>
      )}
      <input
        aria-label="причина перехода"
        placeholder="причина (попадёт в журнал)"
        value={reason}
        onChange={(event) => setReason(event.target.value)}
      />
      <div className="task-actions">
        <button
          type="button"
          disabled={busy || (collects && failed.length === 0)}
          onClick={() => onApply({ reason: reason.trim() || undefined, failedCriteria: failed })}
        >
          подтвердить
        </button>
        <button type="button" className="link" onClick={onCancel}>отмена</button>
      </div>
    </div>
  )
}

/** A step the human owns: mark it, rename it, or block it with a reason. */
function StepRow({
  step,
  current,
  chat,
}: {
  step: TaskStep
  current: boolean
  chat: ChatSession
}) {
  const [renaming, setRenaming] = useState<string | null>(null)

  function block() {
    const note = window.prompt(`Чем заблокирован шаг ${step.index}?`, step.note)
    if (note === null) return
    void chat.editTaskStep(step.index, { status: 'blocked', note })
  }

  return (
    <li className={`task-step task-step-${step.status}${current ? ' now' : ''}`}>
      <span className="task-step-index">{step.index}</span>
      {renaming === null ? (
        <span className="task-step-title">{step.title}</span>
      ) : (
        <input
          aria-label={`название шага ${step.index}`}
          value={renaming}
          autoFocus
          onChange={(event) => setRenaming(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') setRenaming(null)
            if (event.key !== 'Enter') return
            const title = renaming.trim()
            setRenaming(null)
            if (title && title !== step.title) void chat.editTaskStep(step.index, { title })
          }}
        />
      )}
      <select
        aria-label={`статус шага ${step.index}`}
        value={step.status}
        disabled={chat.busy}
        onChange={(event) => void chat.editTaskStep(step.index, {
          status: event.target.value as TaskStepStatus,
        })}
      >
        {STEP_STATUSES.map((status) => (
          <option key={status} value={status}>{STEP_STATUS_LABEL[status]}</option>
        ))}
      </select>
      <button type="button" className="link" disabled={chat.busy} onClick={() => setRenaming(step.title)}>
        переименовать
      </button>
      <button type="button" className="link" disabled={chat.busy} onClick={block}>блокер</button>
      {step.note && <small className="muted task-step-note">{step.note}</small>}
    </li>
  )
}
