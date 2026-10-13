import { useCallback, useEffect, useRef, useState } from 'react'

import {
  approveCall,
  createCheckpoint,
  createSession,
  setSessionUserProfile,
  decideProposal,
  finishTask,
  deleteSession,
  forgetMemoryItem,
  forkSession,
  getSession,
  listSessions,
  moveMemoryItem,
  moveTask,
  sendMessage,
  setTaskPaused,
  startTask,
  updateTaskStep,
  updateSessionMemory,
  updateSessionInvariants,
  updateSessionSpec,
  writeMemoryItem,
  type MemoryItemInput,
  type InvariantRule,
  type MemoryLayer,
  type SessionDetail,
  type SessionSummary,
  type StoredMessage,
  type StreamEvent,
  type TaskStage,
  type TaskStepStatus,
  type TokenUsage,
} from '../api/sessions'
import type { LiveTool } from '../components/tools'

const SESSION_KEY = 'agentbox.session'

/**
 * Everything the chat needs to run: the open session, its transcript and the
 * live state of one streamed answer. Kept out of the components so that the
 * three-zone layout only arranges markup and never owns dialogue state.
 */
export function useChatSession() {
  const [session, setSession] = useState<SessionDetail | null>(null)
  const [messages, setMessages] = useState<StoredMessage[]>([])
  const [streaming, setStreaming] = useState('')
  const [live, setLive] = useState<LiveTool[]>([])
  const [pending, setPending] = useState<LiveTool | null>(null)
  const [busy, setBusy] = useState(false)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  // What the tracker asked for and the human has not answered yet: a proposal
  // is not a transition, so it lives beside the task instead of inside it.
  const [taskProposal, setTaskProposal] = useState<TaskStage | null>(null)
  const [draftVersion, setDraftVersion] = useState(0)
  const draftSpec = useRef<Record<string, unknown> | null>(null)
  const [draftSpecValue, setDraftSpecValue] = useState<Record<string, unknown> | null>(null)
  // undefined leaves the choice to the server: the default profile.
  const draftUserProfile = useRef<string | null | undefined>(undefined)
  const specUpdateQueue = useRef<Promise<void>>(Promise.resolve())

  const refreshSessions = useCallback(async () => {
    try {
      setSessions((await listSessions()).filter((item) => item.messages > 0))
    } catch {
      // The list is a convenience; a failure must not break the open session.
    }
  }, [])

  const show = useCallback((detail: SessionDetail) => {
    localStorage.setItem(SESSION_KEY, detail.id)
    setSession(detail)
    setMessages(detail.messages)
    setStreaming('')
    setLive([])
    setPending(null)
    setTaskProposal(null)
  }, [])

  const send = useCallback(
    async (text: string) => {
      if (busy) return
      setError(null)
      setBusy(true)
      setLive([])
      setPending(null)
      setTaskProposal(null)

      let answer = ''
      try {
        let activeSession = session
        if (!activeSession) {
          setCreating(true)
          const created = await createSession({
            ...(draftSpec.current ? { spec: draftSpec.current } : {}),
            userProfileId: draftUserProfile.current,
          })
          activeSession = await getSession(created.id)
          show(activeSession)
          await refreshSessions()
          setCreating(false)
        }
        setMessages((prev) => [...prev, localMessage(prev, 'user', text)])

        for await (const chunk of sendMessage(activeSession.id, text)) {
          if (chunk.type === 'context') {
            setSession((current) => current && ({ ...current, context: chunk.view }))
          } else if (chunk.type === 'invariant_check') {
            const target = { request: 'запрос', response: 'ответ', tool: 'действие' }[chunk.target]
            if (chunk.outcome === 'allow') {
              setStreaming(`✓ Инварианты: ${target} допустим`)
            } else if (chunk.outcome === 'conflict') {
              setStreaming(`Инварианты: конфликт [${chunk.rule_ids.join(', ')}], формирую отказ…`)
            } else {
              setStreaming('Инварианты: проверка не выполнена, действие остановлено')
            }
          } else if (chunk.type === 'delta') {
            answer += chunk.text
            setStreaming(answer)
          } else if (chunk.type === 'approval') {
            setPending({ ...chunk, arguments: chunk.arguments, status: 'running' })
          } else if (chunk.type === 'tool') {
            setPending(null)
            setLive((prev) => [...prev, { ...chunk, status: 'running' }])
          } else if (chunk.type === 'tool_result') {
            setPending(null)
            setLive((prev) => settle(prev, chunk))
          } else if (chunk.type === 'usage') {
            setSession((current) => current && ({
              ...current,
              totals: {
                requests: chunk.ledger.requests,
                tokens_in: chunk.ledger.tokens_in,
                tokens_out: chunk.ledger.tokens_out,
                reasoning: chunk.ledger.reasoning,
                cost_usd: chunk.ledger.cost_usd,
              },
              usage: [...current.usage, liveUsage(current, chunk)],
            }))
          } else if (chunk.type === 'summary') {
            setSession((current) => current && ({
              ...current,
              summaries: [...current.summaries, chunk],
            }))
          } else if (chunk.type === 'memory') {
            const { layer, ...version } = chunk
            setSession((current) => current && ({
              ...current,
              memory: {
                ...current.memory,
                [layer]: {
                  ...current.memory[layer],
                  versions: [...current.memory[layer].versions, version],
                },
              },
              context: current.context ? { ...current.context, [layer]: version } : null,
            }))
          } else if (chunk.type === 'memory_proposals') {
            const { layer, proposals } = chunk
            setSession((current) => current && ({
              ...current,
              memory: {
                ...current.memory,
                [layer]: { ...current.memory[layer], proposals },
              },
            }))
          } else if (chunk.type === 'memory_rejected') {
            setError(
              `Роутер отклонил ключи: ${chunk.keys.join(', ')} — вид не из своего слоя`
              + ' или поле и значение не по схеме профиля',
            )
          } else if (chunk.type === 'memory_error') {
            setError(`Память не обновлена: ${chunk.message}`)
          } else if (chunk.type === 'task') {
            setSession((current) => current && ({ ...current, task: chunk.task }))
            setTaskProposal(chunk.proposed ?? null)
          } else if (chunk.type === 'task_rejected') {
            setError(`Автомат отклонил трекер: ${chunk.reasons.join('; ')}`)
          } else if (chunk.type === 'task_error') {
            setError(`Трекер не отработал: ${chunk.message}`)
          } else if (chunk.type === 'done') {
            const detail = await getSession(activeSession.id)
            setSession(detail)
            setMessages(detail.messages)
            setStreaming('')
            setLive([])
          } else {
            setStreaming('')
            setLive([])
            setError(chunk.message)
            try {
              const detail = await getSession(activeSession.id)
              setSession(detail)
              setMessages(detail.messages)
            } catch {
              setMessages((prev) => prev.slice(0, -1))
            }
          }
        }
      } catch (cause) {
        setError(String(cause))
        setStreaming('')
      } finally {
        await refreshSessions()
        setPending(null)
        setBusy(false)
        setCreating(false)
      }
    },
    [busy, refreshSessions, session, show],
  )

  const answerApproval = useCallback(
    async (approved: boolean) => {
      if (!session || !pending) return
      setPending(null)
      await approveCall(session.id, pending.call_id, approved)
    },
    [pending, session],
  )

  const newDraft = useCallback(
    async () => {
      if (busy || creating) return
      localStorage.removeItem(SESSION_KEY)
      draftSpec.current = null
      setDraftSpecValue(null)
      setSession(null)
      setMessages([])
      setStreaming('')
      setLive([])
      setPending(null)
      setTaskProposal(null)
      setError(null)
      setDraftVersion((version) => version + 1)
    },
    [busy, creating],
  )

  const updateSpec = useCallback(
    (spec: Record<string, unknown>): Promise<void> => {
      const sessionId = session?.id
      if (!sessionId) {
        draftSpec.current = spec
        setDraftSpecValue(spec)
        return Promise.resolve()
      }

      const update = specUpdateQueue.current
        .catch(() => undefined)
        .then(async () => {
          const detail = await updateSessionSpec(sessionId, spec)
          setSession((current) => current?.id === sessionId ? detail : current)
          void refreshSessions()
        })
      specUpdateQueue.current = update.catch(() => undefined)
      return update
    },
    [refreshSessions, session?.id],
  )

  // Every memory write answers with the whole session, so one applier keeps
  // the panel, the transcript markers and the context estimate in one state.
  const applyMemory = useCallback(
    async (write: (sessionId: string) => Promise<SessionDetail>): Promise<void> => {
      const sessionId = session?.id
      if (!sessionId) throw new Error('Сессия ещё не открыта')
      const detail = await write(sessionId)
      setSession((current) => current?.id === sessionId ? detail : current)
    },
    [session?.id],
  )

  const updateMemory = useCallback(
    (layer: MemoryLayer, items: Record<string, MemoryItemInput>): Promise<void> =>
      applyMemory((id) => updateSessionMemory(id, layer, items)),
    [applyMemory],
  )

  const updateInvariants = useCallback(
    (expectedVersionHash: string, rules: InvariantRule[], replanTask: boolean): Promise<void> =>
      applyMemory((id) => updateSessionInvariants(id, expectedVersionHash, rules, replanTask)),
    [applyMemory],
  )

  const rememberItem = useCallback(
    (layer: MemoryLayer, item: MemoryItemInput & { key: string; seq?: number }): Promise<void> =>
      applyMemory((id) => writeMemoryItem(id, layer, item)),
    [applyMemory],
  )

  const forgetItem = useCallback(
    (layer: MemoryLayer, key: string): Promise<void> =>
      applyMemory((id) => forgetMemoryItem(id, layer, key)),
    [applyMemory],
  )

  const moveItem = useCallback(
    (layer: MemoryLayer, key: string, kind: string): Promise<void> =>
      applyMemory((id) => moveMemoryItem(id, layer, key, kind)),
    [applyMemory],
  )

  /** Every task route answers with the whole session, so one applier is enough. */
  const applyTask = useCallback(
    async (write: (sessionId: string) => Promise<SessionDetail>): Promise<void> => {
      const sessionId = session?.id
      if (!sessionId || busy) return
      setError(null)
      try {
        const detail = await write(sessionId)
        setSession((current) => current?.id === sessionId ? detail : current)
        setTaskProposal(null)
        void refreshSessions()
      } catch (cause) {
        setError(String(cause))
      }
    },
    [busy, refreshSessions, session?.id],
  )

  const openTask = useCallback(
    (): Promise<void> => applyTask((id) => startTask(id)),
    [applyTask],
  )

  const moveTaskStage = useCallback(
    (
      stage: TaskStage,
      options: { reason?: string; failedCriteria?: string[] } = {},
    ): Promise<void> => {
      const task = session?.task
      const invariantHash = session?.invariants.version_hash
      if (!task || !invariantHash) return Promise.resolve()
      return applyTask((id) => moveTask(id, stage, task.revision, invariantHash, options))
    },
    [applyTask, session?.invariants.version_hash, session?.task],
  )

  const pauseTask = useCallback(
    (paused: boolean): Promise<void> => {
      const task = session?.task
      if (!task) return Promise.resolve()
      return applyTask((id) => setTaskPaused(id, paused, task.revision))
    },
    [applyTask, session?.task],
  )

  const editTaskStep = useCallback(
    (
      index: number,
      patch: { status?: TaskStepStatus; title?: string; note?: string },
    ): Promise<void> => {
      const task = session?.task
      if (!task) return Promise.resolve()
      return applyTask((id) => updateTaskStep(id, index, task.revision, patch))
    },
    [applyTask, session?.task],
  )

  const finishWorkingTask = useCallback(
    (promote: Record<string, string>): Promise<void> =>
      applyMemory((id) => finishTask(id, promote)),
    [applyMemory],
  )

  const setDraftUserProfile = useCallback((id: string | null | undefined) => {
    draftUserProfile.current = id
  }, [])

  const loadExperiment = useCallback((spec: Record<string, unknown>, profileId: string | null, detail: SessionDetail | null) => {
    draftSpec.current = spec
    setDraftSpecValue(spec)
    draftUserProfile.current = profileId
    if (detail) show(detail)
    else {
      localStorage.removeItem(SESSION_KEY)
      setSession(null); setMessages([]); setStreaming(''); setLive([]); setPending(null)
      setTaskProposal(null); setError(null)
    }
    setDraftVersion((version) => version + 1)
    void refreshSessions()
  }, [show, refreshSessions])

  const changeUserProfile = useCallback(
    async (userProfileId: string | null): Promise<void> => {
      await applyMemory((id) => setSessionUserProfile(id, userProfileId))
      void refreshSessions()
    },
    [applyMemory, refreshSessions],
  )

  /** Re-read the open session after a change made elsewhere, e.g. its profile. */
  const reloadSession = useCallback(async (): Promise<void> => {
    if (session?.id) await applyMemory((id) => getSession(id))
  }, [applyMemory, session?.id])

  const decideMemory = useCallback(
    (proposalId: number, decision: 'accept' | 'reject'): Promise<void> =>
      applyMemory((id) => decideProposal(id, proposalId, decision)),
    [applyMemory],
  )

  const checkpoint = useCallback(
    async (seq: number, name?: string): Promise<void> => {
      if (!session || busy) return
      setError(null)
      setBusy(true)
      try {
        const detail = await createCheckpoint(session.id, seq, name)
        setSession(detail)
      } catch (cause) {
        setError(String(cause))
      } finally {
        setBusy(false)
      }
    },
    [busy, session],
  )

  const fork = useCallback(
    async (seq: number, title?: string): Promise<void> => {
      if (!session || busy) return
      setError(null)
      setBusy(true)
      try {
        show(await forkSession(session.id, seq, title))
        await refreshSessions()
      } catch (cause) {
        setError(String(cause))
      } finally {
        setBusy(false)
      }
    },
    [busy, refreshSessions, session, show],
  )

  const open = useCallback(
    async (id: string) => {
      if (busy || id === session?.id) return
      setError(null)
      try {
        show(await getSession(id))
      } catch (cause) {
        setError(String(cause))
      }
    },
    [busy, session, show],
  )

  /** Deleting the open session returns to an unsaved draft. */
  const drop = useCallback(
    async (id: string) => {
      if (busy) return
      setError(null)
      try {
        await deleteSession(id)
        if (id === session?.id) {
          localStorage.removeItem(SESSION_KEY)
          draftSpec.current = null
          setDraftSpecValue(null)
          setSession(null)
          setMessages([])
          setStreaming('')
          setLive([])
          setPending(null)
          setDraftVersion((version) => version + 1)
        }
        await refreshSessions()
      } catch (cause) {
        setError(String(cause))
      }
    },
    [busy, refreshSessions, session],
  )

  // oxlint-disable react/set-state-in-effect -- the open session and the
  // session list both arrive from the API after mount; neither has a
  // render-time source to derive them from.
  useEffect(() => {
    let disposed = false
    const restoreSession = async () => {
      const detail = await restoreStored()
      if (disposed) return
      if (detail) {
        setSession(detail)
        setMessages(detail.messages)
      }
    }
    void restoreSession()
    void refreshSessions()
    return () => {
      disposed = true
    }
  }, [refreshSessions])
  // oxlint-enable react/set-state-in-effect

  return {
    session,
    sessions,
    draftVersion,
    draftSpec: draftSpecValue,
    messages,
    streaming,
    live,
    pending,
    busy,
    creating,
    error,
    setError,
    send,
    answerApproval,
    newDraft,
    updateSpec,
    updateMemory,
    updateInvariants,
    rememberItem,
    forgetItem,
    moveItem,
    finishWorkingTask,
    taskProposal,
    openTask,
    moveTaskStage,
    pauseTask,
    editTaskStep,
    decideMemory,
    setDraftUserProfile,
    loadExperiment,
    changeUserProfile,
    reloadSession,
    checkpoint,
    fork,
    open,
    drop,
    refreshSessions,
  }
}

export type ChatSession = ReturnType<typeof useChatSession>

function settle(
  tools: LiveTool[],
  result: Extract<StreamEvent, { type: 'tool_result' }>,
): LiveTool[] {
  const status: LiveTool['status'] = result.ok
    ? 'ok'
    : result.output.startsWith('пользователь отклонил')
      ? 'rejected'
      : 'error'
  const settled = tools.map((tool) =>
    tool.call_id === result.call_id
      ? {
          ...tool,
          status,
          output: result.output,
          exit_code: result.exit_code,
          duration_ms: result.duration_ms,
        }
      : tool,
  )
  // A refused call never produced a `tool` frame, so there is nothing to update.
  if (settled.some((tool) => tool.call_id === result.call_id)) return settled
  return [
    ...settled,
    {
      call_id: result.call_id,
      tool: result.tool,
      arguments: '{}',
      preview: '',
      status,
      output: result.output,
      exit_code: result.exit_code,
      duration_ms: result.duration_ms,
    },
  ]
}

function localMessage(
  previous: StoredMessage[],
  role: 'user' | 'assistant',
  content: string,
): StoredMessage {
  const seq = (previous.at(-1)?.seq ?? 0) + 1
  return { seq, role, content, created_at: new Date().toISOString() }
}

function liveUsage(
  session: SessionDetail,
  event: Extract<StreamEvent, { type: 'usage' }>,
): TokenUsage {
  return {
    id: (session.usage.at(-1)?.id ?? 0) + 1,
    spec_hash: '',
    message_id: null,
    model: session.spec?.model.model ?? '',
    kind: 'chat',
    input_tokens: event.actual.input_tokens,
    output_tokens: event.actual.output_tokens,
    reasoning_tokens: event.actual.reasoning_tokens,
    cost_usd: event.actual.cost_usd,
    role_estimate_tokens: event.estimate.role_tokens,
    history_estimate_tokens: event.estimate.history_tokens,
    request_estimate_tokens: event.estimate.request_tokens,
    input_estimate_tokens: event.estimate.input_tokens,
    context_estimate_tokens: event.context.estimated_tokens,
    context_input_tokens: event.context.input_tokens,
    context_output_tokens: event.context.output_tokens,
    context_limit_tokens: event.context.limit_tokens,
    context_limit_source: event.context.limit_source,
    dropped_messages: event.context.dropped_messages,
    latency_ms: event.actual.latency_ms,
    estimate_delta_tokens: event.actual.estimate_delta_tokens,
    created_at: new Date().toISOString(),
    // The context event of the same turn named the block the call carried.
    user_profile_hash: session.context?.user_profile?.hash ?? null,
    invariant_hash: session.invariants.version_hash,
  }
}

async function restoreStored(): Promise<SessionDetail | null> {
  const stored = localStorage.getItem(SESSION_KEY)
  if (!stored) return null
  try {
    const detail = await getSession(stored)
    if (detail.messages.length > 0) return detail
  } catch {
    // A removed session is indistinguishable from a fresh browser here.
  }
  localStorage.removeItem(SESSION_KEY)
  return null
}
