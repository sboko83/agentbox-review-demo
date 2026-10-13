export type ApprovalMode = 'read-only' | 'ask' | 'auto'
export type ExecutorKind = 'local' | 'docker'
export type TaskStage = 'planning' | 'execution' | 'validation' | 'done' | 'cancelled'
export type TaskStepStatus = 'todo' | 'in_progress' | 'done' | 'blocked'
export type InvariantCategory = 'architecture' | 'decision' | 'stack' | 'business_rule'

export interface InvariantRule {
  id: string
  category: InvariantCategory
  text: string
}

export interface SessionInvariants {
  version_hash: string
  rules: InvariantRule[]
}

export interface InvariantAssignment {
  id: number
  version_hash: string
  previous_hash: string | null
  origin: 'create' | 'edit' | 'migration' | 'fork'
  created_at: string
}

export interface InvariantCheck {
  id: number
  invariant_hash: string
  target: 'request' | 'response' | 'tool'
  outcome: 'allow' | 'conflict' | 'error'
  rule_ids: string[]
  explanation: string
  alternative: string | null
  seq: number
  created_at: string
}

export interface TaskStep {
  index: number
  title: string
  status: TaskStepStatus
  note: string
}

/** One entry of the append-only task journal, stamped with its transcript row. */
export interface TaskTransition {
  id: number
  task_id: string
  kind: string
  from_stage: TaskStage | null
  to_stage: TaskStage
  paused: boolean
  actor: 'user' | 'tracker'
  reason: string | null
  seq: number
  created_at: string
}

/** The active task: the server also says which transitions it would accept. */
export interface Task {
  id: string
  session_id: string
  stage: TaskStage
  paused: boolean
  steps: TaskStep[]
  current_step: number | null
  criteria: string[]
  verdict: 'pass' | 'fail' | null
  plan_text: string
  plan_error: string | null
  revision: number
  result_revision: number
  approved_plan_hash: string | null
  approved_invariant_hash: string | null
  verdict_result_revision: number | null
  expected_action: { actor: 'user' | 'agent'; text: string }
  transitions: TaskTransition[]
  available: { stage: TaskStage; needs_force: boolean; reason: string | null; allowed: boolean }[]
  created_at: string
  updated_at: string
  closed_at: string | null
}

export interface AgentSpec {
  model: {
    provider: string
    model: string
    role: string
    reasoning_effort: 'low' | 'medium' | 'high' | null
  }
  sampling: {
    temperature: number | null
    top_p: number | null
    top_k: number | null
    frequency_penalty: number | null
    presence_penalty: number | null
    seed: number | null
    stop: string
  }
  output: {
    mode: 'free' | 'api' | 'prompt'
    max_output_tokens: number
    format_json: string
    rules: string
  }
  reasoning: { strategy: 'direct' | 'step_by_step' | 'self_prompt' | 'experts' }
  context: {
    mode: 'none' | 'window' | 'tokens' | 'summary'
    window_tokens: number | null
    overflow: 'fail' | 'drop' | 'send'
    budget_usd: number | null
    keep_last_messages?: number
    memory_limit_tokens?: number
    summary_every?: number
    summary_model?: string
    summary_max_tokens?: number
  }
  memory: {
    working_enabled: boolean
    working_max_tokens: number
    longterm_enabled: boolean
    longterm_scope: string
    longterm_max_tokens: number
    longterm_write: LongTermWrite
    model: string
    max_tokens: number
  }
  tools: {
    workspace_path: string | null
    executor: ExecutorKind
    approval_mode: ApprovalMode
    max_tool_steps: number
    command_timeout_s: number
  }
  task: {
    tracker_enabled: boolean
    model: string
    max_tokens: number
    auto_transitions: boolean
  }
  judge: {
    model: string
    criteria: string
    threshold: number
    failure_action: 'as_is' | 'retry' | 'fallback'
    fallback_model: string | null
  }
}

export interface JsonSchema {
  title?: string
  description?: string
  type?: string
  properties?: Record<string, JsonSchema>
  $defs?: Record<string, JsonSchema>
  $ref?: string
  default?: unknown
  enum?: string[]
  oneOf?: JsonSchema[]
  const?: string
  anyOf?: JsonSchema[]
  maximum?: number
  minimum?: number
  maxLength?: number
  items?: JsonSchema
  ui?: string
  group?: string
  capability?: string
  advanced?: boolean
  when?: { field: string; values?: string[]; truthy?: boolean }
}

export interface ModelInfo {
  provider: string
  id: string
  name: string
  family: string
  transport: string
  input_usd_per_million: number | null
  output_usd_per_million: number | null
  context_window: number | null
  /** Sampling parameters this model accepts; the rest are greyed out. */
  supported_params: string[]
  /** False when the price ceiling keeps the model out of the pickers. */
  offered: boolean
}

export interface CostQuote {
  total_usd: number
  agent_calls: number
  judge_calls: number
  by_config: Array<{
    index: number
    name: string
    provider: string
    model: string
    judge_model: string
    maximum_usd: number
  }>
}

export interface MatrixQuote { quote_id: string; quote: CostQuote }

/** The profile version a run configuration sent, fixed when it was quoted. */
export interface RunProfile { hash: string; profile: UserProfile }

export interface ComparisonRow {
  config_index: number
  name: string
  user_profile: RunProfile | null
  provider: string
  model: string
  runs: number
  passed: number
  average_score: number | null
  actual_cost_usd: number
}

export interface RunReport {
  id: string
  status: string
  estimated_cost_usd: number
  totals: Totals
  comparison: ComparisonRow[]
  results: Array<{ config_index: number; repeat_index: number; answer?: string }>
}

export interface PoolAgent {
  id: string
  config_index: number
  repeat_index: number
  name: string
  provider: string
  model: string
  state: 'queued' | 'running' | 'completed' | 'failed' | 'budget_stopped'
  error: string | null
}

export interface PoolDashboard {
  run_id: string
  active: boolean
  agents: PoolAgent[]
  budget: (Totals & { budget_usd: number | null; exhausted: boolean }) | null
  provider: { limit: number; active: number; waiting: number }
  executors: { limit: number; active: number; waiting: number }
}

export interface ChatMessage {
  id?: number
  seq: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

export interface ToolMessage {
  id?: number
  seq: number
  role: 'tool'
  created_at: string
  call_id: string
  tool: string
  arguments_json: string
  status: 'ok' | 'error' | 'rejected'
  content: string
  exit_code: number | null
  duration_ms: number | null
}

export type StoredMessage = ChatMessage | ToolMessage

export interface ContextMessage {
  seq: number
  role: 'user' | 'assistant' | 'tool'
  content: string
  call_id?: string
  tool?: string
  arguments_json?: string
}

export interface ContextView {
  mode: AgentSpec['context']['mode']
  role: { content: string; tokens: number }
  /** Complete rules block; unlike memory it is never trimmed to a sub-budget. */
  invariants: {
    hash: string
    rules: InvariantRule[]
    content: string
    payload_tokens: number
  } | null
  /** The rendered block exactly as the next call carries it. */
  user_profile: { name: string; hash: string; content: string; payload_tokens: number } | null
  /** The task block of the next call; absent while no task is active. */
  task: { stage: TaskStage; paused: boolean; content: string; payload_tokens: number } | null
  summary: {
    id?: number
    version?: number
    covered_seq?: number
    message_count?: number
    text?: string
    tokens?: number
    created_at?: string
    payload_tokens?: number
  } | null
  working: ContextLayer | null
  longterm: ContextLayer | null
  tail: { first_seq: number | null; messages: ContextMessage[]; tokens: number }
  dropped_seqs: number[]
  estimated_tokens: number
  /** What the estimate holds beyond the layers: envelope and tool schemas. */
  overhead_tokens: number
  limit_tokens: number | null
  limit_source: 'spec' | 'model' | null
  overflow: AgentSpec['context']['overflow']
}

/** A memory version as the next call carries it: stored plus its payload cost.
 *
 * `dropped_keys` are the items the layer's token budget left out of this call;
 * they stay in the stored version.
 */
export type ContextLayer = MemoryVersion & {
  payload_tokens?: number
  dropped_keys?: string[]
}

export interface Totals {
  requests: number
  tokens_in: number
  tokens_out: number
  reasoning: number
  cost_usd: number
}

export interface TokenUsage {
  id: number
  spec_hash: string
  message_id: number | null
  model: string
  kind: 'chat' | 'summary' | 'memory' | 'task' | 'invariant'
  input_tokens: number
  output_tokens: number
  reasoning_tokens: number
  cost_usd: number
  role_estimate_tokens: number
  history_estimate_tokens: number
  request_estimate_tokens: number
  input_estimate_tokens: number
  context_estimate_tokens: number
  context_input_tokens: number
  context_output_tokens: number
  context_limit_tokens: number | null
  context_limit_source: 'spec' | 'model' | null
  dropped_messages: number
  latency_ms: number
  estimate_delta_tokens: number
  created_at: string
  /** The profile version this call's payload carried; null when it had none. */
  user_profile_hash: string | null
  /** Rules snapshot fixed at the start of the turn. */
  invariant_hash: string | null
}

export interface SessionSummaryVersion {
  id: number
  version: number
  covered_seq: number
  message_count: number
  summary: string
  summary_tokens: number
  created_at: string
}

/** One item of a memory layer, with the route that put it there. */
export interface MemoryItem {
  value: string
  kind: string
  origin: 'router' | 'manual' | 'promote' | 'migration'
  seq: number
}

export interface MemoryVersion {
  id: number
  version: number
  covered_seq: number
  origin: MemoryItem['origin']
  items: Record<string, MemoryItem>
  tokens: number
  created_at: string
}

export type MemoryLayer = 'working' | 'longterm'

/** What a hand write sends: the value and the kind it joins the layer under. */
export interface MemoryItemInput {
  value: string
  kind: string
}

/** Each layer answers to its own vocabulary; an item names a kind of its layer. */
export const LAYER_KINDS: Record<MemoryLayer, string[]> = {
  working: ['goal', 'constraint', 'step', 'artifact', 'blocker'],
  longterm: ['profile', 'decision', 'knowledge'],
}

/** Router output waiting for a verdict; an empty value proposes forgetting. */
export interface MemoryProposal {
  id: number
  /** A `profile` candidate names a profile field; its value is JSON. */
  layer: MemoryLayer | 'profile'
  key: string
  value: string
  kind: string
  source_seq: number
  created_at: string
}

export type LongTermWrite = 'auto' | 'confirm' | 'manual'

export interface SessionMemory {
  /** Profile learning: candidates only, and only while the router runs. */
  profile: { enabled: boolean; router: boolean; proposals: MemoryProposal[] }
  working: { enabled: boolean; versions: MemoryVersion[]; proposals: MemoryProposal[] }
  longterm: {
    enabled: boolean
    scope: string | null
    /** Sessions joined to that scope: how far the layer reaches past this one. */
    sessions: number
    write: LongTermWrite | null
    versions: MemoryVersion[]
    proposals: MemoryProposal[]
  }
}

export interface Checkpoint {
  session_id: string
  seq: number
  name: string
}

export interface SessionDetail {
  id: string
  title: string | null
  created_at: string
  editable: boolean
  spec_hash: string
  workspace_path: string | null
  executor_kind: ExecutorKind
  task: Task | null
  /** Transitions of every task of the session, closed ones included. */
  task_journal: TaskTransition[]
  spec: AgentSpec | null
  user_profile_id: string | null
  user_profile: UserProfilePointer | null
  invariants: SessionInvariants
  invariant_history: InvariantAssignment[]
  invariant_checks: InvariantCheck[]
  messages: StoredMessage[]
  context: ContextView | null
  totals: Totals
  usage: TokenUsage[]
  summaries: SessionSummaryVersion[]
  memory: SessionMemory
  checkpoints: Checkpoint[]
  parent_id: string | null
  fork_seq: number | null
}

export interface Preset {
  id: string
  name: string
  is_default: boolean
  spec_hash: string
  spec: AgentSpec
  created_at: string
  updated_at: string
}

/** One row of the session list: enough to pick a session without loading it. */
export interface SessionSummary {
  id: string
  title: string | null
  created_at: string
  updated_at: string
  spec: AgentSpec
  executor_kind: ExecutorKind
  task_stage: TaskStage | null
  task_paused: boolean
  parent_id: string | null
  fork_seq: number | null
  messages: number
  cost_usd: number
  user_profile_id: string | null
  user_profile_name: string | null
  invariant_hash: string
}

/** How to answer this person; the form is generated from its JSON Schema. */
export interface UserProfile {
  name: string
  about: string
  language: 'ru' | 'en' | 'auto'
  tone: 'neutral' | 'friendly' | 'strict'
  verbosity: 'brief' | 'normal' | 'detailed'
  structure: 'prose' | 'bullets' | 'steps'
  max_words: number | null
  code_style: 'none' | 'minimal' | 'full'
  constraints: string[]
}

/** A named pointer to an immutable profile version. */
export interface UserProfilePointer {
  id: string
  name: string
  version_hash: string
  is_default: boolean
  profile: UserProfile
  created_at: string
  updated_at: string
}

export type StreamEvent =
  | { type: 'context'; view: ContextView }
  | {
      type: 'invariant_check'
      target: InvariantCheck['target']
      outcome: InvariantCheck['outcome']
      rule_ids: string[]
      explanation: string
      alternative: string | null
      model: string
      usage: { tokens_in: number; tokens_out: number; reasoning: number; cost_usd: number }
      input_estimate_tokens: number
      latency_ms: number
    }
  | { type: 'delta'; text: string }
  | { type: 'approval'; call_id: string; tool: string; arguments: string; preview: string }
  | { type: 'tool'; call_id: string; tool: string; arguments: string; preview: string }
  | {
      type: 'tool_result'
      call_id: string
      tool: string
      output: string
      ok: boolean
      exit_code: number | null
      duration_ms: number
    }
  | {
      type: 'usage'
      estimate: {
        role_tokens: number
        history_tokens: number
        request_tokens: number
        input_tokens: number
      }
      actual: {
        input_tokens: number
        output_tokens: number
        reasoning_tokens: number
        cost_usd: number
        estimate_delta_tokens: number
        latency_ms: number
      }
      context: {
        estimated_tokens: number
        input_tokens: number
        output_tokens: number
        limit_tokens: number | null
        limit_source: 'spec' | 'model' | null
        dropped_messages: number
        chars_per_token: number
      }
      ledger: Totals & { budget_usd: number | null }
    }
  | ({ type: 'summary' } & SessionSummaryVersion)
  | ({ type: 'memory'; layer: MemoryLayer } & MemoryVersion)
  | { type: 'memory_proposals'; layer: MemoryLayer | 'profile'; proposals: MemoryProposal[] }
  | { type: 'memory_rejected'; keys: string[] }
  | { type: 'memory_error'; message: string }
  | { type: 'task'; task: Task | null; proposed?: TaskStage | null }
  | { type: 'task_rejected'; reasons: string[] }
  | { type: 'task_error'; message: string }
  | {
      type: 'done'
      text: string
      usage: Record<string, number>
      ledger: Record<string, number | null>
      memory_tokens: number
      dropped: number
      steps: number
    }
  | { type: 'error'; message: string }

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(errorText(detail) || `HTTP ${response.status}`)
  }
  return (await response.json()) as T
}

/** FastAPI answers `{detail}`: a sentence, a task refusal, or a 422 field list. */
function errorText(body: string): string {
  let parsed: unknown
  try {
    parsed = JSON.parse(body)
  } catch {
    return body
  }
  const detail = (parsed as { detail?: unknown })?.detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const refusal = detail as {
      message?: string
      next_action?: { actor?: string; text?: string }
    }
    if (refusal.message) {
      return refusal.next_action?.text
        ? `${refusal.message}. Следующее действие: ${refusal.next_action.text}`
        : refusal.message
    }
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item: { loc?: unknown[]; msg?: string }) => {
        const field = (item.loc ?? []).filter((part) => part !== 'body').join('.')
        return field ? `${field}: ${item.msg}` : String(item.msg)
      })
      .join('; ')
  }
  return body
}

export interface CreateSessionBody {
  title?: string
  spec?: Record<string, unknown>
  presetId?: string
  /** Omitted: the default profile; null: a session without one. */
  userProfileId?: string | null
}

export async function createSession(options: CreateSessionBody): Promise<{ id: string }> {
  return json(
    await fetch('/api/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: options.title ?? null,
        ...(options.presetId ? { preset_id: options.presetId } : {}),
        ...(options.spec ? { spec: options.spec } : {}),
        ...(options.userProfileId !== undefined ? { user_profile_id: options.userProfileId } : {}),
      }),
    }),
  )
}

export async function getAgentSpecSchema(): Promise<JsonSchema> {
  return json(await fetch('/api/agent-spec-schema'))
}

export async function getModels(): Promise<ModelInfo[]> {
  return json(await fetch('/api/models'))
}

export async function listPresets(): Promise<Preset[]> {
  return json(await fetch('/api/presets'))
}

export async function createPreset(body: {
  name: string
  spec: Record<string, unknown>
  is_default?: boolean
}): Promise<Preset> {
  return json(await fetch('/api/presets', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }))
}

export async function updatePreset(
  id: string,
  body: { name: string; spec: Record<string, unknown> },
): Promise<Preset> {
  return json(await fetch(`/api/presets/${id}`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }))
}

export async function setDefaultPreset(id: string): Promise<Preset> {
  return json(await fetch(`/api/presets/${id}/default`, { method: 'PUT' }))
}

export async function deletePreset(id: string): Promise<void> {
  await json(await fetch(`/api/presets/${id}`, { method: 'DELETE' }))
}

export async function getUserProfileSchema(): Promise<JsonSchema> {
  return json(await fetch('/api/user-profiles/schema'))
}

export async function listUserProfiles(): Promise<UserProfilePointer[]> {
  return json(await fetch('/api/user-profiles'))
}

export async function createUserProfile(
  body: Partial<UserProfile> & { name: string; is_default?: boolean },
): Promise<UserProfilePointer> {
  return json(await fetch('/api/user-profiles', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }))
}

export async function updateUserProfile(id: string, body: UserProfile): Promise<UserProfilePointer> {
  return json(await fetch(`/api/user-profiles/${id}`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }))
}

export async function setDefaultUserProfile(id: string): Promise<UserProfilePointer> {
  return json(await fetch(`/api/user-profiles/${id}/default`, { method: 'PUT' }))
}

export async function deleteUserProfile(id: string): Promise<void> {
  await json(await fetch(`/api/user-profiles/${id}`, { method: 'DELETE' }))
}

export async function setSessionUserProfile(
  id: string,
  userProfileId: string | null,
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_profile_id: userProfileId }),
  }))
}

export async function updateSessionInvariants(
  id: string,
  expectedVersionHash: string,
  rules: InvariantRule[],
  replanTask: boolean,
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/invariants`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      expected_version_hash: expectedVersionHash,
      rules,
      replan_task: replanTask,
    }),
  }))
}

export async function estimateRun(body: {
  prompt: string
  repeats: number
  budget_usd: number | null
  specs?: Record<string, unknown>[]
  preset_ids?: string[]
  user_profile_ids?: string[]
  invariants?: { rules: InvariantRule[] }
}): Promise<MatrixQuote> {
  return json(await fetch('/api/runs/estimate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }))
}

export async function startRun(quoteId: string): Promise<PoolDashboard> {
  return json(await fetch('/api/runs', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ quote_id: quoteId }),
  }))
}

export async function getRun(id: string): Promise<RunReport> {
  return json(await fetch(`/api/runs/${id}`))
}

export async function getRunDashboard(id: string): Promise<PoolDashboard> {
  return json(await fetch(`/api/runs/${id}/dashboard`))
}

export async function getSession(id: string): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}`))
}

export async function updateSessionSpec(
  id: string,
  spec: Record<string, unknown>,
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/spec`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(spec),
  }))
}

/** Rewrite one layer: every item names its own kind, an empty value drops it. */
export async function updateSessionMemory(
  id: string,
  layer: MemoryLayer,
  items: Record<string, MemoryItemInput>,
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/memory/${layer}`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ items }),
  }))
}

/** Remember one thing without resending a layer the browser may have stale. */
export async function writeMemoryItem(
  id: string,
  layer: MemoryLayer,
  item: MemoryItemInput & { key: string; seq?: number },
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/memory/${layer}/items`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(item),
  }))
}

export async function forgetMemoryItem(
  id: string,
  layer: MemoryLayer,
  key: string,
): Promise<SessionDetail> {
  return json(await fetch(
    `/api/sessions/${id}/memory/${layer}/items/${encodeURIComponent(key)}`,
    { method: 'DELETE' },
  ))
}

/** Re-route one item to the other layer under a kind of that layer. */
export async function moveMemoryItem(
  id: string,
  layer: MemoryLayer,
  key: string,
  kind: string,
): Promise<SessionDetail> {
  return json(await fetch(
    `/api/sessions/${id}/memory/${layer}/items/${encodeURIComponent(key)}/move`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind }) },
  ))
}

/** Open a task on the session; the answer is the session with its strip. */
export async function startTask(id: string): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/task`, { method: 'POST' }))
}

/** Ask for one version-checked move; failed criteria become remediation steps. */
export async function moveTask(
  id: string,
  stage: TaskStage,
  expectedRevision: number,
  expectedInvariantHash: string,
  options: { reason?: string; failedCriteria?: string[] } = {},
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/task/transition`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      stage,
      reason: options.reason ?? null,
      failed_criteria: options.failedCriteria ?? [],
      expected_revision: expectedRevision,
      expected_invariant_hash: expectedInvariantHash,
    }),
  }))
}

export async function setTaskPaused(
  id: string,
  paused: boolean,
  expectedRevision: number,
): Promise<SessionDetail> {
  return json(await fetch(
    `/api/sessions/${id}/task/${paused ? 'pause' : 'resume'}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expected_revision: expectedRevision }),
    },
  ))
}

/** A human correction of one step: status, title or the tracker's note. */
export async function updateTaskStep(
  id: string,
  index: number,
  expectedRevision: number,
  patch: { status?: TaskStepStatus; title?: string; note?: string },
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/task/steps/${index}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...patch, expected_revision: expectedRevision }),
  }))
}

/** End the task: promote the named items, then empty the working layer. */
export async function finishTask(
  id: string,
  promote: Record<string, string>,
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/memory/working/finish`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ promote }),
  }))
}

export async function decideProposal(
  id: string,
  proposalId: number,
  decision: 'accept' | 'reject',
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/memory/proposals/${proposalId}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision }),
  }))
}

export async function createCheckpoint(
  id: string,
  seq: number,
  name?: string,
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/checkpoints`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seq, name: name || null }),
  }))
}

export async function forkSession(
  id: string,
  seq: number,
  title?: string,
  spec?: Record<string, unknown>,
): Promise<SessionDetail> {
  return json(await fetch(`/api/sessions/${id}/fork`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seq, title: title || null, ...(spec ? { spec } : {}) }),
  }))
}

export async function listSessions(): Promise<SessionSummary[]> {
  return json(await fetch('/api/sessions'))
}

export async function deleteSession(id: string): Promise<void> {
  await json(await fetch(`/api/sessions/${id}`, { method: 'DELETE' }))
}

/** Answers a pending confirmation; the parked stream resumes on this call. */
export async function approveCall(
  sessionId: string,
  callId: string,
  approved: boolean,
): Promise<void> {
  await fetch(`/api/sessions/${sessionId}/approvals/${callId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved }),
  })
}

/**
 * Streams the answer. EventSource only speaks GET, so the SSE frames are read
 * off a POST response body and split by hand.
 */
export async function* sendMessage(
  sessionId: string,
  text: string,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`/api/sessions/${sessionId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    signal,
  })

  if (!response.ok || !response.body) {
    const detail = await response.text()
    yield { type: 'error', message: detail || `HTTP ${response.status}` }
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const line = frame.split('\n').find((l) => l.startsWith('data: '))
      if (line) yield JSON.parse(line.slice(6)) as StreamEvent
      boundary = buffer.indexOf('\n\n')
    }
  }
}
