import type { ToolMessage } from '../api/sessions'

/** A tool call while the turn is still running; the stored row is its echo. */
export interface LiveTool {
  call_id: string
  tool: string
  arguments: string
  preview: string
  status: 'running' | 'ok' | 'error' | 'rejected'
  output?: string
  exit_code?: number | null
  duration_ms?: number
}

export const LABEL: Record<LiveTool['status'], string> = {
  running: 'выполняется',
  ok: 'готово',
  error: 'ошибка',
  rejected: 'отклонено',
}

export function fromStored(message: ToolMessage): LiveTool {
  return {
    call_id: message.call_id ?? String(message.seq),
    tool: message.tool,
    arguments: message.arguments_json,
    preview: '',
    status: message.status,
    output: message.content,
    exit_code: message.exit_code,
    duration_ms: message.duration_ms ?? undefined,
  }
}
