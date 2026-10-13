import type { TaskStage, TaskStepStatus } from '../api/sessions'

/** One vocabulary for the machine's stages: rail, strip and transcript agree. */
export const STAGE_LABEL: Record<TaskStage, string> = {
  planning: 'планирование',
  execution: 'выполнение',
  validation: 'проверка',
  done: 'завершена',
  cancelled: 'отменена',
}

export const STEP_STATUS_LABEL: Record<TaskStepStatus, string> = {
  todo: 'ждёт',
  in_progress: 'в работе',
  done: 'готов',
  blocked: 'заблокирован',
}

/** The stepper: the line a task walks. `cancelled` is an exit, not a station. */
export const STAGE_TRACK: TaskStage[] = ['planning', 'execution', 'validation', 'done']
