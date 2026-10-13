import type { JsonSchema } from '../api/sessions'

export type SpecValues = Record<string, unknown>

export const SECTION_TITLES: Record<string, string> = {
  model: 'модель', sampling: 'сэмплинг', output: 'ответ', reasoning: 'рассуждение',
  context: 'контекст', memory: 'память', tools: 'инструменты', task: 'задача', judge: 'судья',
}

export const FIELD_TITLES: Record<string, string> = {
  provider: 'Провайдер', model: 'Модель', role: 'Системная роль', reasoning_effort: 'Усилие рассуждения',
  temperature: 'Температура', top_p: 'Вероятностный порог (top_p)', top_k: 'Число кандидатов (top_k)',
  frequency_penalty: 'Штраф за частоту', presence_penalty: 'Штраф за повторение', seed: 'Seed', stop: 'Стоп-последовательности',
  mode: 'Режим', max_output_tokens: 'Лимит токенов ответа', format_json: 'JSON-схема ответа', rules: 'Правила ответа',
  strategy: 'Стратегия решения', window_tokens: 'Размер контекстного окна', overflow: 'При переполнении', budget_usd: 'Бюджет, $',
  keep_last_messages: 'Последних сообщений', memory_limit_tokens: 'Лимит истории в токенах', summary_every: 'Интервал сводки',
  summary_model: 'Модель сводки', summary_max_tokens: 'Лимит токенов сводки', working_enabled: 'Рабочая память',
  working_max_tokens: 'Бюджет рабочей памяти', longterm_enabled: 'Долговременная память', longterm_scope: 'Область долговременной памяти',
  longterm_max_tokens: 'Бюджет долговременной памяти', longterm_write: 'Сохранение в долговременную память', max_tokens: 'Лимит токенов',
  tracker_enabled: 'Трекер задачи', auto_transitions: 'Автоматические переходы', workspace_path: 'Рабочая папка',
  executor: 'Исполнитель', approval_mode: 'Подтверждение действий', max_tool_steps: 'Лимит шагов инструментов',
  command_timeout_s: 'Таймаут команды, с', criteria: 'Критерии оценки', threshold: 'Проходной балл',
  failure_action: 'При неудачной оценке', fallback_model: 'Резервная модель',
}

export function schemaDefaults(schema: JsonSchema): SpecValues {
  return Object.fromEntries(
    Object.entries(schema.properties ?? {}).map(([name, property]) => [
      name,
      property.default ?? defaultFor(property),
    ]),
  )
}

function defaultFor(schema: JsonSchema): unknown {
  if (schema.type === 'object') return {}
  if (schema.type === 'string') return ''
  if (schema.type === 'number' || schema.type === 'integer') return 0
  return null
}

export function decodeSpec(schema: JsonSchema, values: SpecValues): SpecValues {
  return Object.fromEntries(
    Object.entries(schema.properties ?? {}).map(([name, property]) => {
      return [name, decodeValue(schema, property, values[name], name)]
    }),
  )
}

function decodeValue(root: JsonSchema, raw: JsonSchema, value: unknown, name: string): unknown {
  const resolved = resolveSchema(root, raw, value)
  const variant = nonNullVariant(resolved)
  if (isNullable(resolved) && (value === '' || value === null || value === undefined)) return null
  if (name === 'budget_usd' && Number(value) === 0) return null
  if (variant.type === 'number' || variant.type === 'integer') {
    const number = Number(value)
    if (value === '' || !Number.isFinite(number) || (variant.type === 'integer' && !Number.isInteger(number))
      || (variant.minimum !== undefined && number < variant.minimum)
      || (variant.maximum !== undefined && number > variant.maximum)) {
      throw new Error(`Проверьте значение поля «${FIELD_TITLES[name] ?? name}»`)
    }
    return number
  }
  if (variant.type === 'object' && typeof value === 'object' && value !== null) {
    return Object.fromEntries(
      Object.entries(variant.properties ?? {}).map(([child, property]) => [
        child,
        decodeValue(root, property, (value as SpecValues)[child], child),
      ]),
    )
  }
  if (variant.type === 'object' && typeof value === 'string') return JSON.parse(value)
  return value
}

export function resolveSchema(root: JsonSchema, raw: JsonSchema, value?: unknown): JsonSchema {
  if (raw.$ref) return root.$defs?.[raw.$ref.split('/').at(-1) ?? ''] ?? raw
  if (raw.oneOf) {
    const mode = typeof value === 'object' && value !== null
      ? String((value as SpecValues).mode ?? '')
      : ''
    const selected = raw.oneOf.find((variant) => {
      const resolved = resolveSchema(root, variant)
      return resolved.properties?.mode?.const === mode
    }) ?? raw.oneOf[0]
    return resolveSchema(root, selected, value)
  }
  return raw
}

export function nonNullVariant(schema: JsonSchema): JsonSchema {
  return schema.anyOf?.find((item) => item.type !== 'null') ?? schema
}

function isNullable(schema: JsonSchema): boolean {
  return Boolean(schema.anyOf?.some((item) => item.type === 'null'))
}
