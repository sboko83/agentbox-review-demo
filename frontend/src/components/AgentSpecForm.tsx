import { useState } from 'react'

import type { JsonSchema, ModelInfo } from '../api/sessions'
import { FIELD_TITLES, nonNullVariant, resolveSchema, SECTION_TITLES, type SpecValues } from './agentSpec'

const SEGMENTED = new Set(['output.mode', 'reasoning.strategy', 'context.mode'])

export function AgentSpecForm({
  schema,
  values,
  models,
  onChange,
  disabled = false,
  idPrefix = 'spec',
  section: selectedSection,
  search = '',
}: {
  schema: JsonSchema
  values: SpecValues
  models: ModelInfo[]
  onChange: (name: string, value: unknown) => void
  disabled?: boolean
  idPrefix?: string
  section?: string
  search?: string
}) {
  const modelValues = (values.model ?? {}) as SpecValues
  const selectedProvider = String(modelValues.provider ?? '')
  const selectedId = String(modelValues.model ?? '')
  const providerModels = models.filter((model) => model.provider === selectedProvider)
  const providers = [
    ...new Set([...models.map((model) => model.provider), selectedProvider].filter(Boolean)),
  ]
  const selected = providerModels.find((model) => model.id === selectedId)
  const accepted = selected ? new Set(selected.supported_params) : null
  const [openSections, setOpenSections] = useState(
    () => new Set(Object.keys(schema.properties ?? {}).filter((name) => !['sampling', 'task', 'judge'].includes(name))),
  )

  return (
    <div className="spec-form">
      {Object.entries(schema.properties ?? {}).map(([sectionName, rawSection]) => {
        if (!search && selectedSection && selectedSection !== sectionName) return null
        const sectionValues = (values[sectionName] ?? rawSection.default ?? {}) as SpecValues
        const section = resolveSchema(schema, rawSection, sectionValues)
        const properties = contextModeOptions(schema, rawSection, section)
        const visibleFields = Object.entries(properties)
          .filter(([, raw]) => Boolean(search) || isVisible(raw, sectionValues))
          .filter(([name, raw]) => !search || `${SECTION_TITLES[sectionName]} ${sectionName} ${name} ${FIELD_TITLES[name] ?? ''} ${raw.title ?? ''} ${raw.description ?? ''}`.toLowerCase().includes(search.toLowerCase()))
        if (visibleFields.length === 0) return null
        const primaryFields = visibleFields.filter(([, raw]) => !raw.advanced)
        const advancedFields = visibleFields.filter(([, raw]) => raw.advanced)

        return (
          <details
            className="spec-group"
            key={sectionName}
            open={Boolean(search || selectedSection) || openSections.has(sectionName)}
            onToggle={(event) => {
              const opened = event.currentTarget.open
              setOpenSections((current) => {
                if (current.has(sectionName) === opened) return current
                const next = new Set(current)
                if (opened) next.add(sectionName)
                else next.delete(sectionName)
                return next
              })
            }}
          >
            <summary>
              <span>{SECTION_TITLES[sectionName] ?? sectionName}</span>
              <small>{sectionSummary(sectionName, sectionValues)}</small>
            </summary>
            <div className="spec-fields">
              {primaryFields.map(([name, raw]) => (
                <Field
                  key={name}
                  section={sectionName}
                  name={name}
                  raw={isVisible(raw, sectionValues) ? raw : { ...raw, description: `${raw.description ?? ''} Доступно после включения соответствующего режима в этом разделе.` }}
                  value={sectionValues[name] ?? raw.default ?? ''}
                  models={isModelField(sectionName, name) ? providerModels : []}
                  options={sectionName === 'model' && name === 'provider' ? providers : []}
                  refused={accepted !== null && Boolean(raw.capability) && !accepted.has(raw.capability!)}
                  disabled={disabled || !isVisible(raw, sectionValues)}
                  modelId={selectedId}
                  idPrefix={`${idPrefix}-${sectionName}`}
                  onChange={(fieldName, fieldValue) => {
                    let next = { ...sectionValues, [fieldName]: fieldValue }
                    if (sectionName === 'model' && fieldName === 'provider') {
                      const firstModel =
                        models.find((model) => model.provider === fieldValue && model.offered)
                        ?? models.find((model) => model.provider === fieldValue)
                      if (firstModel) next = { ...next, model: firstModel.id }
                    }
                    if (sectionName === 'context' && fieldName === 'mode') {
                      const variant = resolveSchema(schema, rawSection, { mode: fieldValue })
                      next = Object.fromEntries(
                        Object.entries(variant.properties ?? {}).map(([child, childSchema]) => [
                          child,
                          child === 'mode'
                            ? fieldValue
                            : sectionValues[child] ?? childSchema.default ?? '',
                        ]),
                      )
                    }
                    onChange(sectionName, next)
                  }}
                />
              ))}
            </div>
            {advancedFields.length > 0 && (
              <details className="advanced-fields" open={search ? true : undefined}>
                <summary>ещё</summary>
                <div className="spec-fields">
                  {advancedFields.map(([name, raw]) => (
                    <Field
                      key={name}
                      section={sectionName}
                      name={name}
                      raw={isVisible(raw, sectionValues) ? raw : { ...raw, description: `${raw.description ?? ''} Доступно после включения соответствующего режима в этом разделе.` }}
                      value={sectionValues[name] ?? raw.default ?? ''}
                      models={[]}
                      options={[]}
                      refused={accepted !== null && Boolean(raw.capability) && !accepted.has(raw.capability!)}
                      disabled={disabled || !isVisible(raw, sectionValues)}
                      modelId={selectedId}
                      idPrefix={`${idPrefix}-${sectionName}`}
                      onChange={(fieldName, fieldValue) => onChange(
                        sectionName,
                        { ...sectionValues, [fieldName]: fieldValue },
                      )}
                    />
                  ))}
                </div>
              </details>
            )}
          </details>
        )
      })}
    </div>
  )
}

function contextModeOptions(
  root: JsonSchema,
  rawSection: JsonSchema,
  section: JsonSchema,
): Record<string, JsonSchema> {
  const properties = { ...section.properties }
  if (!rawSection.oneOf || !properties.mode) return properties
  const options = rawSection.oneOf
    .map((variant) => resolveSchema(root, variant).properties?.mode?.const)
    .filter((value): value is string => typeof value === 'string')
  properties.mode = { ...properties.mode, const: undefined, enum: options }
  return properties
}

function isVisible(field: JsonSchema, values: SpecValues): boolean {
  if (!field.when) return true
  const sibling = values[field.when.field]
  if (field.when.values) return field.when.values.includes(String(sibling ?? ''))
  if (field.when.truthy) {
    return typeof sibling === 'string' ? sibling.trim().length > 0 : Boolean(sibling)
  }
  return true
}

function sectionSummary(section: string, values: SpecValues): string {
  if (section === 'sampling') {
    return `t ${shortValue(values.temperature)} · top_p ${shortValue(values.top_p)}`
  }
  if (section === 'model') return String(values.model ?? '')
  if (section === 'output' || section === 'context') return String(values.mode ?? '')
  if (section === 'reasoning') return String(values.strategy ?? '')
  if (section === 'memory') {
    const layers = [
      values.working_enabled ? 'рабочая' : null,
      values.longterm_enabled ? 'долговременная' : null,
    ].filter(Boolean)
    return layers.length > 0 ? layers.join(' · ') : 'только диалог'
  }
  if (section === 'tools') return values.workspace_path ? 'включены' : 'выключены'
  if (section === 'task') return values.tracker_enabled ? 'трекер включён' : 'вручную'
  if (section === 'judge') return 'для матричных прогонов'
  return ''
}

function shortValue(value: unknown): string {
  return value === null || value === undefined || value === '' ? '—' : String(value)
}

function isModelField(section: string, name: string): boolean {
  return (
    (section === 'model' && name === 'model')
    || (section === 'context' && name === 'summary_model')
    || (section === 'memory' && name === 'model')
    || (section === 'task' && name === 'model')
    || (section === 'judge' && ['model', 'fallback_model'].includes(name))
  )
}

function Field({
  section,
  name,
  raw,
  value,
  models,
  options,
  refused,
  disabled,
  modelId,
  idPrefix,
  onChange,
}: {
  section: string
  name: string
  raw: JsonSchema
  value: unknown
  models: ModelInfo[]
  options: string[]
  refused: boolean
  disabled: boolean
  modelId: string
  idPrefix: string
  onChange: (name: string, value: unknown) => void
}) {
  const field = nonNullVariant(raw)
  const id = `${idPrefix}-${name}`
  const label = FIELD_TITLES[name] ?? raw.title ?? name
  const hint = refused
    ? `${modelId} не принимает этот параметр — он не уйдёт в запрос.`
    : raw.description
  const set = (next: unknown) => onChange(name, next)
  const isNumber = field.type === 'number' || field.type === 'integer'
  const wide = !isNumber || SEGMENTED.has(`${section}.${name}`)

  return (
    <label
      className={['spec-field', refused ? 'refused' : '', wide ? 'wide-field' : '']
        .filter(Boolean).join(' ')}
      htmlFor={id}
    >
      <span className="field-label">
        {label}
        {hint && (
          <span className="field-hint" aria-label={hint} tabIndex={0}>
            ?
            <span className="field-tooltip" role="tooltip">{hint}</span>
          </span>
        )}
      </span>
      {SEGMENTED.has(`${section}.${name}`) && field.enum ? (
        <span className="segments" id={id}>
          {field.enum.map((option) => (
            <button
              key={option}
              type="button"
              className={String(value) === option ? 'current' : undefined}
              disabled={disabled || refused}
              onClick={() => set(option)}
            >
              {option}
            </button>
          ))}
        </span>
      ) : options.length > 0 ? (
        <select id={id} value={String(value)} disabled={disabled || refused} onChange={(event) => set(event.target.value)}>
          {options.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
      ) : models.length > 0 ? (
        <select id={id} value={String(value)} disabled={disabled || refused} onChange={(event) => set(event.target.value)}>
          {models.every((model) => model.id !== String(value)) && (
            <option value={String(value)}>
              {String(value) ? `${String(value)} — вне каталога` : 'не задано'}
            </option>
          )}
          {familiesOf(pickable(models, String(value))).map(([family, group]) => (
            <optgroup key={family} label={family}>
              {group.map((model) => (
                <option key={`${model.provider}:${model.id}`} value={model.id}>{modelLabel(model)}</option>
              ))}
            </optgroup>
          ))}
        </select>
      ) : field.enum ? (
        <select
          id={id}
          value={String(value)}
          disabled={disabled || refused}
          onChange={(event) => set(event.target.value)}
        >
          {optionsOf(field, raw).map((option) => (
            <option key={option} value={option}>{option === '' ? 'не задано' : option}</option>
          ))}
        </select>
      ) : raw.const ? (
        <input id={id} value={raw.const} disabled />
      ) : field.type === 'object' ? (
        <textarea
          id={id}
          rows={6}
          value={typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
          disabled={disabled || refused}
          onChange={(event) => set(event.target.value)}
          spellCheck={false}
        />
      ) : field.ui === 'textarea' || (field.maxLength ?? 0) > 200 ? (
        <textarea
          id={id}
          rows={3}
          value={String(value)}
          disabled={disabled || refused}
          onChange={(event) => set(event.target.value)}
        />
      ) : (
        <input
          id={id}
          type={isNumber ? 'number' : 'text'}
          min={field.minimum}
          max={field.maximum}
          step={field.type === 'integer' ? 1 : 'any'}
          value={value === null ? '' : String(value)}
          disabled={disabled || refused}
          onChange={(event) => set(event.target.value)}
        />
      )}
    </label>
  )
}

function optionsOf(field: JsonSchema, raw: JsonSchema): string[] {
  const options = field.enum ?? []
  return raw.anyOf?.some((item) => item.type === 'null') ? ['', ...options] : options
}

/** Offered models plus the one already selected, so opening an old session
 *  cannot silently repoint its spec at the first option of the list. */
function pickable(models: ModelInfo[], current: string): ModelInfo[] {
  return models.filter((model) => model.offered || model.id === current)
}

function familiesOf(models: ModelInfo[]): Array<[string, ModelInfo[]]> {
  const groups = new Map<string, ModelInfo[]>()
  for (const model of models) {
    const group = groups.get(model.family)
    if (group) group.push(model)
    else groups.set(model.family, [model])
  }
  return [...groups]
}

function modelLabel(model: ModelInfo): string {
  const prices =
    model.input_usd_per_million === null || model.output_usd_per_million === null
      ? 'цена недоступна'
      : `$${model.input_usd_per_million}/$${model.output_usd_per_million} за 1M`
  const window = model.context_window ? `, окно ${Math.round(model.context_window / 1000)}k` : ''
  const capped = model.offered ? '' : ', вне лимита цены'
  return `${model.name} (${prices}${window}${capped})`
}
