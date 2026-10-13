import { LABEL, type LiveTool } from './tools'

export function ToolCard({ tool }: { tool: LiveTool }) {
  const shown = tool.preview || summarize(tool.arguments)
  return (
    <article className={`tool tool-${tool.status}`}>
      <header>
        <code className="tool-name">{tool.tool}</code>
        <span className="tool-status">{LABEL[tool.status]}</span>
        {tool.exit_code != null && <span className="muted">код {tool.exit_code}</span>}
        {tool.duration_ms != null && <span className="muted">{tool.duration_ms} мс</span>}
      </header>
      {shown && <Preview text={shown} />}
      {tool.output && (
        <details>
          <summary>вывод</summary>
          <pre>{tool.output}</pre>
        </details>
      )}
    </article>
  )
}

export function ApprovalCard({
  request,
  onAnswer,
}: {
  request: LiveTool
  onAnswer: (approved: boolean) => void
}) {
  return (
    <article className="tool tool-approval">
      <header>
        <code className="tool-name">{request.tool}</code>
        <span className="tool-status">нужно подтверждение</span>
      </header>
      <Preview text={request.preview || summarize(request.arguments)} />
      <div className="tool-actions">
        <button type="button" className="primary" onClick={() => onAnswer(true)}>
          разрешить
        </button>
        <button type="button" onClick={() => onAnswer(false)}>
          отклонить
        </button>
      </div>
    </article>
  )
}

/** A unified diff is worth colouring; anything else is shown as it came. */
function Preview({ text }: { text: string }) {
  if (!text.startsWith('---') && !text.startsWith('$ ')) {
    return <pre className="preview">{text}</pre>
  }
  return (
    <pre className="preview">
      {text.split('\n').map((line, index) => (
        <span key={index} className={`line ${lineKind(line)}`}>
          {line}
          {'\n'}
        </span>
      ))}
    </pre>
  )
}

function lineKind(line: string): string {
  if (line.startsWith('+') && !line.startsWith('+++')) return 'added'
  if (line.startsWith('-') && !line.startsWith('---')) return 'removed'
  if (line.startsWith('@@') || line.startsWith('$ ')) return 'meta'
  return ''
}

function summarize(argumentsJson: string): string {
  try {
    const parsed = JSON.parse(argumentsJson) as Record<string, unknown>
    return Object.entries(parsed)
      .map(([key, value]) => `${key}: ${String(value).slice(0, 120)}`)
      .join('\n')
  } catch {
    return argumentsJson
  }
}
