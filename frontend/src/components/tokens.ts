/** How a block's cost is spelled in the panels: absolute, then its share. */
export function formatTokens(value: number): string {
  return `${value.toLocaleString('ru')} ток.`
}

/** What a block costs in the next call, and how much of that call it takes. */
export function cost(tokens: number | undefined, total: number): string {
  if (tokens === undefined) return '— ток.'
  if (total <= 0) return formatTokens(tokens)
  return `${formatTokens(tokens)} · ${Math.round((tokens / total) * 100)}%`
}
