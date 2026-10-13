export function LoadingIndicator({ label }: { label: string }) {
  return (
    <span className="loading-indicator">
      <span className="spinner" aria-hidden="true" />
      {label}
    </span>
  )
}
