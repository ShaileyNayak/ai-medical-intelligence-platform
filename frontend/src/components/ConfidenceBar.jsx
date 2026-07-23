import {
  confidenceBadgeClass,
  confidenceBarClass,
  confidenceTextClass,
  formatConfidencePct,
} from "../utils/confidence.js";

/** Progress bar with color-coded fill and percentage. */
export function ConfidenceBar({ confidence, className = "" }) {
  const pct = Math.min(100, Math.max(0, Number(confidence || 0) * 100));
  return (
    <div className={className}>
      <div className="mb-1.5 flex justify-between text-xs text-clinical-muted">
        <span>Confidence</span>
        <span className={`font-semibold tabular-nums ${confidenceTextClass(confidence)}`}>
          {formatConfidencePct(confidence)}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-sm bg-clinical-soft">
        <div
          className={`h-2 rounded-sm transition-all duration-500 ${confidenceBarClass(confidence)}`}
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={Math.round(pct)}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}

/** Compact badge for tables and recent-activity cells. */
export function ConfidenceBadge({ confidence, className = "" }) {
  return (
    <span
      className={[
        "inline-flex border px-2 py-0.5 text-xs font-semibold tabular-nums",
        confidenceBadgeClass(confidence),
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {formatConfidencePct(confidence)}
    </span>
  );
}

/** Plain colored percentage text (stat cards, inline labels). */
export function ConfidenceText({ confidence, className = "" }) {
  return (
    <span
      className={[
        "tabular-nums font-semibold",
        confidenceTextClass(confidence),
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {formatConfidencePct(confidence)}
    </span>
  );
}
