/**
 * Shared confidence color system (0–1 scale):
 * - green  (high): ≥ 85%
 * - amber  (mid):  60–85% (≥ 60% and < 85%)
 * - red    (low):  < 60%
 */
export const CONFIDENCE_THRESHOLDS = {
  high: 0.85,
  mid: 0.6,
};

export function confidenceLevel(value) {
  const conf = Number(value);
  if (!Number.isFinite(conf)) return "low";
  if (conf >= CONFIDENCE_THRESHOLDS.high) return "high";
  if (conf >= CONFIDENCE_THRESHOLDS.mid) return "mid";
  return "low";
}

export function confidenceTextClass(value) {
  const level = confidenceLevel(value);
  if (level === "high") return "text-conf-high";
  if (level === "mid") return "text-conf-mid";
  return "text-conf-low";
}

export function confidenceBarClass(value) {
  const level = confidenceLevel(value);
  if (level === "high") return "bg-conf-high";
  if (level === "mid") return "bg-conf-mid";
  return "bg-conf-low";
}

export function confidenceBadgeClass(value) {
  const level = confidenceLevel(value);
  if (level === "high") return "border-conf-high/30 bg-conf-high/10 text-conf-high";
  if (level === "mid") return "border-conf-mid/30 bg-conf-mid/10 text-conf-mid";
  return "border-conf-low/30 bg-conf-low/10 text-conf-low";
}

export function formatConfidencePct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${(Math.round(n * 1000) / 10).toFixed(1)}%`;
}
