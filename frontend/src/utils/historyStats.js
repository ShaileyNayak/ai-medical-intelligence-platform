import { getApiBaseUrl } from "../api/baseUrl.js";
import { normalizePredictions } from "../constants/scanTypes.js";

const MS_PER_DAY = 24 * 60 * 60 * 1000;

export function resolveMediaUrl(url) {
  if (!url) return null;
  if (/^https?:\/\//i.test(url)) return url;
  const path = url.startsWith("/") ? url : `/${url}`;
  return `${getApiBaseUrl()}${path}`;
}

/**
 * Aggregate dashboard / analytics metrics from history items.
 * @param {Array} items - History rows from GET /api/history
 * @param {{ total?: number }} [options] - Prefer API `total` for Total Scans when paginated
 */
export function computeHistoryStats(items = [], options = {}) {
  const list = Array.isArray(items) ? items : [];
  const total = options.total != null ? Number(options.total) : list.length;
  const cutoff = new Date(Date.now() - 7 * MS_PER_DAY);

  let scansLast7Days = 0;
  let confSum = 0;
  let confCount = 0;
  const findingCounts = {};
  const scanTypeCounts = {
    chest_xray: 0,
    brain_mri: 0,
    skin_lesion: 0,
  };

  for (const row of list) {
    const preds = normalizePredictions(row);
    const conf = Number(row.confidence);
    if (Number.isFinite(conf)) {
      confSum += conf;
      confCount += 1;
    }

    if (row.scan_type && scanTypeCounts[row.scan_type] !== undefined) {
      scanTypeCounts[row.scan_type] += 1;
    } else if (row.scan_type) {
      scanTypeCounts[row.scan_type] = (scanTypeCounts[row.scan_type] || 0) + 1;
    }

    const created = row.created_at ? new Date(row.created_at) : null;
    if (created && !Number.isNaN(created.getTime()) && created >= cutoff) {
      scansLast7Days += 1;
    }

    for (const p of preds) {
      const label = p.label || "Unknown";
      findingCounts[label] = (findingCounts[label] || 0) + 1;
    }
  }

  let mostCommon = null;
  let mostCommonCount = 0;
  for (const [label, count] of Object.entries(findingCounts)) {
    if (count > mostCommonCount) {
      mostCommon = label;
      mostCommonCount = count;
    }
  }

  return {
    totalScans: total,
    scansLast7Days,
    /** @deprecated use scansLast7Days */
    conditionsThisWeek: scansLast7Days,
    averageConfidence: confCount ? confSum / confCount : 0,
    mostCommonFinding: mostCommon,
    mostCommonCount,
    scanTypeCounts,
    findingCounts,
    isEmpty: list.length === 0 && total === 0,
  };
}
