import { normalizePredictions, scanTypeLabel } from "../constants/scanTypes.js";

function ConditionTags({ row }) {
  const preds = normalizePredictions(row);
  if (!preds.length) {
    return <span className="text-clinical-muted">—</span>;
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {preds.map((p, idx) => (
        <span
          key={`${row.id}-${p.label}-${idx}`}
          className="inline-flex items-center gap-1 border border-clinical-line bg-clinical-soft px-2 py-0.5 text-xs text-clinical-ink"
          title={`${p.label}: ${(Number(p.confidence) * 100).toFixed(1)}%`}
        >
          <span className="font-medium">{p.label}</span>
          <span className="tabular-nums text-clinical-muted">
            {(Number(p.confidence) * 100).toFixed(0)}%
          </span>
        </span>
      ))}
    </div>
  );
}

export default function HistoryTable({ rows, loading }) {
  if (loading) {
    return <p className="text-sm text-clinical-muted">Loading prediction history…</p>;
  }

  if (!rows?.length) {
    return (
      <div className="clinical-panel px-6 py-12 text-center">
        <p className="font-display text-xl text-clinical-ink">No studies yet</p>
        <p className="mt-2 text-sm text-clinical-muted">
          Upload a study on the Analysis page to create the first record.
        </p>
      </div>
    );
  }

  return (
    <div className="clinical-panel overflow-x-auto">
      <table className="w-full min-w-[760px] text-left text-sm">
        <thead>
          <tr className="border-b border-clinical-line bg-clinical-soft/80 text-clinical-muted">
            <th className="px-4 py-3 font-semibold">ID</th>
            <th className="px-4 py-3 font-semibold">Scan type</th>
            <th className="px-4 py-3 font-semibold">Conditions</th>
            <th className="px-4 py-3 font-semibold">Top confidence</th>
            <th className="px-4 py-3 font-semibold">Created</th>
            <th className="px-4 py-3 font-semibold">Report excerpt</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className="border-b border-clinical-line/70 last:border-0">
              <td className="px-4 py-3 tabular-nums text-clinical-muted">{row.id}</td>
              <td className="px-4 py-3 text-clinical-ink">
                {scanTypeLabel(row.scan_type)}
              </td>
              <td className="px-4 py-3">
                <ConditionTags row={row} />
              </td>
              <td className="px-4 py-3 tabular-nums">
                {(Number(row.confidence) * 100).toFixed(1)}%
              </td>
              <td className="px-4 py-3 text-clinical-muted">
                {row.created_at ? new Date(row.created_at).toLocaleString() : "—"}
              </td>
              <td className="max-w-xs truncate px-4 py-3 text-clinical-muted">
                {row.report_text
                  ? `${String(row.report_text).slice(0, 80)}${row.report_text.length > 80 ? "…" : ""}`
                  : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
