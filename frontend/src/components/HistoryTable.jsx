export default function HistoryTable({ rows, loading }) {
  if (loading) {
    return <p className="text-sm text-clinical-muted">Loading prediction history…</p>;
  }

  if (!rows?.length) {
    return (
      <div className="clinical-panel px-6 py-12 text-center">
        <p className="font-display text-xl text-clinical-ink">No studies yet</p>
        <p className="mt-2 text-sm text-clinical-muted">
          Upload a chest X-ray on the Analysis page to create the first record.
        </p>
      </div>
    );
  }

  return (
    <div className="clinical-panel overflow-x-auto">
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead>
          <tr className="border-b border-clinical-line bg-clinical-soft/80 text-clinical-muted">
            <th className="px-4 py-3 font-semibold">ID</th>
            <th className="px-4 py-3 font-semibold">Prediction</th>
            <th className="px-4 py-3 font-semibold">Confidence</th>
            <th className="px-4 py-3 font-semibold">Created</th>
            <th className="px-4 py-3 font-semibold">Report excerpt</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className="border-b border-clinical-line/70 last:border-0">
              <td className="px-4 py-3 tabular-nums text-clinical-muted">{row.id}</td>
              <td className="px-4 py-3 font-medium capitalize text-clinical-ink">
                {row.prediction}
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
