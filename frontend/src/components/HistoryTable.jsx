export default function HistoryTable({ rows, onDelete }) {
  if (!rows?.length) {
    return <p className="text-ink/60">No predictions yet.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-ink/15 text-ink/60">
          <tr>
            <th className="py-2 pr-4 font-medium">ID</th>
            <th className="py-2 pr-4 font-medium">Label</th>
            <th className="py-2 pr-4 font-medium">Confidence</th>
            <th className="py-2 pr-4 font-medium">Model</th>
            <th className="py-2 pr-4 font-medium">Created</th>
            <th className="py-2 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className="border-b border-ink/5">
              <td className="py-3 pr-4">{row.id}</td>
              <td className="py-3 pr-4 capitalize">{row.predicted_label}</td>
              <td className="py-3 pr-4">{(row.confidence_score * 100).toFixed(1)}%</td>
              <td className="py-3 pr-4">{row.model_version}</td>
              <td className="py-3 pr-4">
                {row.created_at ? new Date(row.created_at).toLocaleString() : "—"}
              </td>
              <td className="py-3">
                {onDelete && (
                  <button
                    type="button"
                    className="text-red-700 hover:underline"
                    onClick={() => onDelete(row.id)}
                  >
                    Delete
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
