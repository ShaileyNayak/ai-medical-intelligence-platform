import { useCallback, useEffect, useState } from "react";
import HistoryTable from "../components/HistoryTable.jsx";
import { fetchHistory } from "../api/client.js";

export default function History() {
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHistory(page, pageSize);
      setRows(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      setError(err.message || "Failed to load history");
    } finally {
      setLoading(false);
    }
  }, [page, pageSize]);

  useEffect(() => {
    load();
  }, [load]);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="clinical-label">Archive</p>
          <h1 className="mt-3 font-display text-4xl text-clinical-ink md:text-5xl">
            Prediction history
          </h1>
          <p className="mt-3 max-w-xl text-clinical-muted">
            Prior analyses stored by the API, newest first.
          </p>
        </div>
        <button
          type="button"
          onClick={load}
          className="border border-clinical-teal px-4 py-2 text-sm font-semibold text-clinical-teal hover:bg-cyan-50"
        >
          Refresh
        </button>
      </header>

      <p className="text-sm text-clinical-muted">
        {total} record{total === 1 ? "" : "s"} · page {page} of {totalPages}
      </p>

      {error && (
        <div className="border border-red-200 bg-red-50 px-4 py-3 text-sm text-clinical-danger">
          {error}
        </div>
      )}

      <HistoryTable rows={rows} loading={loading} />

      {totalPages > 1 && (
        <div className="flex items-center gap-3">
          <button
            type="button"
            disabled={page <= 1 || loading}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            className="border border-clinical-line px-3 py-1.5 text-sm disabled:opacity-40"
          >
            Previous
          </button>
          <button
            type="button"
            disabled={page >= totalPages || loading}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            className="border border-clinical-line px-3 py-1.5 text-sm disabled:opacity-40"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
