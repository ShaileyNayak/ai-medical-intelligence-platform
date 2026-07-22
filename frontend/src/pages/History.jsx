import { useCallback, useEffect, useState } from "react";
import HistoryTable from "../components/HistoryTable.jsx";
import { deleteHistoryItem, fetchHistory } from "../api/client.js";

export default function History() {
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    fetchHistory()
      .then((data) => {
        setRows(data.items || []);
        setTotal(data.total || 0);
      })
      .catch((err) => setError(err.message || "Failed to load history"));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function handleDelete(id) {
    try {
      await deleteHistoryItem(id);
      load();
    } catch (err) {
      setError(err.message || "Delete failed");
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl text-ink">History</h1>
        <p className="mt-2 text-ink/60">{total} prediction(s) stored</p>
      </div>
      {error && <p className="text-red-700">{error}</p>}
      <HistoryTable rows={rows} onDelete={handleDelete} />
    </div>
  );
}
