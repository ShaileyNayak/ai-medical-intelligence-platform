import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";
import EmptyState from "../components/EmptyState.jsx";
import { SCAN_TYPE_OPTIONS } from "../constants/scanTypes.js";
import { fetchHistoryWindow } from "../utils/fetchHistoryWindow.js";
import { computeHistoryStats } from "../utils/historyStats.js";
import {
  confidenceTextClass,
  formatConfidencePct,
} from "../utils/confidence.js";

const COLORS = ["#0e7490", "#155e75", "#5b6b7c", "#b45309", "#15803d", "#dc2626"];

export default function Analytics() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const { items: rows } = await fetchHistoryWindow(200);
        if (!cancelled) setItems(rows);
      } catch (err) {
        if (!cancelled) setError(err.message || "Failed to load analytics");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const stats = useMemo(() => computeHistoryStats(items), [items]);

  const modalityData = SCAN_TYPE_OPTIONS.map((opt) => ({
    name: opt.label,
    count: stats.scanTypeCounts[opt.value] || 0,
  }));

  const findingData = Object.entries(stats.findingCounts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  return (
    <div className="space-y-8">
      <header>
        <p className="clinical-label">Insights</p>
        <h1 className="mt-2 font-display text-4xl text-clinical-ink md:text-5xl">Analytics</h1>
        <p className="mt-2 max-w-xl text-clinical-muted">
          Modality mix and finding frequency derived from recent prediction history.
        </p>
      </header>

      {error && (
        <div className="border border-red-200 bg-red-50 px-4 py-3 text-sm text-clinical-danger">
          {error}
        </div>
      )}

      {loading ? (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="clinical-panel skeleton h-80" />
          <div className="clinical-panel skeleton h-80" />
        </div>
      ) : items.length === 0 ? (
        <EmptyState
          title="No analytics yet"
          description="Run a few analyses first — charts populate from stored prediction history."
          actionLabel="Go to New Analysis"
          actionTo="/analysis"
        />
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="clinical-panel p-5">
              <p className="clinical-label">Samples in window</p>
              <p className="mt-2 font-display text-3xl text-clinical-ink">{stats.totalScans}</p>
            </div>
            <div className="clinical-panel p-5">
              <p className="clinical-label">Avg confidence</p>
              <p
                className={`mt-2 font-display text-3xl ${confidenceTextClass(stats.averageConfidence)}`}
              >
                {formatConfidencePct(stats.averageConfidence)}
              </p>
            </div>
            <div className="clinical-panel p-5">
              <p className="clinical-label">Top finding</p>
              <p className="mt-2 font-display text-3xl text-clinical-ink">{stats.mostCommonFinding}</p>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <section className="clinical-panel p-5">
              <p className="clinical-label">Volume</p>
              <h2 className="mt-1 font-display text-2xl text-clinical-ink">By scan type</h2>
              <div className="mt-4 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={modalityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#d7e0ea" />
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#0e7490" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="clinical-panel p-5">
              <p className="clinical-label">Findings</p>
              <h2 className="mt-1 font-display text-2xl text-clinical-ink">Label frequency</h2>
              <div className="mt-4 h-72">
                {findingData.length === 0 ? (
                  <p className="text-sm text-clinical-muted">No labels to chart.</p>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={findingData}
                        dataKey="count"
                        nameKey="name"
                        outerRadius={90}
                        label={({ name, percent }) =>
                          `${name} ${(percent * 100).toFixed(0)}%`
                        }
                      >
                        {findingData.map((entry, idx) => (
                          <Cell key={entry.name} fill={COLORS[idx % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </div>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
