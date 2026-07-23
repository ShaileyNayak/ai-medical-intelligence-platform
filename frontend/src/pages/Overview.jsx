import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { ConfidenceBadge } from "../components/ConfidenceBar.jsx";
import EmptyState from "../components/EmptyState.jsx";
import { fetchHistorySummary } from "../api/client.js";
import {
  SCAN_TYPE_OPTIONS,
  normalizePredictions,
  scanTypeLabel,
} from "../constants/scanTypes.js";
import { fetchHistoryWindow } from "../utils/fetchHistoryWindow.js";
import { computeHistoryStats } from "../utils/historyStats.js";
import {
  confidenceTextClass,
  formatConfidencePct,
} from "../utils/confidence.js";

/** Clinical palette: teal → tealDark → slate muted */
const SCAN_TYPE_COLORS = {
  chest_xray: "#0e7490",
  brain_mri: "#155e75",
  skin_lesion: "#5b6b7c",
};

function resultLabel(row) {
  const preds = normalizePredictions(row);
  if (preds.length) return preds.map((p) => p.label).join(", ");
  return row.prediction || "—";
}

function topCondition(conditions = {}) {
  let best = null;
  let bestCount = 0;
  for (const [label, count] of Object.entries(conditions)) {
    const n = Number(count) || 0;
    if (n > bestCount) {
      best = label;
      bestCount = n;
    }
  }
  return best;
}

/**
 * Dashboard / Overview — default landing page.
 * Stat cards + recent activity from GET /api/history;
 * distribution chart from GET /api/history/summary.
 */
export default function Overview() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [history, summaryData] = await Promise.all([
          fetchHistoryWindow(200),
          fetchHistorySummary(),
        ]);
        if (!cancelled) {
          setItems(history.items || []);
          setTotal(history.total ?? history.items?.length ?? 0);
          setSummary(summaryData);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Failed to load history");
          setItems([]);
          setTotal(0);
          setSummary(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const stats = useMemo(
    () => computeHistoryStats(items, { total }),
    [items, total]
  );

  const recent = useMemo(() => items.slice(0, 5), [items]);

  const modalityData = useMemo(() => {
    if (!summary) return [];
    return SCAN_TYPE_OPTIONS.map((opt) => {
      const bucket = summary[opt.value] || {};
      return {
        key: opt.value,
        name: opt.label,
        count: Number(bucket.total) || 0,
        fill: SCAN_TYPE_COLORS[opt.value] || "#5b6b7c",
        topFinding: topCondition(bucket.conditions),
      };
    }).filter((d) => d.count > 0);
  }, [summary]);

  const categoryBreakdown = useMemo(() => {
    return SCAN_TYPE_OPTIONS.map((opt) => {
      const bucket = summary?.[opt.value] || {};
      return {
        key: opt.value,
        name: opt.label,
        total: Number(bucket.total) || 0,
        topFinding: topCondition(bucket.conditions),
        color: SCAN_TYPE_COLORS[opt.value] || "#5b6b7c",
      };
    });
  }, [summary]);

  const empty = !loading && !error && stats.isEmpty;

  const cards = [
    {
      label: "Total Scans",
      value: empty ? "—" : String(stats.totalScans),
      muted: empty,
    },
    {
      label: "This Week",
      value: empty ? "—" : String(stats.scansLast7Days),
      muted: empty,
    },
    {
      label: "Avg Confidence",
      value: empty ? "—" : formatConfidencePct(stats.averageConfidence),
      muted: empty,
      valueClass: empty ? null : confidenceTextClass(stats.averageConfidence),
    },
    {
      label: "Most Common Finding",
      value: empty ? "No data yet" : stats.mostCommonFinding || "—",
      muted: empty || !stats.mostCommonFinding,
    },
  ];

  return (
    <div className="space-y-8">
      <header>
        <p className="clinical-label">Dashboard</p>
        <h1 className="mt-2 font-display text-3xl text-clinical-ink sm:text-4xl md:text-5xl">
          Overview
        </h1>
        <p className="mt-2 max-w-xl text-clinical-muted">
          Summary of imaging analyses across modalities from prediction history.
        </p>
      </header>

      {error && (
        <div className="border border-red-200 bg-red-50 px-4 py-3 text-sm text-clinical-danger">
          {error}
        </div>
      )}

      {loading ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="clinical-panel p-5">
                <div className="skeleton h-3 w-24" />
                <div className="mt-4 skeleton h-9 w-20" />
              </div>
            ))}
          </div>
          <div className="grid gap-6 lg:grid-cols-5">
            <div className="clinical-panel skeleton min-h-[280px] lg:col-span-3" />
            <div className="clinical-panel skeleton min-h-[280px] lg:col-span-2" />
          </div>
        </>
      ) : empty ? (
        <EmptyState
          title="No scans yet"
          description="Run your first analysis to see overview stats, recent activity, and scan-type distribution here."
          actionLabel="Upload first image"
          actionTo="/analysis"
        />
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {cards.map((card) => (
              <div key={card.label} className="clinical-panel clinical-panel-hover p-5">
                <p className="clinical-label">{card.label}</p>
                <div className="mt-4 flex min-h-12 items-end">
                  <span
                    className={`font-display text-3xl leading-none ${
                      card.valueClass
                        ? card.valueClass
                        : card.muted
                          ? "text-clinical-muted"
                          : "text-clinical-ink"
                    }`}
                  >
                    {card.value}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="grid gap-5 sm:gap-6 lg:grid-cols-5">
            <section className="clinical-panel min-h-0 overflow-hidden lg:col-span-3">
              <div className="border-b border-clinical-line px-4 py-4 sm:px-5">
                <p className="clinical-label">Recent activity</p>
                <h2 className="mt-1 font-display text-xl text-clinical-ink sm:text-2xl">
                  Latest predictions
                </h2>
              </div>

              <div className="space-y-3 p-3 sm:p-4 md:hidden">
                {recent.map((row) => (
                  <article
                    key={row.id}
                    className="border border-clinical-line bg-clinical-soft/40 p-3"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">
                          {scanTypeLabel(row.scan_type)}
                        </p>
                        <p className="mt-1 font-medium text-clinical-ink">{resultLabel(row)}</p>
                      </div>
                      <ConfidenceBadge confidence={row.confidence} />
                    </div>
                    <div className="mt-3 flex items-center justify-between gap-2 text-xs text-clinical-muted">
                      <span className="truncate">
                        {row.created_at
                          ? new Date(row.created_at).toLocaleString()
                          : "—"}
                      </span>
                      <Link
                        to="/history"
                        state={{ focusId: row.id }}
                        className="shrink-0 font-medium text-clinical-teal underline-offset-2 hover:underline"
                      >
                        View
                      </Link>
                    </div>
                  </article>
                ))}
              </div>

              <div className="hidden overflow-x-auto md:block">
                <table className="w-full min-w-[560px] text-left text-sm">
                  <thead>
                    <tr className="border-b border-clinical-line bg-clinical-soft/80 text-clinical-muted">
                      <th className="px-4 py-3 font-semibold">Scan type</th>
                      <th className="px-4 py-3 font-semibold">Result</th>
                      <th className="px-4 py-3 font-semibold">Confidence</th>
                      <th className="px-4 py-3 font-semibold">Timestamp</th>
                      <th className="px-4 py-3 font-semibold">
                        <span className="sr-only">Actions</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {recent.map((row) => (
                      <tr
                        key={row.id}
                        className="border-b border-clinical-line/70 last:border-0 hover:bg-clinical-soft/70"
                      >
                        <td className="px-4 py-3 text-clinical-ink">
                          {scanTypeLabel(row.scan_type)}
                        </td>
                        <td className="px-4 py-3 text-clinical-ink">
                          {resultLabel(row)}
                        </td>
                        <td className="px-4 py-3">
                          <ConfidenceBadge confidence={row.confidence} />
                        </td>
                        <td className="px-4 py-3 text-clinical-muted">
                          {row.created_at
                            ? new Date(row.created_at).toLocaleString()
                            : "—"}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <Link
                            to="/history"
                            state={{ focusId: row.id }}
                            className="font-medium text-clinical-teal underline-offset-2 hover:underline"
                          >
                            View
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="clinical-panel min-h-[240px] p-4 sm:p-5 lg:col-span-2">
              <p className="clinical-label">Distribution</p>
              <h2 className="mt-1 font-display text-xl text-clinical-ink sm:text-2xl">
                By scan type
              </h2>
              {modalityData.length === 0 ? (
                <div className="mt-6 flex h-48 items-center justify-center text-sm text-clinical-muted">
                  No modality data yet
                </div>
              ) : (
                <>
                  <div className="mt-2 h-52">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={modalityData}
                          dataKey="count"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={48}
                          outerRadius={72}
                          paddingAngle={2}
                          stroke="#ffffff"
                          strokeWidth={2}
                        >
                          {modalityData.map((entry) => (
                            <Cell key={entry.key} fill={entry.fill} />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(value, name) => [
                            `${value} scan${value === 1 ? "" : "s"}`,
                            name,
                          ]}
                          contentStyle={{
                            border: "1px solid #d7e0ea",
                            borderRadius: 4,
                            fontSize: 12,
                          }}
                        />
                        <Legend
                          verticalAlign="bottom"
                          height={32}
                          iconType="circle"
                          wrapperStyle={{ fontSize: 12, color: "#5b6b7c" }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <ul className="mt-4 space-y-2 border-t border-clinical-line pt-4">
                    {categoryBreakdown.map((row) => (
                      <li
                        key={row.key}
                        className="flex items-start gap-2 text-xs leading-snug text-clinical-muted"
                      >
                        <span
                          className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
                          style={{ backgroundColor: row.color }}
                          aria-hidden
                        />
                        <span>
                          <span className="font-medium text-clinical-ink">{row.name}</span>
                          {" — most common: "}
                          <span className="text-clinical-ink">
                            {row.total > 0 ? row.topFinding || "—" : "no scans yet"}
                          </span>
                        </span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </section>
          </div>
        </>
      )}
    </div>
  );
}
