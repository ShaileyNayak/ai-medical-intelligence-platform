import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  Search,
  X,
} from "lucide-react";
import EmptyState from "../components/EmptyState.jsx";
import { ConfidenceBadge } from "../components/ConfidenceBar.jsx";
import ImageHeatmapToggle from "../components/ImageHeatmapToggle.jsx";
import { ReportCard } from "../components/ResultCard.jsx";
import ResultCard from "../components/ResultCard.jsx";
import { fetchHistory, fetchHistoryReport, fetchHistorySummary } from "../api/client.js";
import {
  SCAN_TYPE_OPTIONS,
  normalizePredictions,
  scanTypeLabel,
} from "../constants/scanTypes.js";
import { confidenceTextClass } from "../utils/confidence.js";

const PAGE_SIZE = 10;

const CATEGORY_TABS = [
  { value: "", label: "All" },
  ...SCAN_TYPE_OPTIONS,
];

function mostCommonFinding(conditions = {}) {
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

const COLUMNS = [
  { key: "id", label: "ID" },
  { key: "scan_type", label: "Scan type" },
  { key: "prediction", label: "Result" },
  { key: "confidence", label: "Confidence" },
  { key: "created_at", label: "Timestamp" },
];

async function fetchAllHistory(scanType = null) {
  const pageSize = 100;
  let page = 1;
  let items = [];
  let total = Infinity;

  while (items.length < total) {
    const data = await fetchHistory(page, pageSize, scanType || null);
    total = Number(data.total ?? 0);
    const batch = data.items || [];
    items = items.concat(batch);
    if (!batch.length || items.length >= total) break;
    page += 1;
    if (page > 50) break; // safety cap
  }

  return { items, total: total === Infinity ? items.length : total };
}

function ConditionTags({ row }) {
  const preds = normalizePredictions(row);
  if (!preds.length) return <span className="text-clinical-muted">—</span>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {preds.map((p, idx) => (
        <span
          key={`${row.id}-${p.label}-${idx}`}
          className="inline-flex items-center gap-1 border border-clinical-line bg-clinical-soft px-2 py-0.5 text-xs text-clinical-ink"
        >
          <span className="font-medium">{p.label}</span>
          <span className={`tabular-nums font-semibold ${confidenceTextClass(p.confidence)}`}>
            {(Number(p.confidence) * 100).toFixed(0)}%
          </span>
        </span>
      ))}
    </div>
  );
}

/**
 * Side panel: fetch full stored report and render like Analysis results
 * (image/heatmap toggle | condition cards + report).
 */
function DetailPanel({ predictionId, fallbackRow, onClose }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (predictionId == null) return undefined;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [predictionId, onClose]);

  useEffect(() => {
    if (predictionId == null) {
      setReport(null);
      setError(null);
      return undefined;
    }
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      setReport(null);
      try {
        const data = await fetchHistoryReport(predictionId);
        if (!cancelled) setReport(data);
      } catch (err) {
        if (!cancelled) {
          setError(err.response?.data?.detail || err.message || "Failed to load report");
          if (fallbackRow && fallbackRow.id === predictionId) {
            setReport({
              id: fallbackRow.id,
              scan_type: fallbackRow.scan_type,
              predictions: normalizePredictions(fallbackRow),
              report_text: fallbackRow.report_text || "",
              heatmap_url: fallbackRow.heatmap_url,
              image_url: fallbackRow.image_url,
              created_at: fallbackRow.created_at,
            });
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [predictionId, fallbackRow]);

  if (predictionId == null) return null;

  const result = report
    ? {
        ...report,
        prediction: report.predictions?.[0]?.label,
        confidence: report.predictions?.[0]?.confidence,
      }
    : null;

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-clinical-ink/40"
      role="presentation"
    >
      <button
        type="button"
        className="flex-1 cursor-default"
        aria-label="Close detail panel"
        onClick={onClose}
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="history-detail-title"
        className="flex h-full w-full max-w-full flex-col overflow-y-auto border-l border-clinical-line bg-white shadow-panelHover lg:max-w-4xl"
      >
        <div className="sticky top-0 z-10 flex items-start justify-between gap-3 border-b border-clinical-line bg-white px-4 py-4 sm:px-5">
          <div className="min-w-0">
            <p className="clinical-label">Stored report</p>
            <h2
              id="history-detail-title"
              className="mt-1 truncate font-display text-xl text-clinical-ink sm:text-2xl"
            >
              #{predictionId}
              {result?.scan_type ? ` · ${scanTypeLabel(result.scan_type)}` : ""}
            </h2>
            <p className="mt-1 text-xs text-clinical-muted">
              {result?.created_at
                ? new Date(result.created_at).toLocaleString()
                : fallbackRow?.created_at
                  ? new Date(fallbackRow.created_at).toLocaleString()
                  : "—"}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 rounded-sm border border-clinical-line p-2 text-clinical-muted hover:bg-clinical-soft hover:text-clinical-ink"
            aria-label="Close detail"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-5 p-4 sm:p-5">
          {error && !result && (
            <div className="border border-red-200 bg-red-50 px-4 py-3 text-sm text-clinical-danger">
              {error}
            </div>
          )}

          {loading && !result ? (
            <div className="grid grid-cols-1 items-start gap-5 lg:grid-cols-2">
              <div className="clinical-panel overflow-hidden">
                <div className="border-b border-clinical-line px-4 py-3">
                  <div className="skeleton h-4 w-36" />
                </div>
                <div className="skeleton aspect-square w-full rounded-none" />
              </div>
              <div className="space-y-4">
                <div className="clinical-panel space-y-3 p-5">
                  <div className="skeleton h-3 w-24" />
                  <div className="skeleton h-8 w-40" />
                  <div className="skeleton h-2 w-full" />
                </div>
                <div className="clinical-panel space-y-3 p-5">
                  <div className="skeleton h-10 w-10" />
                  <div className="skeleton h-3 w-full" />
                  <div className="skeleton h-3 w-4/5" />
                </div>
              </div>
            </div>
          ) : result ? (
            <div className="grid grid-cols-1 items-start gap-5 sm:gap-6 lg:grid-cols-2">
              <div className="min-w-0">
                <ImageHeatmapToggle
                  key={result.id}
                  imageUrl={result.image_url}
                  heatmapUrl={result.heatmap_url}
                />
              </div>
              <div className="min-w-0 space-y-5">
                <ResultCard result={result} />
                <ReportCard report={result.report_text} />
              </div>
            </div>
          ) : null}
        </div>
      </aside>
    </div>
  );
}

function SortIcon({ active, dir }) {
  if (!active) return <ArrowUpDown className="h-3.5 w-3.5 opacity-50" />;
  return dir === "asc" ? (
    <ArrowUp className="h-3.5 w-3.5 text-clinical-teal" />
  ) : (
    <ArrowDown className="h-3.5 w-3.5 text-clinical-teal" />
  );
}

function sortValue(row, key) {
  if (key === "confidence") return Number(row.confidence || 0);
  if (key === "scan_type") return row.scan_type || "";
  if (key === "prediction") {
    return (
      normalizePredictions(row)[0]?.label ||
      row.prediction ||
      ""
    ).toLowerCase();
  }
  if (key === "id") return Number(row.id || 0);
  return row.created_at ? new Date(row.created_at).getTime() : 0;
}

export default function History() {
  const location = useLocation();
  const [allRows, setAllRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(true);

  const [scanType, setScanType] = useState("");
  const [search, setSearch] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [sortKey, setSortKey] = useState("created_at");
  const [sortDir, setSortDir] = useState("desc");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setSummaryLoading(true);
      try {
        const data = await fetchHistorySummary();
        if (!cancelled) setSummary(data);
      } catch {
        if (!cancelled) setSummary(null);
      } finally {
        if (!cancelled) setSummaryLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const { items } = await fetchAllHistory(scanType || null);
        if (!cancelled) setAllRows(items);
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Failed to load history");
          setAllRows([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [scanType]);

  const filtered = useMemo(() => {
    let list = [...allRows];
    const q = search.trim().toLowerCase();

    if (q) {
      list = list.filter((row) => {
        const labels = normalizePredictions(row)
          .map((p) => String(p.label || "").toLowerCase())
          .join(" ");
        return (
          labels.includes(q) ||
          String(row.prediction || "").toLowerCase().includes(q) ||
          String(row.prediction_label || "").toLowerCase().includes(q)
        );
      });
    }

    if (dateFrom) {
      const from = new Date(dateFrom);
      from.setHours(0, 0, 0, 0);
      list = list.filter((row) => row.created_at && new Date(row.created_at) >= from);
    }

    if (dateTo) {
      const to = new Date(dateTo);
      to.setHours(23, 59, 59, 999);
      list = list.filter((row) => row.created_at && new Date(row.created_at) <= to);
    }

    list.sort((a, b) => {
      const av = sortValue(a, sortKey);
      const bv = sortValue(b, sortKey);
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return 0;
    });

    return list;
  }, [allRows, search, dateFrom, dateTo, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const pageRows = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  useEffect(() => {
    setPage(1);
  }, [scanType, search, dateFrom, dateTo]);

  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  useEffect(() => {
    const focusId = location.state?.focusId;
    if (focusId == null || !allRows.length) return;
    const match = allRows.find((r) => r.id === focusId);
    if (!match) return;
    setSelected(match);
  }, [location.state, allRows]);

  function toggleSort(key) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "created_at" || key === "id" || key === "confidence" ? "desc" : "asc");
    }
  }

  function clearFilters() {
    setSearch("");
    setDateFrom("");
    setDateTo("");
    setPage(1);
  }

  function openDetail(row) {
    setSelected(row);
  }

  const hasFilters = Boolean(search || dateFrom || dateTo);
  const categoryLabel =
    CATEGORY_TABS.find((t) => t.value === scanType)?.label || "All";

  return (
    <div className="space-y-6">
      <header>
        <p className="clinical-label">Archive</p>
        <h1 className="mt-2 font-display text-3xl text-clinical-ink sm:text-4xl md:text-5xl">
          History
        </h1>
        <p className="mt-2 max-w-xl text-sm text-clinical-muted sm:text-base">
          Searchable prediction archive. Click a row to open image, Grad-CAM, findings, and
          report detail.
        </p>
      </header>

      {/* Category summary cards — GET /api/history/summary */}
      <div className="grid gap-3 sm:grid-cols-3">
        {SCAN_TYPE_OPTIONS.map((opt) => {
          const bucket = summary?.[opt.value];
          const total = bucket?.total ?? 0;
          const topFinding = mostCommonFinding(bucket?.conditions);
          const active = scanType === opt.value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => setScanType(opt.value)}
              className={[
                "clinical-panel clinical-panel-hover p-4 text-left transition",
                active ? "border-clinical-teal ring-1 ring-clinical-teal/30" : "",
              ].join(" ")}
              aria-pressed={active}
            >
              <p className="clinical-label">{opt.label}</p>
              {summaryLoading ? (
                <div className="mt-3 space-y-2">
                  <div className="skeleton h-8 w-16" />
                  <div className="skeleton h-3 w-28" />
                </div>
              ) : (
                <>
                  <p className="mt-2 font-display text-3xl text-clinical-ink">
                    {total}
                  </p>
                  <p className="mt-1 text-xs text-clinical-muted">
                    {total === 1 ? "scan" : "scans"} · most common:{" "}
                    <span className="font-medium text-clinical-ink">
                      {topFinding || "—"}
                    </span>
                  </p>
                </>
              )}
            </button>
          );
        })}
      </div>

      {/* Category segmented control — drives GET /api/history?scan_type=… */}
      <div
        className="flex flex-wrap gap-1 border border-clinical-line bg-white p-1"
        role="tablist"
        aria-label="Scan category"
      >
        {CATEGORY_TABS.map((tab) => {
          const active = scanType === tab.value;
          return (
            <button
              key={tab.value || "all"}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setScanType(tab.value)}
              className={[
                "flex-1 min-w-[7rem] px-3 py-2 text-sm font-semibold transition sm:flex-none",
                active
                  ? "bg-clinical-teal text-white"
                  : "text-clinical-muted hover:bg-clinical-soft hover:text-clinical-ink",
              ].join(" ")}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      <div className="clinical-panel grid gap-3 p-3 sm:p-4 md:grid-cols-2 lg:grid-cols-12">
        <label className="text-sm md:col-span-2 lg:col-span-5">
          <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">
            Search by result label
          </span>
          <span className="relative block">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-clinical-muted" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="e.g. Pneumonia, Tumor…"
              className="w-full border border-clinical-line bg-white py-2 pl-9 pr-3 text-sm outline-none focus:border-clinical-teal"
            />
          </span>
        </label>

        <div className="grid grid-cols-2 gap-2 md:col-span-2 lg:col-span-5">
          <label className="text-sm">
            <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">
              From
            </span>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="w-full min-w-0 border border-clinical-line bg-white px-2 py-2 text-sm outline-none focus:border-clinical-teal"
            />
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">
              To
            </span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="w-full min-w-0 border border-clinical-line bg-white px-2 py-2 text-sm outline-none focus:border-clinical-teal"
            />
          </label>
        </div>

        <div className="flex items-end lg:col-span-2">
          <button
            type="button"
            onClick={clearFilters}
            disabled={!hasFilters}
            className="w-full border border-clinical-line px-3 py-2 text-sm text-clinical-muted transition hover:border-clinical-teal hover:text-clinical-teal disabled:opacity-40"
          >
            Clear
          </button>
        </div>
      </div>

      {error && (
        <div className="border border-red-200 bg-red-50 px-4 py-3 text-sm text-clinical-danger">
          {error}
        </div>
      )}

      {loading ? (
        <div className="clinical-panel space-y-3 p-6">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="skeleton h-10 w-full" />
          ))}
        </div>
      ) : allRows.length === 0 ? (
        <EmptyState
          title={scanType ? `No ${categoryLabel} scans yet` : "No scans yet"}
          description={
            scanType
              ? `No predictions in the ${categoryLabel} category. Try another tab or upload a new study.`
              : "Upload your first medical image to start building prediction history."
          }
          actionLabel="Upload first image"
          actionTo="/analysis"
        />
      ) : filtered.length === 0 ? (
        <div className="clinical-panel flex flex-col items-center px-6 py-14 text-center">
          <p className="font-display text-xl text-clinical-ink">No matching records</p>
          <p className="mt-2 max-w-md text-sm text-clinical-muted">
            No results in {categoryLabel} for the current search or date filters.
          </p>
          {hasFilters && (
            <button
              type="button"
              onClick={clearFilters}
              className="mt-6 inline-flex border border-clinical-teal bg-clinical-teal px-4 py-2 text-sm font-semibold text-white transition hover:bg-clinical-tealDark"
            >
              Clear filters
            </button>
          )}
        </div>
      ) : (
        <>
          {/* Mobile sort control */}
          <div className="flex items-center gap-2 md:hidden">
            <label className="flex min-w-0 flex-1 items-center gap-2 text-sm">
              <span className="shrink-0 text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">
                Sort
              </span>
              <select
                value={sortKey}
                onChange={(e) => {
                  const key = e.target.value;
                  setSortKey(key);
                  setSortDir(
                    key === "created_at" || key === "id" || key === "confidence"
                      ? "desc"
                      : "asc"
                  );
                }}
                className="min-w-0 flex-1 border border-clinical-line bg-white px-2 py-2 text-sm outline-none focus:border-clinical-teal"
              >
                {COLUMNS.map(({ key, label }) => (
                  <option key={key} value={key}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              onClick={() => setSortDir((d) => (d === "asc" ? "desc" : "asc"))}
              className="inline-flex items-center gap-1 border border-clinical-line px-3 py-2 text-sm text-clinical-muted"
              aria-label={`Sort ${sortDir === "asc" ? "descending" : "ascending"}`}
            >
              <SortIcon active dir={sortDir} />
            </button>
          </div>

          {/* Mobile: stacked cards */}
          <div className="space-y-3 md:hidden">
            {pageRows.map((row) => (
              <article
                key={row.id}
                role="button"
                tabIndex={0}
                onClick={() => openDetail(row)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    openDetail(row);
                  }
                }}
                className="clinical-panel cursor-pointer p-4 transition hover:border-clinical-teal"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-xs font-semibold uppercase tracking-[0.12em] text-clinical-muted">
                      #{row.id} · {scanTypeLabel(row.scan_type)}
                    </p>
                    <div className="mt-2">
                      <ConditionTags row={row} />
                    </div>
                  </div>
                  <ConfidenceBadge confidence={row.confidence} />
                </div>
                <div className="mt-3 flex items-center justify-between gap-2 border-t border-clinical-line/70 pt-3 text-xs text-clinical-muted">
                  <span className="truncate">
                    {row.created_at ? new Date(row.created_at).toLocaleString() : "—"}
                  </span>
                  <span className="shrink-0 font-medium text-clinical-teal">View</span>
                </div>
              </article>
            ))}
          </div>

          {/* Desktop: table */}
          <div className="clinical-panel hidden overflow-x-auto md:block">
            <table className="w-full min-w-[820px] text-left text-sm">
              <thead>
                <tr className="border-b border-clinical-line bg-clinical-soft/80 text-clinical-muted">
                  {COLUMNS.map(({ key, label }) => (
                    <th key={key} className="px-4 py-3 font-semibold">
                      <button
                        type="button"
                        onClick={() => toggleSort(key)}
                        className={[
                          "inline-flex items-center gap-1.5",
                          sortKey === key ? "text-clinical-ink" : "hover:text-clinical-ink",
                        ].join(" ")}
                        aria-sort={
                          sortKey === key
                            ? sortDir === "asc"
                              ? "ascending"
                              : "descending"
                            : "none"
                        }
                      >
                        {label}
                        <SortIcon active={sortKey === key} dir={sortDir} />
                      </button>
                    </th>
                  ))}
                  <th className="px-4 py-3 font-semibold">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {pageRows.map((row) => (
                  <tr
                    key={row.id}
                    tabIndex={0}
                    onClick={() => openDetail(row)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        openDetail(row);
                      }
                    }}
                    className="cursor-pointer border-b border-clinical-line/70 last:border-0 hover:bg-clinical-soft/70 focus-visible:bg-clinical-soft/70 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-clinical-teal"
                  >
                    <td className="px-4 py-3 tabular-nums text-clinical-muted">{row.id}</td>
                    <td className="px-4 py-3 text-clinical-ink">
                      {scanTypeLabel(row.scan_type)}
                    </td>
                    <td className="px-4 py-3">
                      <ConditionTags row={row} />
                    </td>
                    <td className="px-4 py-3">
                      <ConfidenceBadge confidence={row.confidence} />
                    </td>
                    <td className="px-4 py-3 text-clinical-muted">
                      {row.created_at ? new Date(row.created_at).toLocaleString() : "—"}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        className="font-medium text-clinical-teal underline-offset-2 hover:underline"
                        onClick={(e) => {
                          e.stopPropagation();
                          openDetail(row);
                        }}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex flex-col gap-3 text-sm sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
            <p className="text-clinical-muted">
              Showing {(safePage - 1) * PAGE_SIZE + 1}–
              {Math.min(safePage * PAGE_SIZE, filtered.length)} of {filtered.length}
              {hasFilters ? " matching" : ""} · {allRows.length} total
            </p>
            <div className="flex items-center justify-between gap-2 sm:justify-end">
              <button
                type="button"
                disabled={safePage <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="inline-flex items-center gap-1 border border-clinical-line px-3 py-1.5 disabled:opacity-40"
              >
                <ChevronLeft className="h-4 w-4" /> Previous
              </button>
              <span className="tabular-nums text-clinical-muted">
                Page {safePage} of {totalPages}
              </span>
              <button
                type="button"
                disabled={safePage >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                className="inline-flex items-center gap-1 border border-clinical-line px-3 py-1.5 disabled:opacity-40"
              >
                Next <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </>
      )}

      <DetailPanel
        predictionId={selected?.id ?? null}
        fallbackRow={selected}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}
