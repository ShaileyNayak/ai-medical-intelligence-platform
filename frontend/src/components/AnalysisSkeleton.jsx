/**
 * Loading placeholder that mirrors the Analysis results 2-column layout:
 * left image panel + right finding cards and report card.
 */
export default function AnalysisSkeleton({ localPreview = null }) {
  return (
    <section className="space-y-4" aria-busy="true" aria-label="Loading analysis results">
      <div>
        <p className="clinical-label">Results</p>
        <h2 className="mt-1 font-display text-2xl text-clinical-ink">Study review</h2>
      </div>

      <div className="grid items-start gap-5 sm:gap-6 lg:grid-cols-2">
        {/* Left: image / heatmap panel shape */}
        <div className="clinical-panel min-w-0 overflow-hidden lg:sticky lg:top-24">
          <div className="flex items-center justify-between gap-3 border-b border-clinical-line px-4 py-3">
            <div className="skeleton h-4 w-36" />
            <div className="skeleton h-6 w-28 rounded-full" />
          </div>
          <div className="relative aspect-square w-full bg-slate-950">
            {localPreview ? (
              <img
                src={localPreview}
                alt=""
                className="h-full w-full object-contain opacity-40"
              />
            ) : null}
            <div className="absolute inset-0 skeleton rounded-none opacity-50" />
            <div className="absolute inset-0 flex items-center justify-center">
              <p className="rounded-sm bg-white/90 px-3 py-1.5 text-xs font-semibold text-clinical-muted">
                Processing study…
              </p>
            </div>
          </div>
          <div className="border-t border-clinical-line px-4 py-2">
            <div className="skeleton h-3 w-3/4 max-w-xs" />
          </div>
        </div>

        {/* Right: condition cards + report */}
        <div className="space-y-5">
          <div className="space-y-3">
            <div>
              <div className="skeleton h-3 w-24" />
              <div className="mt-2 skeleton h-7 w-48" />
              <div className="mt-2 skeleton h-3 w-40" />
            </div>
            {[0, 1].map((i) => (
              <div key={i} className="clinical-panel space-y-4 p-4 md:p-5">
                <div className="skeleton h-6 w-40" />
                <div className="space-y-1.5">
                  <div className="flex justify-between">
                    <div className="skeleton h-3 w-16" />
                    <div className="skeleton h-3 w-12" />
                  </div>
                  <div className="skeleton h-2 w-full" />
                </div>
              </div>
            ))}
          </div>

          <div className="clinical-panel space-y-4 p-5 md:p-6">
            <div className="flex items-start gap-3">
              <div className="skeleton h-10 w-10 shrink-0" />
              <div className="space-y-2 pt-1">
                <div className="skeleton h-3 w-28" />
                <div className="skeleton h-5 w-44" />
              </div>
            </div>
            <div className="space-y-2">
              <div className="skeleton h-3 w-full" />
              <div className="skeleton h-3 w-11/12" />
              <div className="skeleton h-3 w-4/5" />
              <div className="skeleton h-3 w-full" />
              <div className="skeleton h-3 w-2/3" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
