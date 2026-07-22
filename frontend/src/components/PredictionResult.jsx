export default function PredictionResult({ result }) {
  if (!result) return null;

  const pct = Math.round(Number(result.confidence || 0) * 1000) / 10;
  const isPneumonia = String(result.prediction || "").toLowerCase().includes("pneumonia");

  return (
    <section className="clinical-panel p-6 md:p-8">
      <p className="clinical-label">Model output</p>
      <div className="mt-4 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="font-display text-3xl text-clinical-ink md:text-4xl">
            {result.prediction}
          </h2>
          <p className="mt-1 text-sm text-clinical-muted">
            Binary classification · Normal vs Pneumonia
          </p>
        </div>
        <div className="min-w-[180px] text-right">
          <p className="text-sm text-clinical-muted">Confidence</p>
          <p
            className={[
              "font-display text-3xl",
              isPneumonia ? "text-clinical-alert" : "text-clinical-teal",
            ].join(" ")}
          >
            {pct.toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="mt-6">
        <div className="mb-2 flex justify-between text-xs text-clinical-muted">
          <span>Softmax probability</span>
          <span>{pct.toFixed(1)}%</span>
        </div>
        <div className="h-2 w-full bg-clinical-soft">
          <div
            className={isPneumonia ? "h-2 bg-clinical-alert" : "h-2 bg-clinical-teal"}
            style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
          />
        </div>
      </div>
    </section>
  );
}
