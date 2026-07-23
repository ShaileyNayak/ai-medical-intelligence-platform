import {
  isAlertLabel,
  normalizePredictions,
  scanTypeLabel,
} from "../constants/scanTypes.js";

function ConfidenceBar({ confidence, alert }) {
  const pct = Math.round(Number(confidence || 0) * 1000) / 10;
  return (
    <div>
      <div className="mb-2 flex justify-between text-xs text-clinical-muted">
        <span>Confidence</span>
        <span>{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 w-full bg-clinical-soft">
        <div
          className={alert ? "h-2 bg-clinical-alert" : "h-2 bg-clinical-teal"}
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
      </div>
    </div>
  );
}

function ConditionCard({ item, rank, multi }) {
  const pct = Math.round(Number(item.confidence || 0) * 1000) / 10;
  const alert = isAlertLabel(item.label);

  return (
    <article className="clinical-panel p-5 md:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          {multi && (
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-clinical-muted">
              Rank {rank}
            </p>
          )}
          <h3
            className={[
              "font-display text-2xl text-clinical-ink md:text-3xl",
              multi ? "mt-1" : "",
            ].join(" ")}
          >
            {item.label}
          </h3>
        </div>
        <p
          className={[
            "font-display text-2xl md:text-3xl",
            alert ? "text-clinical-alert" : "text-clinical-teal",
          ].join(" ")}
        >
          {pct.toFixed(1)}%
        </p>
      </div>
      <div className="mt-5">
        <ConfidenceBar confidence={item.confidence} alert={alert} />
      </div>
    </article>
  );
}

/**
 * Renders a single prediction card (brain MRI / skin) or a ranked list of
 * condition cards (multi-label chest X-ray).
 */
export default function ResultCard({ result }) {
  if (!result) return null;

  const predictions = normalizePredictions(result);
  const multi = predictions.length > 1;
  const modality = scanTypeLabel(result.scan_type);

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="clinical-label">Model output</p>
          <h2 className="mt-2 font-display text-3xl text-clinical-ink md:text-4xl">
            {multi ? "Detected conditions" : "Prediction"}
          </h2>
          <p className="mt-1 text-sm text-clinical-muted">
            {modality}
            {multi
              ? " · multi-label · ranked by confidence"
              : " · single-label classification"}
          </p>
        </div>
      </div>

      {predictions.length === 0 ? (
        <div className="clinical-panel p-6 text-sm text-clinical-muted">
          No conditions cleared the detection threshold. The assistive report may
          describe a normal or inconclusive reading.
        </div>
      ) : multi ? (
        <div className="grid gap-4 md:grid-cols-2">
          {predictions.map((item, idx) => (
            <ConditionCard
              key={`${item.label}-${idx}`}
              item={item}
              rank={idx + 1}
              multi
            />
          ))}
        </div>
      ) : (
        <ConditionCard item={predictions[0]} rank={1} multi={false} />
      )}
    </section>
  );
}
