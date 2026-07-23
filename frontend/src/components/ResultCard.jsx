import { FileText } from "lucide-react";
import { ConfidenceBar } from "./ConfidenceBar.jsx";
import {
  normalizePredictions,
  scanTypeLabel,
} from "../constants/scanTypes.js";

/** Right-column finding cards with confidence progress bars. */
export default function ResultCard({ result }) {
  if (!result) return null;

  const predictions = normalizePredictions(result);
  const multi = predictions.length > 1;
  const modality = scanTypeLabel(result.scan_type);

  return (
    <section className="space-y-3">
      <div>
        <p className="clinical-label">Predictions</p>
        <h2 className="mt-1 font-display text-xl text-clinical-ink md:text-2xl">
          {multi ? "Detected conditions" : "Predicted condition"}
        </h2>
        <p className="mt-1 text-sm text-clinical-muted">
          {modality}
          {multi ? " · ranked by confidence" : ""}
        </p>
      </div>

      {predictions.length === 0 ? (
        <div className="clinical-panel p-5 text-sm text-clinical-muted">
          No conditions cleared the detection threshold. The assistive report may
          describe a normal or inconclusive reading.
        </div>
      ) : (
        <div className="space-y-3">
          {predictions.map((item, idx) => (
            <article
              key={`${item.label}-${idx}`}
              className="clinical-panel clinical-panel-hover p-4 md:p-5"
            >
              <div className="flex items-baseline justify-between gap-3">
                <h3 className="font-display text-xl text-clinical-ink">
                  {item.label}
                </h3>
                {multi && (
                  <span className="shrink-0 text-[11px] font-semibold uppercase tracking-[0.14em] text-clinical-muted">
                    #{idx + 1}
                  </span>
                )}
              </div>
              <ConfidenceBar confidence={item.confidence} className="mt-4" />
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

/** LLM report card with document icon. */
export function ReportCard({ report }) {
  return (
    <section className="clinical-panel p-5 md:p-6">
      <div className="flex items-start gap-3">
        <span
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-sm bg-clinical-soft text-clinical-teal"
          aria-hidden
        >
          <FileText className="h-5 w-5" strokeWidth={1.75} />
        </span>
        <div>
          <p className="clinical-label">Assistive report</p>
          <h2 className="mt-1 font-display text-xl text-clinical-ink">
            Plain-language summary
          </h2>
        </div>
      </div>

      {report ? (
        <div className="mt-5 whitespace-pre-wrap text-[15px] leading-relaxed text-clinical-ink/90">
          {report}
        </div>
      ) : (
        <p className="mt-5 text-sm text-clinical-muted">
          No report was generated for this study.
        </p>
      )}

      <p className="mt-5 border-t border-clinical-line pt-4 text-xs leading-relaxed text-clinical-muted">
        AI-generated for education only. Not a medical diagnosis — confirm with a
        licensed clinician.
      </p>
    </section>
  );
}
