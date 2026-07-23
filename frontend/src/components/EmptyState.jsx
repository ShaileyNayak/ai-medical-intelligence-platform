import { Link } from "react-router-dom";
import { ScanLine } from "lucide-react";

/**
 * Friendly empty state with icon, message, and CTA to New Analysis.
 */
export default function EmptyState({
  title = "No scans yet",
  description = "Upload your first medical image to run a prediction and populate this view.",
  actionLabel = "Go to Analysis",
  actionTo = "/analysis",
  icon: Icon = ScanLine,
}) {
  return (
    <div className="clinical-panel flex flex-col items-center px-6 py-16 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-clinical-soft text-clinical-teal">
        <Icon className="h-8 w-8" strokeWidth={1.5} aria-hidden />
      </div>
      <p className="mt-5 font-display text-2xl text-clinical-ink">{title}</p>
      {description && (
        <p className="mt-2 max-w-md text-sm leading-relaxed text-clinical-muted">
          {description}
        </p>
      )}
      {actionLabel && actionTo && (
        <Link
          to={actionTo}
          className="mt-6 inline-flex border border-clinical-teal bg-clinical-teal px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-clinical-tealDark"
        >
          {actionLabel}
        </Link>
      )}
    </div>
  );
}
