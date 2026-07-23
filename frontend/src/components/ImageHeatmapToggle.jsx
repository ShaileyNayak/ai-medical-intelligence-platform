import { useState } from "react";
import { Image as ImageIcon, Layers } from "lucide-react";
import { resolveMediaUrl } from "../utils/historyStats.js";

/**
 * Left-column study viewer: toggle between original upload and Grad-CAM overlay.
 */
export default function ImageHeatmapToggle({
  imageUrl,
  heatmapUrl,
  localPreview,
}) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const original = resolveMediaUrl(imageUrl) || localPreview || null;
  const heatmap = resolveMediaUrl(heatmapUrl);
  const src = showHeatmap && heatmap ? heatmap : original;

  if (!original && !heatmap) return null;

  return (
    <div className="clinical-panel overflow-hidden">
      <div className="flex flex-col gap-3 border-b border-clinical-line px-3 py-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between sm:px-4">
        <div className="flex items-center gap-2 text-sm font-medium text-clinical-ink">
          {showHeatmap && heatmap ? (
            <>
              <Layers className="h-4 w-4 shrink-0 text-clinical-teal" />
              Grad-CAM overlay
            </>
          ) : (
            <>
              <ImageIcon className="h-4 w-4 shrink-0 text-clinical-teal" />
              Original image
            </>
          )}
        </div>

        <label className="inline-flex cursor-pointer items-center gap-2 self-start text-xs font-semibold text-clinical-muted sm:self-auto">
          <span className={!showHeatmap ? "text-clinical-ink" : ""}>Original</span>
          <span className="relative inline-flex h-6 w-11 shrink-0 items-center">
            <input
              type="checkbox"
              className="peer sr-only"
              checked={Boolean(showHeatmap && heatmap)}
              onChange={(e) => setShowHeatmap(e.target.checked)}
              disabled={!heatmap}
              aria-label="Show Grad-CAM heatmap overlay"
            />
            <span className="absolute inset-0 rounded-full bg-clinical-line transition peer-checked:bg-clinical-teal peer-disabled:opacity-40" />
            <span className="absolute left-0.5 h-5 w-5 rounded-full bg-white shadow transition peer-checked:translate-x-5" />
          </span>
          <span className={showHeatmap ? "text-clinical-ink" : ""}>Heatmap</span>
        </label>
      </div>

      {src ? (
        <img
          src={src}
          alt={showHeatmap ? "Grad-CAM heatmap overlay" : "Uploaded study"}
          className="aspect-square w-full max-h-[70vh] bg-slate-950 object-contain sm:max-h-none"
        />
      ) : (
        <div className="flex aspect-square items-center justify-center text-sm text-clinical-muted">
          Image unavailable
        </div>
      )}

      <p className="border-t border-clinical-line px-3 py-2 text-xs text-clinical-muted sm:px-4">
        Toggle to compare the original study with the Grad-CAM explanation overlay.
      </p>
    </div>
  );
}
