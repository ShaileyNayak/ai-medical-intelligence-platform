import { useState } from "react";
import { RotateCcw } from "lucide-react";
import UploadForm from "../components/UploadForm.jsx";
import ResultCard, { ReportCard } from "../components/ResultCard.jsx";
import ImageHeatmapToggle from "../components/ImageHeatmapToggle.jsx";
import AnalysisSkeleton from "../components/AnalysisSkeleton.jsx";
import { usePrediction } from "../hooks/usePrediction.js";
import { scanTypeLabel } from "../constants/scanTypes.js";

export default function Analysis() {
  const { result, loading, error, runPrediction, reset } = usePrediction();
  const [localPreview, setLocalPreview] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [scanType, setScanType] = useState("chest_xray");

  async function handleSubmit({ file, scanType: selectedScanType }) {
    const nextType = selectedScanType || scanType;
    setScanType(nextType);
    if (localPreview) URL.revokeObjectURL(localPreview);
    setLocalPreview(URL.createObjectURL(file));
    setFileName(file.name);
    await runPrediction(file, nextType);
  }

  function handleReset() {
    if (localPreview) URL.revokeObjectURL(localPreview);
    setLocalPreview(null);
    setFileName(null);
    reset();
  }

  return (
    <div className="space-y-8">
      <header className="max-w-2xl">
        <p className="clinical-label">Workspace</p>
        <h1 className="mt-2 font-display text-3xl text-clinical-ink sm:text-4xl md:text-5xl">
          New analysis
        </h1>
        <p className="mt-2 text-sm text-clinical-muted leading-relaxed sm:text-base">
          Choose a scan type, upload an image, and review Grad-CAM attention with ranked
          findings and an assistive report.
        </p>
      </header>

      <UploadForm
        scanType={scanType}
        onScanTypeChange={setScanType}
        onSubmit={handleSubmit}
        disabled={loading}
      />

      {(fileName || loading) && (
        <div className="flex flex-wrap items-center gap-3 text-sm">
          {fileName && (
            <span className="clinical-panel max-w-full truncate px-3 py-1.5 text-clinical-ink">
              {scanTypeLabel(scanType)} · <strong className="font-medium">{fileName}</strong>
            </span>
          )}
          {loading && (
            <span className="text-clinical-teal">Running inference → Grad-CAM → report…</span>
          )}
          {!loading && result && (
            <button
              type="button"
              onClick={handleReset}
              className="inline-flex items-center gap-1.5 font-semibold text-clinical-teal hover:underline"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Analyze another image
            </button>
          )}
        </div>
      )}

      {error && (
        <div className="border border-red-200 bg-red-50 px-4 py-3 text-sm text-clinical-danger">
          {error}
        </div>
      )}

      {loading && <AnalysisSkeleton localPreview={localPreview} />}

      {result && !loading && (
        <section className="space-y-4" aria-label="Analysis results">
          <div>
            <p className="clinical-label">Results</p>
            <h2 className="mt-1 font-display text-2xl text-clinical-ink">
              Study review
            </h2>
          </div>

          <div className="grid grid-cols-1 items-start gap-5 sm:gap-6 lg:grid-cols-2">
            {/* Left: original / Grad-CAM — stacks above findings on mobile */}
            <div className="min-w-0 lg:sticky lg:top-24">
              <ImageHeatmapToggle
                imageUrl={result.image_url}
                heatmapUrl={result.heatmap_url}
                localPreview={localPreview}
              />
            </div>

            {/* Right: condition cards + report */}
            <div className="min-w-0 space-y-5">
              <ResultCard result={result} />
              <ReportCard report={result.report_text} />
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
