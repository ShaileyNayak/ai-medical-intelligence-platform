import { useState } from "react";
import UploadForm from "../components/UploadForm.jsx";
import ResultCard from "../components/ResultCard.jsx";
import HeatmapViewer from "../components/HeatmapViewer.jsx";
import ReportPanel from "../components/ReportPanel.jsx";
import { usePrediction } from "../hooks/usePrediction.js";
import { scanTypeLabel } from "../constants/scanTypes.js";

export default function Dashboard() {
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
    <div className="space-y-10">
      <header className="max-w-2xl">
        <p className="clinical-label">Multi-disease analysis</p>
        <h1 className="mt-3 font-display text-4xl text-clinical-ink md:text-5xl">
          Upload & interpret
        </h1>
        <p className="mt-3 text-clinical-muted leading-relaxed">
          Choose a scan type, upload an image, and review model findings, Grad-CAM
          attention, and a plain-language assistive report.
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
            <span className="border border-clinical-line bg-white px-3 py-1.5 text-clinical-ink">
              {scanTypeLabel(scanType)} ·{" "}
              <strong className="font-medium">{fileName}</strong>
            </span>
          )}
          {loading && (
            <span className="text-clinical-teal">
              Running inference → Grad-CAM → report…
            </span>
          )}
          {!loading && result && (
            <button
              type="button"
              onClick={handleReset}
              className="text-clinical-teal underline-offset-2 hover:underline"
            >
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

      {(result || localPreview) && !loading && (
        <div className="space-y-8">
          <ResultCard result={result} />
          <HeatmapViewer
            imageUrl={result?.image_url}
            heatmapUrl={result?.heatmap_url}
            localPreview={localPreview}
          />
          <ReportPanel report={result?.report_text} />
        </div>
      )}
    </div>
  );
}
