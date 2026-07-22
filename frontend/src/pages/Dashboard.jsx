import { useState } from "react";
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import UploadCard from "../components/UploadCard.jsx";
import PredictionResult from "../components/PredictionResult.jsx";
import HeatmapViewer from "../components/HeatmapViewer.jsx";
import ReportPanel from "../components/ReportPanel.jsx";
import { usePrediction } from "../hooks/usePrediction.js";

export default function Dashboard() {
  const { result, loading, error, runPrediction } = usePrediction();
  const [localPreview, setLocalPreview] = useState(null);

  async function handleFile(file) {
    setLocalPreview(URL.createObjectURL(file));
    await runPrediction(file);
  }

  const chartData = result
    ? [
        {
          name: result.predicted_label,
          confidence: Number((result.confidence_score * 100).toFixed(1)),
        },
      ]
    : [];

  return (
    <div className="space-y-10">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-accent">Chest X-Ray AI</p>
        <h1 className="mt-2 font-display text-4xl text-ink">Dashboard</h1>
        <p className="mt-2 max-w-xl text-ink/70">
          Upload a chest X-ray for Normal vs Pneumonia prediction, Grad-CAM
          visualization, and an LLM assistive report.
        </p>
      </div>

      <UploadCard onFileSelect={handleFile} disabled={loading} />

      {loading && <p className="text-accent">Running inference → Grad-CAM → report…</p>}
      {error && <p className="text-red-700">{error}</p>}

      {(result || localPreview) && (
        <div className="space-y-8">
          <PredictionResult result={result} />
          {chartData.length > 0 && (
            <div className="h-48 w-full max-w-md">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Bar dataKey="confidence" fill="#1a6b5c" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          <HeatmapViewer
            imageUrl={result?.image_url}
            heatmapUrl={result?.heatmap_url}
            localPreview={localPreview}
          />
          <ReportPanel report={result?.llm_report} />
        </div>
      )}
    </div>
  );
}
