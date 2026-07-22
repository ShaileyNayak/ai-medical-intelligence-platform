export default function PredictionResult({ result }) {
  if (!result) return null;

  return (
    <section className="space-y-2">
      <h2 className="font-display text-2xl text-ink">Prediction</h2>
      <p className="text-2xl capitalize text-accent">{result.predicted_label}</p>
      <p className="text-sm text-ink/70">
        Confidence: {(result.confidence_score * 100).toFixed(1)}%
      </p>
      <p className="text-xs text-ink/50">Model: {result.model_version}</p>
    </section>
  );
}
