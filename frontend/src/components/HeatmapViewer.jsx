const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

function resolveSrc(url, fallback) {
  if (url) return `${API_BASE}${url}`;
  return fallback || null;
}

export default function HeatmapViewer({ imageUrl, heatmapUrl, localPreview }) {
  const original = resolveSrc(imageUrl, localPreview);
  const heatmap = resolveSrc(heatmapUrl, null);

  if (!original && !heatmap) return null;

  return (
    <section className="space-y-4">
      <div>
        <p className="clinical-label">Imaging</p>
        <h2 className="mt-2 font-display text-2xl text-clinical-ink">
          Original & Grad-CAM overlay
        </h2>
        <p className="mt-1 text-sm text-clinical-muted">
          Warmer colors indicate regions that most influenced the model prediction.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <figure className="clinical-panel overflow-hidden">
          <figcaption className="border-b border-clinical-line px-4 py-2 text-sm font-medium text-clinical-muted">
            Original radiograph
          </figcaption>
          {original ? (
            <img
              src={original}
              alt="Uploaded chest X-ray"
              className="aspect-square w-full object-contain bg-slate-950"
            />
          ) : (
            <div className="flex aspect-square items-center justify-center text-sm text-clinical-muted">
              No image
            </div>
          )}
        </figure>

        <figure className="clinical-panel overflow-hidden">
          <figcaption className="border-b border-clinical-line px-4 py-2 text-sm font-medium text-clinical-muted">
            Grad-CAM heatmap
          </figcaption>
          {heatmap ? (
            <img
              src={heatmap}
              alt="Grad-CAM explanation overlay"
              className="aspect-square w-full object-contain bg-slate-950"
            />
          ) : (
            <div className="flex aspect-square items-center justify-center text-sm text-clinical-muted">
              Heatmap pending…
            </div>
          )}
        </figure>
      </div>
    </section>
  );
}
