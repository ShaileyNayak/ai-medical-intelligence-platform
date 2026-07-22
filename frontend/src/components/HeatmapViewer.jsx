const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export default function HeatmapViewer({ imageUrl, heatmapUrl, localPreview }) {
  if (!imageUrl && !heatmapUrl && !localPreview) return null;

  return (
    <section className="grid gap-4 md:grid-cols-2">
      <figure>
        <figcaption className="mb-2 text-sm text-ink/60">Original</figcaption>
        <img
          src={imageUrl ? `${API_BASE}${imageUrl}` : localPreview}
          alt="Uploaded X-ray"
          className="w-full bg-black/5"
        />
      </figure>
      {heatmapUrl && (
        <figure>
          <figcaption className="mb-2 text-sm text-ink/60">Grad-CAM overlay</figcaption>
          <img
            src={`${API_BASE}${heatmapUrl}`}
            alt="Grad-CAM heatmap"
            className="w-full bg-black/5"
          />
        </figure>
      )}
    </section>
  );
}
