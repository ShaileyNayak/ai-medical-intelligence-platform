/** Shared scan-type options for upload UI and history labels. */

export const SCAN_TYPE_OPTIONS = [
  { value: "chest_xray", label: "Chest X-ray" },
  { value: "brain_mri", label: "Brain MRI" },
  { value: "skin_lesion", label: "Skin Lesion" },
];

export function scanTypeLabel(scanType) {
  const found = SCAN_TYPE_OPTIONS.find((o) => o.value === scanType);
  return found?.label || scanType || "—";
}

export function normalizePredictions(result) {
  if (!result) return [];
  if (Array.isArray(result.predictions) && result.predictions.length > 0) {
    return [...result.predictions].sort(
      (a, b) => Number(b.confidence || 0) - Number(a.confidence || 0)
    );
  }
  if (result.prediction) {
    return [
      {
        label: result.prediction,
        confidence: Number(result.confidence || 0),
      },
    ];
  }
  return [];
}

export function isAlertLabel(label) {
  const t = String(label || "").toLowerCase().trim();
  if (t === "no tumor" || t === "normal" || t === "benign") return false;
  return (
    t.includes("pneumonia") ||
    t.includes("covid") ||
    t.includes("tuberculosis") ||
    t === "tumor" ||
    t.includes("malignant")
  );
}
