import { useState } from "react";
import { predictImage } from "../api/client.js";

export function usePrediction() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function runPrediction(file) {
    setLoading(true);
    setError(null);
    try {
      const data = await predictImage(file);
      setResult(data);
      return data;
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg || JSON.stringify(d)).join(", ")
            : err.message || "Prediction failed"
      );
      return null;
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setResult(null);
    setError(null);
  }

  return { result, loading, error, runPrediction, reset };
}
