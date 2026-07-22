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
      setError(err.response?.data?.detail || err.message || "Prediction failed");
      return null;
    } finally {
      setLoading(false);
    }
  }

  return { result, loading, error, runPrediction };
}
