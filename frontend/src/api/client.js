import axios from "axios";

import { getApiBaseUrl } from "./baseUrl.js";

const api = axios.create({
  baseURL: getApiBaseUrl(),
  timeout: 120000,
});

export async function predictImage(file, scanType = "chest_xray") {
  const form = new FormData();
  form.append("file", file);
  form.append("scan_type", scanType);
  const { data } = await api.post("/api/predict", form);
  return data;
}

export async function fetchHistory(page = 1, pageSize = 20, scanType = null) {
  const params = { page, page_size: pageSize };
  if (scanType) params.scan_type = scanType;
  const { data } = await api.get("/api/history", { params });
  return data;
}

export async function fetchHistorySummary() {
  const { data } = await api.get("/api/history/summary");
  return data;
}

export async function fetchHistoryReport(predictionId) {
  const { data } = await api.get(`/api/history/${predictionId}/report`);
  return data;
}

export async function fetchHealth() {
  const { data } = await api.get("/api/health");
  return data;
}

export default api;
