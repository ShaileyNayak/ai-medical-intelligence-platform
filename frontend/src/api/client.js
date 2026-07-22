import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "",
  timeout: 120000,
});

export async function predictImage(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/api/predict", form);
  return data;
}

export async function fetchHistory(page = 1, pageSize = 20) {
  const { data } = await api.get("/api/history", {
    params: { page, page_size: pageSize },
  });
  return data;
}

export async function fetchHealth() {
  const { data } = await api.get("/api/health");
  return data;
}

export default api;
