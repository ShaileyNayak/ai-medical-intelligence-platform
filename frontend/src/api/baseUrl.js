/**
 * Backend origin for API + static media.
 * Prefer VITE_API_URL (Vercel); fall back to VITE_API_BASE_URL (Docker/docs);
 * then localhost for local Vite without an .env file.
 */
export function getApiBaseUrl() {
  const raw =
    import.meta.env.VITE_API_URL ||
    import.meta.env.VITE_API_BASE_URL ||
    "http://localhost:8000";
  return String(raw).trim().replace(/\/+$/, "");
}
