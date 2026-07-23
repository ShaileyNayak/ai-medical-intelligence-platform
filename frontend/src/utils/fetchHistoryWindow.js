import { fetchHistory } from "../api/client.js";

/** Fetch a large recent history window for dashboard / analytics aggregation. */
export async function fetchHistoryWindow(limit = 200) {
  const data = await fetchHistory(1, Math.min(limit, 100));
  const items = data.items || [];
  const total = data.total || items.length;
  if (items.length >= total || items.length >= limit) {
    return { items, total };
  }
  // Second page if needed (API max page_size is 100)
  const page2 = await fetchHistory(2, 100);
  const merged = [...items, ...(page2.items || [])].slice(0, limit);
  return { items: merged, total: page2.total ?? total };
}
