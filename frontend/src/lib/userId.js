const KEY = "jobpilot.userId";

/** Anonymous user id kept in this browser. Falls back to a per-session id if storage is
 * unavailable (private mode, blocked storage). */
export function getUserId() {
  try {
    const existing = localStorage.getItem(KEY);
    if (existing) return existing;
    const id = crypto.randomUUID();
    localStorage.setItem(KEY, id);
    return id;
  } catch {
    return crypto.randomUUID();
  }
}
