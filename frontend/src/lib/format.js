const LAKH = 100_000;

/** Yearly INR salary range -> "12–18 LPA", "12 LPA" or null. */
export function formatSalary(min, max) {
  const lpa = (v) => {
    const n = v / LAKH;
    return Number.isInteger(n) ? String(n) : n.toFixed(1);
  };
  if (min && max && min !== max) return `${lpa(min)}–${lpa(max)} LPA`;
  if (max || min) return `${lpa(max || min)} LPA`;
  return null;
}

/** "today", "3 days ago", "2 weeks ago"... */
export function timeAgo(iso, now = new Date()) {
  if (!iso) return null;
  const days = Math.floor((now - new Date(iso)) / 86_400_000);
  if (Number.isNaN(days)) return null;
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 14) return `${days} days ago`;
  if (days < 60) return `${Math.floor(days / 7)} weeks ago`;
  return `${Math.floor(days / 30)} months ago`;
}
