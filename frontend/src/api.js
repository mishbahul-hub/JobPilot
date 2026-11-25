export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

const FRIENDLY = {
  429: "You're sending requests quickly. Please wait a moment and try again.",
  503: "The AI model is busy right now. Please try again in a minute.",
};

async function request(path, { method = "GET", json, form } = {}) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method,
      headers: json ? { "Content-Type": "application/json" } : undefined,
      body: json ? JSON.stringify(json) : form,
    });
  } catch {
    throw new Error("Can't reach the JobPilot server. Check your connection and try again.");
  }
  if (res.status === 204) return null;
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : null;
    throw new Error(FRIENDLY[res.status] || detail || `Something went wrong (${res.status}).`);
  }
  return data;
}

export const api = {
  chat: (userId, conversationId, message) =>
    request("/chat", {
      method: "POST",
      json: { user_id: userId, conversation_id: conversationId, message },
    }),
  conversations: (userId) => request(`/users/${userId}/conversations`),
  messages: (userId, convId) => request(`/users/${userId}/conversations/${convId}/messages`),
  profile: (userId) => request(`/users/${userId}/profile`),
  saved: (userId) => request(`/users/${userId}/saved`),
  save: (userId, jobId) =>
    request(`/users/${userId}/saved`, { method: "POST", json: { job_id: jobId } }),
  unsave: (userId, jobId) => request(`/users/${userId}/saved/${jobId}`, { method: "DELETE" }),
  feedback: (userId, jobId, rating, messageId) =>
    request("/feedback", {
      method: "POST",
      json: { user_id: userId, job_id: jobId, rating, message_id: messageId ?? null },
    }),
  uploadResume: (userId, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/users/${userId}/resume`, { method: "POST", form });
  },
};
