import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "./api.js";
import Drawer from "./components/Drawer.jsx";
import HistoryPanel from "./components/HistoryPanel.jsx";
import JobItem from "./components/JobItem.jsx";
import ProfilePanel from "./components/ProfilePanel.jsx";
import { getUserId } from "./lib/userId.js";

const SUGGESTIONS = [
  "Backend internships in Bengaluru for someone who knows Python and SQL",
  "Remote frontend jobs using React",
  "Data engineering roles in Hyderabad",
];

export default function App() {
  const [userId] = useState(getUserId);
  const [conversations, setConversations] = useState([]);
  const [convId, setConvId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [profile, setProfile] = useState(null);
  const [saved, setSaved] = useState([]);
  const [ratings, setRatings] = useState({});
  const [drawer, setDrawer] = useState(null); // null | "history" | "profile"
  const endRef = useRef(null);
  const inputRef = useRef(null);

  const refreshSide = useCallback(async () => {
    const [convs, prof, sav] = await Promise.allSettled([
      api.conversations(userId),
      api.profile(userId),
      api.saved(userId),
    ]);
    if (convs.status === "fulfilled") setConversations(convs.value);
    if (prof.status === "fulfilled") setProfile(prof.value);
    if (sav.status === "fulfilled") setSaved(sav.value);
  }, [userId]);

  useEffect(() => {
    refreshSide();
  }, [refreshSide]);

  useEffect(() => {
    if (messages.length) endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const closeDrawer = useCallback(() => setDrawer(null), []);

  async function send(text) {
    const message = text.trim();
    if (!message || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: message }]);
    setLoading(true);
    try {
      const res = await api.chat(userId, convId, message);
      setConvId(res.conversation_id);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.reply, matches: res.matches, messageId: res.message_id },
      ]);
      refreshSide(); // the agent may have updated the profile or saved a job
    } catch (err) {
      setMessages((m) => [...m, { role: "error", content: err.message }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  async function openConversation(id) {
    setDrawer(null);
    try {
      const rows = await api.messages(userId, id);
      setConvId(id);
      setMessages(
        rows.map((r) => ({
          role: r.role,
          content: r.content,
          matches: r.matches || [],
          messageId: r.id,
        })),
      );
    } catch (err) {
      setMessages([{ role: "error", content: err.message }]);
    }
  }

  function newSearch() {
    setConvId(null);
    setMessages([]);
    setDrawer(null);
    inputRef.current?.focus();
  }

  const savedIds = new Set(saved.map((j) => j.id));

  async function toggleSave(jobId) {
    try {
      if (savedIds.has(jobId)) await api.unsave(userId, jobId);
      else await api.save(userId, jobId);
      setSaved(await api.saved(userId));
    } catch (err) {
      setMessages((m) => [...m, { role: "error", content: err.message }]);
    }
  }

  async function rate(jobId, rating, messageId) {
    const previous = ratings[jobId];
    setRatings((r) => ({ ...r, [jobId]: rating }));
    try {
      await api.feedback(userId, jobId, rating, messageId);
    } catch (err) {
      setRatings((r) => ({ ...r, [jobId]: previous }));
      setMessages((m) => [...m, { role: "error", content: err.message }]);
    }
  }

  async function uploadResume(file) {
    const res = await api.uploadResume(userId, file);
    setProfile(res.profile);
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  }

  return (
    <div className="app">
      <header className="bar">
        <button type="button" className="wordmark" onClick={newSearch}>
          JobPilot
        </button>
        <nav className="bar__nav">
          <button type="button" className="link" onClick={() => setDrawer("history")}>
            History
          </button>
          <button type="button" className="link" onClick={() => setDrawer("profile")}>
            Profile{saved.length > 0 && <span className="muted"> · {saved.length} saved</span>}
          </button>
        </nav>
      </header>

      <main className="thread" aria-live="polite">
        {messages.length === 0 && (
          <section className="intro">
            <h1>Find tech jobs that fit you.</h1>
            <p className="muted">
              Describe what you're after: the role, your skills, a city. JobPilot searches current
              postings across India and tells you why each one might suit you.
            </p>
            <ul className="plain-list suggestions">
              {SUGGESTIONS.map((s) => (
                <li key={s}>
                  <button type="button" className="suggestion" onClick={() => send(s)}>
                    {s}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`turn turn--${m.role}`}>
            {m.content && <p className="turn__text">{m.content}</p>}
            {m.matches?.length > 0 && (
              <ol className="jobs">
                {m.matches.map((match) => (
                  <JobItem
                    key={match.job_id}
                    match={match}
                    saved={savedIds.has(match.job_id)}
                    rating={ratings[match.job_id]}
                    onToggleSave={toggleSave}
                    onRate={(jobId, r) => rate(jobId, r, m.messageId)}
                  />
                ))}
              </ol>
            )}
          </div>
        ))}

        {loading && <p className="turn turn--pending muted">Searching…</p>}
        <div ref={endRef} />
      </main>

      <div className="composer-wrap">
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={messages.length ? "Ask a follow-up" : "What kind of job are you looking for?"}
            rows={1}
            maxLength={2000}
            aria-label="Message"
          />
          <button type="submit" className="button" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
        <p className="fineprint">
          Listings from{" "}
          <a href="https://www.adzuna.in" target="_blank" rel="noopener noreferrer">
            Adzuna
          </a>
          . Suggestions can be wrong, so read the posting before you apply.
        </p>
      </div>

      {drawer === "history" && (
        <Drawer title="History" onClose={closeDrawer}>
          <HistoryPanel
            conversations={conversations}
            activeId={convId}
            onOpen={openConversation}
            onNew={newSearch}
          />
        </Drawer>
      )}
      {drawer === "profile" && (
        <Drawer title="Profile" onClose={closeDrawer}>
          <ProfilePanel
            profile={profile}
            saved={saved}
            onUpload={uploadResume}
            onUnsave={toggleSave}
          />
        </Drawer>
      )}
    </div>
  );
}
