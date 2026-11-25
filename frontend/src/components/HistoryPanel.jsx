export default function HistoryPanel({ conversations, activeId, onOpen, onNew }) {
  return (
    <>
      <button type="button" className="button" onClick={onNew}>
        New search
      </button>
      {conversations.length === 0 ? (
        <p className="muted">Your past searches will show up here.</p>
      ) : (
        <ul className="plain-list">
          {conversations.map((c) => (
            <li key={c.id}>
              <button
                type="button"
                className={`history-item${c.id === activeId ? " history-item--on" : ""}`}
                onClick={() => onOpen(c.id)}
              >
                {c.title || "Untitled"}
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
