import { useRef, useState } from "react";

import { formatSalary } from "../lib/format.js";

function Field({ label, items, empty }) {
  return (
    <div className="field">
      <dt>{label}</dt>
      <dd className={items?.length ? "" : "muted"}>{items?.length ? items.join(", ") : empty}</dd>
    </div>
  );
}

export default function ProfilePanel({ profile, saved, onUpload, onUnsave }) {
  const fileRef = useRef(null);
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);

  async function handleFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (file.size > 5 * 1024 * 1024) {
      setStatus("That file is over 5 MB.");
      return;
    }
    setBusy(true);
    setStatus("Reading your resume…");
    try {
      await onUpload(file);
      setStatus("Done. Your profile is updated.");
    } catch (err) {
      setStatus(err.message);
    } finally {
      setBusy(false);
    }
  }

  const p = profile || {};
  return (
    <>
      <section className="section">
        <p className="muted">
          JobPilot ranks jobs using what it knows about you. Upload a resume or just tell it in the
          chat.
        </p>
        <button
          type="button"
          className="button"
          disabled={busy}
          onClick={() => fileRef.current?.click()}
        >
          {busy ? "Reading…" : "Upload resume (PDF)"}
        </button>
        <input ref={fileRef} type="file" accept="application/pdf" hidden onChange={handleFile} />
        {status && <p className="small">{status}</p>}

        <dl>
          <Field label="Looking for" items={p.titles} empty="Not set" />
          <Field label="Skills" items={p.skills} empty="Not set" />
          <Field label="Cities" items={p.locations} empty="Anywhere" />
          {p.years_exp != null && <Field label="Experience" items={[`${p.years_exp} years`]} />}
          {p.min_salary && (
            <Field label="Minimum salary" items={[formatSalary(null, p.min_salary)]} />
          )}
        </dl>
        <p className="small muted">We keep skills and preferences only, never contact details.</p>
      </section>

      <section className="section">
        <h3>Saved jobs</h3>
        {saved.length === 0 ? (
          <p className="muted">Nothing saved yet.</p>
        ) : (
          <ul className="plain-list">
            {saved.map((j) => (
              <li key={j.id} className="saved-item">
                <div>
                  {j.url ? (
                    <a href={j.url} target="_blank" rel="noopener noreferrer">
                      {j.title}
                    </a>
                  ) : (
                    j.title
                  )}
                  <p className="small muted">
                    {[j.company, j.location].filter(Boolean).join(" · ")}
                    {!j.is_active && " · no longer listed"}
                  </p>
                </div>
                <button type="button" className="link" onClick={() => onUnsave(j.id)}>
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
