import { formatSalary, timeAgo } from "../lib/format.js";

export default function JobItem({ match, saved, rating, onToggleSave, onRate }) {
  if (rating === -1) {
    return (
      <li className="job job--dismissed">
        <span>{match.title}</span> won't be shown again.
      </li>
    );
  }

  const meta = [
    match.company,
    match.location,
    match.remote && "Remote",
    formatSalary(match.salary_min, match.salary_max),
    match.posted_at && `posted ${timeAgo(match.posted_at)}`,
  ].filter(Boolean);

  return (
    <li className="job">
      <div className="job__top">
        <h3 className="job__title">
          {match.url ? (
            <a href={match.url} target="_blank" rel="noopener noreferrer">
              {match.title}
            </a>
          ) : (
            match.title
          )}
        </h3>
        <span className="job__fit" title="How well it fits your profile">
          {match.fit_score}/10
        </span>
      </div>
      <p className="job__meta">{meta.join(" · ")}</p>
      <p className="job__why">{match.why_it_fits}</p>
      {match.missing_skills?.length > 0 && (
        <p className="job__missing">Worth brushing up: {match.missing_skills.join(", ")}</p>
      )}
      <div className="job__actions">
        <button type="button" className="link" onClick={() => onToggleSave(match.job_id)}>
          {saved ? "Saved" : "Save"}
        </button>
        <button
          type="button"
          className={`link${rating === 1 ? " link--on" : ""}`}
          onClick={() => onRate(match.job_id, 1)}
        >
          More like this
        </button>
        <button type="button" className="link" onClick={() => onRate(match.job_id, -1)}>
          Not interested
        </button>
        {match.url && (
          <a className="link link--strong" href={match.url} target="_blank" rel="noopener noreferrer">
            Apply ↗
          </a>
        )}
      </div>
    </li>
  );
}
