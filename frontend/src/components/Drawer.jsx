import { useEffect } from "react";

export default function Drawer({ title, onClose, children }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="drawer" role="dialog" aria-modal="true" aria-label={title}>
      <button type="button" className="drawer__backdrop" aria-label="Close" onClick={onClose} />
      <div className="drawer__panel">
        <div className="drawer__head">
          <h2>{title}</h2>
          <button type="button" className="link" onClick={onClose}>
            Close
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
