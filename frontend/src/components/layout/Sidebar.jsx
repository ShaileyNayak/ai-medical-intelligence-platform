import { useEffect } from "react";
import { NavLink } from "react-router-dom";
import {
  BarChart3,
  History,
  LayoutDashboard,
  ScanLine,
  X,
} from "lucide-react";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/analysis", label: "New Analysis", icon: ScanLine },
  { to: "/history", label: "History", icon: History },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
];

export default function Sidebar({ open, onClose }) {
  useEffect(() => {
    if (!open) return undefined;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  return (
    <>
      <div
        className={[
          "fixed inset-0 z-40 bg-clinical-ink/40 transition md:hidden",
          open ? "opacity-100" : "pointer-events-none opacity-0",
        ].join(" ")}
        onClick={onClose}
        aria-hidden={!open}
      />

      <aside
        className={[
          "fixed inset-y-0 left-0 z-50 flex w-[min(18rem,88vw)] flex-col border-r border-clinical-line bg-white shadow-panel transition-transform duration-200 md:static md:z-0 md:w-64 md:translate-x-0 md:shadow-none",
          open ? "translate-x-0" : "-translate-x-full md:translate-x-0",
        ].join(" ")}
        aria-label="Primary"
      >
        <div className="flex items-center justify-between border-b border-clinical-line px-4 py-4 md:hidden">
          <p className="font-display text-base text-clinical-ink">Menu</p>
          <button
            type="button"
            className="rounded-sm p-1.5 text-clinical-muted hover:bg-clinical-soft hover:text-clinical-ink"
            onClick={onClose}
            aria-label="Close menu"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="hidden border-b border-clinical-line px-5 py-5 md:block">
          <p className="clinical-label">Navigate</p>
          <p className="mt-1 font-display text-lg text-clinical-ink">Workspace</p>
        </div>

        <nav className="flex flex-1 flex-col gap-1 overflow-y-auto p-3">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={onClose}
              className={({ isActive }) =>
                [
                  "flex items-center gap-3 rounded-sm px-3 py-2.5 text-sm font-medium transition",
                  isActive
                    ? "bg-clinical-teal/10 text-clinical-teal"
                    : "text-clinical-muted hover:bg-clinical-soft hover:text-clinical-ink",
                ].join(" ")
              }
            >
              <Icon className="h-4 w-4 shrink-0" strokeWidth={1.75} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <p className="border-t border-clinical-line px-5 py-4 text-[11px] leading-relaxed text-clinical-muted">
          Educational prototype — not a medical device.
        </p>
      </aside>
    </>
  );
}
