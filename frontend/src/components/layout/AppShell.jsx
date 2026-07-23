import { useState } from "react";
import { Menu } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import Sidebar from "./Sidebar.jsx";

export default function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-clinical-line/80 bg-white/85 backdrop-blur">
        <div className="flex items-center justify-between gap-3 px-3 py-3 sm:px-4 sm:py-4 md:px-6">
          <div className="flex min-w-0 items-center gap-2 sm:gap-3">
            <button
              type="button"
              className="shrink-0 rounded-sm border border-clinical-line p-2 text-clinical-ink transition hover:border-clinical-teal hover:text-clinical-teal md:hidden"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation menu"
              aria-expanded={mobileOpen}
            >
              <Menu className="h-5 w-5" />
            </button>
            <NavLink to="/" className="min-w-0" onClick={() => setMobileOpen(false)}>
              <p className="truncate font-display text-base leading-tight text-clinical-ink sm:text-lg md:text-xl">
                AI Medical Intelligence
              </p>
              <p className="truncate text-[11px] text-clinical-muted sm:text-xs">
                Multi-disease imaging decision support demo
              </p>
            </NavLink>
          </div>

          <p className="hidden shrink-0 text-xs text-clinical-muted lg:block">
            Research prototype · clinician review required
          </p>
        </div>
      </header>

      <div className="flex min-h-[calc(100vh-4.25rem)]">
        <Sidebar open={mobileOpen} onClose={() => setMobileOpen(false)} />

        <main className="min-w-0 flex-1 overflow-x-hidden px-3 py-5 sm:px-4 sm:py-6 md:px-8 md:py-8">
          <div className="mx-auto w-full max-w-6xl">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
