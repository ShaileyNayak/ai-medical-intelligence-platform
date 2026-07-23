import { NavLink, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import History from "./pages/History.jsx";

const linkClass = ({ isActive }) =>
  [
    "text-sm font-medium transition",
    isActive ? "text-clinical-teal" : "text-clinical-muted hover:text-clinical-ink",
  ].join(" ");

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-clinical-line/80 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-6 px-6 py-4">
          <NavLink to="/" className="min-w-0">
            <p className="font-display text-lg leading-tight text-clinical-ink md:text-xl">
              AI Medical Intelligence
            </p>
            <p className="text-xs text-clinical-muted">Multi-disease imaging decision support demo</p>
          </NavLink>
          <nav className="flex shrink-0 gap-6">
            <NavLink to="/" end className={linkClass}>
              Analysis
            </NavLink>
            <NavLink to="/history" className={linkClass}>
              History
            </NavLink>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10 md:py-12">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>

      <footer className="mx-auto max-w-5xl px-6 pb-10">
        <p className="border-t border-clinical-line pt-6 text-xs leading-relaxed text-clinical-muted">
          Research / educational prototype only. Not a certified medical device. Outputs must
          be reviewed by a licensed clinician and must not be used as the sole basis for
          diagnosis or treatment.
        </p>
      </footer>
    </div>
  );
}
