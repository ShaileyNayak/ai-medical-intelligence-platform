import { Link, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import History from "./pages/History.jsx";

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-ink/10 bg-white/70 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <Link to="/" className="font-display text-xl tracking-tight text-ink">
            AI Medical Intelligence
          </Link>
          <nav className="flex gap-6 text-sm font-medium text-ink/70">
            <Link to="/" className="hover:text-accent">
              Dashboard
            </Link>
            <Link to="/history" className="hover:text-accent">
              History
            </Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-6 py-10">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
      <footer className="mx-auto max-w-5xl px-6 pb-10 text-xs text-ink/45">
        Educational prototype only — not a certified medical device or clinical diagnostic tool.
      </footer>
    </div>
  );
}
