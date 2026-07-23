import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/layout/AppShell.jsx";
import Overview from "./pages/Overview.jsx";
import Analysis from "./pages/Analysis.jsx";
import History from "./pages/History.jsx";
import Analytics from "./pages/Analytics.jsx";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Overview />} />
        <Route path="analysis" element={<Analysis />} />
        <Route path="analyze" element={<Navigate to="/analysis" replace />} />
        <Route path="history" element={<History />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
