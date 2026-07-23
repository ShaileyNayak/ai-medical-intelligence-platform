export default function StatCard({ label, value, hint, icon: Icon }) {
  return (
    <div className="clinical-panel clinical-panel-hover p-5">
      <div className="flex items-start justify-between gap-3">
        <p className="clinical-label">{label}</p>
        {Icon && (
          <span className="rounded-sm bg-clinical-soft p-2 text-clinical-teal">
            <Icon className="h-4 w-4" strokeWidth={1.75} />
          </span>
        )}
      </div>
      <p className="mt-3 font-display text-3xl text-clinical-ink">{value}</p>
      {hint && <p className="mt-1 text-xs text-clinical-muted">{hint}</p>}
    </div>
  );
}
