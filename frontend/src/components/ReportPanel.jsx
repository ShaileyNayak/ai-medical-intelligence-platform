export default function ReportPanel({ report }) {
  if (!report) return null;

  return (
    <section className="space-y-3 border-t border-ink/10 pt-6">
      <h2 className="font-display text-2xl text-ink">LLM Assistive Report</h2>
      <div className="whitespace-pre-wrap text-ink/80 leading-relaxed">{report}</div>
    </section>
  );
}
