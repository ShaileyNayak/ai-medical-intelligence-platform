export default function ReportPanel({ report }) {
  if (!report) return null;

  return (
    <section className="clinical-panel p-6 md:p-8">
      <p className="clinical-label">Assistive report</p>
      <h2 className="mt-2 font-display text-2xl text-clinical-ink">
        Plain-language finding summary
      </h2>
      <div className="mt-5 whitespace-pre-wrap text-[15px] leading-relaxed text-clinical-ink/90">
        {report}
      </div>
      <p className="mt-6 border-t border-clinical-line pt-4 text-xs leading-relaxed text-clinical-muted">
        This narrative is AI-generated for education and triage support only. It is not a
        medical diagnosis and must be confirmed by a licensed clinician.
      </p>
    </section>
  );
}
