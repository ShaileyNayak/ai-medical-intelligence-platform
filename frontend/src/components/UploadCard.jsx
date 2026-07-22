export default function UploadCard({ onFileSelect, disabled }) {
  return (
    <label className="flex cursor-pointer flex-col items-center justify-center border-2 border-dashed border-ink/20 bg-white/60 px-8 py-16 transition hover:border-accent">
      <span className="font-display text-2xl text-ink">Upload chest X-ray</span>
      <span className="mt-2 text-sm text-ink/60">PNG or JPEG, up to 10 MB</span>
      <input
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFileSelect(file);
        }}
      />
    </label>
  );
}
