import { useCallback, useRef, useState } from "react";

const ACCEPT = ["image/png", "image/jpeg", "image/jpg", "image/webp"];

export default function UploadCard({ onFileSelect, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState(null);

  const handleFile = useCallback(
    (file) => {
      if (!file) return;
      if (!ACCEPT.includes(file.type) && !/\.(png|jpe?g|webp)$/i.test(file.name)) {
        setLocalError("Please upload a PNG, JPEG, or WebP chest X-ray image.");
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setLocalError("File is too large. Maximum size is 10 MB.");
        return;
      }
      setLocalError(null);
      onFileSelect(file);
    },
    [onFileSelect]
  );

  function onDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    handleFile(file);
  }

  return (
    <div className="space-y-3">
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onKeyDown={(e) => {
          if (disabled) return;
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragEnter={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragging(false);
        }}
        onDrop={onDrop}
        onClick={() => {
          if (!disabled) inputRef.current?.click();
        }}
        className={[
          "clinical-panel flex cursor-pointer flex-col items-center justify-center px-8 py-14 text-center transition",
          dragging ? "border-clinical-teal bg-cyan-50/40" : "hover:border-clinical-teal/50",
          disabled ? "cursor-wait opacity-60" : "",
        ].join(" ")}
      >
        <p className="clinical-label">Study upload</p>
        <p className="mt-3 font-display text-2xl text-clinical-ink md:text-3xl">
          Drag & drop a chest X-ray
        </p>
        <p className="mt-2 max-w-md text-sm text-clinical-muted">
          or click to browse. Accepts PNG / JPEG / WebP up to 10 MB.
        </p>
        <span className="mt-6 inline-flex border border-clinical-teal px-4 py-2 text-sm font-semibold text-clinical-teal">
          Select image
        </span>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="hidden"
          disabled={disabled}
          onChange={(e) => {
            const file = e.target.files?.[0];
            handleFile(file);
            e.target.value = "";
          }}
        />
      </div>
      {localError && <p className="text-sm text-clinical-danger">{localError}</p>}
    </div>
  );
}
