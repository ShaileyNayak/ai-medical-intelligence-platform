import { useCallback, useRef, useState } from "react";
import { SCAN_TYPE_OPTIONS } from "../constants/scanTypes.js";

const ACCEPT = ["image/png", "image/jpeg", "image/jpg", "image/webp"];

/**
 * Upload form with scan-type selector.
 * Calls ``onSubmit({ file, scanType })`` when an image is chosen.
 */
export default function UploadForm({
  onSubmit,
  onFileSelect,
  disabled,
  scanType: controlledScanType,
  onScanTypeChange,
  defaultScanType = "chest_xray",
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState(null);
  const [internalScanType, setInternalScanType] = useState(defaultScanType);

  const scanType = controlledScanType ?? internalScanType;

  function setScanType(next) {
    if (controlledScanType === undefined) {
      setInternalScanType(next);
    }
    onScanTypeChange?.(next);
  }

  const emit = useCallback(
    (file) => {
      if (onSubmit) {
        onSubmit({ file, scanType });
      } else if (onFileSelect) {
        onFileSelect(file, scanType);
      }
    },
    [onSubmit, onFileSelect, scanType]
  );

  const handleFile = useCallback(
    (file) => {
      if (!file) return;
      if (!ACCEPT.includes(file.type) && !/\.(png|jpe?g|webp)$/i.test(file.name)) {
        setLocalError("Please upload a PNG, JPEG, or WebP image.");
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setLocalError("File is too large. Maximum size is 10 MB.");
        return;
      }
      setLocalError(null);
      emit(file);
    },
    [emit]
  );

  function onDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    handleFile(file);
  }

  const selectedOption = SCAN_TYPE_OPTIONS.find((o) => o.value === scanType);

  return (
    <div className="space-y-4">
      <div className="clinical-panel p-5 md:p-6">
        <p className="clinical-label">Scan type</p>
        <p className="mt-1 text-sm text-clinical-muted">
          Choose the imaging module before uploading. This is sent as{" "}
          <code className="text-clinical-ink">scan_type</code> with the request.
        </p>

        <div
          className="mt-4 hidden flex-wrap gap-2 md:flex"
          role="radiogroup"
          aria-label="Scan type"
        >
          {SCAN_TYPE_OPTIONS.map((opt) => {
            const active = scanType === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                role="radio"
                aria-checked={active}
                disabled={disabled}
                onClick={() => setScanType(opt.value)}
                className={[
                  "border px-4 py-2 text-sm font-semibold transition",
                  active
                    ? "border-clinical-teal bg-clinical-teal text-white"
                    : "border-clinical-line bg-white text-clinical-ink hover:border-clinical-teal/60",
                  disabled ? "opacity-60" : "",
                ].join(" ")}
              >
                {opt.label}
              </button>
            );
          })}
        </div>

        <label className="mt-4 block md:hidden">
          <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.14em] text-clinical-muted">
            Select modality
          </span>
          <select
            className="w-full border border-clinical-line bg-white px-3 py-2 text-sm text-clinical-ink"
            value={scanType}
            disabled={disabled}
            onChange={(e) => setScanType(e.target.value)}
          >
            {SCAN_TYPE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
      </div>

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
          "clinical-panel flex cursor-pointer flex-col items-center justify-center px-5 py-10 text-center transition sm:px-8 sm:py-14",
          dragging ? "border-clinical-teal bg-cyan-50/40" : "hover:border-clinical-teal/50",
          disabled ? "cursor-wait opacity-60" : "",
        ].join(" ")}
      >
        <p className="clinical-label">Study upload</p>
        <p className="mt-3 font-display text-xl text-clinical-ink sm:text-2xl md:text-3xl">
          Drag & drop a {selectedOption?.label?.toLowerCase() || "scan"}
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
