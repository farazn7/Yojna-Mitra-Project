import { useState } from "react";

export default function DocumentUpload({ formData, setFormData, onNext, onBack }) {
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    setUploading(true);
    setFormData((f) => ({ ...f, fileName: "" }));

    // Simulates a real portal's background upload before the step can advance.
    setTimeout(() => {
      setUploading(false);
      setFormData((f) => ({ ...f, fileName: file.name }));
    }, 900);
  };

  const canProceed = Boolean(formData.fileName) && !uploading;

  return (
    <div>
      <h2 className="wizard-heading">Step 2: Income Certificate Verification</h2>

      <div className="form-group">
        <span>Income Certificate Verification:</span>
        <br />
        <label htmlFor="income_cert_input" className="upload-btn">
          📎 Attach Income Certificate (.jpg/.pdf)
        </label>
        <input
          type="file"
          id="income_cert_input"
          name="income_certificate"
          style={{ display: "none" }}
          onChange={handleFileChange}
        />

        {uploading && <p className="status-text">Uploading…</p>}
        {!uploading && formData.fileName && (
          <p className="status-text">✅ Upload Complete: {formData.fileName}</p>
        )}
      </div>

      <div className="actions-row">
        <button className="btn-secondary" onClick={onBack}>
          Back
        </button>
        <button className="btn-primary" disabled={!canProceed} onClick={onNext}>
          Save & Proceed to Step 3
        </button>
      </div>
    </div>
  );
}
