import { useState } from "react";

function fakeAckNumber() {
  const digits = Math.floor(100000 + Math.random() * 900000);
  return `YM-2026-${digits}`;
}

export default function ReviewSubmit({ formData, setFormData, onBack }) {
  const [submitted, setSubmitted] = useState(false);
  const [ackNumber, setAckNumber] = useState("");

  const handleSubmitClick = () => {
    // Real native browser dialog — exercises browser_manager.py's HITL dialog
    // interceptor (page.on("dialog")) instead of a purely cosmetic in-app modal.
    const confirmed = window.confirm("Are you sure you want to submit this application?");
    if (confirmed) {
      setAckNumber(fakeAckNumber());
      setSubmitted(true);
    }
  };

  if (submitted) {
    return (
      <div className="success-box">
        <h2>✅ Application Submitted Successfully</h2>
        <p>Acknowledgement Number: <strong>{ackNumber}</strong></p>
        <p className="hint-text">Please save this number for tracking your application status.</p>
      </div>
    );
  }

  return (
    <div>
      <h2 className="wizard-heading">Step 4: Review and Submit Application</h2>
      <p className="hint-text">Please confirm all details before final submission.</p>

      <div className="summary-box">
        <div><strong>Full Name:</strong> {formData.fullName || "—"}</div>
        <div><strong>State:</strong> {formData.state || "—"}</div>
        <div><strong>District:</strong> {formData.district || "—"}</div>
        <div><strong>Income Certificate:</strong> {formData.fileName || "—"}</div>
        <div><strong>OTP Verified:</strong> {formData.otpVerified ? "Yes" : "No"}</div>
      </div>

      <label className="checkbox-row">
        <input
          type="checkbox"
          name="declaration"
          checked={formData.agreed}
          onChange={(e) => setFormData((f) => ({ ...f, agreed: e.target.checked }))}
        />
        I declare that the above information is correct to the best of my knowledge.
      </label>

      <div className="actions-row">
        <button className="btn-secondary" onClick={onBack}>
          Back
        </button>
        <button className="btn-primary" disabled={!formData.agreed} onClick={handleSubmitClick}>
          Submit Application
        </button>
      </div>
    </div>
  );
}
