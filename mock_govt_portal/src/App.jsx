import { useState } from "react";
import PersonalDetails from "./steps/PersonalDetails.jsx";
import DocumentUpload from "./steps/DocumentUpload.jsx";
import OtpVerification from "./steps/OtpVerification.jsx";
import ReviewSubmit from "./steps/ReviewSubmit.jsx";

const STEP_LABELS = [
  "Personal Details",
  "Document Upload",
  "OTP Verification",
  "Review & Submit"
];

export default function App() {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    fullName: "",
    state: "",
    district: "",
    fileName: "",
    otpVerified: false,
    captchaVerified: false,
    robotChecked: false,
    agreed: false
  });

  const goNext = () => setStep((s) => Math.min(s + 1, 4));
  const goBack = () => setStep((s) => Math.max(s - 1, 1));

  const stepProps = { formData, setFormData, onNext: goNext, onBack: goBack };

  return (
    <div>
      <header className="gov-header">
        Digital Seva Portal — Scheme Application
        <small>Government of India (Mock Demo Portal — not a real government site)</small>
      </header>
      <div className="wizard-shell">
        <div className="progress-bar">
          {STEP_LABELS.map((label, idx) => (
            <span key={label} className={idx + 1 === step ? "active" : ""}>
              {idx + 1}. {label}
            </span>
          ))}
        </div>
        <div className="wizard-step" data-step={step}>
          {step === 1 && <PersonalDetails {...stepProps} />}
          {step === 2 && <DocumentUpload {...stepProps} />}
          {step === 3 && <OtpVerification {...stepProps} />}
          {step === 4 && <ReviewSubmit {...stepProps} />}
        </div>
      </div>
    </div>
  );
}
