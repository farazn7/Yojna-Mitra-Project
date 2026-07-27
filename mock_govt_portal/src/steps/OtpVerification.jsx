import { useState } from "react";

const CAPTCHA_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; // no 0/O/1/I — avoids ambiguity

function generateCaptcha(length = 5) {
  let code = "";
  for (let i = 0; i < length; i++) {
    code += CAPTCHA_CHARS[Math.floor(Math.random() * CAPTCHA_CHARS.length)];
  }
  return code;
}

function randomTransform() {
  const rotate = Math.floor(Math.random() * 30) - 15; // -15deg to 15deg
  const translateY = Math.floor(Math.random() * 7) - 3; // -3px to 3px
  return { transform: `rotate(${rotate}deg) translateY(${translateY}px)` };
}

export default function OtpVerification({ formData, setFormData, onNext, onBack }) {
  const [otpValue, setOtpValue] = useState("");
  const [captchaCode, setCaptchaCode] = useState(() => generateCaptcha());
  const [captchaInput, setCaptchaInput] = useState("");
  const [captchaError, setCaptchaError] = useState("");

  const handleVerifyOtp = () => {
    if (!otpValue.trim()) return;
    setFormData((f) => ({ ...f, otpVerified: true }));
  };

  const handleRefreshCaptcha = () => {
    setCaptchaCode(generateCaptcha());
    setCaptchaInput("");
    setCaptchaError("");
  };

  const handleVerifyCaptcha = () => {
    if (captchaInput.trim().toUpperCase() === captchaCode) {
      setFormData((f) => ({ ...f, captchaVerified: true }));
      setCaptchaError("");
    } else {
      // Regenerate the code but keep the error message visible — don't route through
      // handleRefreshCaptcha(), which clears captchaError and would erase it in the same render.
      setCaptchaError("Incorrect captcha, please try again.");
      setCaptchaCode(generateCaptcha());
      setCaptchaInput("");
    }
  };

  const canProceed =
    formData.otpVerified && formData.captchaVerified && formData.robotChecked;

  return (
    <div>
      <h2 className="wizard-heading">Step 3: Enter OTP</h2>
      <p className="hint-text">
        A One-Time Password (OTP) has been sent to your mobile number ending in 7890.
      </p>

      <div className="form-group">
        <label htmlFor="otp_input">Enter Verification Code</label>
        <input
          type="text"
          id="otp_input"
          name="otp"
          maxLength={6}
          value={otpValue}
          onChange={(e) => setOtpValue(e.target.value)}
          placeholder="6-digit code"
          disabled={formData.otpVerified}
        />
      </div>

      {!formData.otpVerified && (
        <button className="btn-primary" onClick={handleVerifyOtp} disabled={!otpValue.trim()}>
          Verify
        </button>
      )}
      {formData.otpVerified && <p className="status-text">✅ OTP Verified Successfully</p>}

      <hr style={{ margin: "24px 0", border: "none", borderTop: "1px solid #e1e5ea" }} />

      <h3 style={{ fontSize: 16, color: "#0b3d91", marginBottom: 10 }}>Enter Captcha Code</h3>

      <div className="form-group">
        <div className="captcha-box">
          <div className="noise-line" style={{ top: "35%", transform: "rotate(-4deg)" }} />
          <div className="noise-line" style={{ top: "65%", transform: "rotate(3deg)" }} />
          {captchaCode.split("").map((ch, idx) => (
            <span key={idx} className="captcha-char" style={randomTransform()}>
              {ch}
            </span>
          ))}
          <button
            type="button"
            className="captcha-refresh"
            onClick={handleRefreshCaptcha}
            title="Get a new code"
          >
            ↻ Refresh
          </button>
        </div>

        <div style={{ marginTop: 10 }}>
          <input
            type="text"
            id="captcha_input"
            name="captcha"
            maxLength={5}
            value={captchaInput}
            onChange={(e) => setCaptchaInput(e.target.value)}
            placeholder="Type the code shown above"
            disabled={formData.captchaVerified}
          />
        </div>

        {captchaError && <p className="captcha-error">{captchaError}</p>}

        {!formData.captchaVerified && (
          <div style={{ marginTop: 8 }}>
            <button className="btn-primary" onClick={handleVerifyCaptcha} disabled={!captchaInput.trim()}>
              Verify Captcha
            </button>
          </div>
        )}
        {formData.captchaVerified && <p className="status-text">✅ Captcha Verified</p>}
      </div>

      <label className="robot-check-row">
        <input
          type="checkbox"
          name="not_a_robot"
          checked={formData.robotChecked}
          onChange={(e) => setFormData((f) => ({ ...f, robotChecked: e.target.checked }))}
        />
        I'm not a robot
      </label>

      <div className="actions-row">
        <button className="btn-secondary" onClick={onBack}>
          Back
        </button>
        <button className="btn-primary" disabled={!canProceed} onClick={onNext}>
          Save & Proceed to Step 4
        </button>
      </div>
    </div>
  );
}
