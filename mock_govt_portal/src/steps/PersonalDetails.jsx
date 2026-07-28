import { useState } from "react";
import statesDistricts from "../data/states_districts.json";

const STATES = Object.keys(statesDistricts);

/**
 * The District <select> deliberately does not exist in the DOM at all until the
 * async delay resolves — repopulating an existing select's options wouldn't change
 * the page's input/select/textarea element count, and automation_graph.py's
 * execute_node cascade-detection relies on exactly that count changing.
 */
export default function PersonalDetails({ formData, setFormData, onNext }) {
  const [districtLoading, setDistrictLoading] = useState(false);
  const [districtOptions, setDistrictOptions] = useState([]);

  const handleStateChange = (e) => {
    const newState = e.target.value;
    setFormData((f) => ({ ...f, state: newState, district: "" }));
    setDistrictOptions([]);

    if (!newState) return;

    setDistrictLoading(true);
    setTimeout(() => {
      setDistrictOptions(statesDistricts[newState] || []);
      setDistrictLoading(false);
    }, 500);
  };

  const handleDistrictChange = (e) => {
    setFormData((f) => ({ ...f, district: e.target.value }));
  };

  const canProceed = formData.fullName.trim() && formData.state && formData.district;

  return (
    <div>
      <h2 className="wizard-heading">Step 1: Citizen Personal Details</h2>

      <div className="form-group">
        <label htmlFor="applicant_name">Full Name (as per Aadhaar)</label>
        <input
          type="text"
          id="applicant_name"
          name="full_name"
          value={formData.fullName}
          onChange={(e) => setFormData((f) => ({ ...f, fullName: e.target.value }))}
          placeholder="Enter your full name"
          required
        />
      </div>

      <div className="form-group">
        <label htmlFor="state_select">Resident State</label>
        <select id="state_select" name="state" value={formData.state} onChange={handleStateChange}>
          <option value="">-- Select State --</option>
          {STATES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Resident District</label>
        {!formData.state && <p className="hint-text">Select a state first.</p>}
        {formData.state && districtLoading && <p className="hint-text">Loading districts…</p>}
        {formData.state && !districtLoading && districtOptions.length > 0 && (
          <select id="district_select" name="district" value={formData.district} onChange={handleDistrictChange}>
            <option value="">-- Select District --</option>
            {districtOptions.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="actions-row">
        <span />
        <button className="btn-primary" disabled={!canProceed} onClick={onNext}>
          Save & Proceed to Step 2
        </button>
      </div>
    </div>
  );
}
