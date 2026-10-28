const form = document.getElementById("form");
const fieldsEl = document.getElementById("fields");
const submitBtn = document.getElementById("submit");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const noticeEl = document.getElementById("notice");

let schema = null;

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = String(text);
  return div.innerHTML;
}

function showNotice(title, message) {
  noticeEl.innerHTML = `<h2>${escapeHtml(title)}</h2><p>${escapeHtml(message)}</p>`;
  noticeEl.classList.remove("hidden");
}

function fieldHtml(f) {
  const id = `f_${f.name}`;
  if (f.type === "select") {
    const options = f.options
      .map((o) => `<option value="${escapeHtml(o)}"${o === f.default ? " selected" : ""}>${escapeHtml(o)}</option>`)
      .join("");
    return `<label for="${id}">${escapeHtml(f.label)}<select id="${id}" name="${f.name}">${options}</select></label>`;
  }
  const step = f.integer ? "1" : "any";
  const hint = `typical range ${f.min} to ${f.max}`;
  return `<label for="${id}">${escapeHtml(f.label)}
    <input id="${id}" name="${f.name}" type="number" step="${step}" value="${f.default}" required>
    <span class="hint">${hint}</span></label>`;
}

async function init() {
  try {
    const res = await fetch("/api/schema");
    const body = await res.json();
    if (!res.ok) {
      showNotice("Model not ready", body.error || "The server could not load the model.");
      return;
    }
    schema = body;
    document.getElementById("subtitle").textContent =
      `Estimate the chance of: ${schema.outcome_label.toLowerCase()}.`;
    fieldsEl.innerHTML = schema.features.map(fieldHtml).join("");
    form.classList.remove("hidden");

    const m = schema.metrics || {};
    const parts = [];
    if (schema.model_name) parts.push(schema.model_name);
    if (m.roc_auc !== undefined) parts.push(`ROC-AUC ${m.roc_auc} on held-out test data`);
    if (schema.dataset_name) parts.push(`trained on ${schema.dataset_name}`);
    document.getElementById("model-info").textContent = parts.join(" · ");
  } catch (e) {
    showNotice("Cannot reach the server", "Please check that the server is running and try again.");
  }
}

function collectPayload() {
  const payload = {};
  for (const f of schema.features) {
    const el = form.elements[f.name];
    payload[f.name] = f.type === "number" ? el.value : el.value;
  }
  return payload;
}

function showResult(r) {
  document.getElementById("result-title").textContent = r.outcome_label;
  document.getElementById("prob").textContent = r.probability.toFixed(1);
  document.getElementById("prob-caption").textContent = "% estimated chance";
  document.getElementById("bar-fill").style.width = Math.min(100, r.probability) + "%";

  const badge = document.getElementById("risk");
  badge.textContent = r.risk_level + " risk";
  badge.className = "badge " + r.risk_level.toLowerCase();

  document.getElementById("context").textContent =
    `Higher than about ${r.percentile}% of the held-out test patients. ` +
    `Average in the training data: ${r.base_rate}%.`;

  document.getElementById("warnings").innerHTML =
    r.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");

  const max = Math.max(...r.top_factors.map((f) => Math.abs(f.impact)), 0.1);
  document.getElementById("factors").innerHTML = r.top_factors.length
    ? r.top_factors
        .map((f) => {
          const width = (Math.abs(f.impact) / max) * 50;
          const cls = f.impact >= 0 ? "up" : "down";
          const sign = f.impact > 0 ? "+" : "";
          return `<li>
            <span>${escapeHtml(f.feature)}</span>
            <div class="factor-track"><div class="factor-fill ${cls}" style="width:${width}%"></div></div>
            <span class="factor-val">${sign}${f.impact}</span>
          </li>`;
        })
        .join("")
    : "<li>This patient looks like a typical patient, so no single input stands out.</li>";

  resultEl.classList.remove("hidden");
  resultEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.textContent = "";
  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "Prediction failed.");
    showResult(body);
  } catch (err) {
    errorEl.textContent = err.message;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict";
  }
});

init();
