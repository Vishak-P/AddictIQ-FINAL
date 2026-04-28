/**
 * script.js — AddictIQ Frontend Logic
 * Handles: form submission, prediction display, metrics chart
 */

"use strict";

// ──────────────────────────────────────────────
// UTILITIES
// ──────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const show = (el) => el?.classList.remove("d-none");
const hide = (el) => el?.classList.add("d-none");

function showToast(msg, duration = 4000) {
  const existing = $(".toast-msg");
  if (existing) existing.remove();
  const t = document.createElement("div");
  t.className = "toast-msg";
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

// ──────────────────────────────────────────────
// FORM VALIDATION
// ──────────────────────────────────────────────
function validateForm(data) {
  const rules = {
    Age:               { min: 1,  max: 100, label: "Age" },
    Daily_Usage_Hours: { min: 0,  max: 24,  label: "Daily Usage Hours" },
    Social_Media_Apps: { min: 0,  max: 100, label: "Social Media Apps" },
    Screen_Time:       { min: 0,  max: 24,  label: "Screen Time" },
    Sleep_Hours:       { min: 0,  max: 24,  label: "Sleep Hours" },
  };

  for (const [field, rule] of Object.entries(rules)) {
    const raw = data[field];
    if (raw === "" || raw === null || raw === undefined) {
      return `Please enter a value for "${rule.label}".`;
    }
    const val = parseFloat(raw);
    if (isNaN(val)) return `"${rule.label}" must be a number.`;
    if (val < rule.min || val > rule.max) {
      return `"${rule.label}" must be between ${rule.min} and ${rule.max}.`;
    }
  }
  return null; // valid
}

// ──────────────────────────────────────────────
// RESULT RENDERING
// ──────────────────────────────────────────────
function renderResult(data) {
  const panel   = $("#result-panel");
  const inner   = $("#result-inner");
  const addicted = data.prediction === "Addicted";

  const cls     = addicted ? "result-addicted" : "result-not-addicted";
  const vCls    = addicted ? "addicted"          : "ok";
  const fillCls = addicted ? "fill-danger"        : "fill-safe";
  const msg     = addicted
    ? "Your usage patterns suggest a high risk of social media addiction."
    : "Your digital habits appear healthy. Keep maintaining good balance!";

  inner.innerHTML = `
    <div class="${cls}">
      <div class="result-verdict ${vCls}">${data.prediction}</div>
      <div class="result-sub">${msg}</div>

      <div class="prob-bar-wrap">
        <div class="prob-bar-label">
          <span>Addiction Probability</span>
          <span>${data.addicted_prob}%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill ${fillCls}" style="width:0%"
               data-target="${data.addicted_prob}"></div>
        </div>
      </div>

      <div class="prob-bar-wrap">
        <div class="prob-bar-label">
          <span>Confidence</span>
          <span>${data.confidence}%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill fill-safe" style="width:0%"
               data-target="${data.confidence}"></div>
        </div>
      </div>
    </div>
  `;

  show(panel);

  // Animate bars after render
  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-bar-fill").forEach((bar) => {
      const target = bar.dataset.target;
      setTimeout(() => { bar.style.width = target + "%"; }, 50);
    });
  });

  // Smooth scroll to result
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ──────────────────────────────────────────────
// FORM SUBMISSION
// ──────────────────────────────────────────────
$("#predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const form = e.target;
  const btn  = $("#predict-btn");

  // Gather data
  const data = {
    Age:               form.Age.value.trim(),
    Daily_Usage_Hours: form.Daily_Usage_Hours.value.trim(),
    Social_Media_Apps: form.Social_Media_Apps.value.trim(),
    Screen_Time:       form.Screen_Time.value.trim(),
    Sleep_Hours:       form.Sleep_Hours.value.trim(),
  };

  // Client-side validation
  const error = validateForm(data);
  if (error) {
    showToast(error);
    return;
  }

  // UI: loading state
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Analysing...`;
  document.querySelectorAll(".field-input").forEach((i) => i.classList.remove("is-invalid"));

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(data),
    });
    const json = await res.json();

    if (!res.ok || !json.success) {
      throw new Error(json.error || "Prediction failed.");
    }

    renderResult(json);
    loadHistory();

  } catch (err) {
    showToast(err.message);
    hide($("#result-panel"));
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-label">Analyse Behaviour</span><span class="btn-arrow">→</span>`;
  }
});

// ──────────────────────────────────────────────
// METRICS CHART
// ──────────────────────────────────────────────
let accuracyChart = null;

async function loadMetrics() {
  try {
    const res  = await fetch("/metrics");
    const json = await res.json();

    if (!res.ok || json.error) {
      console.warn("Metrics not available:", json.error);
      return;
    }

    const models   = Object.keys(json.model_results);
    const accs     = models.map((m) => json.model_results[m].accuracy * 100);
    const cvMeans  = models.map((m) => json.model_results[m].cv_mean * 100);
    const bestName = json.best_model;

    // ── Chart ──────────────────────────────────
    const ctx = document.getElementById("accuracy-chart").getContext("2d");

    if (accuracyChart) accuracyChart.destroy();

    Chart.defaults.color = "#475569";
    Chart.defaults.font.family = "'DM Sans', sans-serif";

    accuracyChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "Test Accuracy (%)",
            data: accs,
            backgroundColor: models.map((m) =>
              m === bestName
                ? "rgba(14,165,100,0.8)"
                : "rgba(2,132,199,0.5)"
            ),
            borderColor: models.map((m) =>
              m === bestName ? "#0ea564" : "#0284c7"
            ),
            borderWidth: 2,
            borderRadius: 6,
          },
          {
            label: "CV Mean Accuracy (%)",
            data: cvMeans,
            backgroundColor: "rgba(217,119,6,0.15)",
            borderColor: "#d97706",
            borderWidth: 2,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: {
              boxWidth: 12,
              padding: 16,
              font: { size: 11 },
            },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`,
            },
          },
        },
        scales: {
          y: {
            min: 60,
            max: 100,
            grid: { color: "rgba(0,0,0,0.05)" },
            ticks: {
              callback: (v) => v + "%",
              font: { size: 11 },
            },
          },
          x: {
            grid: { display: false },
            ticks: { font: { size: 11 } },
          },
        },
      },
    });

    // ── Model Table ────────────────────────────
    const table = document.createElement("table");
    table.className = "model-table";
    table.innerHTML = `
      <thead>
        <tr>
          <th>Model</th>
          <th>Accuracy</th>
          <th>CV Mean</th>
          <th>CV Std</th>
        </tr>
      </thead>
      <tbody>
        ${models.map((m) => {
          const r   = json.model_results[m];
          const isBest = m === bestName;
          const accPct = (r.accuracy * 100).toFixed(1);
          const pillCls = r.accuracy >= 0.9 ? "acc-high"
                        : r.accuracy >= 0.8 ? "acc-mid"
                        : "acc-low";
          return `
            <tr class="${isBest ? "highlight" : ""}">
              <td>${isBest ? "★ " : ""}${m}</td>
              <td><span class="acc-pill ${pillCls}">${accPct}%</span></td>
              <td>${(r.cv_mean * 100).toFixed(1)}%</td>
              <td>±${(r.cv_std * 100).toFixed(1)}%</td>
            </tr>`;
        }).join("")}
      </tbody>
    `;
    const container = $("#model-table");
    container.innerHTML = "";
    container.appendChild(table);

    // ── Best Model Badge ───────────────────────
    const badge = $("#best-badge");
    const badgeName = $("#best-badge-name");
    if (badge && badgeName) {
      badgeName.textContent = bestName;
      badge.style.display = "flex";
    }

    // ── Best Accuracy Stat ─────────────────────
    const bestAcc = Math.max(...accs);
    const statEl  = $("#stat-best-acc");
    if (statEl) statEl.textContent = bestAcc.toFixed(1) + "%";

  } catch (err) {
    console.error("Failed to load metrics:", err);
  }
}

// ──────────────────────────────────────────────
// HISTORY
// ──────────────────────────────────────────────
async function loadHistory() {
  const tbody = document.getElementById("history-body");
  try {
    const res  = await fetch("/history");
    const json = await res.json();
    if (!json.success || !json.history.length) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:var(--text-dim)">No predictions yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = json.history.map((r, i) => `
      <tr>
        <td>${i + 1}</td>
        <td>${r.age}</td>
        <td>${r.daily_usage_hours}</td>
        <td>${r.social_media_apps}</td>
        <td>${r.screen_time}</td>
        <td>${r.sleep_hours}</td>
        <td><span class="acc-pill ${r.prediction === 'Addicted' ? 'pill-danger' : 'acc-high'}">${r.prediction}</span></td>
        <td>${r.confidence}%</td>
        <td>${r.created_at}</td>
      </tr>`).join("");
  } catch (err) {
    console.error("Failed to load history:", err);
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:var(--text-dim)">Could not load history.</td></tr>`;
  }
}

// ──────────────────────────────────────────────
// INIT
// ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadMetrics();
  loadHistory();

  // Hamburger toggle
  const hamburger  = document.getElementById("hamburger");
  const mobileNav  = document.getElementById("mobile-nav");
  if (hamburger) {
    hamburger.addEventListener("click", () => {
      hamburger.classList.toggle("open");
      mobileNav.classList.toggle("open");
    });
  }
});

function closeMobileNav() {
  document.getElementById("hamburger")?.classList.remove("open");
  document.getElementById("mobile-nav")?.classList.remove("open");
}
