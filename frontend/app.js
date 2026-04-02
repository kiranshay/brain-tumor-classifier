// ===================================
// Configuration
// ===================================

// Change this to your Render URL after deployment
const API_URL = "http://localhost:8000";

const TUMOR_INFO = {
  glioma:
    "Gliomas are tumors that arise from glial cells in the brain or spine. They are the most common type of primary brain tumor and can vary significantly in aggressiveness, from low-grade (slow-growing) to high-grade (fast-growing) forms like glioblastoma.",
  meningioma:
    "Meningiomas originate in the meninges, the membranes surrounding the brain and spinal cord. They are usually slow-growing and often benign, making up about 30% of all brain tumors. Many are discovered incidentally and may only require monitoring.",
  pituitary:
    "Pituitary tumors develop in the pituitary gland at the base of the brain. Most are benign adenomas that can affect hormone production. Depending on their size and hormone activity, they may cause vision problems or hormonal imbalances.",
  no_tumor:
    "No tumor detected in this MRI scan. The brain tissue appears within normal parameters based on the model's classification. Note: this tool is for educational purposes only and should not be used as a medical diagnostic.",
};

const CLASS_ORDER = ["glioma", "meningioma", "pituitary", "no_tumor"];

// ===================================
// State
// ===================================

let historyOffset = 0;
const HISTORY_LIMIT = 20;
let serverReady = false;

// ===================================
// DOM References
// ===================================

const views = document.querySelectorAll(".view");
const navTabs = document.querySelectorAll(".nav-tab");
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const uploadSection = document.getElementById("upload-section");
const resultSection = document.getElementById("result-section");
const loadingSection = document.getElementById("loading-section");
const classifyAgainBtn = document.getElementById("classify-again-btn");
const serverBanner = document.getElementById("server-banner");
const historyGrid = document.getElementById("history-grid");
const historyEmpty = document.getElementById("history-empty");
const loadMoreBtn = document.getElementById("load-more-btn");
const logoLink = document.getElementById("logo-link");

// ===================================
// Initialize
// ===================================

lucide.createIcons();
checkServer();

// ===================================
// Navigation
// ===================================

navTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const viewName = tab.dataset.view;
    switchView(viewName);
  });
});

logoLink.addEventListener("click", (e) => {
  e.preventDefault();
  switchView("classify");
});

function switchView(viewName) {
  navTabs.forEach((t) => t.classList.remove("active"));
  views.forEach((v) => v.classList.remove("active"));

  const activeTab = document.querySelector(`[data-view="${viewName}"]`);
  const activeView = document.getElementById(`view-${viewName}`);
  if (activeTab) activeTab.classList.add("active");
  if (activeView) activeView.classList.add("active");

  if (viewName === "history") loadHistory();
  if (viewName === "stats") loadStats();
}

// ===================================
// Server Health Check
// ===================================

async function checkServer() {
  serverBanner.classList.add("show");
  const maxRetries = 20;

  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        serverReady = true;
        serverBanner.classList.remove("show");
        lucide.createIcons();
        return;
      }
    } catch {
      // Server not ready yet
    }
    await new Promise((r) => setTimeout(r, 3000));
  }

  serverBanner.querySelector("span").textContent =
    "Could not reach the server. Please try refreshing the page.";
}

// ===================================
// File Upload
// ===================================

uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    classifyImage(file);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) {
    classifyImage(fileInput.files[0]);
  }
});

// Sample buttons
document.querySelectorAll(".btn-sample").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const sampleName = btn.dataset.sample;
    try {
      const res = await fetch(`assets/sample_mris/${sampleName}.jpg`);
      const blob = await res.blob();
      const file = new File([blob], `${sampleName}_sample.jpg`, { type: "image/jpeg" });
      classifyImage(file);
    } catch {
      alert("Sample image not found. Please add sample MRI images to assets/sample_mris/.");
    }
  });
});

// Classify again
classifyAgainBtn.addEventListener("click", resetClassify);

function resetClassify() {
  uploadSection.classList.remove("hidden");
  resultSection.classList.add("hidden");
  loadingSection.classList.add("hidden");
  fileInput.value = "";
}

// ===================================
// Classification
// ===================================

async function classifyImage(file) {
  uploadSection.classList.add("hidden");
  resultSection.classList.add("hidden");
  loadingSection.classList.remove("hidden");
  lucide.createIcons();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || `Server error: ${res.status}`);
    }

    const result = await res.json();
    displayResult(result, file);
  } catch (err) {
    loadingSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");
    alert(`Classification failed: ${err.message}`);
  }
}

function displayResult(result, file) {
  loadingSection.classList.add("hidden");
  resultSection.classList.remove("hidden");

  // Image preview
  const imgEl = document.getElementById("result-image");
  imgEl.src = URL.createObjectURL(file);

  // Badge
  const badge = document.getElementById("result-badge");
  badge.textContent = formatClassName(result.predicted_class);
  badge.className = `result-badge ${result.predicted_class}`;

  // Confidence
  document.getElementById("result-confidence").textContent =
    `${(result.confidence * 100).toFixed(1)}%`;

  // Confidence bars
  const barsContainer = document.getElementById("confidence-bars");
  barsContainer.innerHTML = CLASS_ORDER.map((cls) => {
    const pct = ((result.all_confidences[cls] || 0) * 100).toFixed(1);
    return `
      <div class="confidence-bar-row">
        <span class="confidence-label">${formatClassName(cls)}</span>
        <div class="confidence-track">
          <div class="confidence-fill ${cls}" style="width: ${pct}%"></div>
        </div>
        <span class="confidence-pct">${pct}%</span>
      </div>
    `;
  }).join("");

  // Meta
  document.getElementById("result-time").textContent =
    `${result.inference_time_ms.toFixed(0)}ms inference`;
  document.getElementById("result-filename").textContent =
    result.original_filename || file.name;

  // Info blurb
  document.getElementById("result-info").textContent =
    TUMOR_INFO[result.predicted_class] || "";

  lucide.createIcons();
}

// ===================================
// History
// ===================================

async function loadHistory(append = false) {
  if (!append) {
    historyOffset = 0;
    historyGrid.innerHTML = "";
  }

  try {
    const res = await fetch(
      `${API_URL}/predictions?limit=${HISTORY_LIMIT}&offset=${historyOffset}`
    );
    if (!res.ok) throw new Error("Failed to load history");

    const items = await res.json();

    if (items.length === 0 && historyOffset === 0) {
      historyEmpty.classList.remove("hidden");
      loadMoreBtn.classList.add("hidden");
      lucide.createIcons();
      return;
    }

    historyEmpty.classList.add("hidden");

    items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "history-card";

      const thumbSrc = item.thumbnail_base64
        ? `data:image/jpeg;base64,${item.thumbnail_base64}`
        : "";

      const date = new Date(item.created_at);
      const timeStr = date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      });

      card.innerHTML = `
        <div class="history-card-top">
          ${thumbSrc ? `<img class="history-thumb" src="${thumbSrc}" alt="MRI thumbnail">` : ""}
          <div class="history-card-info">
            <span class="history-class ${item.predicted_class}">${formatClassName(item.predicted_class)}</span>
            <div class="history-confidence">${(item.confidence * 100).toFixed(1)}%</div>
            <div class="history-meta">${timeStr} &middot; ${item.inference_time_ms.toFixed(0)}ms</div>
          </div>
        </div>
      `;

      historyGrid.appendChild(card);
    });

    historyOffset += items.length;

    if (items.length < HISTORY_LIMIT) {
      loadMoreBtn.classList.add("hidden");
    } else {
      loadMoreBtn.classList.remove("hidden");
    }

    lucide.createIcons();
  } catch {
    historyEmpty.classList.remove("hidden");
    historyEmpty.querySelector("p").textContent =
      "Could not load history. Is the server running?";
    lucide.createIcons();
  }
}

loadMoreBtn.addEventListener("click", () => loadHistory(true));

// ===================================
// Stats
// ===================================

async function loadStats() {
  try {
    const res = await fetch(`${API_URL}/stats`);
    if (!res.ok) throw new Error("Failed to load stats");

    const stats = await res.json();

    document.getElementById("stat-total").textContent = stats.total_predictions;
    document.getElementById("stat-confidence").textContent =
      `${(stats.avg_confidence * 100).toFixed(1)}%`;
    document.getElementById("stat-speed").textContent =
      `${stats.avg_inference_time_ms.toFixed(0)}ms`;

    // Distribution bars
    const distContainer = document.getElementById("distribution-bars");
    const maxCount = Math.max(...Object.values(stats.class_distribution), 1);

    distContainer.innerHTML = CLASS_ORDER.map((cls) => {
      const count = stats.class_distribution[cls] || 0;
      const pct = (count / maxCount) * 100;
      return `
        <div class="dist-row">
          <span class="dist-label">${formatClassName(cls)}</span>
          <div class="dist-track">
            <div class="dist-fill ${cls}" style="width: ${pct}%"></div>
          </div>
          <span class="dist-count">${count}</span>
        </div>
      `;
    }).join("");
  } catch {
    // Stats will show zeros if server is down
  }
}

// ===================================
// Helpers
// ===================================

function formatClassName(cls) {
  if (cls === "no_tumor") return "No Tumor";
  return cls.charAt(0).toUpperCase() + cls.slice(1);
}
