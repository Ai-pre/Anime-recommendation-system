const selected = new Map();

const $ = (selector) => document.querySelector(selector);
const searchInput = $("#searchInput");
const searchButton = $("#searchButton");
const searchResults = $("#searchResults");
const selectedList = $("#selectedList");
const recommendButton = $("#recommendButton");
const preview = $("#preview");
const statusEl = $("#status");

function renderSelected() {
  selectedList.innerHTML = "";
  selectedList.classList.toggle("empty", selected.size === 0);
  if (selected.size === 0) {
    selectedList.textContent = "No anime selected yet.";
    return;
  }
  for (const item of selected.values()) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = item.title;
    const remove = document.createElement("button");
    remove.textContent = "x";
    remove.addEventListener("click", () => {
      selected.delete(item.id);
      renderSelected();
    });
    chip.appendChild(remove);
    selectedList.appendChild(chip);
  }
}

function animeMeta(item) {
  return `${item.type} - Score ${item.score ?? "?"} - ${item.genres}`;
}

async function searchAnime() {
  const query = searchInput.value.trim();
  searchResults.innerHTML = "";
  if (!query) return;
  statusEl.textContent = "Searching...";
  const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
  const data = await response.json();
  statusEl.textContent = "";
  for (const item of data.results) {
    const button = document.createElement("button");
    button.className = "result-item";
    button.innerHTML = `<strong>${item.title}</strong><span>${animeMeta(item)}</span>`;
    button.addEventListener("click", () => {
      selected.set(item.id, item);
      renderSelected();
    });
    searchResults.appendChild(button);
  }
}

function renderRecommendations(items, shareUrl) {
  preview.innerHTML = "";
  const linkCard = document.createElement("article");
  linkCard.className = "card";
  linkCard.innerHTML = `<div class="rank">Go</div><h3>Share page created</h3><p>${location.origin}${shareUrl}</p><div class="meta"><span>View counter enabled</span></div>`;
  linkCard.addEventListener("click", () => {
    location.href = shareUrl;
  });
  preview.appendChild(linkCard);

  items.forEach((item, idx) => {
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      <div class="rank">#${idx + 1}</div>
      <h3>${item.title}</h3>
      <p>${item.genres}</p>
      <div class="meta">
        <span>${item.type}</span>
        <span>Score ${item.score ?? "?"}</span>
        <span>Match ${(item.taste_score * 100).toFixed(1)}%</span>
      </div>
    `;
    preview.appendChild(card);
  });
}

async function makeRecommendation() {
  if (selected.size === 0) {
    statusEl.textContent = "Select at least one anime first.";
    return;
  }
  statusEl.textContent = "Calculating your taste radar...";
  const response = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      anime_ids: [...selected.keys()],
      min_members: Number($("#minMembers").value || 0),
      top_n: Number($("#topN").value || 12),
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    statusEl.textContent = data.error || "Could not create recommendations.";
    return;
  }
  statusEl.textContent = "Ready. Click the first card to open the share page.";
  renderRecommendations(data.recommendations, data.share_url);
}

searchButton.addEventListener("click", searchAnime);
searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") searchAnime();
});
recommendButton.addEventListener("click", makeRecommendation);
renderSelected();
