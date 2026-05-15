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
    selectedList.textContent = "아직 선택한 애니가 없습니다.";
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
  return `${item.type} - 평점 ${item.score ?? "?"} - ${item.genres}`;
}

async function searchAnime() {
  const query = searchInput.value.trim();
  searchResults.innerHTML = "";
  if (!query) return;
  statusEl.textContent = "검색 중...";
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
  const fullUrl = `${location.origin}${shareUrl}`;
  const linkCard = document.createElement("article");
  linkCard.className = "card share-card";
  linkCard.innerHTML = `
    <div class="rank">공유</div>
    <h3>공유 링크 생성 완료</h3>
    <p class="share-url">${fullUrl}</p>
    <div class="meta">
      <span>조회수 기록 중</span>
      <button class="copy-button" type="button">링크 복사</button>
    </div>
  `;
  const copyButton = linkCard.querySelector(".copy-button");
  copyButton.addEventListener("click", async (event) => {
    event.stopPropagation();
    try {
      await navigator.clipboard.writeText(fullUrl);
      copyButton.textContent = "복사 완료";
    } catch {
      copyButton.textContent = "복사 실패";
    }
  });
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
        <span>평점 ${item.score ?? "?"}</span>
        <span>유사도 ${(item.taste_score * 100).toFixed(1)}%</span>
      </div>
    `;
    preview.appendChild(card);
  });
}

async function makeRecommendation() {
  if (selected.size === 0) {
    statusEl.textContent = "먼저 애니를 하나 이상 선택하세요.";
    return;
  }
  statusEl.textContent = "추천 계산 중...";
  const response = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      anime_ids: [...selected.keys()],
      min_members: Number($("#minMembers").value || 0),
      top_n: Number($("#topN").value || 12),
      avoid_same_series: $("#avoidSameSeries").checked,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    statusEl.textContent = data.error || "추천 생성 실패";
    return;
  }
  statusEl.textContent = "완료. 첫 카드를 누르면 공유 페이지로 이동합니다.";
  renderRecommendations(data.recommendations, data.share_url);
}

searchButton.addEventListener("click", searchAnime);
searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") searchAnime();
});
recommendButton.addEventListener("click", makeRecommendation);
renderSelected();
