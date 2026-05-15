const $ = (selector) => document.querySelector(selector);

const authPanel = $("#authPanel");
const appPanel = $("#appPanel");
const statusEl = $("#status");
const usernameLabel = $("#usernameLabel");
const ratingStatus = $("#ratingStatus");
const personalStatus = $("#personalStatus");
const personalGrid = $("#personalGrid");
const searchInput = $("#searchInput");
const searchResults = $("#searchResults");
const ratingList = $("#ratingList");

let currentUser = null;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "요청 실패");
  return data;
}

function setStatus(message) {
  statusEl.textContent = message || "";
}

function showApp(user) {
  currentUser = user;
  authPanel.classList.add("hidden");
  appPanel.classList.remove("hidden");
  usernameLabel.textContent = user.username;
  updateRatingStatus(user);
}

function updateRatingStatus(user) {
  const remain = Math.max(0, user.threshold - user.rating_count);
  ratingStatus.textContent = user.ready
    ? `평가 ${user.rating_count}개 완료. 개인 추천이 열렸습니다.`
    : `평가 ${user.rating_count}개 완료. ${remain}개 더 평가하면 개인 추천이 열립니다.`;
}

async function refreshMe() {
  const data = await api("/api/me");
  if (!data.authenticated) {
    authPanel.classList.remove("hidden");
    appPanel.classList.add("hidden");
    return;
  }
  showApp(data.user);
  await Promise.all([loadRatings(), loadPersonalRecommendations()]);
}

async function login(username, password) {
  const data = await api("/api/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  showApp(data.user);
  await Promise.all([loadRatings(), loadPersonalRecommendations()]);
}

async function register(username, password) {
  const data = await api("/api/register", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  showApp(data.user);
  await Promise.all([loadRatings(), loadPersonalRecommendations()]);
}

async function logout() {
  await api("/api/logout", { method: "POST", body: "{}" });
  currentUser = null;
  authPanel.classList.remove("hidden");
  appPanel.classList.add("hidden");
  searchResults.innerHTML = "";
  ratingList.innerHTML = "";
  personalGrid.innerHTML = "";
}

function animeMeta(item) {
  return `${item.type} - 평점 ${item.score ?? "?"} - ${item.genres}`;
}

function ratingSelect(item) {
  const options = Array.from({ length: 10 }, (_, index) => {
    const value = index + 1;
    return `<option value="${value}">${value}점</option>`;
  }).join("");
  return `
    <div class="rating-actions">
      <select data-rating-for="${item.id}" aria-label="${item.title} 평점">
        <option value="">평가</option>
        ${options}
      </select>
      <button data-rate="${item.id}" type="button">저장</button>
    </div>
  `;
}

async function searchAnime() {
  const query = searchInput.value.trim();
  searchResults.innerHTML = "";
  if (!query) return;
  setStatus("검색 중...");
  const data = await api(`/api/search?q=${encodeURIComponent(query)}`);
  setStatus("");
  for (const item of data.results) {
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      <h3>${item.title}</h3>
      <p>${animeMeta(item)}</p>
      ${ratingSelect(item)}
    `;
    card.querySelector("[data-rate]").addEventListener("click", async () => {
      const select = card.querySelector(`[data-rating-for="${item.id}"]`);
      if (!select.value) {
        setStatus("평점을 먼저 선택하세요.");
        return;
      }
      await saveRating(item.id, Number(select.value));
    });
    searchResults.appendChild(card);
  }
}

async function saveRating(animeId, rating) {
  const data = await api("/api/ratings", {
    method: "POST",
    body: JSON.stringify({ anime_id: animeId, rating }),
  });
  showApp(data.user);
  setStatus("평가 저장 완료.");
  await Promise.all([loadRatings(), loadPersonalRecommendations()]);
}

async function deleteRating(animeId) {
  const data = await api(`/api/ratings/${animeId}`, { method: "DELETE" });
  showApp(data.user);
  await Promise.all([loadRatings(), loadPersonalRecommendations()]);
}

async function loadRatings() {
  const data = await api("/api/ratings");
  updateRatingStatus(data.user);
  ratingList.innerHTML = "";
  if (data.ratings.length === 0) {
    ratingList.textContent = "아직 평가한 애니가 없습니다.";
    return;
  }
  for (const item of data.ratings) {
    const row = document.createElement("article");
    row.className = "rating-row";
    row.innerHTML = `
      <div>
        <strong>${item.title}</strong>
        <span>${item.user_rating}점 - ${item.genres}</span>
      </div>
      <button type="button">삭제</button>
    `;
    row.querySelector("button").addEventListener("click", () => deleteRating(item.id));
    ratingList.appendChild(row);
  }
}

async function loadPersonalRecommendations() {
  const data = await api("/api/recommendations");
  personalGrid.innerHTML = "";
  if (!data.ready) {
    const remain = Math.max(0, data.threshold - data.rating_count);
    personalStatus.textContent = `아직 개인 추천 전입니다. ${remain}개 더 평가해주세요.`;
    return;
  }
  personalStatus.textContent = `평가 ${data.rating_count}개 기반 추천입니다.`;
  if (data.recommendations.length === 0) {
    personalStatus.textContent = "추천 후보가 부족합니다. 다른 애니를 몇 개 더 평가해보세요.";
    return;
  }
  for (const item of data.recommendations) {
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      <div class="rank">#${item.rank}</div>
      <h3>${item.title}</h3>
      <p>${item.genres}</p>
      <div class="meta">
        <span>${item.type}</span>
        <span>예상 ${item.predicted_score}</span>
        <span>유사도 ${(item.cb_score * 100).toFixed(1)}%</span>
      </div>
    `;
    personalGrid.appendChild(card);
  }
}

$("#loginButton").addEventListener("click", () => {
  login($("#loginName").value.trim(), $("#loginPassword").value).catch((error) => setStatus(error.message));
});

$("#registerButton").addEventListener("click", () => {
  register($("#registerName").value.trim(), $("#registerPassword").value).catch((error) => setStatus(error.message));
});

$("#logoutButton").addEventListener("click", () => {
  logout().catch((error) => setStatus(error.message));
});

$("#searchButton").addEventListener("click", () => {
  searchAnime().catch((error) => setStatus(error.message));
});

searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") searchAnime().catch((error) => setStatus(error.message));
});

refreshMe().catch((error) => setStatus(error.message));
