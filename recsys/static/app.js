"use strict";
const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

// 숫자 문자열이면 숫자로, 아니면 원문 그대로 보내서 서버가 422로 검증하게 한다
function parseInput(text) {
  const t = text.trim();
  return /^-?\d+$/.test(t) ? Number(t) : t;
}

async function api(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch (e) {
    throw new Error("서버에 연결할 수 없습니다. 서버가 켜져 있는지 확인하세요.");
  }
  if (res.ok) return res.json();
  let msg = `요청 실패 (${res.status})`;
  try {
    const body = await res.json();
    msg = humanizeError(res.status, body.detail);
  } catch (_) { /* JSON 이 아니면 기본 메시지 */ }
  throw new Error(msg);
}

const FIELD_NAMES = { user_id: "유저 ID", movie_id: "영화 ID", k: "추천 개수(k)", rating: "평점", ratings: "내 평점" };
function humanizeError(status, detail) {
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      const field = FIELD_NAMES[d.loc[d.loc.length - 1]] || d.loc.join(".");
      if (d.type === "int_parsing" || d.type === "int_from_float" || d.type === "int_type")
        return `${field}: 정수만 입력할 수 있습니다 (입력값: ${JSON.stringify(d.input)})`;
      if (d.type === "greater_than_equal") return `${field}: ${d.ctx.ge} 이상이어야 합니다 (입력값: ${d.input})`;
      if (d.type === "less_than_equal") return `${field}: ${d.ctx.le} 이하여야 합니다 (입력값: ${d.input})`;
      if (d.type === "too_short") return `${field}: ${d.ctx.min_length}개 이상 입력하세요`;
      if (d.type === "multiple_of") return `${field}: 0.5 단위로 입력하세요`;
      return `${field}: ${d.msg}`;
    }).join(" / ");
  }
  return typeof detail === "string" ? detail : `요청 실패 (${status})`;
}

function setStatus(node, text, isError = false) {
  node.textContent = text;
  node.classList.toggle("error", isError);
}

function stars(rating) {
  const full = Math.floor(rating), half = rating - full >= 0.5;
  return "★".repeat(full) + (half ? "½" : "");
}

function genreBadges(genres) {
  const frag = document.createDocumentFragment();
  genres.forEach((g) => frag.append(el("span", { class: "badge genre" }, g)));
  return frag;
}

// ---------- 평점 예측 ----------
function renderPrediction(r) {
  const box = $("predict-result");
  box.replaceChildren();
  const left = el("div", {},
    el("div", { class: "num" }, r.predicted_rating.toFixed(2)),
    el("div", { class: "stars" }, stars(r.predicted_rating)));
  const mid = el("div", { class: "cmp" },
    el("div", { class: "movie-title" }, r.title),
    el("div", {}, genreBadges(r.genres)),
    el("div", {},
      el("span", { class: "badge " + (r.likely_like ? "like" : "dislike") }, r.likely_like ? "좋아할 듯 (≥4.0)" : "별로일 듯 (<4.0)"),
      r.fallback ? el("span", { class: "badge fallback" }, "fallback") : ""));
  const parts = [left, mid];
  if (r.fallback_reason) parts.push(el("div", { class: "muted" }, r.fallback_reason));
  if (r.actual_rating !== null) {
    const diff = r.predicted_rating - r.actual_rating;
    const note = r.actual_source === "train" ? " (학습에 사용된 평점 — 검증용 아님)" : " (test 평점)";
    parts.push(el("div", { class: "cmp" },
      el("div", {}, `실제 평점: ${r.actual_rating.toFixed(1)}${note}`),
      el("div", { class: Math.abs(diff) <= 0.5 ? "err-good" : "err-bad" }, `오차: ${diff >= 0 ? "+" : ""}${diff.toFixed(2)}`)));
  } else {
    parts.push(el("div", { class: "muted" }, "이 유저의 이 영화 실제 평점 기록 없음"));
  }
  box.append(el("div", { class: "big" }, ...parts));
  if (r.tmdb_overview) box.append(el("p", { class: "muted" }, r.tmdb_overview));
  box.append(el("p", { class: "muted" }, `모델 ${r.model_version} · ${r.latency_ms} ms`));
}

async function doPredict() {
  const btn = $("predict-btn");
  btn.disabled = true;
  setStatus($("predict-status"), "예측 중…");
  $("predict-result").replaceChildren();
  try {
    const data = await api("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: parseInput($("user-id").value), movie_id: parseInput($("movie-id").value) }),
    });
    setStatus($("predict-status"), "");
    renderPrediction(data);
  } catch (e) {
    setStatus($("predict-status"), e.message, true);
  } finally {
    btn.disabled = false;
  }
}

// ---------- 영화 검색 ----------
let searchTimer = null;
function onMovieSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const q = $("movie-q").value.trim();
    const list = $("movie-results");
    if (!q) { list.replaceChildren(); return; }
    try {
      const movies = await api(`/movies?q=${encodeURIComponent(q)}&limit=15`);
      list.replaceChildren(...movies.map((m) => el("li", {
        onclick: () => { $("movie-id").value = m.movieId; list.replaceChildren(); $("movie-q").value = m.title; },
      }, `${m.title} (ID ${m.movieId})`)));
      if (!movies.length) list.append(el("li", {}, "검색 결과 없음"));
    } catch (e) {
      setStatus($("predict-status"), e.message, true);
    }
  }, 250);
}

// ---------- 유저 취향 ----------
async function loadHistory() {
  const list = $("history");
  const uid = parseInput($("user-id").value);
  list.replaceChildren();
  if (typeof uid !== "number" || uid < 1) {
    list.append(el("li", {}, "유효한 유저 ID를 입력하면 표시됩니다"));
    return;
  }
  try {
    const items = await api(`/users/${uid}/history?limit=10`);
    if (!items.length) list.append(el("li", {}, "학습 데이터에 없는 유저입니다"));
    items.forEach((m) => list.append(el("li", {
      title: "클릭하면 이 영화로 예측",
      onclick: () => { $("movie-id").value = m.movieId; doPredict(); },
    }, `${m.title} ★${m.rating} · ${m.genres.join(", ")}`)));
  } catch (e) {
    list.append(el("li", {}, e.message));
  }
}

// ---------- 검증 ----------
async function doVerify() {
  const btn = $("verify-btn");
  btn.disabled = true;
  const tbody = $("verify-table").tBodies[0];
  tbody.replaceChildren();
  setStatus($("verify-summary"), "예측 중…");
  try {
    const samples = await api("/samples?n=30");
    const rows = [];
    for (const s of samples) {
      const r = await api("/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: s.user_id, movie_id: s.movie_id }),
      });
      const diff = r.predicted_rating - s.actual_rating;
      rows.push(diff);
      tbody.append(el("tr", {},
        el("td", {}, String(s.user_id)),
        el("td", { class: "title" }, s.title || String(s.movie_id)),
        el("td", {}, r.predicted_rating.toFixed(2)),
        el("td", {}, s.actual_rating.toFixed(1)),
        el("td", { class: Math.abs(diff) <= 0.5 ? "err-good" : "err-bad" }, (diff >= 0 ? "+" : "") + diff.toFixed(2)),
        el("td", {}, el("button", { class: "small secondary", type: "button", onclick: () => {
          $("user-id").value = s.user_id; $("movie-id").value = s.movie_id; loadHistory(); doPredict();
          window.scrollTo({ top: 0, behavior: "smooth" });
        } }, "보기"))));
    }
    const mae = rows.reduce((a, d) => a + Math.abs(d), 0) / rows.length;
    const within = rows.filter((d) => Math.abs(d) <= 0.5).length;
    setStatus($("verify-summary"), `샘플 ${rows.length}건 · 평균 절대 오차 ${mae.toFixed(2)} · 오차 0.5 이내 ${within}건 (${Math.round(100 * within / rows.length)}%)`);
  } catch (e) {
    setStatus($("verify-summary"), e.message, true);
  } finally {
    btn.disabled = false;
  }
}

// ---------- 추천 Top-N ----------
async function doRecommend() {
  const btn = $("rec-btn");
  btn.disabled = true;
  setStatus($("rec-status"), "추천 계산 중…");
  $("rec-list").replaceChildren();
  try {
    const data = await api("/recommend", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: parseInput($("user-id").value), k: parseInput($("rec-k").value),
        exclude_seen: $("rec-exclude").checked,
      }),
    });
    setStatus($("rec-status"), `${data.latency_ms} ms · 모델 ${data.model_version}${data.fallback ? " · 학습에 없는 유저 → 영화 평균 순(fallback)" : ""}`);
    data.items.forEach((m, i) => $("rec-list").append(recCard(m, i + 1)));
  } catch (e) {
    setStatus($("rec-status"), e.message, true);
  } finally {
    btn.disabled = false;
  }
}

function recCard(m, rank, clickable = true) {
  const props = clickable
    ? { class: "card", title: "클릭하면 이 영화로 평점 예측",
        onclick: () => { $("movie-id").value = m.movieId; window.scrollTo({ top: 0, behavior: "smooth" }); doPredict(); } }
    : { class: "card static" };
  const card = el("li", props,
    el("div", { class: "rank" }, String(rank)),
    el("div", {}, el("div", {}, el("strong", {}, m.title)), el("div", {}, genreBadges(m.genres))),
    el("div", { class: "score" }, m.predicted_rating.toFixed(2)));
  if (m.tmdb_overview) {
    const ov = el("div", { class: "ov clamp" }, m.tmdb_overview);
    const more = el("button", { class: "more", type: "button", onclick: (ev) => {
      ev.stopPropagation();
      const clamped = ov.classList.toggle("clamp");
      more.textContent = clamped ? "줄거리 더 보기" : "접기";
    } }, "줄거리 더 보기");
    card.append(ov, more);
  }
  return card;
}


// ---------- 내 평점으로 추천 ----------
const STORE_KEY = "myRatings";
let myRatings = loadMyRatings();   // { movieId: { title, rating } }

function loadMyRatings() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch (_) { return {}; }
}
function saveMyRatings() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(myRatings)); } catch (_) { /* 저장 불가 환경이면 메모리에만 유지 */ }
}

function ratingSelect(current, onChange) {
  const sel = el("select", { "aria-label": "평점" }, el("option", { value: "" }, "평점 선택"));
  for (let v = 0.5; v <= 5; v += 0.5) sel.append(el("option", { value: String(v) }, v.toFixed(1)));
  sel.value = current === undefined ? "" : String(current);
  sel.addEventListener("change", () => onChange(sel.value === "" ? null : Number(sel.value)));
  return sel;
}

function setMyRating(movie, rating) {
  if (rating === null) delete myRatings[movie.movieId];
  else myRatings[movie.movieId] = { title: movie.title, rating };
  saveMyRatings();
  renderMyRatings();
}

function candidateRow(movie) {
  const cur = myRatings[movie.movieId];
  return el("li", {}, el("span", { class: "t" }, movie.title),
    ratingSelect(cur && cur.rating, (v) => setMyRating(movie, v)));
}

function renderMyRatings() {
  const ids = Object.keys(myRatings);
  $("my-count").textContent = `(${ids.length}개${ids.length < 3 ? " · 3개 이상 필요" : ""})`;
  $("my-ratings").replaceChildren(...ids.map((id) => {
    const movie = { movieId: Number(id), title: myRatings[id].title };
    return el("li", {}, el("span", { class: "t" }, movie.title),
      ratingSelect(myRatings[id].rating, (v) => setMyRating(movie, v)),
      el("button", { class: "small secondary", type: "button", onclick: () => setMyRating(movie, null) }, "삭제"));
  }));
}

async function showCandidates(path) {
  const list = $("my-candidates");
  try {
    const movies = await api(path);
    list.replaceChildren(...movies.map(candidateRow));
    if (!movies.length) list.append(el("li", {}, "검색 결과 없음"));
    list.classList.add("scroll");
  } catch (e) {
    setStatus($("my-status"), e.message, true);
  }
}

let mySearchTimer = null;
function onMySearch() {
  clearTimeout(mySearchTimer);
  mySearchTimer = setTimeout(() => {
    const q = $("my-q").value.trim();
    if (!q) { $("my-candidates").replaceChildren(); return; }
    showCandidates(`/movies?q=${encodeURIComponent(q)}&limit=15`);
  }, 250);
}


// 제목 표시용: "Shawshank Redemption, The (1994)" -> { title: "The Shawshank Redemption", year: "1994" }
function prettyTitle(raw) {
  let title = raw, year = "";
  const y = title.match(/^(.*)\s\((\d{4})\)$/);
  if (y) { title = y[1]; year = y[2]; }
  const art = title.match(/^(.*),\s(The|A|An)$/);
  if (art) title = `${art[2]} ${art[1]}`;
  return { title, year };
}

function ratingMeter(value) {
  return el("div", { class: "meter", title: `예상 평점 ${value.toFixed(2)} / 5` },
    el("div", { class: "meter-fill", style: `width:${Math.round((value / 5) * 100)}%` }));
}

function myRecCard(m, rank) {
  const t = prettyTitle(m.title);
  const head = el("div", { class: "rc-head" },
    el("span", { class: "rc-rank" }, String(rank)),
    el("div", { class: "rc-title" }, el("strong", {}, t.title), t.year ? el("span", { class: "muted" }, ` ${t.year}`) : ""),
    el("div", { class: "rc-score" }, el("span", { class: "stars" }, "★"), ` ${m.predicted_rating.toFixed(1)}`));
  const card = el("li", { class: "rec-card" }, head, ratingMeter(m.predicted_rating),
    el("div", { class: "rc-genres" }, genreBadges(m.genres)));
  if (m.tmdb_overview) {
    const ov = el("p", { class: "rc-ov clamp" }, m.tmdb_overview);
    const more = el("button", { class: "more", type: "button", onclick: () => {
      const clamped = ov.classList.toggle("clamp");
      more.textContent = clamped ? "줄거리 더 보기" : "접기";
    } }, "줄거리 더 보기");
    card.append(ov, more);
  }
  return card;
}

function similarUserCard(u) {
  const pct = Math.max(0, Math.min(100, Math.round(u.similarity * 100)));
  return el("li", { class: "su-card" },
    el("div", { class: "su-head" }, el("strong", {}, `유저 #${u.user_id}`), el("span", { class: "su-pct" }, `${pct}% 비슷`)),
    el("div", { class: "meter small" }, el("div", { class: "meter-fill", style: `width:${pct}%` })),
    el("div", { class: "muted" }, `평점 ${u.n_ratings}개 · 이 유저의 최애`),
    el("ul", { class: "su-fav" }, ...u.favorites.map((f) =>
      el("li", {}, `${prettyTitle(f.title).title} `, el("span", { class: "stars" }, `★${f.rating}`)))));
}

async function doMyRecommend() {
  const btn = $("my-rec-btn");
  btn.disabled = true;
  setStatus($("my-status"), "내 취향 계산 중…");
  $("my-similar").replaceChildren();
  $("my-rec-list").replaceChildren();
  const oldTitle = $("my-rec-title");
  if (oldTitle) oldTitle.remove();
  try {
    const ratings = Object.entries(myRatings).map(([id, v]) => ({ movie_id: Number(id), rating: v.rating }));
    const data = await api("/recommend/from-ratings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ratings, k: 10 }),
    });
    setStatus($("my-status"), `내 평점 ${data.n_ratings_used}개 사용${data.n_ratings_skipped ? ` (모델이 모르는 영화 ${data.n_ratings_skipped}개 제외)` : ""} · 안 본 영화 전체에서 상위 10편 · ${data.latency_ms} ms · 모델 ${data.model_version}`);
    $("my-similar").append(el("h3", {}, "나와 취향이 비슷한 유저"),
      el("ul", { class: "su-grid" }, ...data.similar_users.map(similarUserCard)));
    $("my-rec-list").before(el("h3", { id: "my-rec-title" }, `추천 영화 TOP ${data.items.length}`,
      el("span", { class: "muted" }, " · 이 유저들과 내 평점으로 계산한 예상 평점 순")));
    data.items.forEach((m, i) => $("my-rec-list").append(myRecCard(m, i + 1)));
  } catch (e) {
    setStatus($("my-status"), e.message, true);
  } finally {
    btn.disabled = false;
  }
}

// ---------- 초기화 ----------
async function init() {
  try {
    const h = await api("/health");
    $("model-info").textContent = `· 모델 ${h.model_version}`;
    const users = await api("/users");
    $("user-list").replaceChildren(...users.map((u) => el("option", { value: u.user_id }, `평점 ${u.n_ratings}개`)));
    $("random-user").addEventListener("click", () => {
      $("user-id").value = users[Math.floor(Math.random() * users.length)].user_id;
      loadHistory();
    });
  } catch (e) {
    setStatus($("predict-status"), e.message, true);
  }
  $("predict-btn").addEventListener("click", doPredict);
  $("rec-btn").addEventListener("click", doRecommend);
  $("verify-btn").addEventListener("click", doVerify);
  $("movie-q").addEventListener("input", onMovieSearch);
  $("user-id").addEventListener("change", loadHistory);
  $("my-q").addEventListener("input", onMySearch);
  $("my-popular").addEventListener("click", () => showCandidates("/popular?limit=30"));
  $("my-rec-btn").addEventListener("click", doMyRecommend);
  $("my-clear").addEventListener("click", () => { myRatings = {}; saveMyRatings(); renderMyRatings(); });
  renderMyRatings();
  loadHistory();
}
init();
