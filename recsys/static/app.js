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

const FIELD_NAMES = { user_id: "유저 ID", movie_id: "영화 ID", k: "추천 개수(k)" };
function humanizeError(status, detail) {
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      const field = FIELD_NAMES[d.loc[d.loc.length - 1]] || d.loc.join(".");
      if (d.type === "int_parsing" || d.type === "int_from_float" || d.type === "int_type")
        return `${field}: 정수만 입력할 수 있습니다 (입력값: ${JSON.stringify(d.input)})`;
      if (d.type === "greater_than_equal") return `${field}: ${d.ctx.ge} 이상이어야 합니다 (입력값: ${d.input})`;
      if (d.type === "less_than_equal") return `${field}: ${d.ctx.le} 이하여야 합니다 (입력값: ${d.input})`;
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

function recCard(m, rank) {
  const card = el("li", { class: "card", title: "클릭하면 이 영화로 평점 예측",
    onclick: () => { $("movie-id").value = m.movieId; window.scrollTo({ top: 0, behavior: "smooth" }); doPredict(); } },
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
  loadHistory();
}
init();
