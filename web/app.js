"use strict";

const el = (id) => document.getElementById(id);

const dropzone = el("dropzone");
const fileInput = el("file-input");
const runButton = el("run-button");

let selectedFile = null;
let lastResult = null;

// ---- file selection --------------------------------------------------------

function setFile(file) {
  if (!file) return;
  const name = file.name.toLowerCase();
  if (!name.endsWith(".csv") && !name.endsWith(".zip")) {
    showError("Please choose a .csv or .zip export file.");
    return;
  }
  selectedFile = file;
  dropzone.classList.add("has-file");
  el("dropzone-title").textContent = file.name;
  el("dropzone-hint").textContent = `${(file.size / 1024).toFixed(0)} KB — click to change`;
  runButton.disabled = false;
  hideError();
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((type) =>
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((type) =>
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));

// ---- request ---------------------------------------------------------------

function showError(message) {
  const box = el("error-state");
  box.textContent = message;
  box.hidden = false;
}

function hideError() {
  el("error-state").hidden = true;
}

async function run() {
  if (!selectedFile) return;
  runButton.disabled = true;
  runButton.textContent = "Thinking…";
  hideError();
  el("empty-state").hidden = true;
  el("rec-items").innerHTML = "";
  el("loading-state").hidden = false;

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("count", el("opt-count").value);
  form.append("min_votes", el("opt-votes").value);

  try {
    const res = await fetch("/api/recommend", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Request failed (${res.status})`);
    }
    lastResult = data;
    render(data);
  } catch (err) {
    showError(err.message || "Something went wrong.");
    el("empty-state").hidden = false;
  } finally {
    el("loading-state").hidden = true;
    runButton.disabled = false;
    runButton.textContent = "Get recommendations";
  }
}

runButton.addEventListener("click", run);

// ---- rendering ---------------------------------------------------------------

function chip(text) {
  const span = document.createElement("span");
  span.className = "doc-chip";
  span.textContent = text;
  return span;
}

function pill(className, text, title) {
  const span = document.createElement("span");
  span.className = `score-pill ${className}`;
  span.textContent = text;
  if (title) span.title = title;
  return span;
}

function link(href, label) {
  const a = document.createElement("a");
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.textContent = label;
  return a;
}

function render(data) {
  const { profile, recommendations, source } = data;

  el("results-title").textContent =
    source === "imdb" ? "Picks from your IMDb ratings" : "Picks from your Letterboxd diary";

  const mean =
    source === "imdb" ? `${profile.mean_rating10} / 10` : `${profile.mean_rating5} / 5`;
  el("profile-summary").textContent =
    `${profile.matched} of ${profile.parsed} titles matched · you rate ${mean} on average`;

  const chips = el("genre-chips");
  chips.innerHTML = "";
  profile.top_genres.forEach((genre) => chips.append(chip(genre)));
  el("profile-box").hidden = false;

  const unmatchedBox = el("unmatched-box");
  if (data.unmatched_sample.length > 0) {
    el("unmatched-summary").textContent =
      `${profile.unmatched} titles could not be matched`;
    const list = el("unmatched-list");
    list.innerHTML = "";
    data.unmatched_sample.forEach((title) => {
      const li = document.createElement("li");
      li.textContent = title;
      list.append(li);
    });
    unmatchedBox.hidden = false;
  } else {
    unmatchedBox.hidden = true;
  }

  const items = el("rec-items");
  items.innerHTML = "";
  recommendations.forEach((rec) => {
    const row = document.createElement("div");
    row.className = "rec-row";

    const rank = document.createElement("div");
    rank.className = "rec-rank";
    rank.textContent = String(rec.rank);

    const main = document.createElement("div");
    main.className = "rec-main";

    const title = document.createElement("div");
    title.className = "rec-title";
    title.textContent = rec.title + " ";
    if (rec.year) {
      const year = document.createElement("span");
      year.className = "rec-year";
      year.textContent = `(${rec.year})`;
      title.append(year);
    }

    const genres = document.createElement("div");
    genres.className = "rec-genres";
    rec.genres.slice(0, 5).forEach((genre) => genres.append(chip(genre)));

    main.append(title, genres);

    if (rec.because_of.length > 0) {
      const because = document.createElement("div");
      because.className = "rec-because";
      because.append("Because you liked ");
      rec.because_of.forEach((liked, i) => {
        if (i > 0) because.append(" and ");
        const strong = document.createElement("strong");
        strong.textContent = liked;
        because.append(strong);
      });
      main.append(because);
    }

    const links = document.createElement("div");
    links.className = "rec-links";
    if (rec.imdb_url) links.append(link(rec.imdb_url, "IMDb ↗"));
    if (rec.letterboxd_url) links.append(link(rec.letterboxd_url, "Letterboxd ↗"));
    main.append(links);

    const scores = document.createElement("div");
    scores.className = "rec-scores";
    const isImdb = source === "imdb";
    scores.append(
      pill(
        isImdb ? "imdb" : "letterboxd",
        isImdb
          ? `you'd rate ${rec.predicted_imdb10.toFixed(1)} / 10`
          : `you'd rate ${rec.predicted_letterboxd5.toFixed(1)} ★`,
        `Predicted from your own ratings — ${rec.predicted_imdb10.toFixed(1)}/10 on IMDb, ` +
          `${rec.predicted_letterboxd5.toFixed(1)}/5 on Letterboxd`
      )
    );
    const community = document.createElement("div");
    community.className = "rec-community";
    if (rec.community_avg5 != null) {
      community.textContent = `community ${rec.community_avg5.toFixed(2)} ★ · ${rec.community_votes.toLocaleString()} ratings`;
    }
    scores.append(community);

    row.append(rank, main, scores);
    items.append(row);
  });

  el("export-actions").hidden = recommendations.length === 0;
}

// ---- export ---------------------------------------------------------------

function download(filename, mime, content) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function csvEscape(value) {
  const text = value == null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function toCsv(rows) {
  return rows.map((row) => row.map(csvEscape).join(",")).join("\n") + "\n";
}

el("export-csv").addEventListener("click", () => {
  if (!lastResult) return;
  const rows = [
    [
      "rank", "title", "year", "genres", "predicted_imdb_10", "predicted_letterboxd_5",
      "match_pct", "confidence", "because_you_liked", "community_avg_5", "community_votes",
      "imdb_url", "letterboxd_url",
    ],
    ...lastResult.recommendations.map((rec) => [
      rec.rank, rec.title, rec.year, rec.genres.join("|"), rec.predicted_imdb10,
      rec.predicted_letterboxd5, rec.match_pct, rec.confidence, rec.because_of.join("; "),
      rec.community_avg5, rec.community_votes, rec.imdb_url, rec.letterboxd_url,
    ]),
  ];
  download("bir-mov-recommendations.csv", "text/csv;charset=utf-8", toCsv(rows));
});

el("export-json").addEventListener("click", () => {
  if (!lastResult) return;
  download(
    "bir-mov-recommendations.json",
    "application/json",
    JSON.stringify(lastResult, null, 2)
  );
});

el("export-letterboxd").addEventListener("click", () => {
  if (!lastResult) return;
  // Letterboxd list-import format: https://letterboxd.com/about/importing-data/
  const rows = [
    ["imdbID", "Title", "Year"],
    ...lastResult.recommendations.map((rec) => {
      const match = rec.imdb_url ? rec.imdb_url.match(/tt\d+/) : null;
      return [match ? match[0] : "", rec.title, rec.year];
    }),
  ];
  download("bir-mov-letterboxd-import.csv", "text/csv;charset=utf-8", toCsv(rows));
});
