const $ = (id) => document.getElementById(id);

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return sign + (value * 100).toFixed(digits) + "%";
}

function weightPct(value) {
  return (value * 100).toFixed(1) + "%";
}

function cls(value) {
  if (value === null || value === undefined || Math.abs(value) < 0.0000001) return "";
  return value > 0 ? "up" : "down";
}

function show(id) {
  $(id).hidden = false;
}

function headline(state) {
  const objective = state.objective;
  const best = objective.best;
  if (!best) return "The desk does not have a finished window yet.";
  const gap = Math.abs(objective.gap * 100).toFixed(1);
  const start = objective.windowStart;
  const end = objective.windowEnd;
  if (objective.beatEveryIndex) {
    return `From ${start} to ${end}, the desk led every index in the universe. The closest comparison, ${best.name}, trails by ${gap} percentage points of CAGR.`;
  }
  return `From ${start} to ${end}, ${best.name} led this desk by ${gap} percentage points of CAGR. The objective is to lead every index. The table below is the measurement.`;
}

function renderBars(rows) {
  const root = $("bars");
  root.innerHTML = "";
  rows
    .filter((row) => row.weight >= 0.005 || row.group === "cash")
    .forEach((row) => {
      const wrap = document.createElement("div");
      wrap.className = "bar-row";
      const width = Math.max(0, Math.min(100, row.weight * 100));
      wrap.innerHTML = `<span>${row.name}</span><div class="track"><div class="fill" style="width:${width}%"></div></div><span class="weight">${weightPct(row.weight)}</span>`;
      root.appendChild(wrap);
    });
}

function renderIndustries(rows) {
  const body = $("industries").querySelector("tbody");
  body.innerHTML = rows
    .map(
      (row) => `<tr>
        <td>${row.name}</td>
        <td class="${cls(row.ret5)}">${pct(row.ret5)}</td>
        <td class="${cls(row.ret21)}">${pct(row.ret21)}</td>
        <td class="optional ${cls(row.ret63)}">${pct(row.ret63)}</td>
        <td>${weightPct(row.weight)}</td>
      </tr>`
    )
    .join("");
}

function renderBoard(rows) {
  const body = $("board").querySelector("tbody");
  body.innerHTML = rows
    .map(
      (row) => `<tr>
        <td>${row.name}</td>
        <td>${pct(row.cagr)}</td>
        <td>${pct(row.strategyCagr)}</td>
        <td class="${cls(row.gap)}">${pct(row.gap)}</td>
        <td class="optional ${cls(row.gap3y)}">${pct(row.gap3y)}</td>
      </tr>`
    )
    .join("");
}

function polyline(values, color, min, max) {
  const clean = values.map((value, index) => [index, value]).filter((pair) => pair[1] !== null && pair[1] !== undefined);
  if (clean.length < 2) return "";
  const span = max - min || 1;
  const n = values.length - 1 || 1;
  const points = clean
    .map(([index, value]) => {
      const x = (index / n) * 760 + 20;
      const y = 250 - ((value - min) / span) * 210;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return `<polyline fill="none" stroke="${color}" stroke-width="2.2" points="${points}" />`;
}

function renderChart(chart) {
  const svg = $("chart");
  const series = [chart.strategy || [], chart.spy || []];
  const drawBest = chart.bestTicker && chart.bestTicker !== "SPY";
  if (drawBest) series.push(chart.best || []);
  const nums = series.flat().filter((value) => value !== null && value !== undefined);
  if (nums.length < 2) {
    svg.innerHTML = "";
    return;
  }
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  const parts = [
    polyline(chart.strategy || [], "#0b1f3a", min, max),
    polyline(chart.spy || [], "#1f5fae", min, max),
  ];
  if (drawBest) parts.push(polyline(chart.best || [], "#8d2b2b", min, max));
  svg.innerHTML = parts.join("");
  const bestName = chart.bestTicker && chart.bestTicker !== "SPY" ? ` · ${chart.bestTicker} in red` : "";
  $("chart-legend").textContent = `Growth of $1 after the factor choice. Desk in navy, S&P 500 in blue${bestName}.`;
}

function allocationText(state) {
  const lines = [
    `Research Deck index desk`,
    `As of ${state.asOf}`,
    `Instruction: ${state.action.label}`,
    ``,
    `Send these weights:`,
  ];
  state.instruction.forEach((row) => {
    if (row.weight < 0.005 && row.group !== "cash") return;
    lines.push(`  ${row.name} (${row.ticker})  ${weightPct(row.weight)}`);
  });
  if (state.changes.length) {
    lines.push("", "Changes:");
    state.changes.forEach((row) => {
      lines.push(`  ${row.name} (${row.ticker})  ${weightPct(row.from)} -> ${weightPct(row.to)}`);
    });
  }
  lines.push("", state.disclaimer, "", "Merrill Lynch places any trades. This note is not an order.");
  return lines.join("\n");
}

function render(state) {
  if (state.phase === "loading" || state.phase === "refreshing") {
    $("status").textContent = state.message || "Training the allocation model on daily closes.";
    if (state.phase === "refreshing") $("refresh").disabled = true;
    return;
  }
  if (state.phase === "error") {
    $("status").textContent = state.message || "The desk could not train.";
    return;
  }
  $("status").textContent = headline(state);
  $("asof").textContent = `Last close ${state.asOf} · generated ${state.generatedAt}`;
  const qwen = state.qwen || {};
  $("qwen").textContent = qwen.reachable
    ? `${qwen.model} is reachable and may write a note. It does not set weights.`
    : qwen.message || "";
  const pill = $("action");
  pill.textContent = state.action.label;
  pill.className = "pill" + (state.action.code === "shock" ? " shock" : state.action.trade ? " trade" : "");
  $("action-note").textContent = state.action.trade
    ? "These weights are the instruction for this close."
    : "The book carried into this close stays in place.";
  $("reasons").innerHTML = (state.action.reasons || []).map((reason) => `<li>${reason}</li>`).join("");
  renderBars(state.instruction);
  renderIndustries(state.industries || []);
  renderBoard(state.scoreboard || []);
  renderChart(state.chart || {});
  $("model-action").textContent = `Update: ${state.model.action} · risk aversion ${Number(state.model.riskAversion).toFixed(2)} · validation score ${Number(state.model.score).toFixed(4)}`;
  $("model-note").textContent = state.model.note;
  const rejected = (state.model.rejected || []).join(", ") || "none";
  $("factors").textContent = `Kept factors: ${(state.model.features || []).join(", ")}. Set aside: ${rejected}.`;
  $("drivers").innerHTML = (state.drivers || [])
    .map((driver) => {
      const bits = driver.factors
        .map((factor) => `${factor.name} ${factor.effect >= 0 ? "+" : ""}${factor.effect.toFixed(3)}`)
        .join(", ");
      return `<p class="driver"><strong>${driver.name}</strong> ${weightPct(driver.weight)} · expected 21-session excess ${pct(driver.expectedExcess)} · ${bits}</p>`;
    })
    .join("");
  $("copy-block").textContent = allocationText(state);
  $("disclaimer").textContent = state.disclaimer;
  if (state.warning) $("status").textContent += " " + state.warning;
  ["actions", "workspace", "chart-panel", "board-panel", "model-panel", "copy-panel"].forEach(show);
  $("refresh").disabled = false;
}

async function pull() {
  const response = await fetch("/api/state");
  const state = await response.json();
  render(state);
  if (state.phase === "loading" || state.phase === "refreshing") {
    setTimeout(pull, 1500);
  }
}

$("refresh").addEventListener("click", async () => {
  $("refresh").disabled = true;
  await fetch("/api/refresh", { method: "POST" });
  pull();
});

$("copy").addEventListener("click", async () => {
  const text = $("copy-block").textContent;
  try {
    await navigator.clipboard.writeText(text);
    $("copy").textContent = "Copied";
  } catch (error) {
    $("copy").textContent = "Select the text below";
  }
});

pull();
