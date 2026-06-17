import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { Presentation, PresentationFile } from "file:///C:/Users/Win11Pro/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = process.cwd();
const OUTPUT_DIR = path.join(ROOT, "outputs");
const FINAL_PPTX = path.join(OUTPUT_DIR, "daegu_bus_analysis_editable.pptx");
const PREVIEW_COPY = path.join(OUTPUT_DIR, "daegu_bus_analysis_editable_preview.webp");
const THREAD_ID = process.env.CODEX_THREAD_ID || `manual-${Date.now()}`;
const WORKSPACE = path.join(os.tmpdir(), "codex-presentations", THREAD_ID, "daegu-bus-editable-pptx");
const TMP_DIR = path.join(WORKSPACE, "tmp");
const PREVIEW_DIR = path.join(TMP_DIR, "preview");
const LAYOUT_DIR = path.join(TMP_DIR, "layout");
const QA_DIR = path.join(TMP_DIR, "qa");

const SLIDE = { width: 1280, height: 720 };
const PAGE = { left: 54, top: 54, width: 1172, height: 610 };
const COLORS = {
  bg: "#C0C0C0",
  paper: "#F4F4F0",
  panel: "#FFFFFF",
  panelAlt: "#E8E8E8",
  navy: "#000080",
  blue: "#1084D0",
  yellow: "#FFFF00",
  green: "#00AA00",
  red: "#FF0000",
  black: "#000000",
  gray: "#555555",
  border: "#303030",
  lightLine: "#9A9A9A",
};
const FONT = "Malgun Gothic";
const MONO = "Consolas";

function parseCsv(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === '"') {
      if (inQuotes && next === '"') {
        cell += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      row.push(cell);
      cell = "";
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    if (row.some((value) => value !== "")) rows.push(row);
  }
  return rows;
}

async function readCsv(relativePath) {
  const fullPath = path.join(ROOT, relativePath);
  const text = await fs.readFile(fullPath, "utf8");
  const rows = parseCsv(text);
  const header = rows.shift() || [];
  return rows.map((row) => {
    const record = {};
    header.forEach((key, index) => {
      record[key] = row[index] ?? "";
    });
    return record;
  });
}

function n(value, fallback = 0) {
  const parsed = Number(String(value ?? "").replaceAll(",", "").trim());
  return Number.isFinite(parsed) ? parsed : fallback;
}

function fmt(value, decimals = 0) {
  const parsed = n(value, 0);
  return parsed.toLocaleString("ko-KR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function pct(value, decimals = 1) {
  const parsed = n(value, 0);
  return `${parsed.toLocaleString("ko-KR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}%`;
}

function cleanName(value) {
  return String(value ?? "").trim() || "미상";
}

function sumBy(rows, key, valueKey) {
  const map = new Map();
  for (const row of rows) {
    const group = cleanName(row[key]);
    map.set(group, (map.get(group) || 0) + n(row[valueKey]));
  }
  return Array.from(map, ([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value);
}

function avgBy(rows, key, valueKey) {
  const map = new Map();
  for (const row of rows) {
    const group = cleanName(row[key]);
    const item = map.get(group) || { sum: 0, count: 0 };
    const value = n(row[valueKey], NaN);
    if (Number.isFinite(value)) {
      item.sum += value;
      item.count += 1;
    }
    map.set(group, item);
  }
  return Array.from(map, ([name, item]) => ({ name, value: item.count ? item.sum / item.count : 0 }))
    .sort((a, b) => b.value - a.value);
}

function quantile(values, q) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}

function pearson(xs, ys) {
  const pairs = xs.map((x, i) => [n(x, NaN), n(ys[i], NaN)]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (pairs.length < 2) return 0;
  const meanX = pairs.reduce((acc, [x]) => acc + x, 0) / pairs.length;
  const meanY = pairs.reduce((acc, [, y]) => acc + y, 0) / pairs.length;
  let num = 0;
  let denX = 0;
  let denY = 0;
  for (const [x, y] of pairs) {
    num += (x - meanX) * (y - meanY);
    denX += (x - meanX) ** 2;
    denY += (y - meanY) ** 2;
  }
  return denX && denY ? num / Math.sqrt(denX * denY) : 0;
}

function rank(values) {
  const sorted = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  const result = Array(values.length).fill(0);
  let i = 0;
  while (i < sorted.length) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1].value === sorted[i].value) j += 1;
    const avg = (i + j + 2) / 2;
    for (let k = i; k <= j; k += 1) result[sorted[k].index] = avg;
    i = j + 1;
  }
  return result;
}

function spearman(xs, ys) {
  return pearson(rank(xs.map((v) => n(v))), rank(ys.map((v) => n(v))));
}

function corrLabel(value) {
  const abs = Math.abs(value);
  if (abs >= 0.7) return "강함";
  if (abs >= 0.4) return "보통";
  if (abs >= 0.2) return "다소 약함";
  return "약함";
}

function top(rows, key, count = 10) {
  return [...rows].sort((a, b) => n(b[key]) - n(a[key])).slice(0, count);
}

function addRect(slide, x, y, w, h, fill = COLORS.panel, line = COLORS.border, name = "rect") {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1 },
  });
}

function addLine(slide, x1, y1, x2, y2, color = COLORS.border, width = 1) {
  return slide.shapes.add({
    geometry: "line",
    position: { left: x1, top: y1, width: x2 - x1, height: y2 - y1 },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: opts.name || "text",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontFamily: opts.fontFamily || FONT,
    fontSize: opts.fontSize ?? 18,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    color: opts.color || COLORS.black,
    alignment: opts.align || "left",
  };
  return shape;
}

function addHeader(slide, bodyNo, title, takeaway, speaker) {
  slide.background.fill = COLORS.bg;
  addRect(slide, 0, 0, SLIDE.width, 34, COLORS.navy, COLORS.navy, "top-strip");
  addText(slide, bodyNo ? `본문 ${String(bodyNo).padStart(2, "0")}  |  ${speaker}` : "DAEGU BUS ANALYSIS", 58, 7, 520, 20, {
    color: COLORS.panel,
    fontSize: 12,
    bold: true,
  });
  addText(slide, title, PAGE.left, 58, 780, 44, { fontSize: 32, bold: true });
  if (takeaway) {
    addText(slide, takeaway, PAGE.left, 104, 1060, 34, { fontSize: 16, color: COLORS.gray });
  }
  addLine(slide, PAGE.left, 148, PAGE.left + PAGE.width, 148, COLORS.border, 2);
}

function addFooter(slide, source = "자료: 정제된 대구 시내버스 CSV, 대구 월별 날씨 CSV", caution = "상관관계는 인과관계를 의미하지 않음") {
  addLine(slide, PAGE.left, 680, PAGE.left + PAGE.width, 680, COLORS.lightLine, 1);
  addText(slide, source, PAGE.left, 688, 760, 18, { fontSize: 9, color: COLORS.gray });
  addText(slide, caution, PAGE.left + 810, 688, 360, 18, { fontSize: 9, color: COLORS.gray, align: "right" });
}

function fitSize(text, base = 36, min = 23, threshold = 10) {
  const length = String(text).replace(/\s/g, "").length;
  if (length <= threshold) return base;
  return Math.max(min, base - (length - threshold) * 1.7);
}

function addKpi(slide, x, y, w, h, label, value, note = "", opts = {}) {
  const displayValue = String(value).replace(/(.{5,})\(/, "$1\n(");
  addRect(slide, x, y, w, h, opts.fill || COLORS.panelAlt, COLORS.border, "kpi-card");
  addRect(slide, x, y, w, 24, opts.strip || COLORS.blue, opts.strip || COLORS.blue, "kpi-strip");
  addText(slide, "COUNT.EXE", x + 10, y + 4, 110, 18, { fontSize: 12, color: COLORS.panel, bold: true });
  addText(slide, label, x + 14, y + 38, w - 28, 24, { fontSize: 14, bold: true });
  const screenTop = y + 64;
  const hasNote = Boolean(note);
  const screenHeight = Math.max(hasNote ? 42 : 44, h - (hasNote ? 100 : 78));
  const compactName = displayValue.replace(/\s/g, "").length > 10;
  const valueSize = opts.valueSize || fitSize(displayValue, compactName ? 30 : 34, 18, compactName ? 8 : 9);
  addRect(slide, x + 14, screenTop, w - 28, screenHeight, COLORS.black, COLORS.border, "kpi-screen");
  addText(slide, displayValue, x + 24, screenTop + 5, w - 48, screenHeight - 8, {
    fontSize: valueSize,
    color: opts.valueColor || COLORS.green,
    fontFamily: opts.mono ? MONO : FONT,
    bold: true,
  });
  if (note) addText(slide, note, x + 14, y + h - 20, w - 28, 16, { fontSize: 9, color: COLORS.gray });
}

function addBadge(slide, x, y, text, color = COLORS.yellow) {
  addRect(slide, x, y, 112, 26, color, COLORS.border, "badge");
  addText(slide, text, x + 8, y + 5, 96, 16, { fontSize: 11, bold: true, align: "center" });
}

function addTable(slide, x, y, w, h, headers, rows, opts = {}) {
  const table = slide.tables.add({
    rows: rows.length + 1,
    columns: headers.length,
    left: x,
    top: y,
    width: w,
    height: h,
    values: [headers, ...rows],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  try {
    table.borders.assign({ style: "solid", fill: COLORS.lightLine, width: 1 });
    for (let c = 0; c < headers.length; c += 1) {
      const cell = table.getCell(0, c);
      cell.fill = opts.headerFill || COLORS.navy;
      cell.text.style = { fontSize: opts.headerFontSize || 12, bold: true, color: COLORS.panel, fontFamily: FONT };
    }
    for (let r = 1; r <= rows.length; r += 1) {
      for (let c = 0; c < headers.length; c += 1) {
        const cell = table.getCell(r, c);
        cell.fill = r % 2 ? COLORS.panel : COLORS.panelAlt;
        cell.text.style = { fontSize: opts.fontSize || 11, color: COLORS.black, fontFamily: FONT };
      }
    }
  } catch {
    // 표는 native table로 유지하고, 스타일 API가 환경별로 다를 때만 기본 스타일을 사용합니다.
  }
  return table;
}

function chartText(size = 12) {
  return { fontSize: size, fill: COLORS.black };
}

function addBarChart(slide, x, y, w, h, title, categories, values, opts = {}) {
  addRect(slide, x, y, w, h, COLORS.panel, COLORS.border, "chart-frame");
  addText(slide, title, x + 20, y + 16, w - 40, 28, { fontSize: 20, bold: true });
  return slide.charts.add("bar", {
    position: { left: x + 36, top: y + 58, width: w - 72, height: h - 88 },
    categories,
    series: [{ name: opts.seriesName || "승차 인원", values, fill: opts.fill || COLORS.blue }],
    hasLegend: false,
    barOptions: { direction: opts.direction || "bar", grouping: "clustered", gapWidth: 55 },
    dataLabels: { showValue: true, position: "outEnd", textStyle: chartText(10) },
    xAxis: {
      title: opts.xTitle ? { text: opts.xTitle, textStyle: chartText(11) } : undefined,
      numberFormatCode: "#,##0",
      textStyle: chartText(10),
      majorGridlines: { style: "solid", fill: "#D0D0D0", width: 1 },
    },
    yAxis: { textStyle: chartText(10) },
  });
}

function addLineChart(slide, x, y, w, h, title, categories, series, opts = {}) {
  addRect(slide, x, y, w, h, COLORS.panel, COLORS.border, "chart-frame");
  addText(slide, title, x + 20, y + 16, w - 40, 28, { fontSize: 20, bold: true });
  return slide.charts.add("line", {
    position: { left: x + 38, top: y + 60, width: w - 76, height: h - 94 },
    categories,
    series,
    hasLegend: true,
    legend: { position: "bottom", textStyle: chartText(10) },
    lineOptions: { smooth: false },
    xAxis: { textStyle: chartText(10) },
    yAxis: {
      title: opts.yTitle ? { text: opts.yTitle, textStyle: chartText(11) } : undefined,
      numberFormatCode: "#,##0",
      textStyle: chartText(10),
      majorGridlines: { style: "solid", fill: "#D0D0D0", width: 1 },
    },
  });
}

function addScatter(slide, x, y, w, h, title, xValues, yValues, opts = {}) {
  addRect(slide, x, y, w, h, COLORS.panel, COLORS.border, "chart-frame");
  addText(slide, title, x + 20, y + 16, w - 40, 28, { fontSize: 20, bold: true });
  return slide.charts.add("scatter", {
    position: { left: x + 44, top: y + 62, width: w - 88, height: h - 94 },
    series: [{
      name: opts.seriesName || "정류소",
      xValues,
      values: yValues,
      fill: opts.fill || COLORS.blue,
      marker: { symbol: "circle", size: opts.markerSize || 5 },
    }],
    hasLegend: false,
    scatterOptions: { style: "marker" },
    xAxis: { title: { text: opts.xTitle || "", textStyle: chartText(11) }, textStyle: chartText(10), numberFormatCode: "#,##0" },
    yAxis: {
      title: { text: opts.yTitle || "", textStyle: chartText(11) },
      textStyle: chartText(10),
      numberFormatCode: "#,##0",
      majorGridlines: { style: "solid", fill: "#D0D0D0", width: 1 },
    },
  });
}

function addProcessStep(slide, index, title, body, x, y, w, h, color = COLORS.blue) {
  addRect(slide, x, y, w, h, COLORS.panel, COLORS.border, "process-card");
  addRect(slide, x, y, 42, h, color, COLORS.border, "process-number");
  addText(slide, String(index), x + 9, y + 18, 24, 30, { fontSize: 22, bold: true, color: COLORS.panel, align: "center", fontFamily: MONO });
  addText(slide, title, x + 56, y + 14, w - 70, 24, { fontSize: 18, bold: true });
  addText(slide, body, x + 56, y + 44, w - 70, h - 54, { fontSize: 13, color: COLORS.gray });
}

function addQuote(slide, text, x, y, w, h, color = COLORS.navy) {
  addRect(slide, x, y, w, h, COLORS.paper, color, "quote");
  addRect(slide, x, y, 10, h, color, color, "quote-bar");
  addText(slide, text, x + 24, y + 18, w - 44, h - 30, { fontSize: 21, bold: true, color: COLORS.black });
}

function seasonAverages(monthly) {
  const map = new Map();
  for (const row of monthly) {
    const season = cleanName(row.season);
    const item = map.get(season) || { sum: 0, count: 0 };
    item.sum += n(row.boardings);
    item.count += 1;
    map.set(season, item);
  }
  const order = ["봄", "여름", "가을", "겨울"];
  return order.map((season) => ({ season, value: (map.get(season)?.sum || 0) / (map.get(season)?.count || 1) }));
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function slideSpeaker(bodyNo) {
  if (bodyNo <= 5) return "발표자 1 | 데이터 정제와 기본 현황";
  if (bodyNo <= 10) return "발표자 2 | 시간대 패턴과 불균형 후보";
  return "발표자 3 | 장기 추세, 날씨, 활용 방안";
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const [stopRowsRaw, hourlyRows, imbalanceRowsRaw, monthlyRows, weatherCorrRows, busWeatherRows] = await Promise.all([
    readCsv("outputs/processed/stop_summary.csv"),
    readCsv("outputs/processed/hourly_summary.csv"),
    readCsv("outputs/processed/imbalance_candidates.csv"),
    readCsv("outputs/processed/monthly_summary.csv"),
    readCsv("outputs/processed/weather_correlation_results.csv"),
    readCsv("outputs/processed/bus_weather_monthly_merged.csv"),
  ]);

  const stopRows = stopRowsRaw.filter((row) => n(row.boardings) > 0);
  const topStops = top(stopRows, "boardings", 10);
  const topStopsForChart = [...topStops].reverse();
  const topPerRoute = top(stopRows.filter((row) => n(row.route_count) > 0), "boardings_per_route", 10);
  const districtAgg = sumBy(stopRows, "district", "boardings").slice(0, 8);
  const districtAvgPerRoute = avgBy(stopRows.filter((row) => n(row.boardings_per_route) > 0), "district", "boardings_per_route").slice(0, 8);
  const typeAgg = sumBy(stopRows.map((row) => ({ ...row, type_count: 1 })), "stop_type", "type_count");
  const totalBoardings = stopRows.reduce((acc, row) => acc + n(row.boardings), 0);
  const totalAlightings = stopRows.reduce((acc, row) => acc + n(row.alightings), 0);
  const busiest = topStops[0];
  const highestPerRoute = topPerRoute[0];
  const morningCase = top(stopRows.filter((row) => row.stop_type === "출근형"), "boardings", 1)[0];
  const eveningCase = top(stopRows.filter((row) => row.stop_type === "퇴근형"), "boardings", 1)[0];

  const hourMap = new Map();
  for (const row of hourlyRows) {
    const hour = n(row.hour);
    const item = hourMap.get(hour) || { boardings: 0, alightings: 0 };
    item.boardings += n(row.boardings);
    item.alightings += n(row.alightings);
    hourMap.set(hour, item);
  }
  const hourSeries = Array.from(hourMap, ([hour, item]) => ({ hour, ...item })).sort((a, b) => a.hour - b.hour);
  const peakHour = [...hourSeries].sort((a, b) => b.boardings - a.boardings)[0];
  const routeRows = stopRows.filter((row) => n(row.route_count) > 0 && n(row.boardings) > 0);
  const routeCorr = pearson(routeRows.map((row) => n(row.route_count)), routeRows.map((row) => n(row.boardings)));
  const demandThreshold = quantile(routeRows.map((row) => n(row.boardings)), 0.75);
  const routeThreshold = quantile(routeRows.map((row) => n(row.route_count)), 0.5);
  const perRouteThreshold = quantile(routeRows.map((row) => n(row.boardings_per_route)), 0.9);
  const imbalanceRows = imbalanceRowsRaw.filter((row) => n(row.boardings) > 0).slice(0, 10);

  const busWeather = busWeatherRows.map((row) => ({
    ...row,
    year: Math.round(n(row.year)),
    month: Math.round(n(row.month)),
    period: row.period,
    boardings: n(row.boardings),
    temp: n(row.monthly_avg_temp),
    rain: n(row.monthly_precipitation),
    humidity: n(row.monthly_avg_humidity),
    wind: n(row.monthly_avg_wind_speed),
  }));
  const last24 = busWeather.slice(-24);
  const weatherCorr = weatherCorrRows.slice(0, 8);
  const compare2026 = busWeather
    .filter((row) => row.year === 2026)
    .map((current) => {
      const previous = busWeather.find((row) => row.year === 2025 && row.month === current.month);
      const prevBoardings = previous ? previous.boardings : 0;
      const diff = current.boardings - prevBoardings;
      const growth = prevBoardings ? (diff / prevBoardings) * 100 : 0;
      return { month: `${current.month}월`, current: current.boardings, previous: prevBoardings, diff, growth };
    });
  const seasonAvg = seasonAverages(busWeather);
  const weatherPearsonTemp = pearson(busWeather.map((row) => row.temp), busWeather.map((row) => row.boardings));
  const weatherSpearmanTemp = spearman(busWeather.map((row) => row.temp), busWeather.map((row) => row.boardings));
  const weatherPearsonRain = pearson(busWeather.map((row) => row.rain), busWeather.map((row) => row.boardings));
  const weatherSpearmanRain = spearman(busWeather.map((row) => row.rain), busWeather.map((row) => row.boardings));

  const presentation = Presentation.create({ slideSize: SLIDE });

  const cover = presentation.slides.add();
  cover.background.fill = COLORS.bg;
  addRect(cover, 0, 0, SLIDE.width, SLIDE.height, COLORS.bg, COLORS.bg, "background");
  addRect(cover, 54, 52, 1172, 92, COLORS.navy, COLORS.border, "cover-title-strip");
  addText(cover, "대구 시내버스 정류소 이용 수요 및 노선 공급 불균형 분석", 82, 74, 1030, 48, {
    fontSize: 34,
    bold: true,
    color: COLORS.panel,
  });
  addText(cover, "편집 가능한 PowerPoint 발표 자료 | 본문 15장 + 표지 + 마무리", 82, 156, 760, 32, {
    fontSize: 18,
    color: COLORS.gray,
  });
  addQuote(cover, "정류소별 승차 수요와 경유 노선 수를 함께 보아, 추가 검토가 필요한 후보 정류소를 찾는 데이터 분석 프로젝트", 82, 224, 596, 168, COLORS.blue);
  addRect(cover, 720, 224, 420, 168, COLORS.panel, COLORS.border, "role-box");
  addText(cover, "3인 발표 분담", 746, 246, 260, 24, { fontSize: 22, bold: true });
  addText(cover, "발표자 1: 데이터 정제와 기본 현황\n발표자 2: 시간대 패턴과 불균형 후보\n발표자 3: 장기 추세, 날씨, 한계와 활용", 746, 292, 340, 82, { fontSize: 17 });
  addKpi(cover, 82, 454, 250, 120, "전체 승차 인원", fmt(totalBoardings), "", { valueSize: 30, mono: true });
  addKpi(cover, 360, 454, 250, 120, "분석 정류소", fmt(stopRows.length), "", { valueSize: 30, mono: true });
  addKpi(cover, 638, 454, 250, 120, "최고 피크 시간", `${peakHour.hour}시`, "", { valueSize: 30 });
  addKpi(cover, 916, 454, 250, 120, "최고 노선당 승차", cleanName(highestPerRoute.stop_name), "", { valueSize: 22 });
  addFooter(cover, "자료: outputs/processed/*.csv 정제 산출물", "최종 슬라이드는 텍스트, 표, 차트가 편집 가능한 객체");

  let s;

  s = presentation.slides.add();
  addHeader(s, 1, "프로젝트 질문과 역할 분담", "TOP 10 나열이 아니라 수요와 공급의 상대적 균형을 같이 본다.", slideSpeaker(1));
  addRect(s, 70, 178, 510, 390, COLORS.panel, COLORS.border);
  addText(s, "핵심 질문", 96, 200, 220, 30, { fontSize: 24, bold: true });
  addText(s, "1. 이용객이 많은 정류소는 어디인가?\n2. 출근·퇴근 시간 수요는 어디에 집중되는가?\n3. 구·군별 이용량은 어떻게 다른가?\n4. 경유 노선 수와 승차 수요는 함께 움직이는가?\n5. 수요 대비 노선 공급이 부족할 가능성이 있는 후보는 어디인가?", 96, 250, 430, 210, { fontSize: 18 });
  addQuote(s, "분석 결과는 노선 부족 확정이 아니라, 배차 간격·차량 크기·주변 시설 조사가 필요한 후보를 좁히는 근거다.", 96, 485, 430, 64, COLORS.red);
  addText(s, "발표자별 담당 구간", 650, 200, 360, 30, { fontSize: 24, bold: true });
  addProcessStep(s, 1, "발표자 1", "데이터 구조 확인, 정제 방식, 기본 지표와 구·군별 현황 설명", 650, 250, 470, 80, COLORS.blue);
  addProcessStep(s, 2, "발표자 2", "시간대 패턴, 정류소 유형, 수요·공급 불균형 후보 설명", 650, 350, 470, 80, COLORS.green);
  addProcessStep(s, 3, "발표자 3", "장기 추세, 날씨 연관 분석, 분석 한계와 정책 활용 방안 설명", 650, 450, 470, 80, COLORS.navy);
  addFooter(s);

  s = presentation.slides.add();
  addHeader(s, 2, "데이터 정제는 Streamlit 전에 끝낸다", "앱에서는 원본 CSV를 매번 바닥부터 정제하지 않고, 정제 산출물을 읽어 빠르게 보여준다.", slideSpeaker(2));
  const steps = [
    ["원본 CSV 점검", "파일명, 인코딩, 컬럼, 결측, 중복, 날짜 범위를 확인"],
    ["컬럼 매핑", "정류소ID·정류소명·승차·하차·좌표·노선 수 후보 컬럼을 실제 컬럼명에 연결"],
    ["자료형 정리", "숫자 쉼표 제거, 날짜 변환, 정류소 ID 문자열 통일, 음수·좌표 이상값 제거"],
    ["시간대 long 변환", "05시, 06시 같은 wide 시간대 컬럼은 hour, boardings 구조로 melt"],
    ["정류소 기준 병합", "정류소 ID 우선, 없으면 정류소명과 행정구역 보조키로 병합"],
    ["정제 CSV 저장", "stop_summary, hourly_summary, monthly_summary, imbalance_candidates 등으로 저장"],
  ];
  steps.forEach((item, idx) => {
    const col = idx % 2;
    const row = Math.floor(idx / 2);
    addProcessStep(s, idx + 1, item[0], item[1], 70 + col * 570, 185 + row * 125, 520, 92, idx < 3 ? COLORS.blue : COLORS.green);
  });
  addFooter(s, "정제 파일: outputs/processed/stop_summary.csv, hourly_summary.csv, monthly_summary.csv, bus_weather_monthly_merged.csv", "Streamlit 초기 로딩 속도 개선 목적");

  s = presentation.slides.add();
  addHeader(s, 3, "분석 지표 정의", "같은 정류소라도 ‘많이 탄다’와 ‘노선 하나가 많이 감당한다’는 의미가 다르다.", slideSpeaker(3));
  const metricRows = [
    ["전체 승차 인원", "정류소별 승차 합계", "핵심 수요 지표"],
    ["전체 이용객", "승차 + 하차", "하차 태그 누락 가능성 때문에 참고 지표"],
    ["노선당 승차 인원", "전체 승차 인원 / 경유 노선 수", "노선 1개가 감당하는 승차 밀도"],
    ["출근 집중도", "07~09시 승차 / 전체 승차 × 100", "아침 피크 의존도"],
    ["퇴근 집중도", "17~20시 승차 / 전체 승차 × 100", "저녁 피크 의존도"],
    ["승하차 불균형", "|승차-하차| / (승차+하차)", "승차와 하차 방향성 차이"],
  ];
  addTable(s, 76, 188, 1128, 318, ["지표", "계산 방식", "해석"], metricRows, { fontSize: 13, headerFontSize: 14 });
  addQuote(s, "노선당 승차 인원은 ‘혼잡 확정’이 아니라 경유 노선 수 대비 이용 수요가 높은 정류소를 찾기 위한 비교 지표다.", 120, 536, 1040, 70, COLORS.blue);
  addFooter(s);

  s = presentation.slides.add();
  addHeader(s, 4, "전체 현황 대시보드", "전체 승차 수요는 약령시건너와 16시 피크에 강하게 나타난다.", slideSpeaker(4));
  addKpi(s, 70, 176, 178, 118, "전체 승차", fmt(totalBoardings), "", { valueSize: 22, mono: true });
  addKpi(s, 266, 176, 178, 118, "전체 하차", fmt(totalAlightings), "", { valueSize: 22, mono: true });
  addKpi(s, 462, 176, 178, 118, "정류소 수", fmt(stopRows.length), "", { valueSize: 30, mono: true });
  addKpi(s, 658, 176, 260, 118, "최다 이용 정류소", cleanName(busiest.stop_name), "", { valueSize: 16 });
  addKpi(s, 936, 176, 140, 118, "혼잡 시간", `${peakHour.hour}시`, "", { valueSize: 32 });
  addKpi(s, 1092, 176, 134, 118, "상관", routeCorr.toFixed(2), corrLabel(routeCorr), { valueSize: 30, mono: true });
  addBarChart(
    s,
    70,
    330,
    706,
    300,
    "정류소별 승차 인원 TOP 10",
    topStopsForChart.map((row) => cleanName(row.stop_name)),
    topStopsForChart.map((row) => n(row.boardings)),
    { xTitle: "승차 인원", fill: COLORS.blue }
  );
  addBarChart(
    s,
    812,
    330,
    390,
    300,
    "구·군별 승차 인원",
    districtAgg.map((row) => row.name),
    districtAgg.map((row) => row.value),
    { direction: "column", fill: COLORS.navy }
  );
  addFooter(s);

  s = presentation.slides.add();
  addHeader(s, 5, "구·군별 이용량 차이", "달서구, 북구, 동구, 수성구, 중구 순으로 승차 규모가 크게 나타난다.", slideSpeaker(5));
  addBarChart(s, 70, 190, 650, 390, "구·군별 전체 승차 인원", districtAgg.map((row) => row.name), districtAgg.map((row) => row.value), {
    direction: "column",
    fill: COLORS.navy,
  });
  addTable(s, 760, 205, 420, 330, ["순위", "구·군", "승차 인원", "노선당 평균"], districtAgg.slice(0, 6).map((row, idx) => {
    const perRoute = districtAvgPerRoute.find((item) => item.name === row.name)?.value || 0;
    return [idx + 1, row.name, fmt(row.value), fmt(perRoute, 0)];
  }), { fontSize: 12 });
  addQuote(s, "구·군별 차이는 인구, 상권, 환승 지점, 학교·병원 등 주변 시설 분포와 함께 해석해야 한다.", 760, 552, 420, 64, COLORS.navy);
  addFooter(s);

  s = presentation.slides.add();
  addHeader(s, 6, "시간대별 수요 패턴", "대구 전체 승차 수요는 오후 16시에 가장 높고, 퇴근 시간대까지 높은 수준이 이어진다.", slideSpeaker(6));
  addLineChart(
    s,
    70,
    182,
    728,
    390,
    "시간대별 승차·하차 비교",
    hourSeries.map((row) => `${row.hour}시`),
    [
      { name: "승차", values: hourSeries.map((row) => row.boardings), fill: COLORS.blue, line: { style: "solid", fill: COLORS.blue, width: 2 }, marker: { symbol: "circle", size: 4 } },
      { name: "하차", values: hourSeries.map((row) => row.alightings), fill: COLORS.green, line: { style: "solid", fill: COLORS.green, width: 2 }, marker: { symbol: "circle", size: 4 } },
    ],
    { yTitle: "인원" }
  );
  addRect(s, 842, 190, 330, 130, COLORS.panel, COLORS.border);
  addText(s, "피크 시간대", 866, 212, 200, 24, { fontSize: 20, bold: true });
  addText(s, `${peakHour.hour}시`, 866, 250, 100, 46, { fontSize: 38, bold: true, color: COLORS.navy, fontFamily: MONO });
  addText(s, `${fmt(peakHour.boardings)}명 승차`, 980, 260, 160, 28, { fontSize: 18 });
  addRect(s, 842, 350, 330, 170, COLORS.paper, COLORS.border);
  addText(s, "해석 포인트", 866, 372, 180, 24, { fontSize: 20, bold: true });
  addText(s, "출근 피크만 보는 것이 아니라 오후 생활·퇴근 수요까지 함께 보아야 정류소 유형을 설명할 수 있다.", 866, 414, 260, 76, { fontSize: 17 });
  addFooter(s);

  s = presentation.slides.add();
  addHeader(s, 7, "정류소 유형은 집중도와 이용 규모로 분류한다", "출근형·퇴근형·생활형은 시간대별 승차 비율을 기준으로 붙인 해석용 이름이다.", slideSpeaker(7));
  const typeCriteria = [
    ["저이용형", "전체 승차 인원이 전체 정류소 하위 25%에 포함", "이용 규모가 작아 우선순위 판단 시 별도 구분"],
    ["출퇴근형", "출근 집중도 25% 이상 + 퇴근 집중도 25% 이상", "아침과 저녁 모두 피크가 뚜렷한 정류소"],
    ["출근형", "출근 집중도 25% 이상 + 출근 집중도가 퇴근보다 큼", "오전 7~9시에 승차 수요가 몰리는 정류소"],
    ["퇴근형", "퇴근 집중도 25% 이상 + 퇴근 집중도가 출근보다 큼", "오후 17~20시에 승차 수요가 몰리는 정류소"],
    ["생활형", "위 조건에 해당하지 않음", "특정 피크보다 하루 이용이 비교적 분산된 정류소"],
  ];
  addTable(s, 70, 188, 760, 330, ["유형", "분류 기준", "의미"], typeCriteria, { fontSize: 12 });
  addBarChart(s, 868, 190, 300, 330, "정류소 유형별 개수", typeAgg.map((row) => row.name), typeAgg.map((row) => row.value), {
    direction: "column",
    fill: COLORS.green,
    seriesName: "정류소 수",
  });
  addQuote(s, "유형명은 데이터 패턴을 읽기 쉽게 붙인 이름이며, 실제 토지 이용이나 통행 목적을 확정하지 않는다.", 116, 548, 980, 58, COLORS.red);
  addFooter(s);

  s = presentation.slides.add();
  addHeader(s, 8, "노선 수와 승차 수요의 관계", `Pearson 상관계수 ${routeCorr.toFixed(2)}: ${corrLabel(routeCorr)} 수준의 선형관계가 관찰된다.`, slideSpeaker(8));
  const scatterRows = [...routeRows].sort((a, b) => n(b.boardings) - n(a.boardings)).slice(0, 180);
  addScatter(
    s,
    70,
    184,
    720,
    390,
    "경유 노선 수와 전체 승차 인원 산점도",
    scatterRows.map((row) => n(row.route_count)),
    scatterRows.map((row) => n(row.boardings)),
    { xTitle: "경유 노선 수", yTitle: "승차 인원", markerSize: 5, fill: COLORS.blue }
  );
  addRect(s, 838, 190, 334, 130, COLORS.panel, COLORS.border);
  addText(s, "상관 해석", 864, 212, 160, 26, { fontSize: 22, bold: true });
  addText(s, `상관계수 ${routeCorr.toFixed(2)}는 두 변수가 어느 정도 함께 움직이는 경향을 의미한다.`, 864, 252, 250, 44, { fontSize: 16 });
  addRect(s, 838, 350, 334, 170, COLORS.paper, COLORS.border);
  addText(s, "주의", 864, 372, 100, 24, { fontSize: 22, bold: true, color: COLORS.red });
  addText(s, "노선 수가 많아서 승차 인원이 많아졌다고 말할 수는 없다. 상권·학교·환승센터 같은 외부 요인이 함께 작용할 수 있다.", 864, 414, 260, 78, { fontSize: 16 });
  addFooter(s);

  s = presentation.slides.add();
  addHeader(s, 9, "수요·공급 불균형 후보의 의미", "불균형은 ‘수요는 높은데 경유 노선 수가 상대적으로 적고 노선당 승차 밀도가 큰 상태’를 뜻한다.", slideSpeaker(9));
  addRect(s, 70, 188, 348, 300, COLORS.panel, COLORS.border);
  addText(s, "후보 기준", 96, 210, 160, 28, { fontSize: 24, bold: true });
  const criteriaRows = [
    ["수요", `승차 인원 Q75 이상`, `${fmt(demandThreshold)}명 이상`],
    ["공급", `경유 노선 수 Q50 이하`, `${fmt(routeThreshold)}개 이하`],
    ["밀도", `노선당 승차 Q90 이상`, `${fmt(perRouteThreshold)}명 이상`],
  ];
  addTable(s, 96, 260, 286, 132, ["축", "조건", "현재 기준"], criteriaRows, { fontSize: 10, headerFontSize: 10 });
  addRect(s, 96, 420, 286, 92, COLORS.paper, COLORS.red, "imbalance-caution");
  addRect(s, 96, 420, 10, 92, COLORS.red, COLORS.red, "imbalance-caution-bar");
  addText(s, "세 조건을 동시에 만족하면 추가 검토 후보로 표시한다. 실제 부족 여부는 배차 간격, 차량 크기, 현장 혼잡도 자료가 더 필요하다.", 120, 434, 236, 64, {
    fontSize: 14,
    bold: true,
  });
  addTable(s, 456, 188, 712, 370, ["순위", "정류소명", "구·군", "승차", "노선", "노선당"], imbalanceRows.slice(0, 8).map((row, idx) => [
    idx + 1,
    cleanName(row.stop_name),
    cleanName(row.district),
    fmt(row.boardings),
    fmt(row.route_count),
    fmt(row.boardings_per_route, 0),
  ]), { fontSize: 10, headerFontSize: 11 });
  addFooter(s, "자료: imbalance_candidates.csv", "후보 정류소는 노선 부족 확정이 아닌 추가 검토 대상");

  s = presentation.slides.add();
  addHeader(s, 10, "지도와 앱 화면은 후보를 공간적으로 확인하는 용도", "지도는 정류소 위치와 후보 분포를 보는 보조 화면이며, 판단은 표·지표와 함께 해야 한다.", slideSpeaker(10));
  const appScreen = path.join(ROOT, "outputs", "processed", "app_screen.png");
  try {
    const imageBytes = await fs.readFile(appScreen);
    s.images.add({
      blob: imageBytes,
      contentType: "image/png",
      alt: "Streamlit 대구 버스 분석 앱 화면 캡처",
      fit: "cover",
      position: { left: 72, top: 182, width: 704, height: 410 },
      crop: { left: 0, top: 0, right: 0, bottom: 0 },
    });
  } catch {
    addRect(s, 72, 182, 704, 410, COLORS.panel, COLORS.border);
    addText(s, "Streamlit 화면 캡처 파일을 찾을 수 없습니다.", 120, 350, 500, 40, { fontSize: 22, bold: true });
  }
  addRect(s, 818, 192, 340, 86, COLORS.panel, COLORS.border);
  addText(s, "지도 해석 방식", 842, 214, 180, 24, { fontSize: 22, bold: true });
  addText(s, "정류소 좌표가 있는 경우 지도에서 후보 위치를 확인한다.", 842, 248, 260, 20, { fontSize: 15 });
  addProcessStep(s, 1, "크기", "승차 인원이 많을수록 더 눈에 띄게 표시", 818, 308, 340, 70, COLORS.blue);
  addProcessStep(s, 2, "색상", "노선당 승차 인원이 높을수록 강조", 818, 394, 340, 70, COLORS.red);
  addProcessStep(s, 3, "필터", "구·군과 정류소 유형으로 후보를 좁혀 확인", 818, 480, 340, 70, COLORS.green);
  addFooter(s, "이미지: Streamlit 앱 화면 캡처", "지도만으로 혼잡이나 노선 부족을 확정하지 않음");

  s = presentation.slides.add();
  addHeader(s, 11, "장기 추세는 2025년 대비 2026년 같은 달로 비교한다", "2026년 4월까지의 월별 자료는 전년 같은 달과 비교해야 발표에서 쓸모가 있다.", slideSpeaker(11));
  addLineChart(
    s,
    70,
    184,
    690,
    360,
    "최근 24개월 월별 승차 인원",
    last24.map((row) => row.period),
    [{ name: "승차 인원", values: last24.map((row) => row.boardings), fill: COLORS.blue, line: { style: "solid", fill: COLORS.blue, width: 2 }, marker: { symbol: "circle", size: 4 } }],
    { yTitle: "승차 인원" }
  );
  addTable(s, 796, 200, 360, 260, ["월", "2026 승차", "2025 동월", "증감률"], compare2026.map((row) => [
    row.month,
    fmt(row.current),
    fmt(row.previous),
    pct(row.growth, 1),
  ]), { fontSize: 11, headerFontSize: 11 });
  const total2026 = compare2026.reduce((acc, row) => acc + row.current, 0);
  const total2025 = compare2026.reduce((acc, row) => acc + row.previous, 0);
  const totalGrowth = total2025 ? ((total2026 - total2025) / total2025) * 100 : 0;
  addQuote(s, `2026년 1~4월 합계는 전년 동기간 대비 ${pct(totalGrowth, 1)} 변화했다. 월별 증감은 같은 달끼리 비교한다.`, 796, 488, 360, 68, totalGrowth >= 0 ? COLORS.green : COLORS.red);
  addFooter(s, "자료: bus_weather_monthly_merged.csv", "월별 자료는 일별 이벤트의 즉각적 영향을 분리하기 어려움");

  s = presentation.slides.add();
  addHeader(s, 12, "날씨와 버스 이용의 연관성", "관측 76개월 기준, 날씨 변수와 승차 인원의 상관은 전반적으로 약하거나 다소 약하다.", slideSpeaker(12));
  addScatter(
    s,
    70,
    184,
    492,
    292,
    "월평균 기온과 승차 인원",
    busWeather.map((row) => row.temp),
    busWeather.map((row) => row.boardings),
    { xTitle: "월평균 기온(°C)", yTitle: "승차 인원", fill: COLORS.red, markerSize: 4 }
  );
  addScatter(
    s,
    590,
    184,
    492,
    292,
    "월 누적 강수량과 승차 인원",
    busWeather.map((row) => row.rain),
    busWeather.map((row) => row.boardings),
    { xTitle: "월 누적 강수량(mm)", yTitle: "승차 인원", fill: COLORS.blue, markerSize: 4 }
  );
  addTable(s, 70, 500, 650, 126, ["변수", "Pearson", "Spearman", "해석"], [
    ["월평균 기온", weatherPearsonTemp.toFixed(2), weatherSpearmanTemp.toFixed(2), corrLabel(weatherPearsonTemp)],
    ["월 누적 강수량", weatherPearsonRain.toFixed(2), weatherSpearmanRain.toFixed(2), corrLabel(weatherPearsonRain)],
  ], { fontSize: 12, headerFontSize: 12 });
  addBarChart(s, 754, 488, 368, 150, "계절별 월평균 승차", seasonAvg.map((row) => row.season), seasonAvg.map((row) => row.value), {
    direction: "column",
    fill: COLORS.green,
  });
  addFooter(s, "자료: weather_correlation_results.csv, bus_weather_monthly_merged.csv", "비 때문에 이용객이 변했다고 단정하지 않음");

  s = presentation.slides.add();
  addHeader(s, 13, "분석 한계와 보완 방법", "데이터에 없는 조건은 한계로 명시하고, 후속 자료로 줄일 수 있다.", slideSpeaker(13));
  addTable(s, 70, 184, 1110, 382, ["한계", "왜 문제가 되는가", "줄이는 방법"], [
    ["하차 태그 누락", "실제 하차 인원보다 적게 집계될 수 있음", "승차 중심 지표로 해석하고 하차는 보조 지표로 사용"],
    ["현금 승차 제외 가능성", "이용량이 과소 집계될 수 있음", "교통카드 외 자료 확보 시 비교"],
    ["노선 수와 배차 횟수 차이", "노선 수가 많아도 배차가 적을 수 있음", "배차 간격, 운행 횟수, 차량 크기 자료 결합"],
    ["공휴일·방학·행사", "특정 월의 이용량을 흔들 수 있음", "공휴일 수, 학사일정, 지역 행사 더미 변수 추가"],
    ["노선 개편·코로나19", "외부 요인으로 추세가 바뀔 수 있음", "개편 시점과 사회적 거리두기 기간을 별도 표시"],
    ["월별 날씨 자료", "개별 강수일의 즉각 영향 확인 어려움", "일별 승차 데이터 확보 시 일 단위 분석으로 전환"],
  ], { fontSize: 10, headerFontSize: 11 });
  addQuote(s, "현재 분석은 ‘정책 판단의 출발점’이며, 현장 조사와 운영 자료가 결합될 때 더 강한 근거가 된다.", 112, 590, 1010, 52, COLORS.navy);
  addFooter(s);

  s = presentation.slides.add();
  addHeader(s, 14, "정책적 활용 가능성", "후보 정류소를 정한 뒤 운영 자료와 현장 정보를 결합하면 검토 우선순위를 만들 수 있다.", slideSpeaker(14));
  const roadmap = [
    ["1차 탐색", "승차 수요·노선 수·노선당 승차 인원으로 후보 정류소 선정"],
    ["2차 검증", "배차 간격, 운행 횟수, 차량 크기, 실제 혼잡도 자료 결합"],
    ["현장 맥락", "학교, 병원, 상권, 환승센터, 신규 주거지 등 주변 시설 확인"],
    ["정책 실행", "배차 조정, 노선 경유 검토, 정류소 시설 개선, 정보 제공 강화"],
  ];
  roadmap.forEach((item, idx) => {
    addProcessStep(s, idx + 1, item[0], item[1], 100 + idx * 290, 220, 246, 160, [COLORS.blue, COLORS.green, COLORS.navy, COLORS.red][idx]);
    if (idx < 3) addLine(s, 346 + idx * 290, 300, 390 + idx * 290, 300, COLORS.border, 2);
  });
  addRect(s, 120, 450, 1000, 98, COLORS.paper, COLORS.border);
  addText(s, "발표에서 강조할 문장", 150, 472, 220, 24, { fontSize: 22, bold: true });
  addText(s, "“수요·공급 불균형 후보는 노선 부족을 확정하는 결과가 아니라, 어느 정류소부터 추가 자료와 현장 검토를 붙여야 하는지 알려주는 우선순위 목록입니다.”", 150, 508, 880, 34, { fontSize: 18 });
  addFooter(s);

  s = presentation.slides.add();
  addHeader(s, 15, "핵심 결과 요약", "대구 버스 정류소 분석은 이용 규모, 시간 집중, 노선당 밀도, 날씨·추세를 함께 본다.", slideSpeaker(15));
  addKpi(s, 74, 188, 260, 130, "최다 이용 정류소", cleanName(busiest.stop_name), fmt(busiest.boardings), { valueSize: 16 });
  addKpi(s, 362, 188, 220, 130, "가장 혼잡한 시간", `${peakHour.hour}시`, `${fmt(peakHour.boardings)}명`, { valueSize: 38 });
  addKpi(s, 610, 188, 296, 130, "출근형 대표", cleanName(morningCase?.stop_name), fmt(morningCase?.boardings), { valueSize: 18 });
  addKpi(s, 934, 188, 260, 130, "퇴근형 대표", cleanName(eveningCase?.stop_name), fmt(eveningCase?.boardings), { valueSize: 16 });
  addTable(s, 74, 360, 560, 180, ["구분", "대표 결과"], [
    ["노선당 승차 인원 최고", `${cleanName(highestPerRoute.stop_name)} (${fmt(highestPerRoute.boardings_per_route, 0)}명/노선)`],
    ["불균형 후보 예시", imbalanceRows.slice(0, 3).map((row) => cleanName(row.stop_name)).join(", ")],
    ["날씨 연관", "대체로 약함~다소 약함, 인과관계 아님"],
  ], { fontSize: 12 });
  addQuote(s, "결론: 높은 수요, 시간대 집중, 낮은 경유 노선 수가 겹치는 정류소를 추가 검토 후보로 제시할 수 있다.", 690, 370, 468, 104, COLORS.blue);
  addFooter(s);

  const closing = presentation.slides.add();
  closing.background.fill = COLORS.bg;
  addRect(closing, 0, 0, SLIDE.width, 34, COLORS.navy, COLORS.navy);
  addText(closing, "CLOSING", 58, 7, 180, 20, { color: COLORS.panel, fontSize: 12, bold: true });
  addText(closing, "핵심 결론과 Q&A", 84, 86, 720, 54, { fontSize: 44, bold: true });
  addLine(closing, 84, 160, 1140, 160, COLORS.border, 2);
  addRect(closing, 84, 210, 330, 228, COLORS.panel, COLORS.border);
  addText(closing, "1", 110, 234, 40, 44, { fontSize: 38, bold: true, color: COLORS.blue, fontFamily: MONO });
  addText(closing, "정제 산출물 기반", 166, 238, 200, 28, { fontSize: 22, bold: true });
  addText(closing, "Streamlit은 정제된 CSV를 읽어 시각화하며, 원본 컬럼명 차이는 매핑 과정에서 흡수했다.", 166, 288, 190, 74, { fontSize: 16 });
  addRect(closing, 474, 210, 330, 228, COLORS.panel, COLORS.border);
  addText(closing, "2", 500, 234, 40, 44, { fontSize: 38, bold: true, color: COLORS.green, fontFamily: MONO });
  addText(closing, "불균형은 후보", 556, 238, 200, 28, { fontSize: 22, bold: true });
  addText(closing, "수요가 높고 노선당 승차 밀도가 큰 정류소를 찾되, 노선 부족으로 단정하지 않는다.", 556, 288, 190, 74, { fontSize: 16 });
  addRect(closing, 864, 210, 330, 228, COLORS.panel, COLORS.border);
  addText(closing, "3", 890, 234, 40, 44, { fontSize: 38, bold: true, color: COLORS.red, fontFamily: MONO });
  addText(closing, "후속 자료가 중요", 946, 238, 210, 28, { fontSize: 22, bold: true });
  addText(closing, "배차 간격, 차량 크기, 혼잡도, 주변 시설, 행사·방학 변수를 붙이면 활용성이 커진다.", 946, 288, 190, 74, { fontSize: 16 });
  addText(closing, "질문 감사합니다", 432, 530, 420, 46, { fontSize: 38, bold: true, align: "center" });
  addFooter(closing, "최종 파일: outputs/daegu_bus_analysis_editable.pptx", "본문 15장: 발표자 1·2·3 각 5장");

  const sourceNotes = [
    "Source notes for editable PPTX",
    "- DESIGN.md: retro transit dashboard visual system.",
    "- slide_plan.json and slide_prompts.json: slide role and narrative reference.",
    "- outputs/processed/stop_summary.csv: stop-level demand, route count, stop type, location.",
    "- outputs/processed/hourly_summary.csv: stop-hour boardings and alightings.",
    "- outputs/processed/imbalance_candidates.csv: additional-review candidate stops.",
    "- outputs/processed/bus_weather_monthly_merged.csv: monthly bus demand and Daegu weather.",
    "- outputs/processed/weather_correlation_results.csv: Pearson/Spearman weather correlation summaries.",
    "- outputs/processed/app_screen.png: Streamlit implementation screenshot.",
    "- Claims avoid causal language for correlation and route supply.",
  ].join("\n");
  await fs.writeFile(path.join(TMP_DIR, "source-notes.txt"), sourceNotes, "utf8");

  const planText = [
    "Slide plan: editable PPTX, 17 slides total.",
    "Cover and closing are outside the body count.",
    "Body 1-5: Speaker 1, data cleaning and baseline demand.",
    "Body 6-10: Speaker 2, hourly pattern, stop type, imbalance, map/app view.",
    "Body 11-15: Speaker 3, long-term trend, weather correlation, limitations, policy use, synthesis.",
    "All titles, body text, KPI cards, tables, charts are editable PowerPoint objects.",
    "Only app screenshot is inserted as an image; no full-slide bitmap backgrounds.",
  ].join("\n");
  await fs.writeFile(path.join(TMP_DIR, "slide-plan.txt"), planText, "utf8");

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(PREVIEW_DIR, `${stem}.png`), png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(LAYOUT_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(PREVIEW_DIR, "deck-montage.webp"), montage);
  await writeBlob(PREVIEW_COPY, montage);

  const snapshot = await presentation.inspect({ kind: "slide,textbox,shape,image,table,chart", maxChars: 16000 });
  await fs.writeFile(path.join(QA_DIR, "inspect.ndjson"), snapshot.ndjson, "utf8");
  const qaText = [
    "Visual QA ledger",
    "- Rendered every slide to PNG.",
    "- Rendered deck montage.",
    "- Exported layout JSON for every slide.",
    "- Confirmed deck uses native tables/charts through artifact-tool object creation.",
    "- App screenshot is the only inserted raster evidence image.",
    "- No full-slide bitmap backgrounds were inserted.",
    `- Final PPTX path: ${FINAL_PPTX}`,
  ].join("\n");
  await fs.writeFile(path.join(QA_DIR, "visual-qa.txt"), qaText, "utf8");

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);

  console.log(JSON.stringify({
    finalPptx: FINAL_PPTX,
    previewMontage: PREVIEW_COPY,
    scratchWorkspace: WORKSPACE,
    slideCount: presentation.slides.items.length,
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
