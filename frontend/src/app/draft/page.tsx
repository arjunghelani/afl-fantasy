"use client";

import { useEffect, useMemo, useState } from "react";

/* ------------------------- types ------------------------- */
type DraftPick = {
  year: number;
  team_id: number;
  team_name: string;
  owner?: string | null;
  round_num: number | null;
  pick_num: number | null;       // round pick (slot)
  overall_pick: number | null;   // overall pick
  player_name: string;
  position?: string | null;
  pro_team?: string | null;
  is_keeper: boolean;
  auction_price?: number | null;
};

type DraftResponse = {
  year: number;
  league_id: number;
  picks: DraftPick[];
};

type VorpRow = {
  player_name: string;
  team?: string | null;
  fantasy_pos: string;
  g?: number | null;
  fantasy_points_ppr: number;
  vorp_star: number;
  vorp_star_extrap?: number | null;   // NEW
  partial_season?: boolean | null;    // optional (ok if backend doesn’t send)
  vorp_star_rank_overall: number;
  vorp_star_rank_pos: number;
};

type VorpMap = Record<
  string,
  {
    vorp_star: number;
    vorp_star_extrap?: number | null; // NEW
    fantasy_pos: string;
    g?: number | null;                // NEW
  }
>;


type VorpResponse = {
  year: number;
  players: VorpRow[];
  count: number;
  used_ppg: boolean;
};

// type VorpMap = Record<
//   string, // normalized player name
//   { vorp_star: number; fantasy_pos: string }
// >;

/* ---- NEW: relative (smoothed baseline) types ---- */
type RelativePoint = { overall_pick: number; expected: number; n: number; mad?: number | null };
type PlayerResidual = {
  player_name: string;
  team_name: string;
  position?: string | null;
  overall_pick: number;
  vorp_star?: number | null;
  expected?: number | null;
  residual?: number | null;
  z?: number | null;
};
type VorpRelativeResponse = {
  year: number;
  league_id: number;
  team_count: number;
  max_pick: number;
  curve: RelativePoint[];
  players: PlayerResidual[];
};

/* ------------------------- config ------------------------- */
const YEARS = [2020, 2021, 2022, 2024];
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
const REL_WINDOW = 7;
const REL_VERSION = 1; // bump when backend logic changes
const relKeyFor = (year: number) => `${year}|w=${REL_WINDOW}|v=${REL_VERSION}`;


/* ------------------------- helpers ------------------------- */
const normalizeName = (raw: string) => {
  // more forgiving normalizer to improve name matching
  let s = raw.trim().toLowerCase();
  s = s.normalize("NFD").replace(/\p{Diacritic}/gu, "");      // strip accents
  s = s.replace(/[.'’`,\-]/g, " ");                            // punctuation -> space
  s = s.replace(/\b(jr|sr|ii|iii|iv|v)\b/g, "");               // suffixes
  s = s.replace(/\s+/g, " ").trim();                           // collapse spaces
  return s;
};

/** clamp a number into [min, max] */
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/** map value → color (hex). Diverging palette: red → yellow → green */
/** map value → color (hex). White → light green → deep green */
/** map value → color (hex). Diverging: magenta → white → green */
function vorpToColor(v: number, absMax = 6) {
  // normalize to [-1, 1]
  const t = clamp(v, -absMax, absMax) / absMax;

  // gradient stops (position in [-1,1], [r,g,b])
  const stops: Array<{ x: number; c: [number, number, number] }> = [
    { x: -1,   c: [58, 25, 112] },   // deep magenta
    { x: -0.5, c: [128, 73, 215] },  //rgb(128, 73, 215)
    { x:  0,   c: [255, 255, 255] }, // white
    { x:  0.5, c: [22, 163, 74] },   // #16a34a
    { x:  1,   c: [4, 77, 31] },     // deep green
  ];

  // find segment containing t
  let i = 0;
  while (i < stops.length - 1 && t > stops[i + 1].x) i++;

  const a = stops[i];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  const u = (t - a.x) / (b.x - a.x); // 0..1 within segment

  const lerp = (A: number, B: number, U: number) => A + (B - A) * U;
  const r = lerp(a.c[0], b.c[0], u);
  const g = lerp(a.c[1], b.c[1], u);
  const bch = lerp(a.c[2], b.c[2], u);

  const toHex = (n: number) => Math.round(n).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(bch)}`;
}



/** choose black/white text for contrast against a hex background */
function textColorFor(bgHex: string) {
  const n = bgHex.replace("#", "");
  const r = parseInt(n.slice(0, 2), 16);
  const g = parseInt(n.slice(2, 4), 16);
  const b = parseInt(n.slice(4, 6), 16);
  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return luminance > 0.62 ? "text-zinc-900" : "text-white";
}

/** small signed number formatter (for Δ badge) */
function fmtSigned(x: number | null | undefined, digits = 1) {
  if (x == null || !Number.isFinite(x)) return null;
  const s = x.toFixed(digits);
  return x >= 0 ? `+${s}` : s;
}

/* ------------------------- data fetchers ------------------------- */
async function fetchDraft(year: number): Promise<DraftResponse> {
  const res = await fetch(`${API_BASE}/draft/${year}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch draft for ${year}`);
  return res.json();
}

async function fetchVorpMap(year: number): Promise<VorpMap> {
  const res = await fetch(`${API_BASE}/metrics/vorp/${year}?top=500`, { cache: "no-store" });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`VORP* ${year} failed: ${res.status} ${res.statusText} ${txt}`);
  }
  const json: VorpResponse = await res.json();
  const map: VorpMap = {};
  for (const p of json.players) {
    map[normalizeName(p.player_name)] = {
      vorp_star: p.vorp_star,
      vorp_star_extrap: (p as any).vorp_star_extrap ?? null, // tolerate older backend
      fantasy_pos: p.fantasy_pos,
      g: p.g ?? null,
    };
  }
  return map;
}



/* ---- NEW: fetch relative (smoothed) payload ---- */
async function fetchVorpRelative(year: number, window = 7): Promise<VorpRelativeResponse> {
  const res = await fetch(`${API_BASE}/metrics/vorp_relative/${year}?window=${window}`, { cache: "no-store" });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`VORP relative ${year} failed: ${res.status} ${res.statusText} ${txt}`);
  }
  return res.json();
}

/* ===============================================================
   PAGE
   =============================================================== */
type ColorMode = "position" | "vorp" | "vorp_rel";

export default function DraftsPage() {
  const [year, setYear] = useState<number>(YEARS[0]);
  const [data, setData] = useState<DraftResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [view, setView] = useState<"board" | "list">("board");

  // Color modes
  const [colorMode, setColorMode] = useState<ColorMode>("position");

  // VORP* (raw) state
  const [vorpLoading, setVorpLoading] = useState(false);
  const [vorpError, setVorpError] = useState<string | null>(null);
  const [vorpCache, setVorpCache] = useState<Record<number, VorpMap | undefined>>({});
  const vorpMap = vorpCache[year];

  // NEW: relative baseline state
  const [relLoading, setRelLoading] = useState(false);
  const [relError, setRelError] = useState<string | null>(null);
  const [relCache, setRelCache] = useState<Record<string, VorpRelativeResponse | undefined>>({});
  const relKey = relKeyFor(year);
  const rel = relCache[relKey];

  // draft fetch on year change
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchDraft(year)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [year]);

  // prefetch VORP on year change
  useEffect(() => {
    let cancelled = false;
    setVorpError(null);
    if (vorpCache[year]) return;
    setVorpLoading(true);
    fetchVorpMap(year)
      .then((map) => {
        if (!cancelled) setVorpCache((prev) => ({ ...prev, [year]: map }));
      })
      .catch((e) => {
        if (!cancelled) setVorpError(e instanceof Error ? e.message : String(e));
        console.warn("VORP prefetch failed:", e);
      })
      .finally(() => {
        if (!cancelled) setVorpLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [year, vorpCache]);

  // NEW: prefetch relative baseline on year change
  useEffect(() => {
    let cancelled = false;
    setRelError(null);
    if (relCache[relKey]) return;
    setRelLoading(true);
    fetchVorpRelative(year, 7)
      .then((payload) => {
        if (!cancelled) setRelCache((prev) => ({ ...prev, [relKey]: payload }));
      })
      .catch((e) => {
        if (!cancelled) setRelError(e instanceof Error ? e.message : String(e));
        console.warn("VORP relative failed:", e);
      })
      .finally(() => {
        if (!cancelled) setRelLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [year, relCache, relKey]);

  // Group by round for board layout
  const rounds = useMemo(() => {
    const by: Record<number, DraftPick[]> = {};
    for (const p of data?.picks ?? []) {
      const r = p.round_num ?? 0;
      if (!by[r]) by[r] = [];
      by[r].push(p);
    }
    Object.values(by).forEach((arr) =>
      arr.sort((a, b) => {
        const pa = a.pick_num ?? 999, pb = b.pick_num ?? 999;
        if (pa !== pb) return pa - pb;
        const oa = a.overall_pick ?? 9999, ob = b.overall_pick ?? 9999;
        return oa - ob;
      })
    );
    return (Object.entries(by)
      .map(([k, v]) => [Number(k), v] as [number, DraftPick[]])
      .sort((a, b) => a[0] - b[0])) as [number, DraftPick[]][];
  }, [data]);

  // handlers to enter modes safely
  const switchToVorp = async () => {
    setVorpError(null);
    if (!vorpCache[year]) {
      try {
        setVorpLoading(true);
        const map = await fetchVorpMap(year);
        setVorpCache((prev) => ({ ...prev, [year]: map }));
      } catch (e) {
        setVorpError(e instanceof Error ? e.message : String(e));
        return; // stay in position mode
      } finally {
        setVorpLoading(false);
      }
    }
    setColorMode("vorp");
  };

  const switchToVorpRel = async () => {
    setRelError(null);
    if (!relCache[relKey]) {
      try {
        setRelLoading(true);
        const payload = await fetchVorpRelative(year, 7);
        setRelCache((prev) => ({ ...prev, [relKey]: payload }));
      } catch (e) {
        setRelError(e instanceof Error ? e.message : String(e));
        return;
      } finally {
        setRelLoading(false);
      }
    }
    setColorMode("vorp_rel");
  };

  return (
  <main className="mx-auto max-w-none p-6 space-y-8 bg-slate-50 min-h-screen dark:bg-[#0b0f13]">
    {/* Top bar: title + nav (emerald like Home) */}
    <div className="rounded-xl bg-emerald-700 text-white px-4 py-3 flex items-center justify-between shadow-sm">
      <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
        League 86952922 Drafts
      </h1>

      <nav className="flex items-center gap-5">
        <a
          href="/"
          className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline"
        >
          View Standings
        </a>
        <a
          href="/playoffs"
          className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline"
        >
          View Playoffs
        </a>
        <a
          href="/players"
          className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline"
        >
          Player VORP (all years)
        </a>

      </nav>
    </div>

    {/* Controls card */}
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-slate-900/80 p-0">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        {/* Left label */}
        <div className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">
          Draft Controls
        </div>

        {/* Right controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Year */}
          <label className="text-sm text-zinc-600 dark:text-zinc-300">Year</label>
          <select
            className="rounded-lg border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-slate-900 px-3 py-2"
            value={year}
            onChange={(e) => setYear(Number(e.target.value))}
          >
            {YEARS.map((y) => (
              <option value={y} key={y}>
                {y}
              </option>
            ))}
          </select>

          {/* View toggle */}
          <div className="inline-flex rounded-lg overflow-hidden border border-zinc-300 dark:border-zinc-700">
            <button
              className={`px-3 py-2 text-sm ${
                view === "board"
                  ? "bg-emerald-700 text-white"
                  : "bg-white dark:bg-slate-900 text-zinc-800 dark:text-zinc-200"
              }`}
              onClick={() => setView("board")}
            >
              Board
            </button>
            <button
              className={`px-3 py-2 text-sm ${
                view === "list"
                  ? "bg-emerald-700 text-white"
                  : "bg-white dark:bg-slate-900 text-zinc-800 dark:text-zinc-200"
              }`}
              onClick={() => setView("list")}
            >
              List
            </button>
          </div>

          {/* Color-by toggle */}
          <div className="inline-flex rounded-lg overflow-hidden border border-zinc-300 dark:border-zinc-700">
            <button
              className={`px-3 py-2 text-sm ${
                colorMode === "position"
                  ? "bg-emerald-700 text-white"
                  : "bg-white dark:bg-slate-900 text-zinc-800 dark:text-zinc-200"
              }`}
              onClick={() => setColorMode("position")}
            >
              Position
            </button>
            <button
              className={`px-3 py-2 text-sm ${
                colorMode === "vorp"
                  ? "bg-emerald-700 text-white"
                  : "bg-white dark:bg-slate-900 text-zinc-800 dark:text-zinc-200"
              }`}
              disabled={vorpLoading}
              onClick={switchToVorp}
              title={
                vorpLoading
                  ? "Loading WAR…"
                  : !vorpMap
                  ? "Fetches VORP* for this year"
                  : ""
              }
            >
              {vorpLoading ? "Loading…" : "WAR"}
            </button>
            {/* If you re-enable relative mode, style it the same */}
            {/* <button
              className={`px-3 py-2 text-sm ${
                colorMode === "vorp_rel"
                  ? "bg-emerald-700 text-white"
                  : "bg-white dark:bg-slate-900 text-zinc-800 dark:text-zinc-200"
              }`}
              disabled={relLoading}
              onClick={switchToVorpRel}
            >
              VORP vs Pick
            </button> */}
          </div>
        </div>
      </div>
    </div>

    {/* Legends + errors (styled to match) */}
    {colorMode === "vorp" && (
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-white dark:bg-slate-900/80">
        <div className="flex items-center gap-3 text-xs text-zinc-600 dark:text-zinc-300">
          <span>low value</span>
          <div
            className="h-2 w-40 rounded"
            style={{
              background:
                "linear-gradient(90deg, rgb(58,25,112) 0%, rgb(128,73,215) 25%, #ffffff 50%, #16a34a 75%, rgb(4,77,31) 100%)",
            }}
          />
          <span className="font-medium text-zinc-700 dark:text-zinc-200">
                high value
            </span>
        </div>
      </div>
    )}

    {colorMode === "vorp_rel" && (
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-white dark:bg-slate-900/80">
        <div className="flex items-center gap-3 text-xs text-zinc-600 dark:text-zinc-300">
          <span>Vs Pick Expectation:</span>
          <div
            className="h-2 w-40 rounded"
            style={{
              background:
                "linear-gradient(90deg, rgb(58,25,112) 0%, rgb(128,73,215) 25%, #ffffff 50%, #16a34a 75%, rgb(4,77,31) 100%)",
            }}
          />
          <span className="opacity-70">below expected</span>
          <span className="opacity-70 ml-auto">above expected</span>
        </div>
      </div>
    )}

    {loading && (
      <div className="text-zinc-600 dark:text-zinc-300">Loading {year} draft…</div>
    )}
    {error && <div className="text-red-600">{error}</div>}

    {!loading && !error && data && (
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-slate-900/80 p-2">
        {view === "board" ? (
          <DraftBoard
            rounds={rounds}
            colorMode={colorMode}
            vorpMap={vorpMap}
            rel={rel}
          />
        ) : (
          <DraftList picks={data.picks} />
        )}
      </div>
    )}
  </main>
);

}

/* ===============================================================
   BOARD
   =============================================================== */
// Replace your entire DraftBoard component with this:

function DraftBoard({
  rounds,
  colorMode,
  vorpMap,
  rel,
}: {
  rounds: [number, DraftPick[]][];
  colorMode: "position" | "vorp" | "vorp_rel";
  vorpMap?: VorpMap;
  rel?: VorpRelativeResponse;
}) {
  const GUTTER = 84;
  const COL_MIN = 95;
  const CELL_MIN_H = 70;

  const positionColors: Record<string, string> = {
    QB: "bg-[#3a78a7] text-black",
    RB: "bg-[#42a58c] text-black",
    WR: "bg-[#cf853f] text-black",
    TE: "bg-[#84579f] text-black",
    K:  "bg-[#a8c259] text-black",
    DEF:"bg-[#aaaaaa] text-black",
  };
  const normPos = (raw?: string | null) => {
    if (!raw) return "";
    const s = raw.toUpperCase();
    if (s === "D/ST" || s === "DST" || s === "DEFENSE") return "DEF";
    if (s === "PK" || s === "KICKER") return "K";
    return s;
  };

  function getRow(p: DraftPick) {
    if (!vorpMap) return undefined;
    return vorpMap[normalizeName(p.player_name)];
  }
  function getColorInfo(p: DraftPick) {
    const row = getRow(p);
    if (!row) return { isPartial: false, shadeHex: "#ffffff", value: undefined, g: undefined };
    const g = row.g ?? undefined;
    const isPartial = typeof g === "number" && g >= 3 && g < 12 && row.vorp_star_extrap != null;
    const value = isPartial ? (row.vorp_star_extrap as number) : row.vorp_star;
    const shadeHex = vorpToColor(value);
    return { isPartial, shadeHex, value, g };
  }
  function gamesPct(g?: number): number {
    if (!g || g <= 0) return 0;
    return Math.max(0, Math.min(100, (g / 17) * 100));
  }
  function findResidual(p: DraftPick): number | undefined {
    if (!rel || !p.overall_pick) return undefined;
    const row = rel.players.find((x) => x.overall_pick === p.overall_pick);
    return row?.residual ?? undefined;
  }

  const round1 = rounds.find(([r]) => r === 1)?.[1] ?? [];
  const round1Sorted = [...round1].sort((a, b) => (a.pick_num ?? 999) - (b.pick_num ?? 999));
  const headerTeams = round1Sorted.map((p) => p.team_name);
  const maxPickSeen = Math.max(0, ...rounds.flatMap(([, ps]) => ps.map((p) => p.pick_num ?? 0)));
  const teamCount = Math.max(headerTeams.length, maxPickSeen, 12);
  const gridTemplate = `${GUTTER}px repeat(${teamCount}, minmax(${COL_MIN}px, 1fr))`;

  return (
    <div className="relative w-full rounded-md border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-auto">
      {/* Header */}
      <div className="sticky top-0 z-30 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
        <div className="grid" style={{ gridTemplateColumns: gridTemplate }}>
          <div className="h-8 px-2 flex items-center text-[10px] font-semibold text-zinc-500 uppercase">Round</div>
          {Array.from({ length: teamCount }).map((_, idx) => (
            <div
              key={`hdr-slot-${idx}`}
              className="h-auto min-h-8 px-1 py-1 flex items-center justify-center border-l border-zinc-200 dark:border-zinc-800"
            >
              <span className="text-xs font-semibold text-center break-words leading-tight">
                {headerTeams[idx] ?? `Slot ${idx + 1}`}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Body */}
      <div>
        {rounds.map(([roundNum, picks]) => {
          const slots: (DraftPick | null)[] = Array(teamCount).fill(null);
          for (const p of picks) {
            const rp = p.pick_num ?? 0;
            const idx = rp > 0 && rp <= teamCount ? rp - 1 : -1;
            if (idx >= 0) slots[idx] = p;
          }
          const display = roundNum % 2 === 0 ? [...slots].reverse() : slots;

          return (
            <div
              key={roundNum}
              className="grid border-b border-zinc-200 dark:border-zinc-800 last:border-b-0"
              style={{ gridTemplateColumns: gridTemplate }}
            >
              {/* Round label */}
              <div className="sticky left-0 z-20 bg-white dark:bg-zinc-950 border-r border-zinc-200 dark:border-zinc-800">
                <div className="px-2 py-1 flex items-center" style={{ minHeight: CELL_MIN_H }}>
                  <span className="inline-flex rounded bg-zinc-100 dark:bg-zinc-900 px-1.5 py-0.5 text-[10px] font-semibold text-zinc-700 dark:text-zinc-300">
                    R{roundNum}
                    <span className="ml-1 opacity-70">{roundNum % 2 === 1 ? "→" : "←"}</span>
                  </span>
                </div>
              </div>

              {/* Cells */}
              {display.map((p, i) => {
                if (!p) {
                  return (
                    <div
                      key={`${roundNum}-empty-${i}`}
                      className="border-l border-zinc-200 dark:border-zinc-800 flex items-center justify-center text-[10px] text-zinc-400"
                      style={{ minHeight: CELL_MIN_H }}
                    >
                      —
                    </div>
                  );
                }

                const info = getColorInfo(p);
                const posKey = normPos(p.position);

                let cardClass = "relative h-full w-full rounded-sm px-1.5 py-1.5";
                let cardStyle: React.CSSProperties | undefined;
                let textClass = "text-zinc-800";

                if (colorMode === "position") {
                  const cls = positionColors[posKey] ?? "bg-zinc-200 text-zinc-800";
                  cardClass += ` ${cls}`;
                  textClass = cls.includes("text-black") ? "text-black" : "text-zinc-800";
                } else if (colorMode === "vorp") {
                  if (info.isPartial) {
                    // white card + tinted border; **text black**
                    cardClass += " bg-white";
                    cardStyle = { border: `2px solid ${info.shadeHex}` };
                    textClass = "text-zinc-900";
                  } else if (typeof info.value === "number" && Number.isFinite(info.value)) {
                    const bg = info.shadeHex;
                    const txt = textColorFor(bg);
                    cardClass += ` ${txt}`;
                    cardStyle = { backgroundColor: bg };
                    textClass = txt.includes("text-white") ? "text-white" : "text-zinc-900";
                  } else {
                    cardClass += " bg-zinc-100 text-zinc-700";
                    textClass = "text-zinc-700";
                  }
                } else if (colorMode === "vorp_rel") {
                  const r = findResidual(p);
                  if (typeof r === "number" && Number.isFinite(r)) {
                    const bg = vorpToColor(r);
                    const txt = textColorFor(bg);
                    cardClass += ` ${txt}`;
                    cardStyle = { backgroundColor: bg };
                    textClass = txt.includes("text-white") ? "text-white" : "text-zinc-900";
                  } else {
                    cardClass += " bg-zinc-100 text-zinc-700";
                    textClass = "text-zinc-700";
                  }
                }

                const displayV = info.value;
                const showVBadge =
                  colorMode === "vorp" &&
                  typeof displayV === "number" &&
                  Number.isFinite(displayV);

                const g = info.g;

                return (
                  <div
                    key={`${roundNum}-${p.overall_pick ?? i}`}
                    className="border-l border-zinc-200 dark:border-zinc-800 p-1"
                    style={{ minHeight: CELL_MIN_H }}
                  >
                    <div className={cardClass} style={cardStyle}>
                      {/* FIRST-AID BADGE: thicker cross */}
                        {colorMode === "vorp" && info.isPartial && (
                        <span
                            className="absolute -top-2 left-1/2 -translate-x-1/2 inline-flex items-center justify-center w-5 h-5 rounded-full bg-red-500 ring-2 ring-white shadow"
                            title="Partial season (extrapolated)"
                        >
                            <span className="absolute bg-white w-[14px] h-[4.5px]"></span>
                            <span className="absolute bg-white h-[14px] w-[4.5px]"></span>
                        </span>
                        )}


                      {/* TOP ROW */}
                      <div className="flex items-center justify-between text-[10px] mb-0.5">
                        <div className={`font-semibold ${textClass}`}>
                          {p.pick_num ?? "—"}
                          {p.overall_pick != null && <span className="ml-1 opacity-80">#{p.overall_pick}</span>}
                        </div>
                        <div className="flex items-center gap-1">
                          {showVBadge && (
                                <span
                                    className={`inline-flex rounded px-1 py-[1px] text-[9px] font-semibold ${
                                    info.isPartial ? "bg-black/10 text-zinc-900" : "bg-black/15"
                                    }`}
                                    title={
                                    info.isPartial
                                        ? `WAR*17 (extrapolated): ${displayV!.toFixed(1)}`
                                        : `WAR: ${displayV!.toFixed(1)}`
                                    }
                                >
                                    {displayV!.toFixed(1)}
                                </span>
                            )}

                          {p.is_keeper && (
                            <span className="inline-flex rounded bg-black/10 px-1 py-[1px] text-[9px] font-semibold">
                              K
                            </span>
                          )}
                        </div>
                      </div>

                      {/* NAME + META */}
                      <div className={`text-xs font-semibold leading-tight line-clamp-2 ${textClass}`}>
                        {p.player_name}
                      </div>
                      <div className={`mt-0.5 text-[10px] ${textClass}`}>
                        {(p.position?.toUpperCase() || "—")}
                        {p.pro_team ? ` • ${p.pro_team}` : ""}
                      </div>

                      {/* GAMES BAR (partial only) — slightly smaller, bottom-centered */}
                      {colorMode === "vorp" && info.isPartial && typeof g === "number" && (
                        <div className="absolute left-2 right-2" style={{ bottom: -5 }}>
                          <div
                            className="relative rounded-full bg-white"
                            style={{
                              height: 9,               // was 12
                              border: `2px solid ${info.shadeHex}`,
                            }}
                            title={`${g}/17 games`}
                          >
                            <div
                              className="rounded-full h-full"
                              style={{
                                width: `${gamesPct(g)}%`,
                                backgroundColor: info.shadeHex,
                              }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}


/* ===============================================================
   LIST
   =============================================================== */
function DraftList({ picks }: { picks: DraftPick[] }) {
  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-zinc-50 dark:bg-zinc-950">
          <tr className="text-left text-zinc-600 dark:text-zinc-300">
            <th className="py-2 px-3">Ovrl</th>
            <th className="py-2 px-3">Rnd</th>
            <th className="py-2 px-3">Pick</th>
            <th className="py-2 px-3">Player</th>
            <th className="py-2 px-3">Pos</th>
            <th className="py-2 px-3">NFL</th>
            <th className="py-2 px-3">Team</th>
            <th className="py-2 px-3">Owner</th>
            <th className="py-2 px-3">Keeper</th>
          </tr>
        </thead>
        <tbody>
          {picks.map((p) => (
            <tr key={`${p.overall_pick}-${p.team_name}-${p.player_name}`} className="border-t border-zinc-200 dark:border-zinc-800">
              <td className="py-2 px-3">{p.overall_pick ?? "—"}</td>
              <td className="py-2 px-3">{p.round_num ?? "—"}</td>
              <td className="py-2 px-3">{p.pick_num ?? "—"}</td>
              <td className="py-2 px-3">{p.player_name}</td>
              <td className="py-2 px-3">{p.position ?? "—"}</td>
              <td className="py-2 px-3">{p.pro_team ?? "—"}</td>
              <td className="py-2 px-3">{p.team_name}</td>
              <td className="py-2 px-3">{p.owner ?? "—"}</td>
              <td className="py-2 px-3">{p.is_keeper ? "Yes" : "No"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
