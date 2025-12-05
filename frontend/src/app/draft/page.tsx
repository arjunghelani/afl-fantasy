"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

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
  vorp_star_extrap?: number | null;
  partial_season?: boolean | null;    // optional
  vorp_star_rank_overall: number;
  vorp_star_rank_pos: number;
};

type VorpMap = Record<
  string,
  {
    vorp_star: number;
    vorp_star_extrap?: number | null;
    fantasy_pos: string;
    g?: number | null;
  }
>;

type VorpResponse = {
  year: number;
  players: VorpRow[];
  count: number;
  used_ppg: boolean;
};

/* ---- relative (smoothed baseline) types ---- */
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

interface WeeklyStat {
  week: number;
  z_week_ppr: number | null;
  weekly_points_ppr: number | null;
}

interface PlayerWeeklyStatsResponse {
  player_name: string;
  year: number;
  max_week: number;
  weekly_stats: WeeklyStat[];
  total_points?: number | null;
  total_zav?: number | null;
  fantasy_pos?: string | null;
  pos_rank?: number | null;
}

/* ------------------------- config ------------------------- */
const YEARS = [2020, 2021, 2022, 2024, 2025];
// API routes are now proxied through Next.js API routes
const REL_WINDOW = 7;
const REL_VERSION = 1; // bump when backend logic changes
const relKeyFor = (year: number) => `${year}|w=${REL_WINDOW}|v=${REL_VERSION}`;

/* ------------------------- helpers ------------------------- */
// Fantasy points gradient functions - matching ZAV style
function getFantasyPointsGradient(points: number | null): string {
  if (points === null || points === 0) {
    return 'linear-gradient(135deg, rgb(131, 131, 131), rgb(100, 100, 100))';
  }
  
  if (points <= 5) {
    return 'linear-gradient(135deg, rgb(220, 70, 70), rgb(180, 30, 30))';
  } else if (points <= 10) {
    return 'linear-gradient(135deg, rgb(250, 130, 60), rgb(210, 90, 20))';
  } else if (points <= 15) {
    return 'linear-gradient(135deg, rgb(245, 210, 75), rgb(225, 170, 35))';
  } else if (points <= 20) {
    return 'linear-gradient(135deg, rgb(160, 220, 80), rgb(120, 180, 40))';
  } else {
    return 'linear-gradient(135deg, rgb(16, 185, 129), rgb(5, 150, 105))';
  }
}

function getFantasyPointsTextColor(points: number | null): string {
  if (points === null || points === 0) {
    return 'rgb(40, 40, 40)';
  }
  
  // Extremely dark versions matching ZAV text colors
  if (points <= 5) {
    return 'rgb(90, 10, 10)'; // Extremely dark red
  } else if (points <= 10) {
    return 'rgb(110, 35, 5)'; // Extremely dark orange
  } else if (points <= 15) {
    return 'rgb(110, 85, 5)'; // Extremely dark yellow/brown
  } else if (points <= 20) {
    return 'rgb(50, 75, 10)'; // Extremely dark yellow-green
  } else {
    return 'rgb(1, 50, 20)'; // Extremely dark emerald green
  }
}

// ZAV color scale
const ZAV_CUTOFFS = {
  red: -2,
  orange: 2.5,
  yellow: 6,
  yellowGreen: 10,
};

function getZavGradient(zav: number | null): string {
  if (zav === null || zav === 0) {
    return 'linear-gradient(135deg, rgb(131, 131, 131), rgb(100, 100, 100))';
  }
  
  if (zav < ZAV_CUTOFFS.red) {
    return 'linear-gradient(135deg, rgb(220, 70, 70), rgb(180, 30, 30))';
  } else if (zav < ZAV_CUTOFFS.orange) {
    return 'linear-gradient(135deg, rgb(250, 130, 60), rgb(210, 90, 20))';
  } else if (zav < ZAV_CUTOFFS.yellow) {
    return 'linear-gradient(135deg, rgb(245, 210, 75), rgb(225, 170, 35))';
  } else if (zav < ZAV_CUTOFFS.yellowGreen) {
    return 'linear-gradient(135deg, rgb(160, 220, 80), rgb(120, 180, 40))';
  } else {
    return 'linear-gradient(135deg, rgb(16, 185, 129), rgb(5, 150, 105))';
  }
}

function getZavTextColor(zav: number | null): string {
  if (zav === null || zav === 0) {
    return 'rgb(40, 40, 40)';
  }
  
  // Extremely dark versions of bubble colors for text (used inside bubbles)
  if (zav < ZAV_CUTOFFS.red) {
    return 'rgb(90, 10, 10)'; // Extremely dark red
  } else if (zav < ZAV_CUTOFFS.orange) {
    return 'rgb(110, 35, 5)'; // Extremely dark orange
  } else if (zav < ZAV_CUTOFFS.yellow) {
    return 'rgb(110, 85, 5)'; // Extremely dark yellow/brown
  } else if (zav < ZAV_CUTOFFS.yellowGreen) {
    return 'rgb(50, 75, 10)'; // Extremely dark yellow-green
  } else {
    return 'rgb(1, 50, 20)'; // Extremely dark emerald green
  }
}

function getZavBrightColor(zav: number | null): string {
  if (zav === null || zav === 0) {
    return 'rgb(131, 131, 131)'; // Gray for null/zero
  }
  
  // Bright gradient colors (used for stats display)
  if (zav < ZAV_CUTOFFS.red) {
    return 'rgb(220, 70, 70)'; // Red
  } else if (zav < ZAV_CUTOFFS.orange) {
    return 'rgb(250, 130, 60)'; // Orange
  } else if (zav < ZAV_CUTOFFS.yellow) {
    return 'rgb(245, 210, 75)'; // Yellow
  } else if (zav < ZAV_CUTOFFS.yellowGreen) {
    return 'rgb(160, 220, 80)'; // Yellow-green
  } else {
    return 'rgb(16, 185, 129)'; // Emerald green
  }
}

const normalizeName = (raw: string) => {
  let s = raw.trim().toLowerCase();
  s = s.normalize("NFD").replace(/\p{Diacritic}/gu, "");
  s = s.replace(/[.'’`,\-]/g, " ");
  s = s.replace(/\b(jr|sr|ii|iii|iv|v)\b/g, "");
  s = s.replace(/\s+/g, " ").trim();
  return s;
};

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

function vorpToColor(v: number, absMax = 15) {
  const t = clamp(v, -absMax, absMax) / absMax;
  const stops: Array<{ x: number; c: [number, number, number] }> = [
    { x: -1,   c: [58, 25, 112] },
    { x: -0.5, c: [128, 73, 215] },
    { x:  0,   c: [255, 255, 255] },
    { x:  0.5, c: [22, 163, 74] },
    { x:  1,   c: [4, 77, 31] },
  ];
  let i = 0;
  while (i < stops.length - 1 && t > stops[i + 1].x) i++;
  const a = stops[i];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  const u = (t - a.x) / (b.x - a.x);
  const lerp = (A: number, B: number, U: number) => A + (B - A) * U;
  const r = lerp(a.c[0], b.c[0], u);
  const g = lerp(a.c[1], b.c[1], u);
  const bch = lerp(a.c[2], b.c[2], u);
  const toHex = (n: number) => Math.round(n).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(bch)}`;
}

function textColorFor(bgHex: string) {
  const n = bgHex.replace("#", "");
  const r = parseInt(n.slice(0, 2), 16);
  const g = parseInt(n.slice(2, 4), 16);
  const b = parseInt(n.slice(4, 6), 16);
  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return luminance > 0.62 ? "text-slate-900" : "text-white";
}

function fmtSigned(x: number | null | undefined, digits = 1) {
  if (x == null || !Number.isFinite(x)) return null;
  const s = x.toFixed(digits);
  return x >= 0 ? `+${s}` : s;
}

/* ------------------------- data fetchers ------------------------- */
async function fetchDraft(year: number): Promise<DraftResponse> {
  const res = await fetch(`/api/draft/${year}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch draft for ${year}`);
  return res.json();
}

async function fetchVorpMap(year: number): Promise<VorpMap> {
  const res = await fetch(`/api/metrics/vorp/${year}?top=2000`, { cache: "no-store" });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`VORP* ${year} failed: ${res.status} ${res.statusText} ${txt}`);
  }
  const json: VorpResponse = await res.json();
  const map: VorpMap = {};
  for (const p of json.players) {
    map[normalizeName(p.player_name)] = {
      vorp_star: p.vorp_star,
      vorp_star_extrap: (p as any).vorp_star_extrap ?? null,
      fantasy_pos: p.fantasy_pos,
      g: p.g,
    };
  }
  return map;
}



/* ---- relative (smoothed) payload ---- */
async function fetchVorpRelative(year: number, window = 7): Promise<VorpRelativeResponse> {
  const res = await fetch(`/api/metrics/vorp_relative/${year}?window=${window}`, { cache: "no-store" });
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

  // VORP* (season) state
  const [vorpLoading, setVorpLoading] = useState(false);
  const [vorpError, setVorpError] = useState<string | null>(null);
  const [vorpCache, setVorpCache] = useState<Record<number, VorpMap | undefined>>({});
  const vorpMapSeason = vorpCache[year];


  // relative baseline state
  const [relLoading, setRelLoading] = useState(false);
  const [relError, setRelError] = useState<string | null>(null);
  const [relCache, setRelCache] = useState<Record<string, VorpRelativeResponse | undefined>>({});
  const relKey = relKeyFor(year);
  const rel = relCache[relKey];

  // player popup state
  const [selectedPlayers, setSelectedPlayers] = useState<Map<string, {playerName: string, year: number, selectedYear: number, availableYears: number[], position?: {x: number, y: number}, stats?: PlayerWeeklyStatsResponse, loading?: boolean, headshotUrl?: string | null}>>(new Map());

  // draft fetch on year change
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchDraft(year)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [year]);

  // prefetch season VORP on year change
  useEffect(() => {
    let cancelled = false;
    setVorpError(null);
    if (vorpCache[year]) return;
    setVorpLoading(true);
    fetchVorpMap(year)
      .then((map) => { if (!cancelled) setVorpCache((prev) => ({ ...prev, [year]: map })); })
      .catch((e) => {
        if (!cancelled) setVorpError(e instanceof Error ? e.message : String(e));
        console.warn("VORP prefetch failed:", e);
      })
      .finally(() => { if (!cancelled) setVorpLoading(false); });
    return () => { cancelled = true; };
  }, [year, vorpCache]);

  // prefetch relative baseline on year change
  useEffect(() => {
    let cancelled = false;
    setRelError(null);
    if (relCache[relKey]) return;
    setRelLoading(true);
    fetchVorpRelative(year, 7)
      .then((payload) => { if (!cancelled) setRelCache((prev) => ({ ...prev, [relKey]: payload })); })
      .catch((e) => {
        if (!cancelled) setRelError(e instanceof Error ? e.message : String(e));
        console.warn("VORP relative failed:", e);
      })
      .finally(() => { if (!cancelled) setRelLoading(false); });
    return () => { cancelled = true; };
  }, [year, relCache, relKey]);

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
        return;
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

  const hasPlayerData = (stats: PlayerWeeklyStatsResponse): boolean => {
    // Check if there's at least one week with actual data
    return stats.weekly_stats.some(stat => 
      stat.z_week_ppr !== null || stat.weekly_points_ppr !== null
    );
  };

  const handlePlayerClick = async (playerName: string, year: number, event?: React.MouseEvent<HTMLDivElement>) => {
    const playerKey = `${playerName}_${year}`;
    
    // Check if this player is already open
    if (selectedPlayers.has(playerKey)) {
      return; // Don't reopen if already open
    }
    
    // Always center pop-ups, stack vertically when multiple
      const existingPopups = Array.from(selectedPlayers.values());
    
    // Add player to map with loading state
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      newMap.set(playerKey, { playerName, year, selectedYear: year, availableYears: [year], position: { x: 0, y: 0 }, loading: true });
      return newMap;
    });
    
    try {
      // Fetch stats and headshot in parallel
      // encodeURIComponent handles special characters like "/" in "D/ST"
      const encodedName = encodeURIComponent(playerName);
      const [statsResponse, headshotResponse] = await Promise.all([
        fetch(`/api/players/${encodedName}/weekly-stats?year=${year}`),
        fetch(`/api/players/${encodedName}/headshot`)
      ]);
      
      if (!statsResponse.ok) {
        const errorText = await statsResponse.text();
        console.error(`Failed to fetch stats for ${playerName} (encoded: ${encodedName}):`, statsResponse.status, errorText);
        throw new Error(`Failed to fetch player stats: ${statsResponse.status}`);
      }
      const data: PlayerWeeklyStatsResponse = await statsResponse.json();
      
      // Get headshot URL
      let headshotUrl: string | null = null;
      if (headshotResponse.ok) {
        const headshotData = await headshotResponse.json();
        headshotUrl = headshotData.headshot_url || null;
      }
      
      // Check all years to see which ones have data
      const availableYears: number[] = [];
      const yearChecks = YEARS.map(async (checkYear) => {
        try {
          const checkResponse = await fetch(`/api/players/${encodedName}/weekly-stats?year=${checkYear}`);
          if (checkResponse.ok) {
            const checkData: PlayerWeeklyStatsResponse = await checkResponse.json();
            if (hasPlayerData(checkData)) {
              availableYears.push(checkYear);
            }
          }
        } catch (err) {
          // Silently skip years that fail
        }
      });
      
      await Promise.all(yearChecks);
      
      // Sort available years
      availableYears.sort();
      
      // Update with stats, headshot, and available years
      setSelectedPlayers(prev => {
        const newMap = new Map(prev);
        const existing = newMap.get(playerKey);
        if (existing) {
          newMap.set(playerKey, { 
            ...existing, 
            stats: data, 
            headshotUrl: headshotUrl,
            availableYears: availableYears.length > 0 ? availableYears : [year], // Fallback to at least the current year
            loading: false 
          });
        }
        return newMap;
      });
    } catch (err) {
      console.error('Error fetching player stats:', err);
      
      // Remove on error
      setSelectedPlayers(prev => {
        const newMap = new Map(prev);
        newMap.delete(playerKey);
        return newMap;
      });
    }
  };

  const closePlayerPopup = (playerKey: string) => {
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      newMap.delete(playerKey);
      return newMap;
    });
  };

  const handleYearChange = async (playerKey: string, newYear: number) => {
    const playerData = selectedPlayers.get(playerKey);
    if (!playerData) return;

    // Update selected year and set loading
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      const existing = newMap.get(playerKey);
      if (existing) {
        newMap.set(playerKey, { ...existing, selectedYear: newYear, loading: true });
      }
      return newMap;
    });

    try {
      const response = await fetch(`/api/players/${encodeURIComponent(playerData.playerName)}/weekly-stats?year=${newYear}`);
      if (!response.ok) {
        throw new Error('Failed to fetch player stats');
      }
      const data: PlayerWeeklyStatsResponse = await response.json();
      
      // Update with new stats
      setSelectedPlayers(prev => {
        const newMap = new Map(prev);
        const existing = newMap.get(playerKey);
        if (existing) {
          newMap.set(playerKey, { ...existing, stats: data, loading: false });
        }
        return newMap;
      });
    } catch (err) {
      console.error('Error fetching player stats:', err);
      
      // Revert to previous state on error
      setSelectedPlayers(prev => {
        const newMap = new Map(prev);
        const existing = newMap.get(playerKey);
        if (existing) {
          newMap.set(playerKey, { ...existing, loading: false });
        }
        return newMap;
      });
    }
  };

  // pick active VORP map based on mode
  const activeVorpMap: VorpMap | undefined =
    colorMode === "vorp" ? vorpMapSeason :
    undefined;

    // Group by round for board layout (⬅ place this before the return)
    const draftRounds = useMemo<[number, DraftPick[]][]>(() => {
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


  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <div className="max-w-[1600px] mx-auto px-3 py-6 sm:px-6">
        {/* Header */}
        <div className="text-center mb-7">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent tracking-tight">
            Drafts
          </h1>
          
          {/* Navigation */}
          <nav className="flex justify-center items-center gap-3 mb-6 flex-wrap">
            <Link href="/" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Home
            </Link>
            <Link href="/players" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Players
            </Link>
            <Link href="/trades" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Trades
            </Link>
            <Link href="/waivers" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Waivers
            </Link>
            <Link href="/scoreboard" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Scoreboard
            </Link>
            <Link href="/draft" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Draft
            </Link>
          </nav>
          
          {/* Year Selector */}
          <div className="flex justify-center gap-2 mb-5 flex-wrap">
            {YEARS.map(yearOption => (
              <button
                key={yearOption}
                onClick={() => setYear(yearOption)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  year === yearOption
                    ? 'bg-white text-black'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {yearOption}
              </button>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="mb-5">
          <div className="flex flex-col md:flex-row md:items-center gap-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Draft Controls</h2>
            
            <div className="flex flex-wrap items-center gap-3">
              {/* View toggle */}
              <div className="inline-flex rounded-full overflow-hidden border border-gray-200 dark:border-gray-700">
                <button
                  className={`px-3 py-2 text-sm font-medium transition-colors ${
                    view === "board" 
                      ? "bg-white text-black" 
                      : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                  }`}
                  onClick={() => setView("board")}
                >
                  Board
                </button>
                <button
                  className={`px-3 py-2 text-sm font-medium transition-colors ${
                    view === "list" 
                      ? "bg-white text-black" 
                      : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                  }`}
                  onClick={() => setView("list")}
                >
                  List
                </button>
              </div>

              {/* Color-by toggle */}
              <div className="inline-flex rounded-full overflow-hidden border border-gray-200 dark:border-gray-700">
                <button
                  className={`px-3 py-2 text-sm font-medium transition-colors ${
                    colorMode === "position" 
                      ? "bg-white text-black" 
                      : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                  }`}
                  onClick={() => setColorMode("position")}
                >
                  Position
                </button>
                <button
                  className={`px-3 py-2 text-sm font-medium transition-colors ${
                    colorMode === "vorp" 
                      ? "bg-white text-black" 
                      : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                  }`}
                  disabled={vorpLoading}
                  onClick={switchToVorp}
                  title={vorpLoading ? "Loading ZAV…" : !vorpMapSeason ? "Fetches ZAV (season)" : ""}
                >
                  {vorpLoading ? "Loading…" : "ZAV"}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Legends + errors */}
        {colorMode === "vorp" && (
          <div className="mb-6">
            <div className="flex items-center gap-3 text-sm text-slate-300">
              <span>low value</span>
              <div
                className="h-3 w-48 rounded"
                style={{ background: "linear-gradient(90deg, rgb(58,25,112) 0%, rgb(128,73,215) 25%, #ffffff 50%, #16a34a 75%, rgb(4,77,31) 100%)" }}
              />
              <span className="font-medium text-slate-200">high value</span>
            </div>
            {vorpError && (
              <div className="mt-3 text-sm text-red-600 dark:text-red-400">
                {vorpError}
              </div>
            )}
          </div>
        )}

        {colorMode === "vorp_rel" && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 mb-6">
            <div className="flex items-center gap-3 text-sm text-gray-600 dark:text-gray-300">
              <span>Vs Pick Expectation:</span>
              <div
                className="h-3 w-48 rounded"
                style={{ background: "linear-gradient(90deg, rgb(58,25,112) 0%, rgb(128,73,215) 25%, #ffffff 50%, #16a34a 75%, rgb(4,77,31) 100%)" }}
              />
              <span className="opacity-70">below expected</span>
              <span className="opacity-70 ml-auto">above expected</span>
            </div>
            {relError && <div className="mt-3 text-sm text-red-600 dark:text-red-400">{relError}</div>}
          </div>
        )}

        {loading && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 text-center">
            <div className="text-gray-600 dark:text-gray-300">Loading {year} draft…</div>
          </div>
        )}
        
        {error && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 text-center">
            <div className="text-red-600 dark:text-red-400">{error}</div>
          </div>
        )}

        {!loading && !error && data && (
          <div className="rounded-2xl border border-gray-200/70 dark:border-white/10 bg-white/95 dark:bg-gray-900/85 shadow-2xl overflow-hidden backdrop-blur">
          {view === "board" ? (
            <DraftBoard
              rounds={draftRounds}
              colorMode={colorMode}
              vorpMap={activeVorpMap}
              rel={rel}
              onPlayerClick={handlePlayerClick}
              year={year}
            />
          ) : (
            <DraftList picks={data.picks} onPlayerClick={handlePlayerClick} year={year} />
          )}
          </div>
        )}

        {/* Player Weekly Stats Popups - Multiple can be open */}
        {Array.from(selectedPlayers.entries()).map(([playerKey, playerData], index) => {
          // Calculate position for this pop-up in the centered stack
          const allPopups = Array.from(selectedPlayers.entries());
          const popupHeight = 200;
          const popupSpacing = 20;
          const totalHeight = allPopups.length * (popupHeight + popupSpacing) - popupSpacing;
          const startY = -totalHeight / 2;
          const thisPopupY = startY + index * (popupHeight + popupSpacing);
          
          return (
          <div 
            key={playerKey} 
            className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto pointer-events-none"
            onClick={() => closePlayerPopup(playerKey)}
          >
            <div 
              className="relative bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-2xl p-6 transition-all duration-300 ease-out pointer-events-auto"
              style={{
                transform: `translateY(${thisPopupY}px)`,
                maxWidth: 'calc(100vw - 32px)',
                width: 'auto',
                minWidth: '400px',
                animation: 'fadeIn 0.3s ease-out forwards',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-4 relative">
                {/* Headshot - positioned to left of name, top 1/3 above card */}
                {playerData.headshotUrl && (
                  <div className="absolute -top-16 left-0 z-10" style={{ width: '80px', height: '120px' }}>
                    <img 
                      src={playerData.headshotUrl} 
                      alt={playerData.playerName}
                      className="w-full h-full object-cover"
                      style={{
                        clipPath: 'polygon(0 0, 100% 0, 100% 85%, 50% 100%, 0 85%)',
                        filter: 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3))',
                      }}
                    />
                  </div>
                )}
                <div className="flex-1" style={{ marginLeft: playerData.headshotUrl ? '100px' : '0' }}>
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                    {playerData.playerName}
                  </h2>
                  <div className="flex items-center gap-2 mt-2">
                    <label className="text-sm text-gray-500 dark:text-gray-400">Year:</label>
                    <select
                      value={playerData.selectedYear}
                      onChange={(e) => handleYearChange(playerKey, parseInt(e.target.value))}
                      className="text-sm bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-3 py-1.5 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {playerData.availableYears.map(year => (
                        <option key={year} value={year}>{year}</option>
                      ))}
                    </select>
                  </div>
                </div>
                <button
                  onClick={() => closePlayerPopup(playerKey)}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors ml-4"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {playerData.loading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="text-sm text-gray-500 dark:text-gray-400">Loading...</div>
                </div>
              ) : playerData.stats ? (
                <>
                  <div className="flex gap-2 items-center overflow-x-auto" style={{ maxWidth: 'calc(100vw - 80px)', scrollbarWidth: 'thin' }}>
                  {playerData.stats.weekly_stats.map((stat: WeeklyStat, statIndex: number) => {
                    const hasData = stat.z_week_ppr !== null && stat.weekly_points_ppr !== null;
                      const points = stat.weekly_points_ppr;
                      const gradient = getFantasyPointsGradient(points);
                      const textColor = getFantasyPointsTextColor(points);

                    return (
                        <div key={stat.week} className="flex items-center gap-2">
                        <div
                            className={`rounded-md p-2 border border-gray-200 dark:border-gray-600 flex-shrink-0 w-16 ${hasData ? '' : 'opacity-50'}`}
                            style={hasData ? { background: gradient } : {}}
                        >
                            <div className={`text-xs font-semibold mb-1 text-center`} style={hasData ? { color: textColor } : {}}>
                            W{stat.week}
                          </div>
                          {hasData ? (
                            <>
                                <div className={`text-base font-bold text-center mb-0.5`} style={{ color: textColor }}>
                                {stat.weekly_points_ppr?.toFixed(1) ?? 'N/A'}
                              </div>
                                <div className={`text-xs text-center italic opacity-90`} style={{ color: textColor }}>
                                  z: {stat.z_week_ppr?.toFixed(2) ?? 'N/A'}
                              </div>
                            </>
                          ) : (
                              <div className={`text-[10px] text-center text-gray-500 dark:text-gray-400`}>
                              N/A
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600 flex items-center justify-center gap-6">
                    {playerData.stats.total_points !== null && playerData.stats.total_points !== undefined && (() => {
                      // Calculate games played (weeks with non-null points, excluding week 0)
                      const gamesPlayed = playerData.stats.weekly_stats.filter(
                        stat => stat.week !== 0 && stat.weekly_points_ppr !== null && stat.weekly_points_ppr !== undefined
                      ).length;
                      const ppg = gamesPlayed > 0 ? playerData.stats.total_points / gamesPlayed : 0;
                      return (
                        <div className="flex items-center gap-2">
                          <span className="text-white font-bold text-lg">PPG:</span>
                          <span style={{ color: getZavBrightColor(playerData.stats.total_zav ?? null) }} className="font-semibold text-xl">
                            {ppg.toFixed(1)}
                          </span>
                        </div>
                      );
                    })()}
                    {playerData.stats.total_zav !== null && playerData.stats.total_zav !== undefined && (
                      <div className="flex items-center gap-2 ml-6">
                        <span className="text-white font-bold text-lg">ZAV:</span>
                        <span style={{ color: getZavBrightColor(playerData.stats.total_zav) }} className="font-semibold text-xl">
                          {playerData.stats.total_zav.toFixed(2)}
                        </span>
                      </div>
                    )}
                    {playerData.stats.fantasy_pos && playerData.stats.pos_rank !== null && playerData.stats.pos_rank !== undefined && (
                      <div className="flex items-center gap-0 ml-6">
                        <span className="text-white font-bold text-lg">{playerData.stats.fantasy_pos}</span>
                        <span style={{ color: getZavBrightColor(playerData.stats.total_zav ?? null) }} className="font-semibold text-xl">
                          {playerData.stats.pos_rank}
                        </span>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="text-center py-8 text-sm text-gray-500 dark:text-gray-400">
                  No weekly stats available
                </div>
              )}
            </div>
          </div>
          );
        })}
      </div>
    </div>
  );
}

/* ===============================================================
   BOARD
   =============================================================== */
function DraftBoard({
  rounds,
  colorMode,
  vorpMap,
  rel,
  onPlayerClick,
  year,
}: {
  rounds: [number, DraftPick[]][];
  colorMode: "position" | "vorp" | "vorp_rel";
  vorpMap?: VorpMap;
  rel?: VorpRelativeResponse;
  onPlayerClick?: (playerName: string, year: number, event?: React.MouseEvent<HTMLDivElement>) => void;
  year: number;
}) {
  const GUTTER = 60; // pull columns closer so the board fits on screen
  const COL_MIN = 95; // slightly smaller column width to "zoom out"
  const CELL_MIN_H = 66;

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
    if (!row && colorMode === "vorp") {
      console.log(`No VORP data for: ${p.player_name} (normalized: ${normalizeName(p.player_name)})`);
    }
    if (!row) return { isPartial: false, shadeHex: "#ffffff", value: undefined, g: undefined };
  
    const g = row.g ?? undefined;
  
    // 🩹 Updated partial-season logic
    let isPartial = false;
    if (typeof g === "number") {
      if (year === 2025) {
        // For 2025 only: mark as partial if under 5 games
        isPartial = g < 6 && row.vorp_star_extrap != null;
      } else {
        // Other years: original logic (between 3–10 games)
        isPartial = g >= 3 && g < 11 && row.vorp_star_extrap != null;
      }
    }
  
    const value = row.vorp_star; // Always show true WAR, not extrapolated
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
    <div className="relative w-full rounded-2xl shadow-xl shadow-slate-950/50 border border-slate-800 bg-slate-900 overflow-auto">
      {/* Header */}
      <div className="sticky top-0 z-30 border-b border-slate-800 bg-gradient-to-r from-slate-800 to-slate-700">
        <div className="grid" style={{ gridTemplateColumns: gridTemplate }}>
          <div className="h-8 px-2 flex items-center text-[10px] font-semibold text-white uppercase">Round</div>
          {Array.from({ length: teamCount }).map((_, idx) => (
            <div
              key={`hdr-slot-${idx}`}
              className="h-auto min-h-8 px-1 py-1 flex items-center justify-center border-l border-slate-700"
            >
              <span className="text-xs font-semibold text-white text-center break-words leading-tight">
                {headerTeams[idx] ?? `Slot ${idx + 1}`}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Body */}
      <div className="mt-2">
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
              className="grid border-b border-slate-800 last:border-b-0"
              style={{ gridTemplateColumns: gridTemplate }}
            >
              {/* Round label */}
              <div className="sticky left-0 z-20 bg-slate-900 border-r border-slate-800">
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
                      className="border-l border-slate-800 flex items-center justify-center text-[10px] text-slate-500"
                      style={{ minHeight: CELL_MIN_H }}
                    >
                      —
                    </div>
                  );
                }

                const info = getColorInfo(p);
                const posKey = normPos(p.position);

                let cardClass = "relative w-full rounded-sm px-1.5 py-1.5 h-[4.75rem] flex flex-col justify-between";
                let cardStyle: React.CSSProperties | undefined;
                let textClass = "text-zinc-800";

                if (colorMode === "position") {
                  const cls = positionColors[posKey] ?? "bg-zinc-200 text-zinc-800";
                  cardClass += ` ${cls}`;
                  textClass = cls.includes("text-black") ? "text-black" : "text-zinc-800";
                } else if (colorMode === "vorp") {
                  if (colorMode === "vorp" && info.isPartial) {
                    // partial-season badge only for season VORP (extrapolated rows)
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
                    className="border-l border-slate-800 p-1"
                    style={{ minHeight: CELL_MIN_H }}
                  >
                    <div className={cardClass} style={cardStyle}>
                      {/* Partial-season badge (only for season VORP mode) */}
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
                          {p.round_num && p.pick_num ? `${p.round_num}.${p.pick_num}` : "—"}
                        </div>
                        <div className="flex items-center gap-1">
                          {showVBadge && (
                            <span
                              className={`inline-flex px-0 py-0 text-[11px] font-semibold leading-none ${textClass}`}
                              title={
                                colorMode === "vorp" && info.isPartial
                                  ? `ZAV (${g}/17 games): ${displayV!.toFixed(1)}`
                                  : `ZAV: ${displayV!.toFixed(1)}`
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
                      <div 
                        className={`text-xs font-semibold leading-tight line-clamp-2 ${textClass} ${onPlayerClick ? 'cursor-pointer hover:scale-105 transition-all duration-200' : ''}`}
                        onClick={(e) => onPlayerClick?.(p.player_name, year, e)}
                      >
                        {p.player_name}
                      </div>
                      <div className={`mt-0.5 text-[10px] ${textClass}`}>
                        {(p.position?.toUpperCase() || "—")}
                        {p.pro_team ? ` • ${p.pro_team}` : ""}
                      </div>

                      {/* Games bar for partial season (season VORP only) */}
                      {colorMode === "vorp" && info.isPartial && typeof g === "number" && (
                        <div className="absolute left-2 right-2" style={{ bottom: -5 }}>
                          <div
                            className="relative rounded-full bg-white"
                            style={{ height: 9, border: `2px solid ${info.shadeHex}` }}
                            title={`${g}/17 games`}
                          >
                            <div
                              className="rounded-full h-full"
                              style={{ width: `${gamesPct(g)}%`, backgroundColor: info.shadeHex }}
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


function DraftList({ picks, onPlayerClick, year }: { picks: DraftPick[]; onPlayerClick?: (playerName: string, year: number, event?: React.MouseEvent<HTMLDivElement>) => void; year: number }) {
  return (
    <div className="rounded-2xl shadow-xl shadow-slate-950/50 border border-slate-800 bg-slate-900 overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-gradient-to-r from-slate-800 to-slate-700">
          <tr className="text-left text-white">
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
            <tr key={`${p.overall_pick}-${p.team_name}-${p.player_name}`} className="border-t border-slate-800">
              <td className="py-2 px-3 text-slate-200">{p.overall_pick ?? "—"}</td>
              <td className="py-2 px-3 text-slate-200">{p.round_num ?? "—"}</td>
              <td className="py-2 px-3 text-slate-200">{p.pick_num ?? "—"}</td>
              <td className="py-2 px-3">
                <div 
                  className={`text-slate-200 ${onPlayerClick ? 'cursor-pointer hover:text-indigo-400 transition-colors' : ''}`}
                  onClick={(e) => onPlayerClick?.(p.player_name, year, e)}
                >
                  {p.player_name}
                </div>
              </td>
              <td className="py-2 px-3 text-slate-200">{p.position ?? "—"}</td>
              <td className="py-2 px-3 text-slate-200">{p.pro_team ?? "—"}</td>
              <td className="py-2 px-3 text-slate-200">{p.team_name}</td>
              <td className="py-2 px-3 text-slate-200">{p.owner ?? "—"}</td>
              <td className="py-2 px-3 text-slate-200">{p.is_keeper ? "Yes" : "No"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
