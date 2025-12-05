"use client";

import { useEffect, useMemo, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";

/* --------------------------- types --------------------------- */
type PlayerRow = {
  player_name: string;
  team?: string | null;
  fantasy_pos: string;
  fantasy_points_ppr?: number;
  ppr_per_game?: number;       // NEW
  g?: number;                  // NEW (from /metrics/vorp)
  vorp_star?: number;
  true_vorp_star?: number;
  delta_vorp_star_mean?: number;
  delta_vorp_star_p10?: number;
  delta_vorp_star_p90?: number;
  adj_vorp_star?: number;
  weeks_played?: number;
  missed_weeks?: number;
  year?: number;
};

type VorpResponse = {
  year: number;
  players: PlayerRow[];
  count: number;
  used_ppg: boolean;
};

type DraftPick = {
  year: number;
  team_id: number;
  team_name: string;              // fantasy drafter
  round_num: number | null;
  pick_num: number | null;
  overall_pick: number | null;
  player_name: string;
};

type DraftResponse = {
  year: number;
  league_id: number;
  picks: DraftPick[];
};

/* NEW: extrapolated API types */
type ExtrapolatedRow = {
  player_name: string;
  team?: string | null;
  fantasy_pos: string;
  fantasy_points_ppr: number;  // NEW
  ppr_per_game?: number;       // NEW
  true_vorp_star: number;
  delta_vorp_star_mean: number;
  delta_vorp_star_p10: number;
  delta_vorp_star_p90: number;
  adj_vorp_star: number;
  weeks_played?: number;
  missed_weeks?: number;
};

type ExtrapolatedResponse = {
  year: number;
  sims: number;
  weeks_in_season: number;
  count: number;
  rows: ExtrapolatedRow[];
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

/* -------------------------- config -------------------------- */
const YEARS = [2020, 2021, 2022, 2024, 2025] as const;
type YearChoice = (typeof YEARS)[number] | "ALL";
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

// Map team_id → display name (fill in with your own labels)
const TEAM_NAME_MAP: Record<number, string> = {
  1: "PJ",
  2: "Conan",
  3: "Victor",
  4: "Evan",
  5: "Logan",
  6: "Jackson",
  7: "Jon",
  8: "Dylan",
  9: "Gavin",
  10: "Arjun",
  12: "Owen",
  14: "Aidan",
  // ...
};

// Names to exclude from WAR totals
const EXCLUDE_DRAFTER_NAMES = new Set<string>([
  "Team Ned",
]);

/* ------------------------ helpers --------------------------- */
const normalizeName = (raw: string) =>
  raw
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .replace(/[.''`,\-]/g, " ")
    .replace(/\b(jr|sr|ii|iii|iv|v)\b/g, "")
    .replace(/\s+/g, " ")
    .trim();

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

// Fantasy points color scale
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

/* ------------------------ data fetchers ---------------------- */
// async function fetchVorp(year: number): Promise<VorpResponse> {
//   const res = await fetch(`${API_BASE}/metrics/vorp/${year}?top=500`, { cache: "no-store" });
//   if (!res.ok) {
//     const body = await res.text().catch(() => "");
//     throw new Error(`VORP ${year} failed: ${res.status} ${res.statusText} ${body}`);
//   }
//   return res.json();
// }

async function fetchDraft(year: number): Promise<DraftResponse> {
  const res = await fetch(`${API_BASE}/draft/${year}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Draft ${year} failed: ${res.status} ${res.statusText} ${body}`);
  }
  return res.json();
}

// Helper to calculate optimal API limits based on filters
function calculateOptimalLimit(positionFilter?: Set<string>, yearCount: number = 1): number {
  const POS_LIMITS = { QB: 30, RB: 75, WR: 75, TE: 30 };
  
  if (!positionFilter || positionFilter.size === 4) {
    // All positions: sum all position limits (30+75+75+30 = 210)
    const totalPerYear = Object.values(POS_LIMITS).reduce((a, b) => a + b, 0);
    return totalPerYear * yearCount;
  } else {
    // Specific positions: sum only selected
    const totalPerYear = Array.from(positionFilter).reduce((sum, pos) => {
      return sum + (POS_LIMITS[pos as keyof typeof POS_LIMITS] || 0);
    }, 0);
    return totalPerYear * yearCount;
  }
}

/* NEW: extrapolated endpoint fetcher with dynamic limits */
async function fetchExtrapolated(year: number, positionFilter?: Set<string>): Promise<ExtrapolatedResponse> {
  // Calculate optimal limit based on position filter
  const limit = calculateOptimalLimit(positionFilter, 1); // 1 year
  
  const params = new URLSearchParams({
    sims: String(1000),
    weeks_in_season: String(17),
    limit: String(limit),
    // Add position filter to API call if not all positions
    ...(positionFilter && positionFilter.size < 4 ? { pos: Array.from(positionFilter).join(",") } : {}),
  });
  const res = await fetch(`${API_BASE}/metrics/war-extrapolated/${year}?${params.toString()}`, {
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Extrapolated WAR ${year} failed: ${res.status} ${res.statusText} ${body}`);
  }
  return res.json();
}

async function fetchVorp(year: number, positionFilter?: Set<string>): Promise<VorpResponse> {
  // Calculate optimal limit for regular VORP too
  const limit = calculateOptimalLimit(positionFilter, 1); // 1 year
  
  const res = await fetch(`${API_BASE}/metrics/vorp/${year}?top=${limit}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`VORP ${year} failed: ${res.status} ${res.statusText} ${body}`);
  }
  return res.json();
}

/* --------------------------- page --------------------------- */
function PlayersPageContent() {
  const searchParams = useSearchParams();
  const yearFromQuery = searchParams.get('year');
  
  // Initialize year from query param if present, otherwise default to "ALL"
  const [year, setYear] = useState<YearChoice>(() => {
    if (yearFromQuery && YEARS.includes(Number(yearFromQuery) as typeof YEARS[number])) {
      return Number(yearFromQuery) as YearChoice;
    }
    return "ALL";
  });
  
  // Update year when query param changes
  useEffect(() => {
    if (yearFromQuery && YEARS.includes(Number(yearFromQuery) as typeof YEARS[number])) {
      setYear(Number(yearFromQuery) as YearChoice);
    } else if (!yearFromQuery) {
      setYear("ALL");
    }
  }, [yearFromQuery]);
  
  const [extrapolate, setExtrapolate] = useState(false); // NEW toggle
  const [searchQuery, setSearchQuery] = useState<string>(""); // Search query state

  const [data, setData] = useState<PlayerRow[]>([]);
  const [drafts, setDrafts] = useState<DraftResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [draftLoading, setDraftLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draftError, setDraftError] = useState<string | null>(null);
  const [selectedPlayers, setSelectedPlayers] = useState<Map<string, {playerName: string, year: number, selectedYear: number, availableYears: number[], position?: {x: number, y: number}, stats?: PlayerWeeklyStatsResponse, loading?: boolean, headshotUrl?: string | null}>>(new Map());

  // Sorting state
  type SortColumn = 'player_name' | 'fantasy_pos' | 'drafter' | 'round' | 'fantasy_points_ppr' | 'ppr_per_game' | 'vorp_star' | 'true_vorp_star' | 'adj_vorp_star' | 'delta_vorp_star_mean' | 'weeks_played' | 'missed_weeks' | 'year';
  type SortDirection = 'asc' | 'desc';
  const [sortColumn, setSortColumn] = useState<SortColumn>('vorp_star');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  // Update default sort column when extrapolate changes
  useEffect(() => {
    if (extrapolate) {
      setSortColumn('adj_vorp_star');
    } else {
      setSortColumn('vorp_star');
    }
  }, [extrapolate]);

  // multi-select positions
  const POS_ALL = ["QB", "RB", "WR", "TE"] as const;
  const [posSet, setPosSet] = useState<Set<string>>(new Set(POS_ALL));

  const togglePos = (p: string) =>
    setPosSet((prev) => {
      const next = new Set(prev);
      if (next.has(p)) next.delete(p);
      else next.add(p);
      return next.size === 0 ? new Set(POS_ALL) : next;
    });

  const handlePlayerClick = async (playerName: string, year: number, event?: React.MouseEvent<HTMLDivElement>) => {
    const playerKey = `${playerName}_${year}`;
    
    // Check if this player is already open
    if (selectedPlayers.has(playerKey)) {
      return; // Don't reopen if already open
    }
    
    // Always center pop-ups, stack vertically when multiple
      const existingPopups = Array.from(selectedPlayers.values());
    const popupHeight = 200; // Increased size
    const popupSpacing = 20;
    
    // Add player to map with loading state
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      newMap.set(playerKey, { playerName, year, selectedYear: year, availableYears: YEARS as number[], position: { x: 0, y: 0 }, loading: true });
      return newMap;
    });
    
    try {
      // Fetch stats and headshot in parallel
      const [statsResponse, headshotResponse] = await Promise.all([
        fetch(`${API_BASE}/players/${encodeURIComponent(playerName)}/weekly-stats?year=${year}`),
        fetch(`${API_BASE}/players/${encodeURIComponent(playerName)}/headshot`)
      ]);
      
      if (!statsResponse.ok) {
        throw new Error('Failed to fetch player stats');
      }
      const data: PlayerWeeklyStatsResponse = await statsResponse.json();
      
      // Get headshot URL
      let headshotUrl: string | null = null;
      if (headshotResponse.ok) {
        const headshotData = await headshotResponse.json();
        headshotUrl = headshotData.headshot_url || null;
      }
      
      // Update with stats and headshot
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
        const existing = newMap.get(playerKey);
        if (existing) {
          newMap.set(playerKey, { ...existing, stats: data, headshotUrl: headshotUrl, loading: false });
        }
        return newMap;
      });
    } catch (err) {
      console.error('Error fetching player stats:', err);
      setError('Failed to load player weekly stats');
      
      // Remove on error
      setSelectedPlayers(prev => {
        const newMap = new Map(prev);
        newMap.delete(playerKey);
        return newMap;
      });
    }
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
      const response = await fetch(`${API_BASE}/players/${encodeURIComponent(playerData.playerName)}/weekly-stats?year=${newYear}`);
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

  const closePlayerPopup = (playerKey: string) => {
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      newMap.delete(playerKey);
      return newMap;
    });
  };

  // fetch players (season or extrapolated) + drafts
  useEffect(() => {
    let cancelled = false;

    async function runPlayers() {
      setLoading(true);
      setError(null);
      try {
        // Determine position filter for API calls
        const positionFilter = posSet.size === POS_ALL.length ? undefined : posSet;
        const yearCount = year === "ALL" ? YEARS.length : 1;
        
        if (year === "ALL") {
          if (extrapolate) {
            const all = await Promise.all(YEARS.map(y => fetchExtrapolated(y, positionFilter)));
            const rows = all.flatMap((r) =>
              (r.rows ?? []).map((p) => ({
                // project into unified PlayerRow
                player_name: p.player_name,
                team: p.team ?? null,
                fantasy_pos: p.fantasy_pos,
                fantasy_points_ppr: p.fantasy_points_ppr,                                              // NEW
                ppr_per_game: p.ppr_per_game ?? (p.weeks_played ? p.fantasy_points_ppr / p.weeks_played : undefined), // NEW
                true_vorp_star: p.true_vorp_star,
                delta_vorp_star_mean: p.delta_vorp_star_mean,
                delta_vorp_star_p10: p.delta_vorp_star_p10,
                delta_vorp_star_p90: p.delta_vorp_star_p90,
                adj_vorp_star: p.adj_vorp_star,
                weeks_played: p.weeks_played,
                missed_weeks: p.missed_weeks,
                year: r.year,
              }))
            );
            if (!cancelled) setData(rows);
          } else {
            const all = await Promise.all(YEARS.map(y => fetchVorp(y, positionFilter)));
            const rows = all.flatMap((r) =>
              (r.players ?? []).map((p) => ({
                ...p,
                // compute PPG client-side from totals and games
                ppr_per_game: p.fantasy_points_ppr && (p as any).g ? p.fantasy_points_ppr / (p as any).g : undefined, // NEW
                year: r.year,
              }))
            );
            if (!cancelled) setData(rows);
          }
        } else {
          if (extrapolate) {
            const r = await fetchExtrapolated(year, positionFilter);
            const rows = (r.rows ?? []).map((p) => ({
              player_name: p.player_name,
              team: p.team ?? null,
              fantasy_pos: p.fantasy_pos,
              fantasy_points_ppr: p.fantasy_points_ppr,            // NEW
              ppr_per_game: p.ppr_per_game ?? (p.weeks_played ? p.fantasy_points_ppr / p.weeks_played : undefined), // NEW
              true_vorp_star: p.true_vorp_star,
              delta_vorp_star_mean: p.delta_vorp_star_mean,
              delta_vorp_star_p10: p.delta_vorp_star_p10,
              delta_vorp_star_p90: p.delta_vorp_star_p90,
              adj_vorp_star: p.adj_vorp_star,
              weeks_played: p.weeks_played,
              missed_weeks: p.missed_weeks,
              year: r.year,
            }));

            if (!cancelled) setData(rows);
          } else {
            const r = await fetchVorp(year, positionFilter);
            const rows = (r.players ?? []).map((p) => ({
              ...p,
              // compute PPG client-side from totals and games
              ppr_per_game: p.fantasy_points_ppr && (p as any).g ? p.fantasy_points_ppr / (p as any).g : undefined, // NEW
              year: r.year,
            }));
            if (!cancelled) setData(rows);
          }
        }
      } catch (e: any) {
        if (!cancelled) setError(e?.message || "Failed to load players");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    async function runDrafts() {
      setDraftLoading(true);
      setDraftError(null);
      try {
        if (year === "ALL") {
          const ds = await Promise.all(YEARS.map(fetchDraft));
          if (!cancelled) setDrafts(ds);
        } else {
          const d = await fetchDraft(year);
          if (!cancelled) setDrafts([d]);
        }
      } catch (e: any) {
        if (!cancelled) {
          setDraftError(e?.message || "Failed to load draft data");
          setDrafts([]);
        }
      } finally {
        if (!cancelled) setDraftLoading(false);
      }
    }

    runPlayers();
    runDrafts();
    return () => {
      cancelled = true;
    };
  }, [year, extrapolate, posSet]); // NEW deps: extrapolate, posSet

  // (year|player) -> { team_id, drafter, round }
  const draftIndex = useMemo(() => {
    const m: Record<string, { team_id: number; drafter: string; round: number }> = {};
    for (const d of drafts) {
      for (const p of d.picks ?? []) {
        if (!p.player_name) continue;
        const key = `${d.year}|${normalizeName(p.player_name)}`;
        const round = typeof p.round_num === "number" ? p.round_num : 99;
        m[key] = { team_id: p.team_id, drafter: p.team_name, round };
      }
    }
    return m;
  }, [drafts]);

  // Handle sort
  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      // Toggle direction if clicking the same column
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      // Set new column and default to descending
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  // apply filters and sort
  const rows = useMemo(() => {
    const want = posSet;
    let r = data.filter((x) => want.has(x.fantasy_pos));
    
    // Apply search filter if search query exists
    if (searchQuery.trim()) {
      const query = searchQuery.trim().toLowerCase();
      r = r.filter((x) => x.player_name.toLowerCase().includes(query));
    }
    
    // Sort based on sortColumn and sortDirection
    return r.sort((a, b) => {
      let aValue: string | number | null | undefined;
      let bValue: string | number | null | undefined;
      
      switch (sortColumn) {
        case 'player_name':
          aValue = a.player_name.toLowerCase();
          bValue = b.player_name.toLowerCase();
          if (sortDirection === 'asc') {
            return aValue.localeCompare(bValue as string);
      } else {
            return (bValue as string).localeCompare(aValue as string);
          }
        
        case 'fantasy_pos':
          aValue = a.fantasy_pos;
          bValue = b.fantasy_pos;
          if (sortDirection === 'asc') {
            return (aValue as string).localeCompare(bValue as string);
          } else {
            return (bValue as string).localeCompare(aValue as string);
          }
        
        case 'drafter': {
          const yA = a.year ?? 0;
          const yB = b.year ?? 0;
          const dA = yA ? draftIndex[`${yA}|${normalizeName(a.player_name)}`] : undefined;
          const dB = yB ? draftIndex[`${yB}|${normalizeName(b.player_name)}`] : undefined;
          aValue = dA?.drafter ?? '—';
          bValue = dB?.drafter ?? '—';
          if (sortDirection === 'asc') {
            return (aValue as string).localeCompare(bValue as string);
          } else {
            return (bValue as string).localeCompare(aValue as string);
          }
        }
        
        case 'round': {
          const yA = a.year ?? 0;
          const yB = b.year ?? 0;
          const dA = yA ? draftIndex[`${yA}|${normalizeName(a.player_name)}`] : undefined;
          const dB = yB ? draftIndex[`${yB}|${normalizeName(b.player_name)}`] : undefined;
          aValue = dA?.round ?? 99;
          bValue = dB?.round ?? 99;
          if (sortDirection === 'asc') {
            return (aValue as number) - (bValue as number);
          } else {
            return (bValue as number) - (aValue as number);
          }
        }
        
        case 'fantasy_points_ppr':
          aValue = a.fantasy_points_ppr ?? 0;
          bValue = b.fantasy_points_ppr ?? 0;
          if (sortDirection === 'asc') {
            return (aValue as number) - (bValue as number);
          } else {
            return (bValue as number) - (aValue as number);
          }
        
        case 'ppr_per_game':
          aValue = a.ppr_per_game ?? 0;
          bValue = b.ppr_per_game ?? 0;
          if (sortDirection === 'asc') {
            return (aValue as number) - (bValue as number);
          } else {
            return (bValue as number) - (aValue as number);
          }
        
        case 'vorp_star':
          aValue = a.vorp_star ?? 0;
          bValue = b.vorp_star ?? 0;
          if (sortDirection === 'asc') {
            return (aValue as number) - (bValue as number);
          } else {
            return (bValue as number) - (aValue as number);
          }
        
        case 'true_vorp_star':
          aValue = a.true_vorp_star ?? 0;
          bValue = b.true_vorp_star ?? 0;
          if (sortDirection === 'asc') {
            return (aValue as number) - (bValue as number);
          } else {
            return (bValue as number) - (aValue as number);
          }
        
        case 'adj_vorp_star':
          aValue = a.adj_vorp_star ?? 0;
          bValue = b.adj_vorp_star ?? 0;
          if (sortDirection === 'asc') {
            return (aValue as number) - (bValue as number);
          } else {
            return (bValue as number) - (aValue as number);
          }
        
        case 'delta_vorp_star_mean':
          aValue = a.delta_vorp_star_mean ?? 0;
          bValue = b.delta_vorp_star_mean ?? 0;
          if (sortDirection === 'asc') {
            return (aValue as number) - (bValue as number);
          } else {
            return (bValue as number) - (aValue as number);
          }
        
        case 'weeks_played':
          aValue = a.weeks_played ?? 0;
          bValue = b.weeks_played ?? 0;
          if (sortDirection === 'asc') {
            return (aValue as number) - (bValue as number);
          } else {
            return (bValue as number) - (aValue as number);
          }
        
        case 'missed_weeks':
          aValue = a.missed_weeks ?? 0;
          bValue = b.missed_weeks ?? 0;
          if (sortDirection === 'asc') {
            return (aValue as number) - (bValue as number);
          } else {
            return (bValue as number) - (aValue as number);
          }
        
        case 'year':
          aValue = a.year ?? 0;
          bValue = b.year ?? 0;
          if (sortDirection === 'asc') {
            return (aValue as number) - (bValue as number);
          } else {
            return (bValue as number) - (aValue as number);
          }
        
        default:
          return 0;
      }
    });
  }, [data, posSet, extrapolate, searchQuery, sortColumn, sortDirection, draftIndex]);

  // Total WAR by team_id; if extrapolate, use adjusted WAR, else true WAR
  const warTotalsByTeam = useMemo(() => {
    const acc: Record<number, number> = {};

    for (const r of rows) {
      const y = r.year ?? 0;
      if (!y) continue;

      const di = draftIndex[`${y}|${normalizeName(r.player_name)}`];
      if (!di) continue;
      if (EXCLUDE_DRAFTER_NAMES.has(di.drafter)) continue;

      let baseWar: number;
      if (extrapolate) {
        baseWar = r.adj_vorp_star ?? r.true_vorp_star ?? 0;
      } else {
        baseWar = r.vorp_star ?? 0;
      }

      const round = di.round ?? 99;
      let effective = baseWar;
      if (round >= 9 && baseWar < 0) {
        effective = baseWar / 2;
      }

      acc[di.team_id] = (acc[di.team_id] ?? 0) + effective;
    }

    return Object.entries(acc)
      .filter(([id]) => TEAM_NAME_MAP[Number(id)] != null)
      .map(([id, total]) => ({
        team_id: Number(id),
        display: TEAM_NAME_MAP[Number(id)]!,
        total,
      }))
      .sort((a, b) => b.total - a.total);
  }, [rows, draftIndex, extrapolate]);

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="max-w-none mx-0 pr-8 pt-8 pb-8 pl-5 space-y-8">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold text-white mb-4">
          Fantasy Football Dashboard
        </h1>
        <p className="text-xl text-slate-400">
          League 86952922 - Complete Analytics & Insights
        </p>
      </div>

      {/* Navigation */}
      <nav className="flex justify-center items-center gap-3 mb-12 flex-wrap">
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

      {/* Filters */}
      <div className="mb-6 flex items-center justify-center gap-4 flex-wrap">
        {/* Year dropdown with ALL */}
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-400">Year:</label>
          <select
            className="bg-slate-800 text-slate-100 text-sm font-medium rounded-lg px-3 py-1.5 border border-slate-700 hover:border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-colors cursor-pointer"
            value={year}
            onChange={(e) => setYear(e.target.value === "ALL" ? "ALL" : (Number(e.target.value) as YearChoice))}
          >
            <option value="ALL">ALL</option>
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>

        {/* Position multi-filter */}
        <div className="flex items-center gap-1 bg-slate-800 rounded-md p-1 border border-slate-700">
          {(["QB", "RB", "WR", "TE"] as const).map((p) => {
            const active = posSet.has(p);
            return (
              <button
                key={p}
                onClick={() => togglePos(p)}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  active ? "bg-indigo-600 text-white font-semibold" : "text-slate-300 hover:text-white hover:bg-slate-700"
                }`}
              >
                {p}
              </button>
            );
          })}
        </div>
      </div>

      {/* WAR totals by drafter */}
      {/* <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-slate-900/80 p-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">
            Total {extrapolate ? "Adjusted ZAV" : "ZAV"} by Drafter
          </h2>
          <div className="flex items-center gap-2">
            {(loading || draftLoading) && (
              <span className="text-xs font-medium text-white bg-emerald-600 px-2 py-0.5 rounded">Loading…</span>
            )}
            {(error || draftError) && (
              <span className="text-xs font-medium text-white bg-rose-600 px-2 py-0.5 rounded">
                {error || draftError}
              </span>
            )}
          </div>
        </div>

        <div className="rounded ring-1 ring-zinc-100 dark:ring-zinc-800 overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-emerald-600 text-white">
              <tr>
                <th className="text-left p-2">#</th>
                <th className="text-left p-2">Drafter</th>
                <th className="text-right p-2">{extrapolate ? "Adjusted ZAV" : "Total ZAV"}</th>
              </tr>
            </thead>
            <tbody className="text-zinc-700 dark:text-zinc-200">
              {warTotalsByTeam.length > 0 ? (
                warTotalsByTeam.map((row, i) => (
                  <tr
                    key={row.team_id}
                    className={`border-t border-zinc-200 dark:border-zinc-800 ${
                      i % 2 === 1 ? "bg-slate-50/60 dark:bg-slate-800/40" : ""
                    }`}
                  >
                    <td className="p-2 font-medium">{i + 1}</td>
                    <td className="p-2">{row.display}</td>
                    <td className="p-2 text-right font-semibold text-emerald-700 dark:text-emerald-400">
                      {row.total.toFixed(2)}
                    </td>
                  </tr>
                ))
              ) : (
                <tr className="border-t border-zinc-200 dark:border-zinc-800">
                  <td colSpan={3} className="p-3 text-center text-zinc-500 dark:text-zinc-400">
                    {loading || draftLoading ? "Loading…" : "No data for current filters."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div> */}

      {/* Players table */}
      <div className="bg-slate-900 rounded-2xl shadow-xl shadow-slate-950/50 border border-slate-800 overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 gap-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-slate-100">
              {year === "ALL" ? "All Years" : year} •{" "}
              {posSet.size === POS_ALL.length ? "All Positions" : Array.from(posSet).join(", ")}
            </h2>
            <span className="inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-sm">
              {rows.length} players
            </span>
          </div>
          
          {/* Search input */}
          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="Search players..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="rounded-md bg-slate-800 text-slate-100 px-3 py-1.5 text-sm outline-none ring-1 ring-slate-700 hover:ring-slate-600 focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-colors min-w-[200px]"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="text-slate-400 hover:text-slate-200 transition-colors"
                title="Clear search"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gradient-to-r from-slate-800 to-slate-700">
                <th 
                  className="text-left px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                  onClick={() => handleSort('player_name')}
                >
                  <div className="flex items-center gap-2">
                    Player
                    {sortColumn === 'player_name' && (
                      <span className="text-xs">
                        {sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                <th 
                  className="text-left px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                  onClick={() => handleSort('fantasy_pos')}
                >
                  <div className="flex items-center gap-2">
                    Pos
                    {sortColumn === 'fantasy_pos' && (
                      <span className="text-xs">
                        {sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                <th 
                  className="text-left px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                  onClick={() => handleSort('drafter')}
                >
                  <div className="flex items-center gap-2">
                    Drafted By
                    {sortColumn === 'drafter' && (
                      <span className="text-xs">
                        {sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                <th 
                  className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                  onClick={() => handleSort('round')}
                >
                  <div className="flex items-center justify-end gap-2">
                    Rnd
                    {sortColumn === 'round' && (
                      <span className="text-xs">
                        {sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                {!extrapolate && (
                  <>
                    <th 
                      className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                      onClick={() => handleSort('fantasy_points_ppr')}
                    >
                      <div className="flex items-center justify-end gap-2">
                        PPR Points
                        {sortColumn === 'fantasy_points_ppr' && (
                          <span className="text-xs">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                    <th 
                      className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                      onClick={() => handleSort('ppr_per_game')}
                    >
                      <div className="flex items-center justify-end gap-2">
                        PPG
                        {sortColumn === 'ppr_per_game' && (
                          <span className="text-xs">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                    <th 
                      className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                      onClick={() => handleSort('vorp_star')}
                    >
                      <div className="flex items-center justify-end gap-2">
                        ZAV
                        {sortColumn === 'vorp_star' && (
                          <span className="text-xs">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                  </>
                )}
                {extrapolate && (
                  <>
                    <th 
                      className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                      onClick={() => handleSort('fantasy_points_ppr')}
                    >
                      <div className="flex items-center justify-end gap-2">
                        PPR Points
                        {sortColumn === 'fantasy_points_ppr' && (
                          <span className="text-xs">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                    <th 
                      className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                      onClick={() => handleSort('ppr_per_game')}
                    >
                      <div className="flex items-center justify-end gap-2">
                        PPG
                        {sortColumn === 'ppr_per_game' && (
                          <span className="text-xs">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                    <th 
                      className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                      onClick={() => handleSort('true_vorp_star')}
                    >
                      <div className="flex items-center justify-end gap-2">
                        True ZAV
                        {sortColumn === 'true_vorp_star' && (
                          <span className="text-xs">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                    <th 
                      className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                      onClick={() => handleSort('delta_vorp_star_mean')}
                    >
                      <div className="flex items-center justify-end gap-2">
                        Injury Δ (μ)
                        {sortColumn === 'delta_vorp_star_mean' && (
                          <span className="text-xs">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                    <th 
                      className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                      onClick={() => handleSort('adj_vorp_star')}
                    >
                      <div className="flex items-center justify-end gap-2">
                        Adj ZAV
                        {sortColumn === 'adj_vorp_star' && (
                          <span className="text-xs">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                    <th 
                      className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                      onClick={() => handleSort('weeks_played')}
                    >
                      <div className="flex items-center justify-end gap-2">
                        Wks Played
                        {sortColumn === 'weeks_played' && (
                          <span className="text-xs">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                    <th 
                      className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                      onClick={() => handleSort('missed_weeks')}
                    >
                      <div className="flex items-center justify-end gap-2">
                        Wks Missed
                        {sortColumn === 'missed_weeks' && (
                          <span className="text-xs">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                  </>
                )}
                <th 
                  className="text-right px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                  onClick={() => handleSort('year')}
                >
                  <div className="flex items-center justify-end gap-2">
                    Year
                    {sortColumn === 'year' && (
                      <span className="text-xs">
                        {sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {rows.map((r, i) => {
                const y = r.year ?? 0;
                const d = y ? draftIndex[`${y}|${normalizeName(r.player_name)}`] : undefined;
                return (
                  <tr
                    key={`${r.player_name}-${r.year}-${i}`}
                    className="group hover:bg-slate-800/50 transition-all duration-200 ease-in-out"
                  >
                    <td className="px-6 py-5">
                      <div 
                        className="cursor-pointer font-semibold text-slate-100 hover:text-indigo-400 transition-colors"
                        onClick={(e) => handlePlayerClick(r.player_name, r.year ?? 2024, e)}
                      >
                        {r.player_name}
                      </div>
                    </td>
                    <td className="px-6 py-5 text-slate-300">{r.fantasy_pos}</td>
                    <td className="px-6 py-5 text-slate-300">{d?.drafter ?? "—"}</td>
                    <td className="px-6 py-5 text-right text-slate-300">{d?.round ?? "—"}</td>

                    {!extrapolate && (
                      <>
                        <td className="px-6 py-5 text-right text-slate-300">{((r.fantasy_points_ppr ?? 0)).toFixed(1)}</td>
                        <td className="px-6 py-5 text-right text-slate-300">
                          {typeof r.ppr_per_game === "number" ? (r.ppr_per_game).toFixed(2) : "—"}
                        </td>
                        <td className="px-6 py-5 text-right font-semibold text-emerald-400">
                          {typeof r.vorp_star === "number" ? r.vorp_star.toFixed(2) : "—"}
                        </td>
                      </>
                    )}

                    {extrapolate && (
                      <>
                        <td className="px-6 py-5 text-right text-slate-300">{((r.fantasy_points_ppr ?? 0)).toFixed(1)}</td>
                        <td className="px-6 py-5 text-right text-slate-300">
                          {typeof r.ppr_per_game === "number" ? (r.ppr_per_game).toFixed(2) : "—"}
                        </td>
                        <td className="px-6 py-5 text-right font-semibold text-emerald-400">
                          {typeof r.true_vorp_star === "number" ? r.true_vorp_star.toFixed(2) : "—"}
                        </td>
                        <td className="px-6 py-5 text-right text-slate-300">
                          {typeof r.delta_vorp_star_mean === "number" ? 
                            (r.delta_vorp_star_mean >= 0 ? "+" : "") + r.delta_vorp_star_mean.toFixed(2) : "—"}
                        </td>
                        <td className="px-6 py-5 text-right font-bold text-blue-400">
                          {typeof r.adj_vorp_star === "number" ? r.adj_vorp_star.toFixed(2) : "—"}
                        </td>
                        <td className="px-6 py-5 text-right text-slate-400">
                          {typeof r.weeks_played === "number" ? r.weeks_played : "—"}
                        </td>
                        <td className="px-6 py-5 text-right text-slate-400">
                          {typeof r.missed_weeks === "number" ? r.missed_weeks : "—"}
                        </td>
                      </>
                    )}

                    <td className="px-6 py-5 text-right text-slate-300">{r.year ?? "—"}</td>
                  </tr>
                );
              })}

              {!loading && !error && rows.length === 0 && (
                <tr>
                  <td colSpan={extrapolate ? 12 : 8} className="px-6 py-12 text-center">
                    <div className="text-slate-400">
                      No players match the current filters.
                    </div>
                  </td>
                </tr>
              )}
              {(error || draftError) && (
                <tr>
                  <td colSpan={extrapolate ? 12 : 8} className="px-6 py-12 text-center">
                    <div className="text-rose-400">
                      {error || draftError}
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

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
                      {statIndex > 0 && playerData.stats && playerData.stats.weekly_stats[statIndex - 1] && 
                          playerData.stats.weekly_stats[statIndex - 1].week < playerData.selectedYear && 
                          stat.week >= playerData.selectedYear && (
                        <div className="w-0.5 h-16 bg-white mx-0.5"></div>
                      )}
                    </div>
                  );
                })}
              </div>
                {playerData.stats && (
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
                )}
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
    </main>
  );
  }

export default function PlayersPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">Loading...</div>}>
      <PlayersPageContent />
    </Suspense>
  );
}
