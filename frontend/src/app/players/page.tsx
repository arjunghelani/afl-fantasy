"use client";

import { useEffect, useMemo, useState } from "react";

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
  weeks_in_season: int;
  count: number;
  rows: ExtrapolatedRow[];
};

/* -------------------------- config -------------------------- */
const YEARS = [2020, 2021, 2022, 2024] as const;
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
export default function PlayersPage() {
  const [year, setYear] = useState<YearChoice>("ALL");
  const [extrapolate, setExtrapolate] = useState(false); // NEW toggle

  const [data, setData] = useState<PlayerRow[]>([]);
  const [drafts, setDrafts] = useState<DraftResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [draftLoading, setDraftLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draftError, setDraftError] = useState<string | null>(null);

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

  // apply filters and sort (by chosen WAR)
  const rows = useMemo(() => {
    const want = posSet;
    const r = data.filter((x) => want.has(x.fantasy_pos));
    return r.sort((a, b) => {
      let aWar: number;
      let bWar: number;
      
      if (extrapolate) {
        aWar = a.adj_vorp_star ?? a.true_vorp_star ?? 0;
        bWar = b.adj_vorp_star ?? b.true_vorp_star ?? 0;
      } else {
        aWar = a.vorp_star ?? 0;
        bWar = b.vorp_star ?? 0;
      }
      
      return bWar - aWar;
    });
  }, [data, posSet, extrapolate]);

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
    <main className="mx-auto max-w-5xl p-6 space-y-8 bg-slate-50 min-h-screen dark:bg-[#0b0f13]">
      {/* Top bar */}
      <div className="rounded-xl bg-emerald-700 text-white px-4 py-3 flex items-center justify-between shadow-sm">
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Players • WAR</h1>

        <div className="flex items-center gap-4">
          {/* Year dropdown with ALL */}
          <label className="hidden sm:block text-sm text-white/80">Year</label>
          <select
            className="rounded-md bg-white/15 text-white px-2 py-1 text-sm outline-none ring-1 ring-white/20 hover:bg-white/20"
            value={year}
            onChange={(e) => setYear(e.target.value === "ALL" ? "ALL" : (Number(e.target.value) as YearChoice))}
          >
            <option value="ALL" className="text-black">ALL</option>
            {YEARS.map((y) => (
              <option key={y} value={y} className="text-black">{y}</option>
            ))}
          </select>

          {/* Position multi-filter */}
          <div className="hidden sm:flex items-center gap-1 bg-white/10 rounded-md p-1">
            {(["QB", "RB", "WR", "TE"] as const).map((p) => {
              const active = posSet.has(p);
              return (
                <button
                  key={p}
                  onClick={() => togglePos(p)}
                  className={`px-2 py-1 rounded text-sm ${
                    active ? "bg-white text-emerald-700 font-semibold" : "text-white/90 hover:text-white"
                  }`}
                >
                  {p}
                </button>
              );
            })}
          </div>

          {/* NEW: Extrapolate injuries toggle */}
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="accent-white"
              checked={extrapolate}
              onChange={(e) => setExtrapolate(e.target.checked)}
            />
            <span className="text-white/90">Extrapolate injuries</span>
          </label>

          {/* Nav */}
          <nav className="flex items-center gap-5">
            <a href="/" className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline">
              Standings
            </a>
            <a href="/draft" className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline">
              Drafts
            </a>
          </nav>
        </div>
      </div>

      {/* WAR totals by drafter */}
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-slate-900/80 p-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">
            Total {extrapolate ? "Adjusted WAR" : "WAR"} by Drafter
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
                <th className="text-right p-2">{extrapolate ? "Adjusted WAR" : "Total WAR"}</th>
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
      </div>

      {/* Players table */}
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-slate-900/80 p-0 overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">
              {year === "ALL" ? "All Years" : year} •{" "}
              {posSet.size === POS_ALL.length ? "All Positions" : Array.from(posSet).join(", ")}
            </h2>
            <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-900/40 dark:text-fuchsia-300">
              {rows.length} players
            </span>
          </div>
        </div>

        <div className="h-[60vh] md:h-[60vh] overflow-auto rounded-t-lg ring-1 ring-zinc-100 dark:ring-zinc-800">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-emerald-600 text-white">
              <tr>
                <th className="text-left p-2">Player</th>
                <th className="text-left p-2">Pos</th>
                <th className="text-left p-2">NFL Team</th>
                <th className="text-left p-2">Drafted By</th>
                <th className="text-right p-2">Rnd</th>
                {!extrapolate && (
                  <>
                    <th className="text-right p-2">PPR Points</th>
                    <th className="text-right p-2">PPG</th> {/* NEW */}
                    <th className="text-right p-2">WAR</th>
                  </>
                )}
                {extrapolate && (
                  <>
                    <th className="text-right p-2">PPR Points</th> {/* NEW */}
                    <th className="text-right p-2">PPG</th>         {/* NEW */}
                    <th className="text-right p-2">True WAR</th>
                    <th className="text-right p-2">Injury Δ (μ)</th>
                    <th className="text-right p-2">Adj WAR</th>
                    <th className="text-right p-2">Wks Played</th>
                    <th className="text-right p-2">Wks Missed</th>
                  </>
                )}
                <th className="text-right p-2">Year</th>
              </tr>
            </thead>
            <tbody className="text-zinc-700 dark:text-zinc-200">
              {rows.map((r, i) => {
                const y = r.year ?? 0;
                const d = y ? draftIndex[`${y}|${normalizeName(r.player_name)}`] : undefined;
                return (
                  <tr
                    key={`${r.player_name}-${r.year}-${i}`}
                    className={`border-t border-zinc-200 dark:border-zinc-800 ${
                      i % 2 === 1 ? "bg-slate-50/60 dark:bg-slate-800/40" : ""
                    }`}
                  >
                    <td className="p-2">{r.player_name}</td>
                    <td className="p-2">{r.fantasy_pos}</td>
                    <td className="p-2">{r.team ?? "—"}</td>
                    <td className="p-2">{d && TEAM_NAME_MAP[d.team_id] ? TEAM_NAME_MAP[d.team_id] : "—"}</td>
                    <td className="p-2 text-right">{d?.round ?? "—"}</td>

                    {!extrapolate && (
                      <>
                        <td className="p-2 text-right">{(r.fantasy_points_ppr ?? 0).toFixed(1)}</td>
                        <td className="p-2 text-right">
                          {typeof r.ppr_per_game === "number" ? r.ppr_per_game.toFixed(2) : "—"}
                        </td>
                        <td className="p-2 text-right font-semibold text-emerald-700 dark:text-emerald-400">
                          {typeof r.vorp_star === "number" ? r.vorp_star.toFixed(2) : "—"}
                        </td>
                      </>
                    )}

                    {extrapolate && (
                      <>
                        <td className="p-2 text-right">{(r.fantasy_points_ppr ?? 0).toFixed(1)}</td>
                        <td className="p-2 text-right">
                          {typeof r.ppr_per_game === "number" ? r.ppr_per_game.toFixed(2) : "—"}
                        </td>
                        <td className="p-2 text-right font-semibold text-emerald-700 dark:text-emerald-400">
                          {typeof r.true_vorp_star === "number" ? r.true_vorp_star.toFixed(2) : "—"}
                        </td>
                        <td className="p-2 text-right">
                          {typeof r.delta_vorp_star_mean === "number" ? 
                            (r.delta_vorp_star_mean >= 0 ? "+" : "") + r.delta_vorp_star_mean.toFixed(2) : "—"}
                        </td>
                        <td className="p-2 text-right font-bold text-blue-700 dark:text-blue-400">
                          {typeof r.adj_vorp_star === "number" ? r.adj_vorp_star.toFixed(2) : "—"}
                        </td>
                        <td className="p-2 text-right text-zinc-500">
                          {typeof r.weeks_played === "number" ? r.weeks_played : "—"}
                        </td>
                        <td className="p-2 text-right text-zinc-500">
                          {typeof r.missed_weeks === "number" ? r.missed_weeks : "—"}
                        </td>
                      </>
                    )}

                    <td className="p-2 text-right">{r.year ?? "—"}</td>
                  </tr>
                );
              })}

              {!loading && !error && rows.length === 0 && (
                <tr className="border-t border-zinc-200 dark:border-zinc-800">
                  <td colSpan={extrapolate ? 13 : 9} className="p-4 text-center text-zinc-500 dark:text-zinc-400">
                    No players match the current filters.
                  </td>
                </tr>
              )}
              {(error || draftError) && (
                <tr className="border-t border-zinc-200 dark:border-zinc-800">
                  <td colSpan={extrapolate ? 13 : 9} className="p-4 text-center text-zinc-500 dark:text-zinc-400">
                    {error || draftError}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}
