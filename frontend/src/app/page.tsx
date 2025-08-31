"use client";

import { useEffect, useState } from "react";

type TeamRow = {
  team_id: number;
  team_name: string;
  wins: number;
  losses: number;
  ties: number;
  win_percentage: number;
  points_for: number;
  points_against: number;
  expected_wins?: number;        // NEW
};

type StandingsResponse = {
  year: number;
  num_teams: number;
  teams: TeamRow[];
};

const YEARS = [2020, 2021, 2022, 2024] as const;
type YearChoice = (typeof YEARS)[number] | "ALL";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

// NEW: exclude these team names (case-insensitive)
const EXCLUDED_TEAM_NAMES = new Set(["team ned"]);
const isExcludedTeamName = (name?: string) =>
  !!name && EXCLUDED_TEAM_NAMES.has(name.trim().toLowerCase());


async function fetchStandings(year: number): Promise<StandingsResponse> {
  const res = await fetch(`${API_BASE}/standings/${year}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch standings for ${year}`);
  return res.json();
}

/** Aggregate multiple StandingsResponse objects by team_id (sums xW too). */
function aggregateStandingsByTeamId(all: StandingsResponse[]): StandingsResponse {
  // team_id -> accumulator
  const acc: Record<number, {
    team_id: number;
    team_name: string;      // from most recent year seen
    latest_year: number;
    wins: number;
    losses: number;
    ties: number;
    points_for: number;
    points_against: number;
    expected_wins: number;  // NEW
  }> = {};

  for (const resp of all) {
    const y = resp.year ?? 0;
    for (const t of resp.teams) {
      if (isExcludedTeamName(t.team_name)) continue
      const cur = acc[t.team_id] ?? {
        team_id: t.team_id,
        team_name: t.team_name,
        latest_year: y,
        wins: 0, losses: 0, ties: 0,
        points_for: 0, points_against: 0,
        expected_wins: 0, // NEW
      };

      // use most recent team_name (names can change)
      if (y >= cur.latest_year && t.team_name) {
        cur.team_name = t.team_name;
        cur.latest_year = y;
      }

      cur.wins += t.wins;
      cur.losses += t.losses;
      cur.ties += t.ties;
      cur.points_for += t.points_for;
      cur.points_against += t.points_against;
      cur.expected_wins += t.expected_wins ?? 0; // NEW

      acc[t.team_id] = cur;
    }
  }

  const teams: TeamRow[] = Object.values(acc).map((r) => {
    const games = r.wins + r.losses + r.ties;
    const winpct = games > 0 ? Math.round((r.wins / games) * 1000) / 10 : 0; // one decimal
    return {
      team_id: r.team_id,
      team_name: r.team_name,
      wins: r.wins,
      losses: r.losses,
      ties: r.ties,
      win_percentage: winpct,
      points_for: r.points_for,
      points_against: r.points_against,
      expected_wins: r.expected_wins, // NEW
    };
  });

  // sort similar to API (Win% desc, then PF desc)
  teams.sort((a, b) => {
    if (b.win_percentage !== a.win_percentage) return b.win_percentage - a.win_percentage;
    return b.points_for - a.points_for;
  });

  return {
    year: 0,                // not used when "ALL" is selected
    num_teams: teams.length,
    teams,
  };
}

export default function Home() {
  const [year, setYear] = useState<YearChoice>(YEARS[YEARS.length - 1]); // default to latest
  const [data, setData] = useState<StandingsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        if (year === "ALL") {
          const all = await Promise.all(YEARS.map(fetchStandings));
          const agg = aggregateStandingsByTeamId(all);
          if (!cancelled) setData(agg);
        } else {
          const d = await fetchStandings(year);
          // NEW: remove Team Ned rows and fix team count
          const filteredTeams = d.teams.filter(t => !isExcludedTeamName(t.team_name));
          if (!cancelled) setData({ ...d, teams: filteredTeams, num_teams: filteredTeams.length });

        }
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message || e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [year]);

  return (
    <main className="mx-auto max-w-5xl p-6 space-y-8 bg-slate-50 min-h-screen dark:bg-[#0b0f13]">
      {/* Top bar: title + nav + year picker */}
      <div className="rounded-xl bg-emerald-700 text-white px-4 py-3 flex items-center justify-between shadow-sm">
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
          League 86952922 Standings
        </h1>

        <div className="flex items-center gap-4">
          {/* Year dropdown */}
          <label className="hidden sm:block text-sm text-white/80">Year</label>
          <select
            className="rounded-md bg-white/15 text-white px-2 py-1 text-sm outline-none ring-1 ring-white/20 hover:bg-white/20"
            value={year}
            onChange={(e) => {
              const v = e.target.value;
              setYear(v === "ALL" ? "ALL" : Number(v) as YearChoice);
            }}
          >
            <option value="ALL" className="text-black">ALL</option>
            {YEARS.map((y) => (
              <option value={y} key={y} className="text-black">
                {y}
              </option>
            ))}
          </select>

          {/* Nav */}
          <nav className="flex items-center gap-5">
            <a
              href="/draft"
              className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline"
            >
              View Drafts
            </a>
            <a
              href="/playoffs"
              className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline"
            >
              View Playoffs
            </a>
          </nav>
        </div>
      </div>

      {/* Standings card (single year or ALL aggregated) */}
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-slate-900/80 flex-1">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xl font-semibold text-zinc-800 dark:text-zinc-100 flex items-center gap-2">
            {year === "ALL" ? "ALL YEARS" : (data?.year ?? "")}
            {data && (
              <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-900/40 dark:text-fuchsia-300">
                {data.num_teams} teams
              </span>
            )}
          </h2>

          {/* Loading / Error badges */}
          {loading && (
            <span className="text-xs font-medium text-white bg-emerald-600 px-2 py-0.5 rounded">
              Loading…
            </span>
          )}
          {error && (
            <span className="text-xs font-medium text-white bg-rose-600 px-2 py-0.5 rounded">
              {error}
            </span>
          )}
        </div>

        {/* Full table, no scroll */}
        <div className="rounded-lg ring-1 ring-zinc-100 dark:ring-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-emerald-600 text-white">
              <tr>
                <th className="text-left p-2">Team</th>
                <th className="text-right p-2">W-L-T</th>
                <th className="text-right p-2">Win%</th>
                <th className="text-right p-2">xW</th>           {/* NEW */}
                <th className="text-right p-2">PF</th>
                <th className="text-right p-2">PA</th>
              </tr>
            </thead>
            <tbody className="text-zinc-700 dark:text-zinc-200">
              {!loading && !error && data?.teams?.length
                ? data.teams.map((t: TeamRow, i: number) => (
                    <tr
                      key={t.team_id}
                      className={`
                        border-t border-zinc-200 dark:border-zinc-800
                        ${i % 2 === 1 ? "bg-slate-50/60 dark:bg-slate-800/40" : ""}
                      `}
                    >
                      <td className="p-2">{t.team_name}</td>
                      <td className="p-2 text-right">
                        {t.wins}-{t.losses}-{t.ties}
                      </td>
                      <td className="p-2 text-right font-semibold text-emerald-700 dark:text-emerald-400">
                        {t.win_percentage}%
                      </td>
                      <td className="p-2 text-right">
                        {typeof t.expected_wins === "number" ? t.expected_wins.toFixed(1) : "—"}
                      </td>
                      <td className="p-2 text-right">{t.points_for.toFixed(1)}</td>
                      <td className="p-2 text-right">{t.points_against.toFixed(1)}</td>
                    </tr>
                  ))
                : (
                  <tr className="border-t border-zinc-200 dark:border-zinc-800">
                    <td className="p-3 text-center text-zinc-500 dark:text-zinc-400" colSpan={6}>
                      {loading ? "Loading…" : error ? "Failed to load standings." : "No data."}
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
