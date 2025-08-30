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
};

type StandingsResponse = {
  year: number;
  num_teams: number;
  teams: TeamRow[];
};

const YEARS = [2020, 2021, 2022, 2024];
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

async function fetchStandings(year: number): Promise<StandingsResponse> {
  const res = await fetch(`${API_BASE}/standings/${year}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch standings for ${year}`);
  return res.json();
}

export default function Home() {
  const [year, setYear] = useState<number>(YEARS[YEARS.length - 1]); // default to latest
  const [data, setData] = useState<StandingsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchStandings(year)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
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
            onChange={(e) => setYear(Number(e.target.value))}
          >
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

      {/* Single standings card */}
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-slate-900/80 flex-1">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xl font-semibold text-zinc-800 dark:text-zinc-100 flex items-center gap-2">
            {data?.year ?? year}
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
                      <td className="p-2 text-right">{t.points_for.toFixed(1)}</td>
                      <td className="p-2 text-right">{t.points_against.toFixed(1)}</td>
                    </tr>
                  ))
                : (
                  <tr className="border-t border-zinc-200 dark:border-zinc-800">
                    <td className="p-3 text-center text-zinc-500 dark:text-zinc-400" colSpan={5}>
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
