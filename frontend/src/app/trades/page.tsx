"use client";

import { useEffect, useState } from "react";

/* --------------------------- types --------------------------- */
type TradeRow = {
  trade_week: number;
  player_name: string;
  player_id: number;
  from_team_name: string;
  to_team_name: string;
  tenure_weeks: number;
  weeks_with_data: number;
  total_vorp_star: number;
  avg_weekly_vorp_star: number;
  direction: string;
};

type TradeSummary = {
  trade_week: number;
  team_a: string;
  team_b: string;
  team_a_vorp_received: number;
  team_b_vorp_received: number;
  net_advantage: number;
  winner: string;
  players_to_a: string;
  players_to_b: string;
};

type TradesResponse = {
  year: number;
  trade_values: TradeRow[];
  trade_summary: TradeSummary[];
  count: number;
};

/* -------------------------- config -------------------------- */
const YEARS = [2020, 2021, 2022, 2024] as const;
type YearChoice = (typeof YEARS)[number] | "ALL";
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

/* ------------------------ data fetchers ---------------------- */
async function fetchTrades(year: number): Promise<TradesResponse> {
  const res = await fetch(`${API_BASE}/trades/${year}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Trades ${year} failed: ${res.status} ${res.statusText} ${body}`);
  }
  return res.json();
}

/* --------------------------- page --------------------------- */
export default function TradesPage() {
  const [year, setYear] = useState<YearChoice>("ALL");
  const [data, setData] = useState<TradesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // fetch trades
  useEffect(() => {
    let cancelled = false;

    async function runTrades() {
      setLoading(true);
      setError(null);
      try {
        if (year === "ALL") {
          // For now, just show 2024 data when "ALL" is selected
          const r = await fetchTrades(2024);
          if (!cancelled) setData(r);
        } else {
          const r = await fetchTrades(year);
          if (!cancelled) setData(r);
        }
      } catch (e: any) {
        if (!cancelled) setError(e?.message || "Failed to load trades");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    runTrades();
    return () => {
      cancelled = true;
    };
  }, [year]);

  return (
    <main className="mx-auto max-w-6xl p-6 space-y-8 bg-slate-50 min-h-screen dark:bg-[#0b0f13]">
      {/* Top bar */}
      <div className="rounded-xl bg-emerald-700 text-white px-4 py-3 flex items-center justify-between shadow-sm">
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Trades • Analysis</h1>

        <div className="flex items-center gap-4">
          {/* Year dropdown */}
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

          {/* Nav */}
          <nav className="flex items-center gap-5">
            <a href="/" className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline">
              Standings
            </a>
            <a href="/players" className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline">
              Players
            </a>
            <a href="/draft" className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline">
              Drafts
            </a>
          </nav>
        </div>
      </div>

      {/* Trade Summary */}
      {data && data.trade_summary && data.trade_summary.length > 0 && (
        <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-slate-900/80 p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">
              Trade Winners Summary
            </h2>
            <div className="flex items-center gap-2">
              {loading && (
                <span className="text-xs font-medium text-white bg-emerald-600 px-2 py-0.5 rounded">Loading…</span>
              )}
              {error && (
                <span className="text-xs font-medium text-white bg-rose-600 px-2 py-0.5 rounded">
                  {error}
                </span>
              )}
            </div>
          </div>

          <div className="rounded ring-1 ring-zinc-100 dark:ring-zinc-800 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-emerald-600 text-white">
                <tr>
                  <th className="text-left p-2">Week</th>
                  <th className="text-left p-2">Trade</th>
                  <th className="text-right p-2">Team A VORP*</th>
                  <th className="text-right p-2">Team B VORP*</th>
                  <th className="text-center p-2">Winner</th>
                  <th className="text-right p-2">Advantage</th>
                </tr>
              </thead>
              <tbody className="text-zinc-700 dark:text-zinc-200">
                {data.trade_summary.map((trade, i) => (
                  <tr
                    key={`${trade.trade_week}-${trade.team_a}-${trade.team_b}`}
                    className={`border-t border-zinc-200 dark:border-zinc-800 ${
                      i % 2 === 1 ? "bg-slate-50/60 dark:bg-slate-800/40" : ""
                    }`}
                  >
                    <td className="p-2 font-medium">{trade.trade_week}</td>
                    <td className="p-2">
                      <div className="text-sm">
                        <div className="font-medium">{trade.team_a} ↔ {trade.team_b}</div>
                        <div className="text-xs text-zinc-500 dark:text-zinc-400">
                          {trade.players_to_a} ↔ {trade.players_to_b}
                        </div>
                      </div>
                    </td>
                    <td className="p-2 text-right font-semibold text-emerald-700 dark:text-emerald-400">
                      {trade.team_a_vorp_received.toFixed(2)}
                    </td>
                    <td className="p-2 text-right font-semibold text-emerald-700 dark:text-emerald-400">
                      {trade.team_b_vorp_received.toFixed(2)}
                    </td>
                    <td className="p-2 text-center">
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        trade.winner === "Tie" 
                          ? "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"
                          : "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300"
                      }`}>
                        {trade.winner}
                      </span>
                    </td>
                    <td className="p-2 text-right font-semibold text-blue-700 dark:text-blue-400">
                      {trade.net_advantage.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Detailed Trade Values */}
      {data && data.trade_values && data.trade_values.length > 0 && (
        <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-slate-900/80 p-0 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">
                Detailed Trade Values
              </h2>
              <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-900/40 dark:text-fuchsia-300">
                {data.trade_values.length} player movements
              </span>
            </div>
          </div>

          <div className="h-[60vh] overflow-auto rounded-t-lg ring-1 ring-zinc-100 dark:ring-zinc-800">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-emerald-600 text-white">
                <tr>
                  <th className="text-left p-2">Week</th>
                  <th className="text-left p-2">Player</th>
                  <th className="text-left p-2">Direction</th>
                  <th className="text-right p-2">Weeks Owned</th>
                  <th className="text-right p-2">Total VORP*</th>
                  <th className="text-right p-2">Avg Weekly VORP*</th>
                </tr>
              </thead>
              <tbody className="text-zinc-700 dark:text-zinc-200">
                {data.trade_values.map((trade, i) => (
                  <tr
                    key={`${trade.trade_week}-${trade.player_name}-${trade.player_id}`}
                    className={`border-t border-zinc-200 dark:border-zinc-800 ${
                      i % 2 === 1 ? "bg-slate-50/60 dark:bg-slate-800/40" : ""
                    }`}
                  >
                    <td className="p-2 font-medium">{trade.trade_week}</td>
                    <td className="p-2">{trade.player_name}</td>
                    <td className="p-2 text-sm text-zinc-600 dark:text-zinc-400">
                      {trade.direction}
                    </td>
                    <td className="p-2 text-right">{trade.tenure_weeks}</td>
                    <td className="p-2 text-right font-semibold text-emerald-700 dark:text-emerald-400">
                      {trade.total_vorp_star.toFixed(2)}
                    </td>
                    <td className="p-2 text-right">
                      {trade.avg_weekly_vorp_star.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* No data state */}
      {!loading && !error && (!data || data.trade_values.length === 0) && (
        <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-slate-900/80 p-8 text-center">
          <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100 mb-2">
            No Trades Found
          </h3>
          <p className="text-zinc-500 dark:text-zinc-400">
            No trades were detected for the selected year.
          </p>
        </div>
      )}
    </main>
  );
}
