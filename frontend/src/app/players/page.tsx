"use client";

import { useEffect, useMemo, useState } from "react";

/* --------------------------- types --------------------------- */
type PlayerRow = {
  player_name: string;
  team?: string | null;           // NFL team
  fantasy_pos: string;
  fantasy_points_ppr: number;
  vorp_star: number;
  year?: number;                  // added when aggregating "ALL"
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

/* -------------------------- config -------------------------- */
const YEARS = [2020, 2021, 2022, 2024] as const;
type YearChoice = (typeof YEARS)[number] | "ALL";
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

/* ------------------------ helpers --------------------------- */
const normalizeName = (raw: string) =>
  raw
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .replace(/[.'’`,\-]/g, " ")
    .replace(/\b(jr|sr|ii|iii|iv|v)\b/g, "")
    .replace(/\s+/g, " ")
    .trim();

/* ------------------------ data fetchers ---------------------- */
async function fetchVorp(year: number): Promise<VorpResponse> {
  const res = await fetch(`${API_BASE}/metrics/vorp/${year}?top=500`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`VORP ${year} failed: ${res.status} ${res.statusText} ${body}`);
  }
  return res.json();
}

async function fetchDraft(year: number): Promise<DraftResponse> {
  const res = await fetch(`${API_BASE}/draft/${year}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Draft ${year} failed: ${res.status} ${res.statusText} ${body}`);
  }
  return res.json();
}

/* --------------------------- page --------------------------- */
export default function PlayersPage() {
  const [year, setYear] = useState<YearChoice>("ALL");
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
      // never allow empty set; if empty, restore ALL
      if (next.size === 0) return new Set(POS_ALL);
      return next;
    });

  // fetch VORP + Draft data for selected year (or ALL)
  useEffect(() => {
    let cancelled = false;

    async function runPlayers() {
      setLoading(true);
      setError(null);
      try {
        if (year === "ALL") {
          const all = await Promise.all(YEARS.map(fetchVorp));
          const rows = all.flatMap((r) =>
            (r.players ?? []).map((p) => ({ ...p, year: r.year }))
          );
          if (!cancelled) setData(rows);
        } else {
          const r = await fetchVorp(year);
          const rows = (r.players ?? []).map((p) => ({ ...p, year: r.year }));
          if (!cancelled) setData(rows);
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
        if (!cancelled) setDraftError(e?.message || "Failed to load draft data");
        if (!cancelled) setDrafts([]);
      } finally {
        if (!cancelled) setDraftLoading(false);
      }
    }

    runPlayers();
    runDrafts();
    return () => {
      cancelled = true;
    };
  }, [year]);

  // Build a quick lookup: (year|player) -> { drafter, round }
  const draftIndex = useMemo(() => {
    const m: Record<string, { drafter: string; round: number }> = {};
    for (const d of drafts) {
      for (const p of d.picks ?? []) {
        if (!p.player_name) continue;
        const key = `${d.year}|${normalizeName(p.player_name)}`;
        const round = typeof p.round_num === "number" ? p.round_num : 99;
        m[key] = { drafter: p.team_name, round };
      }
    }
    return m;
  }, [drafts]);

  // apply filters (positions) and sort by true WAR desc
  const rows = useMemo(() => {
    const want = posSet;
    const r = data.filter((x) => want.has(x.fantasy_pos));
    return r.sort((a, b) => (b.vorp_star ?? -Infinity) - (a.vorp_star ?? -Infinity));
  }, [data, posSet]);

  // Total WAR by drafter name with late-round penalty floor (R1–8: full WAR, R9+: floor at –1)
  const warTotalsByDrafter = useMemo(() => {
    const acc: Record<string, number> = {};

    for (const r of rows) {
      const y = r.year ?? 0;
      if (!y) continue;

      const di = draftIndex[`${y}|${normalizeName(r.player_name)}`];
      if (!di) continue;

      const round = di.round ?? 99;
      const v = typeof r.vorp_star === "number" ? r.vorp_star : 0; // TRUE WAR ONLY

      let effective = v;
      if (round >= 9) {
        // From round 9 on, floor downside at –1
        effective = Math.max(v, -1.5);
      }

      const drafter = di.drafter ?? "—"; // use drafter name directly
      acc[drafter] = (acc[drafter] ?? 0) + effective;
    }

    return Object.entries(acc)
      .map(([drafter, total]) => ({ drafter, total }))
      .sort((a, b) => b.total - a.total);
  }, [rows, draftIndex]);



  return (
    <main className="mx-auto max-w-5xl p-6 space-y-8 bg-slate-50 min-h-screen dark:bg-[#0b0f13]">
      {/* Top bar: title + nav + controls */}
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

          {/* Nav */}
          <nav className="flex items-center gap-5">
            <a
              href="/"
              className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline"
            >
              Standings
            </a>
            <a
              href="/draft"
              className="text-sm md:text-base text-white/90 hover:text-white underline-offset-4 hover:underline"
            >
              Drafts
            </a>
          </nav>
        </div>
      </div>

      {/* WAR totals by drafter (R1–8 full; R9+ floored at –1) */}
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-slate-900/80 p-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">
            Total WAR by Drafter
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

        <div className="overflow-auto rounded ring-1 ring-zinc-100 dark:ring-zinc-800 max-h-[30vh]">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-emerald-600 text-white">
              <tr>
                <th className="text-left p-2">Drafter</th>
                <th className="text-right p-2">Total WAR</th>
              </tr>
            </thead>
            <tbody className="text-zinc-700 dark:text-zinc-200">
              {warTotalsByDrafter.length > 0 ? (
                warTotalsByDrafter.map((row, i) => (
                  <tr
                    key={`${row.drafter}-${i}`}
                    className={`border-t border-zinc-200 dark:border-zinc-800 ${
                      i % 2 === 1 ? "bg-slate-50/60 dark:bg-slate-800/40" : ""
                    }`}
                  >
                    <td className="p-2">{row.drafter}</td>
                    <td className="p-2 text-right font-semibold text-emerald-700 dark:text-emerald-400">
                      {row.total.toFixed(2)}
                    </td>
                  </tr>
                ))
              ) : (
                <tr className="border-t border-zinc-200 dark:border-zinc-800">
                  <td colSpan={2} className="p-3 text-center text-zinc-500 dark:text-zinc-400">
                    {loading || draftLoading ? "Loading…" : "No data for current filters."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>


      {/* Players table — fills the screen first, scrolls inside */}
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-slate-900/80 p-0 overflow-hidden">
        {/* Header row inside card */}
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

          {/* Mobile pos multi-select (fallback) */}
          <div className="sm:hidden">
            <select
              multiple
              className="rounded-md bg-slate-100 dark:bg-slate-800 text-sm px-2 py-1"
              value={Array.from(posSet)}
              onChange={(e) => {
                const opts = Array.from(e.target.selectedOptions).map((o) => o.value);
                setPosSet(new Set(opts.length ? opts : POS_ALL));
              }}
            >
              {POS_ALL.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Scroll area */}
        <div className="h-[60vh] overflow-auto rounded-t-lg ring-1 ring-zinc-100 dark:ring-zinc-800">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-emerald-600 text-white">
              <tr>
                <th className="text-left p-2">Player</th>
                <th className="text-left p-2">Pos</th>
                <th className="text-left p-2">NFL Team</th>
                <th className="text-left p-2">Drafted By</th>
                <th className="text-right p-2">Rnd</th>
                <th className="text-right p-2">PPR Points</th>
                <th className="text-right p-2">WAR</th>
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
                    <td className="p-2">{d?.drafter ?? "—"}</td>
                    <td className="p-2 text-right">{d?.round ?? "—"}</td>
                    <td className="p-2 text-right">{(r.fantasy_points_ppr ?? 0).toFixed(1)}</td>
                    <td className="p-2 text-right font-semibold text-emerald-700 dark:text-emerald-400">
                      {typeof r.vorp_star === "number" ? r.vorp_star.toFixed(2) : "—"}
                    </td>
                    <td className="p-2 text-right">{r.year ?? "—"}</td>
                  </tr>
                );
              })}

              {!loading && !error && rows.length === 0 && (
                <tr className="border-t border-zinc-200 dark:border-zinc-800">
                  <td colSpan={8} className="p-4 text-center text-zinc-500 dark:text-zinc-400">
                    No players match the current filters.
                  </td>
                </tr>
              )}
              {(error || draftError) && (
                <tr className="border-t border-zinc-200 dark:border-zinc-800">
                  <td colSpan={8} className="p-4 text-center text-zinc-500 dark:text-zinc-400">
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
