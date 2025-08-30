"use client";

import { useEffect, useMemo, useState } from "react";

type DraftPick = {
  year: number;
  team_id: number;
  team_name: string;
  owner?: string | null;
  round_num: number | null;
  pick_num: number | null;       // this is the ROUND pick (P:…)
  overall_pick: number | null;   // already computed in backend
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

const YEARS = [2020, 2021, 2022, 2024];

async function fetchDraft(year: number): Promise<DraftResponse> {
  const base = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
  const res = await fetch(`${base}/draft/${year}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch draft for ${year}`);
  return res.json();
}

export default function DraftsPage() {
  const [year, setYear] = useState<number>(YEARS[0]);
  const [data, setData] = useState<DraftResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"board" | "list">("board");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchDraft(year)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [year]);

  // Group by round for board layout
  const rounds = useMemo(() => {
    const by: Record<number, DraftPick[]> = {};
    for (const p of data?.picks ?? []) {
      const r = p.round_num ?? 0;
      if (!by[r]) by[r] = [];
      by[r].push(p);
    }
    // Sort picks within a round by pick_num then overall
    Object.values(by).forEach((arr) =>
      arr.sort((a, b) => {
        const pa = a.pick_num ?? 999, pb = b.pick_num ?? 999;
        if (pa !== pb) return pa - pb;
        const oa = a.overall_pick ?? 9999, ob = b.overall_pick ?? 9999;
        return oa - ob;
      })
    );
    const entries = Object.entries(by)
      .map(([k, v]) => [Number(k), v] as [number, DraftPick[]])
      .sort((a, b) => a[0] - b[0]);
    return entries;
  }, [data]);

  return (
    <main className="mx-auto max-w-none px-2 space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
  {/* Title + tiny link */}
  <div className="flex items-center gap-3">
    <h1 className="text-3xl font-bold">League Drafts</h1>
    <a
      href="/"
      className="text-sm text-zinc-400 hover:text-zinc-600 transition-colors"
    >
      View standings
    </a>
  </div>

  {/* Year + view toggle (unchanged) */}
  <div className="flex items-center gap-3">
    <label className="text-sm text-zinc-600 dark:text-zinc-300">Year</label>
    <select
      className="rounded-lg border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-3 py-2"
      value={year}
      onChange={(e) => setYear(Number(e.target.value))}
    >
      {YEARS.map((y) => (
        <option value={y} key={y}>{y}</option>
      ))}
    </select>

    <div className="ml-2 inline-flex rounded-lg overflow-hidden border border-zinc-300 dark:border-zinc-700">
      <button
        className={`px-3 py-2 text-sm ${view === "board" ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900" : "bg-white dark:bg-zinc-900"}`}
        onClick={() => setView("board")}
      >
        Board
      </button>
      <button
        className={`px-3 py-2 text-sm ${view === "list" ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900" : "bg-white dark:bg-zinc-900"}`}
        onClick={() => setView("list")}
      >
        List
      </button>
    </div>
  </div>
</div>


      {loading && (
        <div className="text-zinc-500">Loading {year} draft…</div>
      )}
      {error && (
        <div className="text-red-600">Error: {error}</div>
      )}

      {!loading && !error && data && (
        <>
          {view === "board" ? (
            <DraftBoard rounds={rounds} />
          ) : (
            <DraftList picks={data.picks} />
          )}
        </>
      )}
    </main>
  );
}

// Soft colored card: solid base + subtle radial blobs + optional noise overlay
function ColorTile({
  baseHex,
  seed,            // use something stable per-card (e.g., overall_pick)
  className = "",
  children,
}: {
  baseHex: string;
  seed: number | string;
  className?: string;
  children: React.ReactNode;
}) {
  const { r, g, b } = hexToRgb(baseHex);

  // stable pseudo-random (0..1) from seed + salt
  const rnd = (s: number) => {
    const str = String(seed) + ":" + s;
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    // map to [0,1]
    return (h >>> 0) / 2 ** 32;
  };

  // two soft blobs at pseudo-random positions (but consistent for this seed)
  const x1 = 20 + rnd(1) * 60; // 20%..80%
  const y1 = 25 + rnd(2) * 50; // 25%..75%
  const x2 = 15 + rnd(3) * 70;
  const y2 = 20 + rnd(4) * 60;

  // same hue, tiny alpha → looks like gentle “Gaussian” variation
  const cLight = `rgba(${r}, ${g}, ${b}, 0.20)`; // faint light blob
  const cDark  = `rgba(0, 0, 0, 0.08)`;          // faint dark blob for depth

  const style: React.CSSProperties = {
    backgroundColor: baseHex,
    backgroundImage: `
      radial-gradient(180px 140px at ${x1}% ${y1}%, ${cLight}, transparent 65%),
      radial-gradient(220px 180px at ${x2}% ${y2}%, ${cDark},  transparent 70%)
    `,
    backgroundBlendMode: "overlay, normal",
  };

  return (
    <div className={`relative rounded-md shadow-md ring-1 ring-black/10 ${className}`} style={style}>
      {/* optional super-faint noise texture (drop a tiny PNG at /public/textures/noise.png) */}
      <div
        className="pointer-events-none absolute inset-0 rounded-md opacity-10"
        style={{ backgroundImage: "url(/textures/noise.png)" }}
      />
      {children}
    </div>
  );
}

// helpers
function hexToRgb(hex: string) {
  const n = hex.replace("#", "");
  const bigint = parseInt(n.length === 3 ? n.split("").map(c => c + c).join("") : n, 16);
  return { r: (bigint >> 16) & 255, g: (bigint >> 8) & 255, b: bigint & 255 };
}
function DraftBoard({ rounds }: { rounds: [number, DraftPick[]][] }) {
  // compact layout knobs
  const GUTTER = 84;      // left "Round" column
  const COL_MIN = 95;    // min width per column
  const CELL_MIN_H = 70;  // card height

  // colors you chose
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

  // ---- Build columns by Round 1 order (for header labels) ----
  const round1 = rounds.find(([r]) => r === 1)?.[1] ?? [];
  const round1Sorted = [...round1].sort(
    (a, b) => (a.pick_num ?? 999) - (b.pick_num ?? 999)
  );
  // Header labels = team names in slot order from round 1
  const headerTeams = round1Sorted.map((p) => p.team_name);
  const teamCountGuessFromR1 = headerTeams.length;

  // fallback: use largest pick_num seen if round 1 missing/short
  const maxPickSeen = Math.max(
    0,
    ...rounds.flatMap(([, ps]) => ps.map((p) => p.pick_num ?? 0))
  );
  const teamCount = Math.max(teamCountGuessFromR1, maxPickSeen, 12);

  const gridTemplate = `${GUTTER}px repeat(${teamCount}, minmax(${COL_MIN}px, 1fr))`;

  return (
    <div className="relative w-full rounded-md border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-auto">
      {/* Header: show Round-1 team names per slot (wrap to two lines) */}
      <div className="sticky top-0 z-30 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
        <div className="grid" style={{ gridTemplateColumns: gridTemplate }}>
          <div className="h-8 px-2 flex items-center text-[10px] font-semibold text-zinc-500 uppercase">
            Round
          </div>
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

      {/* Body (snake per round) */}
      <div>
        {rounds.map(([roundNum, picks]) => {
          // slots by round pick position (1..N); index = pick_num - 1
          const slots: (DraftPick | null)[] = Array(teamCount).fill(null);
          for (const p of picks) {
            const rp = p.pick_num ?? 0;
            const idx = rp > 0 && rp <= teamCount ? rp - 1 : -1;
            if (idx >= 0) slots[idx] = p;
          }

          // snake: reverse visible order on even rounds
          const display = roundNum % 2 === 0 ? [...slots].reverse() : slots;

          return (
            <div
              key={roundNum}
              className="grid border-b border-zinc-200 dark:border-zinc-800 last:border-b-0"
              style={{ gridTemplateColumns: gridTemplate }}
            >
              {/* Round label + direction cue */}
              <div className="sticky left-0 z-20 bg-white dark:bg-zinc-950 border-r border-zinc-200 dark:border-zinc-800">
                <div
                  className="px-2 py-1 flex items-center"
                  style={{ minHeight: CELL_MIN_H }}
                >
                  <span className="inline-flex rounded bg-zinc-100 dark:bg-zinc-900 px-1.5 py-0.5 text-[10px] font-semibold text-zinc-700 dark:text-zinc-300">
                    R{roundNum}
                    <span className="ml-1 opacity-70">
                      {roundNum % 2 === 1 ? "→" : "←"}
                    </span>
                  </span>
                </div>
              </div>

              {/* Row cells (reversed on even rounds) */}
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
                const posKey = normPos(p.position);
                const posClass =
                  positionColors[posKey] ?? "bg-zinc-200 text-zinc-800";

                return (
                  <div
                    key={`${roundNum}-${p.overall_pick ?? i}`}
                    className="border-l border-zinc-200 dark:border-zinc-800 p-1"
                    style={{ minHeight: CELL_MIN_H }}
                  >
                    <div className={`h-full w-full rounded-sm ${posClass} px-1.5 py-1.5`}>
                      <div className="flex items-center justify-between text-[10px] mb-0.5">
                        <div className="font-semibold">
                          {p.pick_num ?? "—"}
                          {p.overall_pick != null && (
                            <span className="ml-1 opacity-80">#{p.overall_pick}</span>
                          )}
                        </div>
                        {p.is_keeper && (
                          <span className="inline-flex rounded bg-black/10 px-1 py-[1px] text-[9px] font-semibold">
                            K
                          </span>
                        )}
                      </div>
                      <div className="text-xs font-semibold leading-tight line-clamp-2">
                        {p.player_name}
                      </div>
                      <div className="mt-0.5 text-[10px]">
                        {(posKey || "—")}{p.pro_team ? ` • ${p.pro_team}` : ""}
                      </div>
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
