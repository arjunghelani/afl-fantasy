// app/playoffs/page.tsx

type Game = {
  home_team: string;
  away_team: string;
  home_score?: number | null;
  away_score?: number | null;
  winner?: string | null;
  round_name: string;
  week?: number | null;
};

type Season = {
  year: number;
  games: Game[];
};

async function fetchPlayoffs(year: number) {
  const base = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
  const res = await fetch(`${base}/playoffs/${year}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch playoffs for ${year}`);
  return res.json() as Promise<Season>;
}

// ---------- Helpers to build a proper bracket (with BYE cards) ----------

type Match = {
  id?: string;
  home_team: string;
  away_team: string;
  home_score?: number | null;
  away_score?: number | null;
  winner?: string | null;
  __is_bye__?: boolean; // for rendering BYE as a single-line card
};

type Round = { name: string; matches: Match[] };

const ROUND_ORDER: Record<string, number> = {
  Quarterfinals: 1,
  Semifinals: 2,
  Championship: 3,
  "Winners Consolation": 60,
  Consolation: 70,
  Playoff: 90,
};

function isPlaceholderName(name?: string | null): boolean {
  if (!name) return true;
  const s = name.trim().toLowerCase();
  return (
    s === "tbd" ||
    s.startsWith("winner q") ||
    s.startsWith("winner sf") ||
    s.startsWith("winner of") ||
    s === "winner" ||
    s.includes("winner")
  );
}

function normalizeRounds(games: Game[]): Round[] {
  // Group by round name
  const byRound = new Map<string, Game[]>();
  for (const g of games) {
    const key = g.round_name || "Playoff";
    if (!byRound.has(key)) byRound.set(key, []);
    byRound.get(key)!.push(g);
  }
  // Sort within round
  for (const [, arr] of byRound) {
    arr.sort((a, b) => (a.week ?? 0) - (b.week ?? 0));
  }
  // Order rounds
  const names = Array.from(byRound.keys()).sort(
    (a, b) => (ROUND_ORDER[a] ?? 999) - (ROUND_ORDER[b] ?? 999)
  );

  const rounds: Round[] = names.map((name) => ({
    name,
    matches: byRound.get(name)!,
  }));

  return injectByes(rounds);
}

/**
 * Detect 6-team playoffs (top-2 byes). If Quarterfinals contain placeholder
 * matches (TBD vs TBD), replace them with a single BYE card using the team
 * that appears in Semifinals but not in Quarterfinals.
 */
function injectByes(rounds: Round[]): Round[] {
  const qf = rounds.find((r) => r.name === "Quarterfinals");
  const sf = rounds.find((r) => r.name === "Semifinals");
  if (!qf || !sf) return rounds;

  const qfTeams = new Set<string>();
  for (const m of qf.matches) {
    if (!isPlaceholderName(m.home_team)) qfTeams.add(m.home_team);
    if (!isPlaceholderName(m.away_team)) qfTeams.add(m.away_team);
  }

  const sfTeams: string[] = [];
  for (const m of sf.matches) {
    if (!isPlaceholderName(m.home_team)) sfTeams.push(m.home_team);
    if (!isPlaceholderName(m.away_team)) sfTeams.push(m.away_team);
  }

  const byeTeams = sfTeams.filter((t) => !qfTeams.has(t));
  if (byeTeams.length === 0) return rounds;

  const placeholderIdxs: number[] = [];
  qf.matches.forEach((m, idx) => {
    const bothPlaceholder =
      isPlaceholderName(m.home_team) && isPlaceholderName(m.away_team);
    if (bothPlaceholder) placeholderIdxs.push(idx);
  });

  const qfMatches = qf.matches.slice();
  for (let i = 0; i < Math.min(byeTeams.length, placeholderIdxs.length); i++) {
    const byeName = byeTeams[i];
    const idx = placeholderIdxs[i];
    qfMatches[idx] = {
      home_team: byeName,
      away_team: "BYE",
      home_score: null,
      away_score: null,
      winner: byeName,
      __is_bye__: true,
    };
  }

  return rounds.map((r) =>
    r.name === "Quarterfinals" ? { ...r, matches: qfMatches } : r
  );
}

// ---------- Bracket UI (columns + connectors, with label spacing) ----------

function Bracket({ rounds }: { rounds: Round[] }) {
  // Layout constants
  const slotH = 84;      // vertical space per matchup
  const colW = 280;      // column width
  const padX = 24;       // space between columns
  const labelHeight = 32; // reserved space above cards for the round title

  const baseCount = rounds[0]?.matches.length ?? 0;

  // Vertical centers for each match; add labelHeight so cards start below title
  const positions: number[][] = rounds.map((round, r) => {
    const stride = Math.pow(2, r); // 1, 2, 4, ...
    const block = slotH * stride;
    const offset = block / 2;
    return round.matches.map((_, i) => i * block + offset + labelHeight);
  });

  const canvasH = Math.max(1, baseCount) * slotH + labelHeight + 40;
  const canvasW = rounds.length * (colW + padX) + 80;

  const MatchCard = ({ m, x, y }: { m: Match; x: number; y: number }) => {
    const homeWins = !!m.winner && m.winner === m.home_team;
    const awayWins = !!m.winner && m.winner === m.away_team;

    if (m.__is_bye__) {
      return (
        <div
          className="absolute w-[220px] rounded-2xl shadow p-3 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800"
          style={{ left: x, top: y - 24 }}
        >
          <div className="flex items-center justify-between">
            <span className="font-semibold">{m.home_team}</span>
            <span className="text-xs px-2 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800">
              BYE (advances)
            </span>
          </div>
        </div>
      );
    }

    return (
      <div
        className="absolute w-[220px] rounded-2xl shadow p-3 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800"
        style={{ left: x, top: y - 32 }}
      >
        <div
          className={`flex justify-between items-center rounded px-2 py-1 ${
            homeWins ? "bg-green-100 dark:bg-green-900 font-semibold" : ""
          }`}
        >
          <span className="truncate">{m.home_team}</span>
          {m.home_score != null && (
            <span className="tabular-nums">{m.home_score}</span>
          )}
        </div>
        <div
          className={`flex justify-between items-center rounded px-2 py-1 mt-1 ${
            awayWins ? "bg-green-100 dark:bg-green-900 font-semibold" : ""
          }`}
        >
          <span className="truncate">
            {m.away_team === "BYE" ? "—" : m.away_team}
          </span>
          {m.away_score != null && m.away_team !== "BYE" && (
            <span className="tabular-nums">{m.away_score}</span>
          )}
        </div>
      </div>
    );
  };

  const Connectors = () => {
    const width = canvasW;
    const height = canvasH;
    const paths: JSX.Element[] = [];

    for (let r = 0; r < rounds.length - 1; r++) {
      const fromXs = 16 + r * (colW + padX) + 220; // right edge of card
      const toXs = 16 + (r + 1) * (colW + padX);   // left edge next col
      const ysFrom = positions[r];
      const ysTo = positions[r + 1];

      for (let i = 0; i < ysTo.length; i++) {
        const a = ysFrom[i * 2];
        const b = ysFrom[i * 2 + 1];
        if (a == null || b == null) continue;
        const mid = ysTo[i];

        const gap = 22;
        const x1 = fromXs;
        const xMid = toXs - gap;
        const x2 = toXs;

        const pathA = `M ${x1} ${a} H ${xMid} V ${mid}`;
        const pathB = `M ${x1} ${b} H ${xMid} V ${mid}`;
        const pathC = `M ${xMid} ${mid} H ${x2}`;

        paths.push(
          <path
            key={`p-${r}-${i}-a`}
            d={pathA}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            opacity={0.5}
          />
        );
        paths.push(
          <path
            key={`p-${r}-${i}-b`}
            d={pathB}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            opacity={0.5}
          />
        );
        paths.push(
          <path
            key={`p-${r}-${i}-c`}
            d={pathC}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            opacity={0.5}
          />
        );
      }
    }

    return (
      <svg
        className="absolute inset-0 text-zinc-300 dark:text-zinc-700 pointer-events-none"
        width={width}
        height={height}
      >
        {paths}
      </svg>
    );
  };

  return (
    <div className="relative overflow-auto rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 p-4">
      <div className="relative" style={{ height: canvasH, minWidth: canvasW }}>
        <Connectors />

        {/* Titles per column, aligned with reserved space */}
        {rounds.map((round, r) => {
          const x = 16 + r * (colW + padX);
          return (
            <div
              key={`${round.name}-title`}
              className="absolute text-center"
              style={{ left: x, top: 0, width: 220 }}
            >
              <div className="mb-2 text-sm font-semibold text-zinc-600 dark:text-zinc-300">
                {round.name}
              </div>
            </div>
          );
        })}

        {/* Cards */}
        {rounds.map((round, r) => {
          const x = 16 + r * (colW + padX);
          return (
            <div key={round.name}>
              {round.matches.map((m, i) => (
                <MatchCard
                  key={`${round.name}-${i}-${m.home_team}-${m.away_team}`}
                  m={m}
                  x={x}
                  y={positions[r][i]}
                />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------- Page ----------

export default async function PlayoffsPage() {
  const years = [2020, 2021, 2022, 2024];
  const seasons = await Promise.all(years.map((y) => fetchPlayoffs(y)));

  return (
    <main className="mx-auto max-w-7xl p-6 space-y-8">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">League 86952922 Playoff Brackets</h1>
        <a
          href="/"
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
        >
          View Standings
        </a>
      </div>

      <div className="space-y-10">
        {seasons.map((d) => {
          const rounds = normalizeRounds(d.games).filter((r) =>
            ["Quarterfinals", "Semifinals", "Championship"].includes(r.name)
          );

          return (
            <section
              key={d.year}
              className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-900"
            >
              <h2 className="text-xl font-semibold mb-4 text-center">{d.year}</h2>
              {rounds.length === 0 ? (
                <p className="text-sm text-zinc-500 text-center">
                  No bracket data available.
                </p>
              ) : (
                <Bracket rounds={rounds} />
              )}
            </section>
          );
        })}
      </div>
    </main>
  );
}
