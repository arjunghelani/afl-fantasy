async function fetchStandings(year: number) {
  const base = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
  const res = await fetch(`${base}/standings/${year}`, { next: { revalidate: 60 } });
  if (!res.ok) {
    throw new Error(`Failed to fetch standings for ${year}`);
  }
  return res.json();
}

export default async function Home() {
  const years = [2020, 2021, 2022, 2024];
  const data = await Promise.all(years.map((y) => fetchStandings(y)));

  return (
    <main className="mx-auto max-w-5xl p-6 space-y-8 bg-slate-50 min-h-screen dark:bg-[#0b0f13]">
      {/* Top bar: title + nav */}
      <div className="rounded-xl bg-emerald-700 text-white px-4 py-3 flex items-center justify-between shadow-sm">
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
          League 86952922 Standings
        </h1>

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

      {/* Standings grid */}
      <div className="grid gap-6 md:grid-cols-3">
        {data.map((d) => (
          <div
            key={d.year}
            className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-slate-900/80"
          >
            <h2 className="text-xl font-semibold mb-2 text-zinc-800 dark:text-zinc-100 flex items-center gap-2">
              {d.year}
              <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-900/40 dark:text-fuchsia-300">
                {d.num_teams} teams
              </span>
            </h2>

            <div className="max-h-72 overflow-auto rounded-lg ring-1 ring-zinc-100 dark:ring-zinc-800">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-emerald-600 text-white">
                  <tr>
                    <th className="text-left p-2">Team</th>
                    <th className="text-right p-2">W-L-T</th>
                    <th className="text-right p-2">Win%</th>
                    <th className="text-right p-2">PF</th>
                    <th className="text-right p-2">PA</th>
                  </tr>
                </thead>
                <tbody className="text-zinc-700 dark:text-zinc-200">
                  {d.teams.map((t: any, i: number) => (
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
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
