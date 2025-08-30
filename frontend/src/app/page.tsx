import Image from "next/image";

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
    <main className="mx-auto max-w-5xl p-6 space-y-8">
     {/* Top bar: title + text links */}
      <div className="rounded-xl bg-zinc-900 text-white px-4 py-3 flex items-center justify-between">
        <h1 className="text-2xl md:text-3xl font-bold">League 86952922 Standings</h1>

        <nav className="flex items-center gap-5">
          <a
            href="/draft"
            className="text-sm md:text-base text-white hover:text-zinc-400 transition-colors"
          >
            View Drafts
          </a>
          <a
            href="/playoffs"
            className="text-sm md:text-base text-white hover:text-zinc-400 transition-colors"
          >
            View Playoffs
          </a>
        </nav>
      </div>
      <div className="grid gap-6 md:grid-cols-3">
        {data.map((d) => (
          <div key={d.year} className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-900">
            <h2 className="text-xl font-semibold mb-2">{d.year}</h2>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-3">Teams: {d.num_teams}</p>
            <div className="max-h-72 overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-zinc-100 dark:bg-zinc-800">
                  <tr>
                    <th className="text-left p-1">Team</th>
                    <th className="text-right p-1">W-L-T</th>
                    <th className="text-right p-1">Win%</th>
                    <th className="text-right p-1">PF</th>
                    <th className="text-right p-1">PA</th>
                  </tr>
                </thead>
                <tbody>
                  {d.teams.map((t: any) => (
                    <tr key={t.team_id} className="border-t border-zinc-200 dark:border-zinc-800">
                      <td className="p-1">{t.team_name}</td>
                      <td className="p-1 text-right">{t.wins}-{t.losses}-{t.ties}</td>
                      <td className="p-1 text-right font-medium">{t.win_percentage}%</td>
                      <td className="p-1 text-right">{t.points_for.toFixed(1)}</td>
                      <td className="p-1 text-right">{t.points_against.toFixed(1)}</td>
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
