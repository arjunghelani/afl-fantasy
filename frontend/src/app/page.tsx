'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

type TeamRow = {
  team_id: number;
  team_name: string;
  wins: number;
  losses: number;
  ties: number;
  win_percentage: number;
  points_for: number;
  points_against: number;
  expected_wins?: number;
};

type StandingsResponse = {
  year: number;
  num_teams: number;
  teams: TeamRow[];
};

const YEARS = [2020, 2021, 2022, 2024, 2025] as const;
type YearChoice = (typeof YEARS)[number] | "ALL";

// API routes are now proxied through Next.js API routes
const LEAGUE_ID_STORAGE_KEY = 'fantasy_league_id';

// exclude these team names (case-insensitive)
const EXCLUDED_TEAM_NAMES = new Set(["team ned"]);
const isExcludedTeamName = (name?: string) =>
  !!name && EXCLUDED_TEAM_NAMES.has(name.trim().toLowerCase());

// League ID management
function getLeagueId(): number | null {
  if (typeof window === 'undefined') return null;
  const stored = localStorage.getItem(LEAGUE_ID_STORAGE_KEY);
  return stored ? parseInt(stored, 10) : null;
}

function setLeagueId(leagueId: number): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(LEAGUE_ID_STORAGE_KEY, leagueId.toString());
}

async function fetchStandings(year: number, leagueId?: number | null): Promise<StandingsResponse> {
  const leagueIdParam = leagueId ? `?league_id=${leagueId}` : '';
  const res = await fetch(`/api/standings/${year}${leagueIdParam}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch standings for ${year}`);
  return res.json();
}

/** Aggregate multiple StandingsResponse objects by team_id */
function aggregateStandingsByTeamId(all: StandingsResponse[]): StandingsResponse {
  const acc: Record<number, {
    team_id: number;
    team_name: string;
    latest_year: number;
    wins: number;
    losses: number;
    ties: number;
    points_for: number;
    points_against: number;
    expected_wins: number;
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
        expected_wins: 0,
      };

      if (y >= cur.latest_year && t.team_name) {
        cur.team_name = t.team_name;
        cur.latest_year = y;
      }

      cur.wins += t.wins;
      cur.losses += t.losses;
      cur.ties += t.ties;
      cur.points_for += t.points_for;
      cur.points_against += t.points_against;
      cur.expected_wins += t.expected_wins ?? 0;

      acc[t.team_id] = cur;
    }
  }

  const teams: TeamRow[] = Object.values(acc).map((r) => {
    const games = r.wins + r.losses + r.ties;
    const winpct = games > 0 ? Math.round((r.wins / games) * 1000) / 10 : 0;
    return {
      team_id: r.team_id,
      team_name: r.team_name,
      wins: r.wins,
      losses: r.losses,
      ties: r.ties,
      win_percentage: winpct,
      points_for: r.points_for,
      points_against: r.points_against,
      expected_wins: r.expected_wins,
    };
  });

  teams.sort((a, b) => {
    if (b.win_percentage !== a.win_percentage) return b.win_percentage - a.win_percentage;
    return b.points_for - a.points_for;
  });

  return {
    year: 0,
    num_teams: teams.length,
    teams,
  };
}

type PlayerVorp = {
  player_name: string;
  team?: string | null;
  fantasy_pos: string;
  g?: number | null;
  fantasy_points_ppr: number;
  vorp_star: number;
  vorp_star_rank_overall: number;
  vorp_star_rank_pos: number;
  partial_season?: boolean | null;
  vorp_star_extrap?: number | null;
};

type VorpResponse = {
  year: number;
  players: PlayerVorp[];
  count: number;
  used_ppg: boolean;
};

type TeamZAV = {
  team_id: number;
  team_name: string;
  total_zav: number;
};

type TeamZAVResponse = {
  year: number;
  teams: TeamZAV[];
};

type PlayerRoster = {
  player_name: string;
  position: string;
  pos_rank: number | null;
  vorp_star: number | null;  // Seasonal ZAV
};

type TeamRosterDetail = {
  team_id: number;
  team_name: string;
  players: PlayerRoster[];
};

type TeamRostersResponse = {
  year: number;
  teams: TeamRosterDetail[];
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

async function fetchTeamRosters(year: number): Promise<TeamRosterDetail[]> {
  const res = await fetch(`/api/teams/${year}/rosters`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch team rosters for ${year}`);
  const data: TeamRostersResponse = await res.json();
  return data.teams;
}

const ZAV_CUTOFFS = {
  red: -2,        // ZAV < this value
  orange: 2.5,     // ZAV < this value
  yellow: 6,    // ZAV < this value
  yellowGreen: 10, // ZAV < this value
  // ZAV >= yellowGreen uses green
};

function getZavColor(vorp: number | null): string {
  if (vorp === null || vorp === 0) return 'rgb(131, 131, 131)'; // Gray for null/zero
  
  // Same RGB colors as weekly stats popup and VORP in trade cards
  if (vorp < ZAV_CUTOFFS.red) {
    return 'rgb(200, 50, 50)'; // Red
  } else if (vorp < ZAV_CUTOFFS.orange) {
    return 'rgb(230, 110, 40)'; // Orange
  } else if (vorp < ZAV_CUTOFFS.yellow) {
    return 'rgb(235, 190, 55)'; // Yellow
  } else if (vorp < ZAV_CUTOFFS.yellowGreen) {
    return 'rgb(140, 200, 60)'; // Yellow-green
  } else {
    return 'rgb(40, 150, 70)'; // Green
  }
}

function getZavGradient(vorp: number | null): string {
  if (vorp === null || vorp === 0) {
    return 'linear-gradient(135deg, rgb(131, 131, 131), rgb(100, 100, 100))';
  }
  
  // Gradient colors using similar shades
  if (vorp < ZAV_CUTOFFS.red) {
    return 'linear-gradient(135deg, rgb(220, 70, 70), rgb(180, 30, 30))'; // Red gradient
  } else if (vorp < ZAV_CUTOFFS.orange) {
    return 'linear-gradient(135deg, rgb(250, 130, 60), rgb(210, 90, 20))'; // Orange gradient
  } else if (vorp < ZAV_CUTOFFS.yellow) {
    return 'linear-gradient(135deg, rgb(245, 210, 75), rgb(225, 170, 35))'; // Yellow gradient
  } else if (vorp < ZAV_CUTOFFS.yellowGreen) {
    return 'linear-gradient(135deg, rgb(160, 220, 80), rgb(120, 180, 40))'; // Yellow-green gradient
  } else {
    return 'linear-gradient(135deg, rgb(16, 185, 129), rgb(5, 150, 105))'; // Emerald green gradient
  }
}

function getZavTextColor(vorp: number | null): string {
  if (vorp === null || vorp === 0) {
    return 'rgb(40, 40, 40)'; // Extremely dark gray for null/zero
  }
  
  // Extremely dark versions of bubble colors for text (used inside bubbles)
  if (vorp < ZAV_CUTOFFS.red) {
    return 'rgb(90, 10, 10)'; // Extremely dark red
  } else if (vorp < ZAV_CUTOFFS.orange) {
    return 'rgb(110, 35, 5)'; // Extremely dark orange
  } else if (vorp < ZAV_CUTOFFS.yellow) {
    return 'rgb(110, 85, 5)'; // Extremely dark yellow/brown
  } else if (vorp < ZAV_CUTOFFS.yellowGreen) {
    return 'rgb(50, 75, 10)'; // Extremely dark yellow-green
  } else {
    return 'rgb(1, 50, 20)'; // Extremely dark emerald green
  }
}

function getZavBrightColor(vorp: number | null): string {
  if (vorp === null || vorp === 0) {
    return 'rgb(131, 131, 131)'; // Gray for null/zero
  }
  
  // Bright gradient colors (used for stats display)
  if (vorp < ZAV_CUTOFFS.red) {
    return 'rgb(220, 70, 70)'; // Red
  } else if (vorp < ZAV_CUTOFFS.orange) {
    return 'rgb(250, 130, 60)'; // Orange
  } else if (vorp < ZAV_CUTOFFS.yellow) {
    return 'rgb(245, 210, 75)'; // Yellow
  } else if (vorp < ZAV_CUTOFFS.yellowGreen) {
    return 'rgb(160, 220, 80)'; // Yellow-green
  } else {
    return 'rgb(16, 185, 129)'; // Emerald green
  }
}

function getFantasyPointsGradient(points: number | null): string {
  if (points === null) {
    return 'linear-gradient(135deg, rgb(131, 131, 131), rgb(100, 100, 100))';
  }
  
  // Gradient colors matching the ZAV gradient style
  if (points <= 5) {
    return 'linear-gradient(135deg, rgb(220, 70, 70), rgb(180, 30, 30))'; // Red gradient
  } else if (points <= 10) {
    return 'linear-gradient(135deg, rgb(250, 130, 60), rgb(210, 90, 20))'; // Orange gradient
  } else if (points <= 15) {
    return 'linear-gradient(135deg, rgb(245, 210, 75), rgb(225, 170, 35))'; // Yellow gradient
  } else if (points <= 20) {
    return 'linear-gradient(135deg, rgb(160, 220, 80), rgb(120, 180, 40))'; // Yellow-green gradient
  } else {
    return 'linear-gradient(135deg, rgb(16, 185, 129), rgb(5, 150, 105))'; // Emerald green gradient
  }
}

function getFantasyPointsTextColor(points: number | null): string {
  if (points === null) {
    return 'rgb(40, 40, 40)'; // Extremely dark gray for null
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

async function fetchTop10ZAV(year: number, leagueId?: number | null): Promise<PlayerVorp[]> {
  const leagueIdParam = leagueId ? `&league_id=${leagueId}` : '';
  const res = await fetch(`/api/metrics/vorp/${year}?top=10${leagueIdParam}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch top 10 ZAV for ${year}`);
  const data: VorpResponse = await res.json();
  return data.players;
}

type PlayerInRecentWaiver = {
  player_name: string;
  vorp_star: number | null;
  fantasy_pos: string | null;
};

type RecentWaiverItem = {
  transaction_id: number | null;
  transaction_date: string;
  team_name: string;
  added_players: PlayerInRecentWaiver[];
  dropped_players: PlayerInRecentWaiver[];
};

type RecentWaiversResponse = {
  year: number;
  transactions: RecentWaiverItem[];
  count: number;
};

async function fetchRecentWaivers(year: number, leagueId?: number | null): Promise<RecentWaiversResponse> {
  const leagueIdParam = leagueId ? `?league_id=${leagueId}` : '';
  const res = await fetch(`/api/recent-waivers/${year}${leagueIdParam}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch recent waivers for ${year}`);
  return res.json();
}

function RecentWaiversBox({ onPlayerClick, leagueId }: { onPlayerClick?: (playerName: string, year: number, event?: React.MouseEvent<HTMLSpanElement>) => void; leagueId?: number | null }) {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [transactions, setTransactions] = useState<RecentWaiverItem[]>([]);

  useEffect(() => {
    if (!leagueId) {
      setLoading(false);
      return;
    }
    
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchRecentWaivers(2025, leagueId);
        if (!cancelled) setTransactions(data.transactions);
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message || e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="w-full mt-6 flex flex-col flex-1 min-h-0">
      <div className="flex flex-col flex-1 min-h-0">
        {/* Header */}
        <div className="mb-3 flex items-center justify-center gap-3">
          <h2 className="text-lg font-bold bg-gradient-to-r from-slate-100 to-slate-300 bg-clip-text text-transparent">
            Recent Waiver Activity
          </h2>
          <Link 
            href="/waivers" 
            className="text-xs text-slate-400 hover:text-slate-300 underline-offset-2 hover:underline transition-colors"
          >
            View All
          </Link>
        </div>

        {/* Waivers List */}
        <div className="flex-1 flex flex-col">
          {loading ? (
            <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 text-center">
              <div className="flex items-center justify-center gap-2 text-slate-400">
                <div className="w-4 h-4 border-2 border-slate-600 border-t-slate-300 rounded-full animate-spin"></div>
                <span className="text-xs">Loading...</span>
              </div>
            </div>
          ) : error ? (
            <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 text-center">
              <div className="text-xs text-rose-400">{error}</div>
            </div>
          ) : transactions.length === 0 ? (
            <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 text-center">
              <div className="text-xs text-slate-400">No recent waiver activity</div>
            </div>
          ) : (
            <div className="bg-slate-900 rounded-lg border border-slate-800 overflow-hidden flex-1 flex flex-col">
              {/* Single card with all transactions grouped by date */}
              <div className="px-3 py-2 space-y-4 flex-1 overflow-y-auto">
                {(() => {
                  // Group transactions by date
                  const groupedByDate = transactions.reduce((acc, transaction) => {
                    const date = formatDate(transaction.transaction_date);
                    if (!acc[date]) {
                      acc[date] = [];
                    }
                    acc[date].push(transaction);
                    return acc;
                  }, {} as Record<string, typeof transactions>);

                  return Object.entries(groupedByDate).map(([date, dateTransactions]) => (
                    <div key={date} className="space-y-2">
                      {/* Date as main subtitle */}
                      <div className="text-sm font-semibold text-slate-300 border-b border-slate-800 pb-1">
                        {date}
                      </div>

                      {/* Added Players Section */}
                      {dateTransactions.some(t => t.added_players.length > 0) && (
                        <div>
                          <div className="text-xs font-semibold text-green-400 mb-1">
                            Added:
                          </div>
                          <div className="space-y-1">
                            {dateTransactions.flatMap((transaction, transIdx) =>
                              transaction.added_players.map((player, playerIdx) => (
                                <div 
                                  key={`added-${date}-${transIdx}-${playerIdx}`} 
                                  className="flex items-center justify-between py-1"
                                >
                                  <div 
                                    className={`text-sm font-medium text-slate-200 ${onPlayerClick ? 'cursor-pointer hover:text-indigo-400 transition-colors' : ''}`}
                                    onClick={(e) => onPlayerClick?.(player.player_name, 2025, e)}
                                  >
                                    {player.player_name}
                                  </div>
                                  <div className="text-xs text-slate-400">
                                    {transaction.team_name}
                                  </div>
                                </div>
                              ))
                            )}
                          </div>
                        </div>
                      )}

                      {/* Dropped Players Section */}
                      {dateTransactions.some(t => t.dropped_players.length > 0) && (
                        <div>
                          <div className="text-xs font-semibold text-red-400 mb-1">
                            Dropped:
                          </div>
                          <div className="space-y-1">
                            {dateTransactions.flatMap((transaction, transIdx) =>
                              transaction.dropped_players.map((player, playerIdx) => (
                                <div 
                                  key={`dropped-${date}-${transIdx}-${playerIdx}`} 
                                  className="flex items-center justify-between py-1"
                                >
                                  <div 
                                    className={`text-sm font-medium text-slate-200 ${onPlayerClick ? 'cursor-pointer hover:text-indigo-400 transition-colors' : ''}`}
                                    onClick={(e) => onPlayerClick?.(player.player_name, 2025, e)}
                                  >
                                    {player.player_name}
                                  </div>
                                  <div className="text-xs text-slate-400">
                                    {transaction.team_name}
                                  </div>
                                </div>
                              ))
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ));
                })()}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Top10ZAVTable({ onPlayerClick, year, leagueId }: { onPlayerClick?: (playerName: string, year: number, event?: React.MouseEvent<HTMLDivElement>) => void; year: number; leagueId?: number | null }) {
  const [topPlayers, setTopPlayers] = useState<PlayerVorp[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!leagueId) {
      setLoading(false);
      return;
    }
    
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const players = await fetchTop10ZAV(year, leagueId);
        if (!cancelled) setTopPlayers(players);
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message || e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [leagueId]);

  // Calculate max ZAV for progress bar scaling
  const maxZav = topPlayers.length > 0 
    ? Math.max(...topPlayers.map(p => p.vorp_star || 0))
    : 30; // Default max if no players

    return (
      <div className="w-full">
        <div>
          {/* Header */}
          <div className="mb-6 flex items-center justify-center gap-3">
            <h2 className="text-xl font-bold bg-gradient-to-r from-slate-100 to-slate-300 bg-clip-text text-transparent">
              Top 10 Players
            </h2>
            <Link 
              href="/players" 
              className="text-xs text-slate-400 hover:text-slate-300 underline-offset-2 hover:underline transition-colors"
            >
              View All
            </Link>
          </div>
  
          {/* Players List */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
            {loading ? (
              <div className="p-6 text-center">
                <div className="flex items-center justify-center gap-2 text-slate-400">
                  <div className="w-4 h-4 border-2 border-slate-600 border-t-slate-300 rounded-full animate-spin"></div>
                  <span className="text-sm">Loading...</span>
                </div>
              </div>
            ) : error ? (
              <div className="p-6 text-center">
                <div className="text-sm text-rose-400">Failed to load top players</div>
              </div>
            ) : topPlayers.length === 0 ? (
              <div className="p-6 text-center">
                <div className="text-sm text-slate-400">No data available</div>
              </div>
            ) : (
              <div className="divide-y divide-slate-800">
                {topPlayers.map((player, i) => {
                  const zav = player.vorp_star || 0;
                  const zavPercentage = maxZav > 0 ? (zav / maxZav) * 100 : 0;
                  
                  // Use the same gradient and text color functions as the rest of the app
                  const barGradient = getZavGradient(zav);
                  const zavTextColor = getZavTextColor(zav);
                  
                  const rankColor = 
                    i === 0 ? 'text-amber-400' :
                    i === 1 ? 'text-slate-300' :
                    i === 2 ? 'text-orange-400' :
                    'text-slate-500';
  
                  return (
                    <div
                      key={player.player_name}
                      className="group hover:bg-slate-800/50 transition-all duration-200 p-2.5"
                    >
                      <div className="flex items-center gap-2 mb-1.5">
                        {/* Rank */}
                        <div className={`text-sm font-bold ${rankColor} w-5 text-center`}>
                          {i + 1}
                        </div>
                        
                        {/* Player Info */}
                        <div className="flex-1 min-w-0">
                          <div 
                            className={`font-semibold text-sm text-slate-100 truncate ${onPlayerClick ? 'cursor-pointer hover:text-indigo-400 transition-colors' : ''}`}
                            onClick={(e) => onPlayerClick?.(player.player_name, year, e)}
                          >
                            {player.player_name}
                            {player.fantasy_pos && (
                              <span className="text-slate-400 text-xs font-normal ml-1.5">{player.fantasy_pos}</span>
                            )}
                            {player.team && (
                              <>
                                <span className="text-slate-500 text-xs mx-1">•</span>
                                <span className="text-slate-500 text-xs font-normal">{player.team}</span>
                              </>
                            )}
                          </div>
                        </div>
                        
                        {/* ZAV Value */}
                        <div className="text-sm font-bold text-white tabular-nums">
                          {zav.toFixed(2)}
                        </div>
                      </div>
                      
                      {/* Progress Bar */}
                      <div className="relative h-1.5 bg-slate-800 rounded-full overflow-hidden ml-7">
                        <div 
                          className="relative h-full rounded-full transition-all duration-700 ease-out"
                          style={{ 
                            width: `${Math.min(100, zavPercentage)}%`,
                            background: barGradient
                          }}
                        >
                          {/* Gloss overlay */}
                          <div 
                            className="absolute inset-0 rounded-full opacity-30"
                            style={{
                              background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.4) 0%, rgba(255, 255, 255, 0) 50%, rgba(0, 0, 0, 0.2) 100%)'
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }


type SortColumn = 'team_name' | 'wins' | 'win_percentage' | 'points_for' | 'points_against';
type SortDirection = 'asc' | 'desc';

// Re-populate Confirmation Modal
function RepopulateConfirmModal({
  leagueId,
  onYes,
  onNo
}: {
  leagueId: number;
  onYes: () => void;
  onNo: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Data Already Exists
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          Data already exists for league ID {leagueId}. Do you want to re-populate it?
        </p>
        <div className="flex gap-3">
          <button
            onClick={onYes}
            className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
          >
            Yes
          </button>
          <button
            onClick={onNo}
            className="flex-1 bg-gray-300 hover:bg-gray-400 text-gray-800 font-semibold py-2 px-4 rounded-lg transition-colors"
          >
            No
          </button>
        </div>
      </div>
    </div>
  );
}

// League ID Modal Component
function LeagueIdModal({ 
  onSubmit, 
  onCancel 
}: { 
  onSubmit: (leagueId: number) => void; 
  onCancel?: () => void;
}) {
  const [leagueId, setLeagueId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [showRepopulateConfirm, setShowRepopulateConfirm] = useState<boolean>(false);
  const [pendingLeagueId, setPendingLeagueId] = useState<number | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    
    const id = parseInt(leagueId.trim(), 10);
    if (isNaN(id) || id <= 0) {
      setError('Please enter a valid league ID (positive number)');
      return;
    }

    setSubmitting(true);
    try {
      // Check if data exists
      const statusRes = await fetch(`/api/api/league-status/${id}`);
      if (!statusRes.ok) throw new Error('Failed to check league status');
      const status = await statusRes.json();
      
      if (status.status === 'ready') {
        // Data exists, show confirmation modal
        setPendingLeagueId(id);
        setShowRepopulateConfirm(true);
        setSubmitting(false);
        return;
      } else if (status.status === 'initializing') {
        setError('This league is already being initialized. Please wait.');
        setSubmitting(false);
        return;
      } else {
        // Start initialization
        const initRes = await fetch(`/api/api/initialize-league/${id}`, {
          method: 'POST'
        });
        if (!initRes.ok) {
          const errorData = await initRes.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Failed to start initialization');
        }
        onSubmit(id);
      }
    } catch (e: any) {
      setError(e.message || 'Failed to initialize league');
      setSubmitting(false);
    }
  };

  const handleRepopulateYes = async () => {
    if (!pendingLeagueId) return;
    setShowRepopulateConfirm(false);
    setSubmitting(true);
    try {
      // Initialize with force=true
      const initRes = await fetch(`/api/api/initialize-league/${pendingLeagueId}?force=true`, {
        method: 'POST'
      });
      if (!initRes.ok) throw new Error('Failed to start initialization');
      onSubmit(pendingLeagueId);
    } catch (e: any) {
      setError(e.message || 'Failed to start initialization');
      setSubmitting(false);
    }
  };

  const handleRepopulateNo = () => {
    if (!pendingLeagueId) return;
    setShowRepopulateConfirm(false);
    // User said no, just use existing data
    onSubmit(pendingLeagueId);
  };

  if (showRepopulateConfirm && pendingLeagueId) {
    return (
      <RepopulateConfirmModal
        leagueId={pendingLeagueId}
        onYes={handleRepopulateYes}
        onNo={handleRepopulateNo}
      />
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Enter League ID
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          Please enter your ESPN Fantasy Football League ID to get started.
        </p>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="leagueId" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              League ID
            </label>
            <input
              id="leagueId"
              type="number"
              value={leagueId}
              onChange={(e) => setLeagueId(e.target.value)}
              placeholder="e.g., 86952922"
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              required
              disabled={submitting}
            />
            {error && (
              <p className="mt-2 text-sm text-red-600 dark:text-red-400">{error}</p>
            )}
          </div>
          <div className="flex gap-3">
            <button
              type="submit"
              disabled={submitting}
              className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
            >
              {submitting ? 'Starting...' : 'Continue'}
            </button>
            {onCancel && (
              <button
                type="button"
                onClick={onCancel}
                disabled={submitting}
                className="flex-1 bg-gray-300 hover:bg-gray-400 disabled:bg-gray-200 text-gray-800 font-semibold py-2 px-4 rounded-lg transition-colors"
              >
                Cancel
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

// Initialization Status Component
function InitializationStatus({ leagueId, onComplete }: { leagueId: number; onComplete: () => void }) {
  const [status, setStatus] = useState<{ status: string; message: string; progress?: string } | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;
    let pollInterval: NodeJS.Timeout;

    async function checkStatus() {
      try {
        const res = await fetch(`/api/api/league-status/${leagueId}`);
        if (!res.ok) throw new Error('Failed to check status');
        const data = await res.json();
        
        if (!cancelled) {
          setStatus(data);
          setLoading(false);
          
          if (data.status === 'ready') {
            onComplete();
            return; // Stop polling
          } else if (data.status === 'error') {
            // Keep showing error, don't auto-complete
            return; // Stop polling
          } else if (data.status === 'initializing') {
            // Continue polling
            pollInterval = setTimeout(checkStatus, 2000); // Poll every 2 seconds
          }
        }
      } catch (e) {
        if (!cancelled) {
          setLoading(false);
          setStatus({ status: 'error', message: 'Failed to check initialization status' });
        }
      }
    }

    checkStatus();
    
    return () => {
      cancelled = true;
      if (pollInterval) clearTimeout(pollInterval);
    };
  }, [leagueId, onComplete]);

  if (loading || !status) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Initializing...</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">Setting up your league data</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (status.status === 'error') {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
          <h3 className="text-lg font-semibold text-red-600 dark:text-red-400 mb-2">Initialization Error</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">{status.message}</p>
          <button
            onClick={() => window.location.reload()}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
          >
            Reload Page
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Initializing...</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">{status.progress || status.message}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [leagueId, setLeagueId] = useState<number | null>(null);
  const [showModal, setShowModal] = useState<boolean>(false);
  const [initializing, setInitializing] = useState<boolean>(false);
  const [selectedYear, setSelectedYear] = useState<number | 'ALL'>(2025);
  const [data, setData] = useState<StandingsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Sorting state
  const [sortColumn, setSortColumn] = useState<SortColumn>('win_percentage');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  
  // Team roster state
  const [rosterTeams, setRosterTeams] = useState<TeamRosterDetail[]>([]);
  const [rosterLoading, setRosterLoading] = useState<boolean>(true);
  const [rosterError, setRosterError] = useState<string | null>(null);
  const [expandedTeams, setExpandedTeams] = useState<Set<number>>(new Set());
  const [selectedPlayers, setSelectedPlayers] = useState<Map<string, {playerName: string, year: number, selectedYear: number, availableYears: number[], position?: {x: number, y: number}, stats?: PlayerWeeklyStatsResponse, loading?: boolean, headshotUrl?: string | null}>>(new Map());

  // Check for league ID on mount
  useEffect(() => {
    const storedLeagueId = getLeagueId();
    if (storedLeagueId) {
      // Check if data is ready
      fetch(`/api/api/league-status/${storedLeagueId}`)
        .then(res => res.json())
        .then(status => {
          if (status.status === 'ready') {
            setLeagueId(storedLeagueId);
          } else if (status.status === 'initializing') {
            setLeagueId(storedLeagueId);
            setInitializing(true);
          } else {
            setShowModal(true);
          }
        })
        .catch(() => {
          setShowModal(true);
        });
    } else {
      setShowModal(true);
    }
  }, []);

  const handleLeagueIdSubmit = (id: number) => {
    setLeagueId(id);
    setLeagueId(id); // Store in localStorage
    setShowModal(false);
    // Check if we need to initialize or if data is already ready
    fetch(`/api/api/league-status/${id}`)
      .then(res => res.json())
      .then(status => {
        if (status.status === 'initializing') {
          setInitializing(true);
        } else if (status.status === 'ready') {
          // Data is ready, just show dashboard
          setInitializing(false);
        } else {
          // Shouldn't happen, but set initializing just in case
          setInitializing(true);
        }
      })
      .catch(() => {
        // If check fails, assume we need to initialize
        setInitializing(true);
      });
  };

  const handleInitializationComplete = () => {
    setInitializing(false);
    // Just update state, don't reload - this will show the dashboard
    // The useEffect will automatically reload data when leagueId is set
  };

  useEffect(() => {
    if (!leagueId) return;
    
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        if (selectedYear === 'ALL') {
          // Fetch standings for all years and aggregate
          const allStandings = await Promise.all(
            YEARS.map(year => fetchStandings(year, leagueId))
          );
          const aggregated = aggregateStandingsByTeamId(allStandings);
          const filteredTeams = aggregated.teams.filter(t => !isExcludedTeamName(t.team_name));
          if (!cancelled) setData({ ...aggregated, teams: filteredTeams, num_teams: filteredTeams.length });
        } else {
          // Fetch standings for specific year
          const d = await fetchStandings(selectedYear, leagueId);
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
  }, [selectedYear, leagueId]);

  // Load team rosters (only when a specific year is selected, not "ALL")
  useEffect(() => {
    if (selectedYear === 'ALL') {
      // Don't fetch rosters when "ALL" is selected
      setRosterTeams([]);
      setRosterLoading(false);
      setRosterError(null);
      return;
    }
    
    let cancelled = false;
    async function loadRosters() {
      setRosterLoading(true);
      setRosterError(null);
      try {
        const teamData = await fetchTeamRosters(selectedYear as number);
        if (!cancelled) setRosterTeams(teamData);
      } catch (e: any) {
        if (!cancelled) setRosterError(String(e?.message || e));
      } finally {
        if (!cancelled) setRosterLoading(false);
      }
    }
    loadRosters();
    return () => { cancelled = true; };
  }, [selectedYear]);

  const toggleTeam = (teamId: number) => {
    setExpandedTeams(prev => {
      const newSet = new Set(prev);
      if (newSet.has(teamId)) {
        newSet.delete(teamId);
      } else {
        newSet.add(teamId);
      }
      return newSet;
    });
  };

  // Helper function to check if player has data
  const hasPlayerData = (stats: PlayerWeeklyStatsResponse): boolean => {
    // Check if there's at least one week with actual data
    return stats.weekly_stats.some(stat => 
      stat.z_week_ppr !== null || stat.weekly_points_ppr !== null
    );
  };

  const handlePlayerClick = async (playerName: string, year: number, event?: React.MouseEvent<HTMLSpanElement>) => {
    const playerKey = `${playerName}_${year}`;
    
    if (selectedPlayers.has(playerKey)) {
      return;
    }
    
    // Always center pop-ups, stack vertically when multiple
      const existingPopups = Array.from(selectedPlayers.values());
    const popupHeight = 200; // Increased size
    const popupSpacing = 20;
    const totalHeight = (existingPopups.length + 1) * (popupHeight + popupSpacing) - popupSpacing;
    const startY = -totalHeight / 2;
    const offsetY = existingPopups.length * (popupHeight + popupSpacing);
    
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      newMap.set(playerKey, { playerName, year, selectedYear: year, availableYears: [year], position: { x: 0, y: startY + offsetY }, loading: true });
      return newMap;
    });
    
    try {
      // Fetch stats and headshot in parallel
      const [statsResponse, headshotResponse] = await Promise.all([
        fetch(`/api/players/${encodeURIComponent(playerName)}/weekly-stats?year=${year}`),
        fetch(`/api/players/${encodeURIComponent(playerName)}/headshot`)
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
      
      // Check all years to see which ones have data
      const availableYears: number[] = [];
      const yearChecks = YEARS.map(async (checkYear) => {
        try {
          const checkResponse = await fetch(`/api/players/${encodeURIComponent(playerName)}/weekly-stats?year=${checkYear}`);
          if (checkResponse.ok) {
            const checkData: PlayerWeeklyStatsResponse = await checkResponse.json();
            if (hasPlayerData(checkData)) {
              availableYears.push(checkYear);
            }
          }
        } catch (err) {
          // Silently skip years that fail
        }
      });
      
      await Promise.all(yearChecks);
      
      // Sort available years
      availableYears.sort();
      
      // Update with stats, headshot, and available years
      setSelectedPlayers(prev => {
        const newMap = new Map(prev);
        const existing = newMap.get(playerKey);
        if (existing) {
          newMap.set(playerKey, { 
            ...existing, 
            stats: data, 
            headshotUrl: headshotUrl,
            availableYears: availableYears.length > 0 ? availableYears : [year], // Fallback to at least the current year
            loading: false 
          });
        }
        return newMap;
      });
    } catch (err) {
      console.error('Error fetching player stats:', err);
      setSelectedPlayers(prev => {
        const newMap = new Map(prev);
        newMap.delete(playerKey);
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
      const response = await fetch(`/api/players/${encodeURIComponent(playerData.playerName)}/weekly-stats?year=${newYear}`);
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

  const groupPlayersByPosition = (players: PlayerRoster[]) => {
    const grouped: Record<string, PlayerRoster[]> = {
      'QB': [],
      'RB': [],
      'WR': [],
      'TE': [],
      'D/ST': []
    };
    
    players.forEach(player => {
      const pos = player.position || 'OTHER';
      // Handle D/ST variations
      if (pos === 'D/ST' || pos === 'DST' || pos === 'DEF') {
        grouped['D/ST'].push(player);
      } else if (grouped[pos]) {
        grouped[pos].push(player);
      }
    });
    
    // Sort each position group by ZAV (highest first, null values last)
    Object.keys(grouped).forEach(pos => {
      grouped[pos].sort((a, b) => {
        // Handle null values - put them at the end
        if (a.vorp_star === null && b.vorp_star === null) return 0;
        if (a.vorp_star === null) return 1; // a goes to end
        if (b.vorp_star === null) return -1; // b goes to end
        
        // Sort by ZAV descending (highest first)
        return (b.vorp_star || 0) - (a.vorp_star || 0);
      });
    });
    
    return grouped;
  };

  const formatPlayerName = (fullName: string): string => {
    const parts = fullName.trim().split(/\s+/);
    if (parts.length === 1) return fullName; // Single name, return as is
    if (parts.length === 2) {
      // First name + Last name
      return `${parts[0][0]}. ${parts[1]}`;
    }
    
    // Handle suffixes like Jr., Sr., II, III, etc.
    const suffixPattern = /^(jr|sr|ii|iii|iv|v|vi|vii|viii|ix|x)\.?$/i;
    const lastPart = parts[parts.length - 1];
    const secondLastPart = parts[parts.length - 2];
    
    // Check if last part is a suffix
    if (suffixPattern.test(lastPart)) {
      // Include second to last + suffix (e.g., "J. Doe Jr.")
      return `${parts[0][0]}. ${secondLastPart} ${lastPart}`;
    }
    
    // Check if second to last part looks like part of a compound last name
    // (e.g., "Van", "De", "La", "Mc", "O'", etc.)
    const compoundPrefixes = /^(van|de|la|le|du|da|del|der|von|mc|mac|o')$/i;
    if (compoundPrefixes.test(secondLastPart)) {
      // Include the compound last name parts (e.g., "J. Van Der Berg")
      return `${parts[0][0]}. ${parts.slice(1).join(' ')}`;
    }
    
    // If there are 3+ parts and no suffix/compound detected, check if last two parts might be compound
    // For names like "John De La Cruz" or "John Van Der Berg", include all parts after first name
    if (parts.length >= 3) {
      // Check if any middle parts are compound prefixes
      for (let i = 1; i < parts.length - 1; i++) {
        if (compoundPrefixes.test(parts[i])) {
          // Found a compound prefix, include everything from this point to the end
          return `${parts[0][0]}. ${parts.slice(i).join(' ')}`;
        }
      }
    }
    
    // Default: first initial + last name
    return `${parts[0][0]}. ${lastPart}`;
  };

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

  const getSortedTeams = (): TeamRow[] => {
    if (!data?.teams) return [];
    
    const sorted = [...data.teams].sort((a, b) => {
      let aValue: number | string;
      let bValue: number | string;
      
      switch (sortColumn) {
        case 'team_name':
          aValue = a.team_name.toLowerCase();
          bValue = b.team_name.toLowerCase();
          break;
        case 'wins':
          aValue = a.wins;
          bValue = b.wins;
          break;
        case 'win_percentage':
          aValue = a.win_percentage;
          bValue = b.win_percentage;
          break;
        case 'points_for':
          aValue = a.points_for;
          bValue = b.points_for;
          break;
        case 'points_against':
          aValue = a.points_against;
          bValue = b.points_against;
          break;
        default:
          return 0;
      }
      
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        if (sortDirection === 'asc') {
          return aValue.localeCompare(bValue);
        } else {
          return bValue.localeCompare(aValue);
        }
      } else {
        if (sortDirection === 'asc') {
          return (aValue as number) - (bValue as number);
        } else {
          return (bValue as number) - (aValue as number);
        }
      }
    });
    
    return sorted;
  };

  // Show modal if no league ID
  if (showModal) {
    return (
      <LeagueIdModal 
        onSubmit={handleLeagueIdSubmit}
      />
    );
  }

  // Show initialization status if initializing
  if (initializing && leagueId) {
    return (
      <InitializationStatus 
        leagueId={leagueId} 
        onComplete={handleInitializationComplete}
      />
    );
  }

  // Don't render main content until league ID is set
  if (!leagueId) {
    return null;
  }

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <div className="max-w-none mx-0 pr-8 pt-8 pb-8 pl-5">

        {/* Header Navigation */}
        <header className="mb-12">
          <nav className="flex justify-center items-center gap-3 flex-wrap">
            <Link 
              href="/" 
              className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]"
            >
              Home
            </Link>
            <Link 
              href="/players" 
              className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]"
            >
            Players
          </Link>
            <Link 
              href="/trades" 
              className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]"
            >
            Trades
          </Link>
            <Link 
              href="/waivers" 
              className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]"
            >
            Waivers
          </Link>
            <Link 
              href="/scoreboard" 
              className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]"
            >
            Scoreboard
          </Link>
            <Link 
              href="/draft" 
              className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]"
            >
            Draft
          </Link>
        </nav>
        </header>

        {/* Standings and Top 10 ZAV Section - Side by Side */}
        <div className="mt-12 flex gap-6 items-stretch">
          {/* Standings Section - Left */}
          <div className="flex-1 flex flex-col">
            {/* Header */}
            <div className="mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 flex-1 justify-center">
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-slate-100 to-slate-300 bg-clip-text text-transparent">
                    League Standings
                  </h2>
                </div>
                
                {/* Year Selector */}
                <div className="flex items-center gap-2">
                  <select
                    value={selectedYear}
                    onChange={(e) => {
                      const value = e.target.value;
                      setSelectedYear(value === 'ALL' ? 'ALL' : parseInt(value, 10));
                    }}
                    className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98] cursor-pointer focus:outline-none focus:ring-2 focus:ring-slate-300"
                  >
                    {YEARS.map((year) => (
                      <option key={year} value={year}>
                        {year}
                      </option>
                    ))}
                    <option value="ALL">ALL</option>
                  </select>
                </div>
                
                <div className="flex items-center gap-3">
                  {loading && (
                    <span className="inline-flex items-center gap-2 text-sm font-medium text-slate-300">
                      <div className="w-4 h-4 border-2 border-slate-600 border-t-slate-300 rounded-full animate-spin"></div>
                      Loading
                    </span>
                  )}
                  {error && (
                    <span className="text-sm font-medium text-rose-400 bg-rose-900/20 px-3 py-1 rounded-full">
                      {error}
                    </span>
                  )}
              </div>
            </div>
          </div>

            {/* Table Container */}
            <div className="bg-slate-900 rounded-2xl shadow-xl shadow-slate-950/50 border border-slate-800 overflow-hidden flex-1 flex flex-col min-h-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-gradient-to-r from-slate-800 to-slate-700">
                      <th 
                        className="text-left px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                        onClick={() => handleSort('team_name')}
                      >
                        <div className="flex items-center gap-2">
                          Team
                          {sortColumn === 'team_name' && (
                            <span className="text-xs">
                              {sortDirection === 'asc' ? '↑' : '↓'}
                            </span>
                          )}
                        </div>
                      </th>
                      <th 
                        className="text-center px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                        onClick={() => handleSort('wins')}
                      >
                        <div className="flex items-center justify-center gap-2">
                          Record
                          {sortColumn === 'wins' && (
                            <span className="text-xs">
                              {sortDirection === 'asc' ? '↑' : '↓'}
                            </span>
                          )}
                        </div>
                      </th>
                      <th 
                        className="text-center px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                        onClick={() => handleSort('win_percentage')}
                      >
                        <div className="flex items-center justify-center gap-2">
                          Win %
                          {sortColumn === 'win_percentage' && (
                            <span className="text-xs">
                              {sortDirection === 'asc' ? '↑' : '↓'}
                            </span>
                          )}
                        </div>
                      </th>
                      <th 
                        className="text-center px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                        onClick={() => handleSort('points_for')}
                      >
                        <div className="flex items-center justify-center gap-2">
                          Points For
                          {sortColumn === 'points_for' && (
                            <span className="text-xs">
                              {sortDirection === 'asc' ? '↑' : '↓'}
                            </span>
                          )}
                        </div>
                      </th>
                      <th 
                        className="text-center px-6 py-4 text-sm font-semibold text-white uppercase tracking-wider cursor-pointer hover:bg-slate-700/50 transition-colors"
                        onClick={() => handleSort('points_against')}
                      >
                        <div className="flex items-center justify-center gap-2">
                          Points Against
                          {sortColumn === 'points_against' && (
                            <span className="text-xs">
                              {sortDirection === 'asc' ? '↑' : '↓'}
                            </span>
                          )}
                        </div>
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {!loading && !error && data?.teams?.length ? (
                      getSortedTeams().map((team: TeamRow, index: number) => {
                        const winPctColor = 
                          team.win_percentage >= 70 ? 'text-emerald-400' :
                          team.win_percentage >= 50 ? 'text-amber-400' :
                          'text-rose-400';
                        
                        const isExpanded = expandedTeams.has(team.team_id);
                        const teamRoster = rosterTeams.find(t => t.team_id === team.team_id);
                        const groupedPlayers = teamRoster ? groupPlayersByPosition(teamRoster.players) : {};
                        const positionOrder = ['QB', 'RB', 'WR', 'TE', 'D/ST'];
                        
                        return (
                          <>
                            <tr
                              key={team.team_id}
                              className="group hover:bg-slate-800/50 transition-all duration-200 ease-in-out"
                            >
                              <td className="px-6 py-5">
                                {selectedYear !== 'ALL' ? (
                                  <button
                                    onClick={() => toggleTeam(team.team_id)}
                                    className="font-semibold text-slate-100 text-base group-hover:text-indigo-400 transition-colors cursor-pointer text-left"
                                  >
                                    {team.team_name}
                                  </button>
                                ) : (
                                  <div className="font-semibold text-slate-100 text-base">
                                    {team.team_name}
                                  </div>
                                )}
                              </td>
                              
                              <td className="px-6 py-5 text-center">
                                <div className="inline-flex items-center gap-2 bg-slate-800 px-4 py-2 rounded-lg">
                                  <span className="font-mono font-bold text-slate-100">
                                    {team.wins}
                                  </span>
                                  <span className="text-slate-500">-</span>
                                  <span className="font-mono font-bold text-slate-100">
                                    {team.losses}
                                  </span>
                                  {team.ties > 0 && (
                                    <>
                                      <span className="text-slate-500">-</span>
                                      <span className="font-mono font-bold text-slate-100">
                                        {team.ties}
                                      </span>
                                    </>
                                  )}
                                </div>
                              </td>
                              
                              <td className="px-6 py-5 text-center">
                                <div className="flex flex-col items-center gap-2">
                                  <span className={`font-bold text-lg ${winPctColor}`}>
                                    {team.win_percentage.toFixed(1)}%
                                  </span>
                                  <div className="w-20 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                    <div 
                                      className={`h-full rounded-full transition-all duration-500 ${
                                        team.win_percentage >= 70 ? 'bg-gradient-to-r from-emerald-500 to-emerald-600' :
                                        team.win_percentage >= 50 ? 'bg-gradient-to-r from-amber-500 to-amber-600' :
                                        'bg-gradient-to-r from-rose-500 to-rose-600'
                                      }`}
                                      style={{ width: `${team.win_percentage}%` }}
                                    />
                                  </div>
                                </div>
                              </td>
                              
                              <td className="px-6 py-5 text-center">
                                <div className="font-semibold text-slate-300">
                                  {team.points_for.toFixed(1)}
                                </div>
                              </td>
                              
                              <td className="px-6 py-5 text-center">
                                <div className="font-semibold text-slate-300">
                                  {team.points_against.toFixed(1)}
                                </div>
                              </td>
                            </tr>
                            
                            {/* Roster Dropdown - Only show when a specific year is selected */}
                            {selectedYear !== 'ALL' && isExpanded && teamRoster && (
                              <tr key={`${team.team_id}-roster`}>
                                <td colSpan={5} className="px-6 py-4 bg-slate-800/30">
                                  <div className="flex flex-nowrap gap-x-6 gap-y-4 items-start overflow-x-auto">
                                    {positionOrder.map((pos) => {
                                      const players = groupedPlayers[pos] || [];
                                      if (players.length === 0) return null;
                                      
                                      return (
                                        <div key={pos} className="flex flex-col gap-2 flex-shrink-0">
                                          <div className="text-xs font-semibold text-slate-400 uppercase mb-1 text-center w-40">
                                            {pos}
                                          </div>
                                          <div className="flex flex-col gap-2">
                                            {players.map((player, idx) => {
                                              const zavGradient = getZavGradient(player.vorp_star);
                                              const zavTextColor = getZavTextColor(player.vorp_star);
                                              const displayName = pos === 'D/ST' ? player.player_name : formatPlayerName(player.player_name);
                                              return (
                                                <div key={`${player.player_name}-${idx}`} className="flex items-center gap-0">
                                                  <span 
                                                    onClick={(e) => {
                                                      const year = typeof selectedYear === 'number' ? selectedYear : 2025;
                                                      handlePlayerClick(player.player_name, year, e);
                                                    }}
                                                    className="text-white text-base cursor-pointer hover:text-indigo-400 transition-colors w-40"
                                                  >
                                                    {displayName}
                                                  </span>
                                                  {player.vorp_star !== null && (
                                                    <span
                                                      className="inline-flex items-center justify-center px-1.5 py-0.5 rounded text-sm font-bold min-w-[2.5rem] -ml-1"
                                                      style={{ background: zavGradient, color: zavTextColor }}
                                                    >
                                                      {player.vorp_star.toFixed(2)}
                                                    </span>
                                                  )}
                                                </div>
                                              );
                                            })}
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </td>
                              </tr>
                            )}
                          </>
                        );
                      })
                    ) : (
                      <tr>
                        <td colSpan={5} className="px-6 py-12 text-center">
                          <div className="text-slate-400">
                            {loading ? (
                              <div className="flex items-center justify-center gap-2">
                                <div className="w-5 h-5 border-2 border-slate-600 border-t-slate-300 rounded-full animate-spin"></div>
                                Loading standings...
                              </div>
                            ) : error ? (
                              <div className="text-rose-400">Failed to load standings</div>
                            ) : (
                              'No data available'
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Top 10 ZAV Section - Right */}
          <div className="flex-1 max-w-md pt-[0.5rem] flex flex-col">
            <Top10ZAVTable onPlayerClick={handlePlayerClick} year={selectedYear === 'ALL' ? 2025 : selectedYear} leagueId={leagueId} />
            <div className="flex-1 flex flex-col min-h-0">
              <RecentWaiversBox onPlayerClick={handlePlayerClick} leagueId={leagueId} />
            </div>
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
              <div className="flex items-center justify-between mb-3 relative">
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
                      </div>
                    );
                  })}
                </div>
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
    </div>
  );
}