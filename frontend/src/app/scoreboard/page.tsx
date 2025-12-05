'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
const YEARS = [2020, 2021, 2022, 2024, 2025] as const;

const ZAV_CUTOFFS = {
  red: -2,
  orange: 2.5,
  yellow: 6,
  yellowGreen: 10,
};

function getZavBrightColor(vorp: number | null): string {
  if (vorp === null || vorp === 0) {
    return 'rgb(131, 131, 131)';
  }
  if (vorp < ZAV_CUTOFFS.red) {
    return 'rgb(220, 70, 70)';
  } else if (vorp < ZAV_CUTOFFS.orange) {
    return 'rgb(250, 130, 60)';
  } else if (vorp < ZAV_CUTOFFS.yellow) {
    return 'rgb(245, 210, 75)';
  } else if (vorp < ZAV_CUTOFFS.yellowGreen) {
    return 'rgb(160, 220, 80)';
  } else {
    return 'rgb(16, 185, 129)';
  }
}

function getFantasyPointsGradient(points: number | null): string {
  if (points === null) {
    return 'linear-gradient(135deg, rgb(131, 131, 131), rgb(100, 100, 100))';
  }
  if (points <= 5) {
    return 'linear-gradient(135deg, rgb(220, 70, 70), rgb(180, 30, 30))';
  } else if (points <= 10) {
    return 'linear-gradient(135deg, rgb(250, 130, 60), rgb(210, 90, 20))';
  } else if (points <= 15) {
    return 'linear-gradient(135deg, rgb(245, 210, 75), rgb(225, 170, 35))';
  } else if (points <= 20) {
    return 'linear-gradient(135deg, rgb(160, 220, 80), rgb(120, 180, 40))';
  } else {
    return 'linear-gradient(135deg, rgb(16, 185, 129), rgb(5, 150, 105))';
  }
}

function getFantasyPointsTextColor(points: number | null): string {
  if (points === null) {
    return 'rgb(40, 40, 40)';
  }
  if (points <= 5) {
    return 'rgb(90, 10, 10)';
  } else if (points <= 10) {
    return 'rgb(110, 35, 5)';
  } else if (points <= 15) {
    return 'rgb(110, 85, 5)';
  } else if (points <= 20) {
    return 'rgb(50, 75, 10)';
  } else {
    return 'rgb(1, 50, 20)';
  }
}

type GameResult = {
  week: number;
  opponent: string;
  score: number;
  opponent_score: number;
  result: string; // "W" or "L"
  margin: number;
  is_playoff: boolean;
  matchup_type: string;
};

type TeamScoreboard = {
  team_name: string;
  wins: number;
  losses: number;
  total_points: number;
  win_percentage: number;
  games: GameResult[];
};

type TopScoringWeek = {
  team_name: string;
  points: number;
  week: number;
};

type ScoreboardResponse = {
  year: number;
  teams: TeamScoreboard[];
  top_scoring_week?: TopScoringWeek | null;
};

type PlayerScore = {
  player_name: string;
  position: string;
  points: number;
  projected_points: number;
};

type TeamRoster = {
  team_name: string;
  total_score: number;
  players: PlayerScore[];
};

type MatchupDetail = {
  year: number;
  week: number;
  home_team: TeamRoster;
  away_team: TeamRoster;
  is_playoff: boolean;
};

type GameDetail = {
  week: number;
  team1: string;
  team2: string;
  score1: number;
  score2: number;
  winner: string;
  margin: number;
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

type SelectedPlayerData = {
  headshotUrl?: string | null;
  playerName: string;
  year: number;
  selectedYear: number;
  availableYears: number[];
  position?: { x: number; y: number };
  loading: boolean;
  stats?: PlayerWeeklyStatsResponse;
};

async function fetchScoreboard(year: number): Promise<ScoreboardResponse> {
  const res = await fetch(`${API_BASE}/scoreboard/${year}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch scoreboard for ${year}`);
  return res.json();
}

async function fetchMatchupDetail(year: number, week: number, team1: string, team2: string): Promise<MatchupDetail> {
  const res = await fetch(`${API_BASE}/matchup/${year}/${week}?team1=${encodeURIComponent(team1)}&team2=${encodeURIComponent(team2)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch matchup details for ${team1} vs ${team2}`);
  return res.json();
}

function GameBubble({ 
  game, 
  week, 
  onClick,
  isEliminated = false,
  isChampionship = false,
  isTopScoringWeek = false
}: { 
  game: GameResult | null; 
  week: number; 
  onClick: () => void;
  isEliminated?: boolean;
  isChampionship?: boolean;
  isTopScoringWeek?: boolean;
}) {
  if (!game) {
    return (
      <div 
        className="w-12 h-12 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center cursor-pointer hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
        onClick={onClick}
      >
        <span className="text-xs text-gray-500 dark:text-gray-400">BYE</span>
      </div>
    );
  }

  const isWin = game.result === "W";
  const isPlayoff = game.is_playoff;
  
  // If this is the top scoring week, show a white star with rounded edges
  if (isTopScoringWeek) {
    return (
      <div 
        className="w-12 h-12 flex items-center justify-center cursor-pointer transition-all hover:scale-110 relative"
        onClick={onClick}
      >
        <svg 
          className="absolute inset-0 w-full h-full text-white" 
          fill="currentColor" 
          viewBox="0 0 24 24"
          style={{ 
            filter: 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3)) drop-shadow(0 0 4px rgba(255, 255, 255, 0.5))',
            transform: 'scale(1.15)',
          }}
        >
          <path 
            d="M12 2.5l2.8 5.7 6.2.9-4.5 4.4 1.1 6.3-5.6-2.9-5.6 2.9 1.1-6.3-4.5-4.4 6.2-.9L12 2.5z"
            style={{ 
              fillRule: 'evenodd', 
              clipRule: 'evenodd',
              stroke: 'rgba(255, 255, 255, 0.8)',
              strokeWidth: '0.5',
              strokeLinejoin: 'round',
              strokeLinecap: 'round',
            }}
          />
        </svg>
        <span className="text-black font-bold text-xs relative z-10">W</span>
      </div>
    );
  }
  
  return (
    <div 
      className={`w-12 h-12 rounded-full flex items-center justify-center cursor-pointer transition-all hover:scale-110 ${
        isChampionship && isWin
          ? 'bg-gradient-to-br from-yellow-400 to-yellow-600 hover:from-yellow-300 hover:to-yellow-500 text-white shadow-2xl animate-pulse'
          : isPlayoff && isEliminated
          ? (isWin 
              ? 'bg-emerald-300 hover:bg-emerald-400 text-white shadow-lg opacity-60' 
              : 'bg-violet-300 hover:bg-violet-400 text-white shadow-lg opacity-60')
          : isPlayoff
          ? (isWin 
              ? 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-lg' 
              : 'bg-violet-600 hover:bg-violet-700 text-white shadow-lg')
          : (isWin 
              ? 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-lg' 
              : 'bg-violet-600 hover:bg-violet-700 text-white shadow-lg')
      }`}
      onClick={onClick}
      style={isChampionship && isWin ? {
        boxShadow: '0 0 20px rgba(251, 191, 36, 0.8), 0 0 40px rgba(251, 191, 36, 0.6), 0 0 60px rgba(251, 191, 36, 0.4)'
      } : {}}
    >
      <span className="text-xs font-semibold">{game.result}</span>
    </div>
  );
}

function GameDetailModal({ 
  matchup, 
  isOpen, 
  onClose,
  loading = false,
  onPlayerClick,
  year
}: { 
  matchup: MatchupDetail | null; 
  isOpen: boolean; 
  onClose: () => void;
  loading?: boolean;
  onPlayerClick?: (playerName: string, year: number, event?: React.MouseEvent<HTMLSpanElement>) => void;
  year: number;
}) {
  if (!isOpen) return null;

  if (loading) {
    return (
      <div className="fixed inset-0 backdrop-blur-sm bg-white/20 dark:bg-black/20 flex items-center justify-center z-50">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto mb-4"></div>
            <p className="text-gray-600 dark:text-gray-400">Loading matchup details...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!matchup) return null;

  const positionOrder = ['QB', 'RB', 'RB', 'WR', 'WR', 'TE', 'FLEX', 'D/ST', 'K'];
  
  const getStarters = (team: TeamRoster) => {
    const starters: PlayerScore[] = [];
    const usedPlayers = new Set<string>();
    
    // Get players by position, handling multiple RBs and WRs
    const qb = team.players.find(p => p.position === 'QB' && !usedPlayers.has(p.player_name));
    const rbs = team.players.filter(p => p.position === 'RB' && !usedPlayers.has(p.player_name)).slice(0, 2);
    const wrs = team.players.filter(p => p.position === 'WR' && !usedPlayers.has(p.player_name)).slice(0, 2);
    const te = team.players.find(p => p.position === 'TE' && !usedPlayers.has(p.player_name));
    const flex = team.players.find(p => p.position == 'RB/WR/TE' && !usedPlayers.has(p.player_name));
    const dst = team.players.find(p => p.position === 'D/ST' && !usedPlayers.has(p.player_name));
    const k = team.players.find(p => p.position === 'K' && !usedPlayers.has(p.player_name));
    
    // Add players in the correct order
    if (qb) {
      starters.push(qb);
      usedPlayers.add(qb.player_name);
    }
    
    // Add RBs (up to 2)
    rbs.forEach(rb => {
      if (rb) {
        starters.push(rb);
        usedPlayers.add(rb.player_name);
      }
    });
    
    // Add WRs (up to 2)
    wrs.forEach(wr => {
      if (wr) {
        starters.push(wr);
        usedPlayers.add(wr.player_name);
      }
    });
    
    if (te) {
      starters.push(te);
      usedPlayers.add(te.player_name);
    }
    
    if (flex) {
      starters.push(flex);
      usedPlayers.add(flex.player_name);
    }
    
    if (dst) {
      starters.push(dst);
      usedPlayers.add(dst.player_name);
    }
    
    if (k) {
      starters.push(k);
      usedPlayers.add(k.player_name);
    }
    
    
    return starters;
  };

  const homeStarters = getStarters(matchup.home_team);
  const awayStarters = getStarters(matchup.away_team);

  const getPlayerComparison = (homePlayer: PlayerScore, awayPlayer: PlayerScore) => {
    // Use the same points field that's displayed in the table
    const homePoints = homePlayer.points || 0;
    const awayPoints = awayPlayer.points || 0;
    
    // Convert to numbers to ensure proper comparison
    const homeNum = Number(homePoints);
    const awayNum = Number(awayPoints);
    
    return {
      homeWins: homeNum > awayNum,
      awayWins: awayNum > homeNum,
      homePoints: homeNum,
      awayPoints: awayNum
    };
  };

  return (
    <div className="fixed inset-0 backdrop-blur-sm bg-white/20 dark:bg-black/20 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
              Week {matchup.week} Matchup
            </h3>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          {/* Team Scores Header */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className={`text-center p-4 rounded-lg ${
              matchup.home_team.total_score > matchup.away_team.total_score 
                ? 'bg-emerald-50 dark:bg-emerald-900/20 border-2 border-emerald-200 dark:border-emerald-700' 
                : 'bg-gray-50 dark:bg-gray-700'
            }`}>
              <h4 className={`text-lg font-semibold mb-2 ${
                matchup.home_team.total_score > matchup.away_team.total_score 
                  ? 'text-emerald-800 dark:text-emerald-200' 
                  : 'text-gray-900 dark:text-white'
              }`}>
                {matchup.home_team.team_name}
              </h4>
              <div className={`text-3xl font-bold ${
                matchup.home_team.total_score > matchup.away_team.total_score 
                  ? 'text-emerald-700 dark:text-emerald-300' 
                  : 'text-gray-900 dark:text-white'
              }`}>
                {matchup.home_team.total_score.toFixed(1)}
              </div>
            </div>
            <div className={`text-center p-4 rounded-lg ${
              matchup.away_team.total_score > matchup.home_team.total_score 
                ? 'bg-emerald-50 dark:bg-emerald-900/20 border-2 border-emerald-200 dark:border-emerald-700' 
                : 'bg-gray-50 dark:bg-gray-700'
            }`}>
              <h4 className={`text-lg font-semibold mb-2 ${
                matchup.away_team.total_score > matchup.home_team.total_score 
                  ? 'text-emerald-800 dark:text-emerald-200' 
                  : 'text-gray-900 dark:text-white'
              }`}>
                {matchup.away_team.team_name}
              </h4>
              <div className={`text-3xl font-bold ${
                matchup.away_team.total_score > matchup.home_team.total_score 
                  ? 'text-emerald-700 dark:text-emerald-300' 
                  : 'text-gray-900 dark:text-white'
              }`}>
                {matchup.away_team.total_score.toFixed(1)}
              </div>
            </div>
          </div>

          {/* Lineup Table */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gradient-to-r from-slate-800 to-slate-700">
                  <th className="text-left py-3 px-4 text-sm font-semibold text-white uppercase tracking-wider">
                    {matchup.home_team.team_name}
                  </th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-white uppercase tracking-wider">
                    Points
                  </th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-white uppercase tracking-wider">
                    Pos
                  </th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-white uppercase tracking-wider">
                    Points
                  </th>
                  <th className="text-right py-3 px-4 text-sm font-semibold text-white uppercase tracking-wider">
                    {matchup.away_team.team_name}
                  </th>
                </tr>
              </thead>
              <tbody>
                {positionOrder.map((position, index) => {
                  const homePlayer = homeStarters[index];
                  const awayPlayer = awayStarters[index];
                  
                  const comparison = homePlayer && awayPlayer ? getPlayerComparison(homePlayer, awayPlayer) : null;
                  
                  return (
                    <tr key={`${position}-${index}`} className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-3 px-4 text-gray-900 dark:text-white">
                        <div className="flex items-center gap-2">
                          <span 
                            className={homePlayer && onPlayerClick ? 'cursor-pointer hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors' : ''}
                            onClick={(e) => homePlayer && onPlayerClick?.(homePlayer.player_name, year, e)}
                          >
                            {homePlayer ? homePlayer.player_name : '—'}
                          </span>
                          {comparison && comparison.homeWins && (
                            <span className="text-emerald-600 dark:text-emerald-400 text-sm">✓</span>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-4 text-center font-semibold text-gray-900 dark:text-white">
                        {homePlayer ? (
                          <span>
                            {homePlayer.points.toFixed(1)}
                            <sub className="text-xs text-gray-500 dark:text-gray-400 ml-1">
                              {homePlayer.projected_points.toFixed(1)}
                            </sub>
                          </span>
                        ) : '—'}
                      </td>
                      <td className="py-3 px-4 text-center font-medium text-gray-700 dark:text-gray-300">
                        {position}
                      </td>
                      <td className="py-3 px-4 text-center font-semibold text-gray-900 dark:text-white">
                        {awayPlayer ? (
                          <span>
                            {awayPlayer.points.toFixed(1)}
                            <sub className="text-xs text-gray-500 dark:text-gray-400 ml-1">
                              {awayPlayer.projected_points.toFixed(1)}
                            </sub>
                          </span>
                        ) : '—'}
                      </td>
                      <td className="py-3 px-4 text-right text-gray-900 dark:text-white">
                        <div className="flex items-center justify-end gap-2">
                          <span 
                            className={awayPlayer && onPlayerClick ? 'cursor-pointer hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors' : ''}
                            onClick={(e) => awayPlayer && onPlayerClick?.(awayPlayer.player_name, year, e)}
                          >
                            {awayPlayer ? awayPlayer.player_name : '—'}
                          </span>
                          {comparison && comparison.awayWins && (
                            <span className="text-emerald-600 dark:text-emerald-400 text-sm">✓</span>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ScoreboardPage() {
  const [selectedYear, setSelectedYear] = useState(2024);
  const [scoreboard, setScoreboard] = useState<ScoreboardResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMatchup, setSelectedMatchup] = useState<MatchupDetail | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [loadingMatchup, setLoadingMatchup] = useState(false);
  const [selectedPlayers, setSelectedPlayers] = useState<Map<string, SelectedPlayerData>>(new Map());

  const years = [2020, 2021, 2022, 2024, 2025];

  const hasPlayerData = (data: PlayerWeeklyStatsResponse): boolean => {
    return data.weekly_stats.some(stat => stat.weekly_points_ppr !== null || stat.z_week_ppr !== null);
  };

  const handlePlayerClick = async (playerName: string, year: number, event?: React.MouseEvent<HTMLSpanElement>) => {
    const playerKey = `${playerName}_${year}`;
    
    if (selectedPlayers.has(playerKey)) {
      return;
    }
    
    // Always center pop-ups, stack vertically when multiple
    const existingPopups = Array.from(selectedPlayers.values());
    
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      newMap.set(playerKey, { playerName, year, selectedYear: year, availableYears: [year], position: { x: 0, y: 0 }, loading: true });
      return newMap;
    });
    
    try {
      // Fetch stats and headshot in parallel
      const [statsResponse, headshotResponse] = await Promise.all([
        fetch(`${API_BASE}/players/${encodeURIComponent(playerName)}/weekly-stats?year=${year}`),
        fetch(`${API_BASE}/players/${encodeURIComponent(playerName)}/headshot`)
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
      
      const availableYears: number[] = [];
      const yearChecks = YEARS.map(async (checkYear) => {
        try {
          const checkResponse = await fetch(`${API_BASE}/players/${encodeURIComponent(playerName)}/weekly-stats?year=${checkYear}`);
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
      availableYears.sort();
      
      setSelectedPlayers(prev => {
        const newMap = new Map(prev);
        const existing = newMap.get(playerKey);
        if (existing) {
          newMap.set(playerKey, { 
            ...existing, 
            stats: data, 
            headshotUrl: headshotUrl,
            availableYears: availableYears.length > 0 ? availableYears : [year],
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

    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      const existing = newMap.get(playerKey);
      if (existing) {
        newMap.set(playerKey, { ...existing, selectedYear: newYear, loading: true });
      }
      return newMap;
    });

    try {
      const response = await fetch(`${API_BASE}/players/${encodeURIComponent(playerData.playerName)}/weekly-stats?year=${newYear}`);
      if (!response.ok) {
        throw new Error('Failed to fetch player stats');
      }
      const data: PlayerWeeklyStatsResponse = await response.json();
      
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

  useEffect(() => {
    loadScoreboard();
  }, [selectedYear]);

  const loadScoreboard = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchScoreboard(selectedYear);
      setScoreboard(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load scoreboard');
    } finally {
      setLoading(false);
    }
  };

  const handleGameClick = async (team: TeamScoreboard, game: GameResult) => {
    setLoadingMatchup(true);
    setIsModalOpen(true);
    try {
      const matchup = await fetchMatchupDetail(selectedYear, game.week, team.team_name, game.opponent);
      setSelectedMatchup(matchup);
    } catch (err) {
      console.error('Failed to load matchup details:', err);
      // Fallback to simple game detail
      const gameDetail: GameDetail = {
        week: game.week,
        team1: team.team_name,
        team2: game.opponent,
        score1: game.score,
        score2: game.opponent_score,
        winner: game.result === "W" ? team.team_name : game.opponent,
        margin: game.margin
      };
      setSelectedMatchup({
        year: selectedYear,
        week: game.week,
        home_team: {
          team_name: team.team_name,
          total_score: game.score,
          players: []
        },
        away_team: {
          team_name: game.opponent,
          total_score: game.opponent_score,
          players: []
        },
        is_playoff: false
      });
    } finally {
      setLoadingMatchup(false);
    }
  };

  const getMaxWeek = () => {
    if (!scoreboard) return 0;
    return Math.max(...scoreboard.teams.flatMap(team => team.games.map(game => game.week)));
  };

  const getGameForWeek = (team: TeamScoreboard, week: number) => {
    return team.games.find(game => game.week === week) || null;
  };

  const getRegularSeasonWeeks = () => {
    if (!scoreboard) return [];
    const allWeeks = scoreboard.teams.flatMap(team => 
      team.games.filter(game => !game.is_playoff).map(game => game.week)
    );
    return [...new Set(allWeeks)].sort((a, b) => a - b);
  };

  const getPlayoffWeeks = () => {
    if (!scoreboard) return [];
    const allWeeks = scoreboard.teams.flatMap(team => 
      team.games.filter(game => game.is_playoff).map(game => game.week)
    );
    return [...new Set(allWeeks)].sort((a, b) => a - b);
  };

  const isTeamEliminated = (team: TeamScoreboard, week: number) => {
    if (!scoreboard) return false;
    
    // Only check elimination for playoff weeks (week 15+)
    if (week < 15) return false;
    
    // Get all playoff games for this team up to and including the current week
    const playoffGames = team.games.filter(game => game.is_playoff && game.week <= week);
    
    // Debug logging for week 15
    // if (week === 15) {
    //   console.log(`\n=== ${team.team_name} Week 15 Elimination Check ===`);
    //   playoffGames.forEach(game => {
    //     console.log(`Week ${game.week}: vs ${game.opponent} - ${game.result} - Type: ${game.matchup_type} - Playoff: ${game.is_playoff}`);
    //   });
    // }
    
    // Check if team is in consolation bracket (any consolation game means they're eliminated)
    const hasConsolationGame = playoffGames.some(game => 
      game.matchup_type === 'LOSERS_CONSOLATION_LADDER' || 
      game.matchup_type === 'WINNERS_CONSOLATION_LADDER'
    );
    
    // Check if team has lost in main playoff bracket (before consolation)
    const hasPlayoffLoss = playoffGames.some(game => 
      game.result === "L" && 
      (game.matchup_type === 'WINNERS_BRACKET' || game.matchup_type === 'NONE')
    );
    
    return hasConsolationGame || hasPlayoffLoss;
  };

  const isChampionshipWinner = (team: TeamScoreboard, week: number) => {
    if (!scoreboard) return false;
    
    // Determine the championship week based on year
    const championshipWeek = selectedYear === 2020 ? 16 : 17;
    
    if (week !== championshipWeek) return false;
    
    // Check if this team won the championship game (winners bracket)
    const championshipGame = team.games.find(game => 
      game.week === championshipWeek && 
      game.is_playoff && 
      game.result === "W" &&
      (game.matchup_type === 'WINNERS_BRACKET' || game.matchup_type === 'NONE')
    );
    
    return !!championshipGame;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading scoreboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400 mb-4">{error}</p>
          <button
            onClick={loadScoreboard}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!scoreboard) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 dark:text-gray-400">No scoreboard data available</p>
        </div>
      </div>
    );
  }

  const maxWeek = getMaxWeek();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Scoreboard
          </h1>
          
          {/* Navigation */}
          <nav className="flex justify-center items-center gap-3 mb-6 flex-wrap">
            <Link href="/" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Home
            </Link>
            <Link href="/players" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Players
            </Link>
            <Link href="/trades" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Trades
            </Link>
            <Link href="/waivers" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Waivers
            </Link>
            <Link href="/scoreboard" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Scoreboard
            </Link>
            <Link href="/draft" className="px-5 py-2.5 rounded-lg bg-white/90 text-slate-700 text-sm font-semibold hover:bg-slate-100 active:bg-slate-200 transition-all duration-150 shadow-sm hover:shadow active:shadow-none active:scale-[0.98]">
              Draft
            </Link>
          </nav>
          
          {/* Year Selector */}
          <div className="flex justify-center gap-2 mb-6">
            {years.map(year => (
              <button
                key={year}
                onClick={() => setSelectedYear(year)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedYear === year
                    ? 'bg-indigo-600 text-white'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                {year}
              </button>
            ))}
          </div>
        </div>

        {/* Scoreboard Table */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gradient-to-r from-slate-800 to-slate-700">
                  <th className="sticky left-0 z-10 bg-gradient-to-r from-slate-800 to-slate-700 px-4 py-3 text-left text-sm font-semibold text-white uppercase tracking-wider">
                    Team
                  </th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-white uppercase tracking-wider">
                    Record
                  </th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-white uppercase tracking-wider">
                    Points
                  </th>
                  {/* Regular Season Weeks */}
                  {getRegularSeasonWeeks().map(week => (
                    <th key={`reg-${week}`} className="px-2 py-3 text-center text-sm font-semibold text-white uppercase tracking-wider min-w-[60px]">
                      W{week}
                    </th>
                  ))}
                  {/* Separator header */}
                  <th className="px-1 py-3 text-center relative bg-gradient-to-r from-slate-800 to-slate-700">
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-300 dark:bg-gray-600 transform -translate-x-1/2"></div>
                  </th>
                  {/* Playoff Weeks */}
                  {getPlayoffWeeks().map(week => (
                    <th key={`playoff-${week}`} className="px-2 py-3 text-center text-sm font-semibold text-white uppercase tracking-wider min-w-[60px]">
                      W{week}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {scoreboard.teams.map((team, index) => (
                  <tr key={team.team_name} className="hover:bg-gray-50 dark:hover:bg-gray-700 group">
                    <td className="sticky left-0 z-10 bg-white dark:bg-gray-800 group-hover:bg-gray-50 dark:group-hover:bg-gray-700 px-4 py-3 border-r border-gray-200 dark:border-gray-700">
                      <div className="flex items-center">
                        <span 
                          className={`text-sm font-bold ${
                            isChampionshipWinner(team, getMaxWeek()) 
                              ? 'bg-gradient-to-r from-yellow-200 to-yellow-400 bg-clip-text text-transparent animate-pulse' 
                              : 'text-gray-900 dark:text-white'
                          }`}
                          style={isChampionshipWinner(team, getMaxWeek()) ? {
                            textShadow: '0 0 8px #fbbf24, 0 0 16px #f59e0b',
                            filter: 'drop-shadow(0 0 24px #fbbf24)'
                          } : {}}
                        >
                          {team.team_name}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className="text-sm text-gray-900 dark:text-white">
                        {team.wins}-{team.losses}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className="text-sm text-gray-900 dark:text-white">
                        {team.total_points.toFixed(1)}
                      </span>
                    </td>
                    {/* Regular Season Games */}
                    {getRegularSeasonWeeks().map(week => {
                      const game = getGameForWeek(team, week);
                      const isTopScoringWeek = scoreboard?.top_scoring_week && 
                                               scoreboard.top_scoring_week.team_name === team.team_name &&
                                               scoreboard.top_scoring_week.week === week;
                      return (
                        <td key={`reg-${week}`} className="px-2 py-3 text-center">
                          <GameBubble
                            game={game}
                            week={week}
                            onClick={() => game && handleGameClick(team, game)}
                            isTopScoringWeek={isTopScoringWeek}
                          />
                        </td>
                      );
                    })}
                    {/* Separator between regular season and playoffs */}
                    <td className="px-1 py-3 text-center relative">
                      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-300 dark:bg-gray-600 transform -translate-x-1/2"></div>
                    </td>
                    {/* Playoff Games */}
                    {getPlayoffWeeks().map(week => {
                      const game = getGameForWeek(team, week);
                      const isEliminated = isTeamEliminated(team, week);
                      const isChampionship = isChampionshipWinner(team, week);
                      const isTopScoringWeek = scoreboard?.top_scoring_week && 
                                               scoreboard.top_scoring_week.team_name === team.team_name &&
                                               scoreboard.top_scoring_week.week === week;
                      return (
                        <td key={`playoff-${week}`} className="px-2 py-3 text-center">
                          <GameBubble
                            game={game}
                            week={week}
                            onClick={() => game && handleGameClick(team, game)}
                            isEliminated={isEliminated}
                            isChampionship={isChampionship}
                            isTopScoringWeek={isTopScoringWeek}
                          />
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Legend */}
        <div className="mt-6 flex justify-center">
          <div className="flex items-center gap-6 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-emerald-600"></div>
              <span>Win</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-violet-600"></div>
              <span>Loss</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-emerald-300 opacity-60"></div>
              <span>Eliminated Team Win</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-violet-300 opacity-60"></div>
              <span>Eliminated Team Loss</span>
            </div>
            {/* <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-gray-200 dark:bg-gray-700"></div>
              <span>No Game</span>
            </div> */}
          </div>
        </div>
      </div>

      {/* Game Detail Modal */}
      <GameDetailModal
        matchup={selectedMatchup}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        loading={loadingMatchup}
        onPlayerClick={handlePlayerClick}
        year={selectedYear}
      />

      {/* Player Weekly Stats Popups */}
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
            <div className="flex items-center justify-between mb-4 relative">
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
  );
}
