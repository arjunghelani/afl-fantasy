'use client';

import { useState, useEffect } from 'react';

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

type ScoreboardResponse = {
  year: number;
  teams: TeamScoreboard[];
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

async function fetchScoreboard(year: number): Promise<ScoreboardResponse> {
  const base = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
  const res = await fetch(`${base}/scoreboard/${year}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch scoreboard for ${year}`);
  return res.json();
}

async function fetchMatchupDetail(year: number, week: number, team1: string, team2: string): Promise<MatchupDetail> {
  const base = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
  const res = await fetch(`${base}/matchup/${year}/${week}?team1=${encodeURIComponent(team1)}&team2=${encodeURIComponent(team2)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch matchup details for ${team1} vs ${team2}`);
  return res.json();
}

function GameBubble({ 
  game, 
  week, 
  onClick,
  isEliminated = false
}: { 
  game: GameResult | null; 
  week: number; 
  onClick: () => void;
  isEliminated?: boolean;
}) {
  if (!game) {
    return (
      <div 
        className="w-12 h-12 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center cursor-pointer hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
        onClick={onClick}
      >
        <span className="text-xs text-gray-500 dark:text-gray-400">—</span>
      </div>
    );
  }

  const isWin = game.result === "W";
  const isPlayoff = game.is_playoff;
  
  return (
    <div 
      className={`w-12 h-12 rounded-full flex items-center justify-center cursor-pointer transition-all hover:scale-110 ${
        isPlayoff && isEliminated
          ? (isWin 
              ? 'bg-green-300 hover:bg-green-400 text-white shadow-lg opacity-60' 
              : 'bg-red-300 hover:bg-red-400 text-white shadow-lg opacity-60')
          : isPlayoff
          ? (isWin 
              ? 'bg-green-500 hover:bg-green-600 text-white shadow-lg' 
              : 'bg-red-500 hover:bg-red-600 text-white shadow-lg')
          : (isWin 
              ? 'bg-green-500 hover:bg-green-600 text-white shadow-lg' 
              : 'bg-red-500 hover:bg-red-600 text-white shadow-lg')
      }`}
      onClick={onClick}
    >
      <span className="text-xs font-semibold">{game.result}</span>
    </div>
  );
}

function GameDetailModal({ 
  matchup, 
  isOpen, 
  onClose,
  loading = false
}: { 
  matchup: MatchupDetail | null; 
  isOpen: boolean; 
  onClose: () => void;
  loading?: boolean;
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
    console.log(`\n=== ${team.team_name} Players ===`);
    team.players.forEach((player, index) => {
      console.log(`${index}: ${player.player_name} - Position: "${player.position}" - Points: ${player.points}`);
    });
    
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
    
    // Fill remaining slots with any available players
    const remainingPlayers = team.players.filter(p => !usedPlayers.has(p.player_name));
    while (starters.length < 9 && remainingPlayers.length > 0) {
      const player = remainingPlayers.shift();
      if (player) {
        starters.push(player);
        usedPlayers.add(player.player_name);
      }
    }
    
    console.log(`\n=== ${team.team_name} Starters ===`);
    starters.forEach((starter, index) => {
      console.log(`${index}: ${starter.player_name} - Position: "${starter.position}" - Points: ${starter.points}`);
    });
    
    return starters;
  };

  const homeStarters = getStarters(matchup.home_team);
  const awayStarters = getStarters(matchup.away_team);

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
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {matchup.home_team.team_name}
              </h4>
              <div className="text-3xl font-bold text-gray-900 dark:text-white">
                {matchup.home_team.total_score.toFixed(1)}
              </div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {matchup.away_team.team_name}
              </h4>
              <div className="text-3xl font-bold text-gray-900 dark:text-white">
                {matchup.away_team.total_score.toFixed(1)}
              </div>
            </div>
          </div>

          {/* Lineup Table */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-white">
                    Position
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-white">
                    {matchup.home_team.team_name}
                  </th>
                  <th className="text-center py-3 px-4 font-medium text-gray-900 dark:text-white">
                    Points
                  </th>
                  <th className="text-center py-3 px-4 font-medium text-gray-900 dark:text-white">
                    Proj
                  </th>
                  <th className="text-center py-3 px-4 font-medium text-gray-900 dark:text-white">
                    Proj
                  </th>
                  <th className="text-center py-3 px-4 font-medium text-gray-900 dark:text-white">
                    Points
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-900 dark:text-white">
                    {matchup.away_team.team_name}
                  </th>
                </tr>
              </thead>
              <tbody>
                {positionOrder.map((position, index) => {
                  const homePlayer = homeStarters[index];
                  const awayPlayer = awayStarters[index];
                  
                  return (
                    <tr key={position} className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-3 px-4 font-medium text-gray-700 dark:text-gray-300">
                        {position}
                      </td>
                      <td className="py-3 px-4 text-gray-900 dark:text-white">
                        {homePlayer ? homePlayer.player_name : '—'}
                      </td>
                      <td className="py-3 px-4 text-center font-semibold text-gray-900 dark:text-white">
                        {homePlayer ? homePlayer.points.toFixed(1) : '—'}
                      </td>
                      <td className="py-3 px-4 text-center text-sm text-gray-500 dark:text-gray-400">
                        {homePlayer ? homePlayer.projected_points.toFixed(1) : '—'}
                      </td>
                      <td className="py-3 px-4 text-center text-sm text-gray-500 dark:text-gray-400">
                        {awayPlayer ? awayPlayer.projected_points.toFixed(1) : '—'}
                      </td>
                      <td className="py-3 px-4 text-center font-semibold text-gray-900 dark:text-white">
                        {awayPlayer ? awayPlayer.points.toFixed(1) : '—'}
                      </td>
                      <td className="py-3 px-4 text-right text-gray-900 dark:text-white">
                        {awayPlayer ? awayPlayer.player_name : '—'}
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

  const years = [2020, 2021, 2022, 2024];

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
    if (week === 15) {
      console.log(`\n=== ${team.team_name} Week 15 Elimination Check ===`);
      playoffGames.forEach(game => {
        console.log(`Week ${game.week}: vs ${game.opponent} - ${game.result} - Type: ${game.matchup_type} - Playoff: ${game.is_playoff}`);
      });
    }
    
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
    
    const isEliminated = hasConsolationGame || hasPlayoffLoss;
    
    if (week === 15) {
      console.log(`Has consolation game: ${hasConsolationGame}`);
      console.log(`Has playoff loss: ${hasPlayoffLoss}`);
      console.log(`Is eliminated: ${isEliminated}`);
    }
    
    return isEliminated;
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
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-900 dark:text-white">
                    Team
                  </th>
                  <th className="px-4 py-3 text-center text-sm font-medium text-gray-900 dark:text-white">
                    Record
                  </th>
                  <th className="px-4 py-3 text-center text-sm font-medium text-gray-900 dark:text-white">
                    Points
                  </th>
                  {/* Regular Season Weeks */}
                  {getRegularSeasonWeeks().map(week => (
                    <th key={`reg-${week}`} className="px-2 py-3 text-center text-sm font-medium text-gray-900 dark:text-white min-w-[60px]">
                      W{week}
                    </th>
                  ))}
                  {/* Playoff Weeks */}
                  {getPlayoffWeeks().map(week => (
                    <th key={`playoff-${week}`} className="px-2 py-3 text-center text-sm font-medium text-gray-900 dark:text-white min-w-[60px]">
                      W{week}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {scoreboard.teams.map((team, index) => (
                  <tr key={team.team_name} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                    <td className="px-4 py-3">
                      <div className="flex items-center">
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
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
                      return (
                        <td key={`reg-${week}`} className="px-2 py-3 text-center">
                          <GameBubble
                            game={game}
                            week={week}
                            onClick={() => game && handleGameClick(team, game)}
                          />
                        </td>
                      );
                    })}
                    {/* Playoff Games */}
                    {getPlayoffWeeks().map(week => {
                      const game = getGameForWeek(team, week);
                      const isEliminated = isTeamEliminated(team, week);
                      return (
                        <td key={`playoff-${week}`} className="px-2 py-3 text-center">
                          <GameBubble
                            game={game}
                            week={week}
                            onClick={() => game && handleGameClick(team, game)}
                            isEliminated={isEliminated}
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
              <div className="w-4 h-4 rounded-full bg-green-500"></div>
              <span>Win</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500"></div>
              <span>Loss</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-300 opacity-60"></div>
              <span>Eliminated Team Win</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-300 opacity-60"></div>
              <span>Eliminated Team Loss</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-gray-200 dark:bg-gray-700"></div>
              <span>No Game</span>
            </div>
          </div>
        </div>
      </div>

      {/* Game Detail Modal */}
      <GameDetailModal
        matchup={selectedMatchup}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        loading={loadingMatchup}
      />
    </div>
  );
}
