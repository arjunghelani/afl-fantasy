'use client';

import { useState, useEffect } from 'react';

interface TradePackage {
  week: number;
  team1_id: number;
  team1_name: string;
  team2_id: number;
  team2_name: string;
  team1_players: string[];
  team2_players: string[];
  total_players: number;
  is_trade_like: boolean;
}

interface TradeSummary {
  trade_week: number;
  team_a: string;
  team_b: string;
  team_a_vorp_received: number;
  team_b_vorp_received: number;
  net_advantage: number;
  winner: string;
  players_to_a: string;
  players_to_b: string;
}

interface TradesResponse {
  year: number;
  trade_packages: TradePackage[];
  trade_summary: TradeSummary[];
  count: number;
}

const SUPPORTED_YEARS = [2020, 2021, 2022, 2024];
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

const MiniTradeCard = ({ trade, year, showYear = false }: { trade: TradePackage; year: number; showYear?: boolean }) => {
  return (
    <div className="flex-shrink-0 w-80 h-40 bg-gradient-to-br from-white to-gray-50 dark:from-gray-900 dark:to-gray-800 rounded-xl border border-gray-200/60 dark:border-gray-700/60 p-3 hover:shadow-xl hover:shadow-blue-100/30 dark:hover:shadow-blue-900/20 hover:scale-[1.02] transition-all duration-300">
      {/* Team Names Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex-1 text-center">
        <h3 className="text-xs font-bold text-orange-600 dark:text-orange-400 leading-tight break-words hyphens-auto max-w-full overflow-hidden" style={{display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical'}}>
          {trade.team1_name}
        </h3>
          <div className="w-full h-0.5 bg-gradient-to-r from-orange-500 to-amber-400 rounded-full mt-1"></div>
        </div>
        
        {/* Week/Year Circle */}
        <div className="mx-4 flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex flex-col items-center justify-center shadow-xl ring-1 ring-indigo-300/50 dark:ring-indigo-400/50">
            <div className="text-[11px] font-bold text-white leading-tight">W{trade.week}</div>
            {showYear && <div className="text-[8px] font-bold text-indigo-100 leading-tight">{year}</div>}
          </div>
        </div>
        
        <div className="flex-1 text-center">
        <h3 className="text-xs font-bold text-teal-600 dark:text-teal-400 leading-tight break-words hyphens-auto max-w-full overflow-hidden" style={{display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical'}}>
          {trade.team2_name}
        </h3>
          <div className="w-full h-0.5 bg-gradient-to-r from-teal-500 to-cyan-400 rounded-full mt-1"></div>
        </div>
      </div>

      {/* Players Section */}
      <div className="flex gap-4 -mt-2">
        {/* Team 1 Players */}
        <div className="flex-1">
          <div className="space-y-0">
            {trade.team1_players.length > 0 ? (
              trade.team1_players.map((player, index) => (
                <div key={index} className="text-center border-b border-gray-300/30 dark:border-gray-600/30 last:border-b-0">
                  <div className="text-sm font-medium text-white break-words hyphens-auto max-w-full overflow-hidden py-1" style={{display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical'}}>
                    {player}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-xs text-gray-400 dark:text-gray-500 italic text-center py-2">
                No players
              </div>
            )}
          </div>
        </div>

        {/* Trade Arrow */}
        <div className="flex items-center justify-center px-1">
          <div className="w-6 h-6 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-xl ring-1 ring-indigo-300/50 dark:ring-indigo-400/50">
            <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
            </svg>
          </div>
        </div>

        {/* Team 2 Players */}
        <div className="flex-1">
          <div className="space-y-0">
            {trade.team2_players.length > 0 ? (
              trade.team2_players.map((player, index) => (
                <div key={index} className="text-center border-b border-gray-300/30 dark:border-gray-600/30 last:border-b-0">
                  <div className="text-sm font-medium text-white break-words hyphens-auto max-w-full overflow-hidden py-1" style={{display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical'}}>
                    {player}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-xs text-gray-400 dark:text-gray-500 italic text-center py-2">
                No players
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

type ViewMode = 'year' | 'team';

export default function TradesPage() {
  const [viewMode, setViewMode] = useState<ViewMode>('year');
  const [selectedYear, setSelectedYear] = useState(2024);
  const [selectedTeam, setSelectedTeam] = useState<string>('all');
  const [data, setData] = useState<TradesResponse | null>(null);
  const [allYearsData, setAllYearsData] = useState<{[year: number]: TradesResponse}>({});
  const [loading, setLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState<{current: number, total: number} | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedTeams, setExpandedTeams] = useState<Set<number>>(new Set());

  const toggleTeamExpansion = (teamId: number) => {
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

  const fetchTrades = async (year: number) => {
    setLoading(true);
    setError(null);
    try {
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      const response = await fetch(`${API_BASE}/trades/${year}`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch trades: ${response.statusText}`);
      }
      const tradesData: TradesResponse = await response.json();
      setData(tradesData);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Request timed out. Please try again.');
      } else {
        setError(err instanceof Error ? err.message : 'Failed to fetch trades');
      }
    } finally {
      setLoading(false);
    }
  };

  const fetchAllYearsData = async () => {
    setLoading(true);
    setLoadingProgress({current: 0, total: SUPPORTED_YEARS.length});
    setError(null);
    try {
      // Fetch years sequentially to avoid overwhelming the server
      const yearsData: {[year: number]: TradesResponse} = {};
      
      for (let i = 0; i < SUPPORTED_YEARS.length; i++) {
        const year = SUPPORTED_YEARS[i];
        setLoadingProgress({current: i + 1, total: SUPPORTED_YEARS.length});
        
        try {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 45000); // Longer timeout for team view
          
          const response = await fetch(`${API_BASE}/trades/${year}`, {
            signal: controller.signal
          });
          
          clearTimeout(timeoutId);
          
          if (!response.ok) {
            console.warn(`Failed to fetch trades for ${year}: ${response.statusText}`);
            continue; // Skip this year but continue with others
          }
          
          const data = await response.json() as TradesResponse;
          yearsData[year] = data;
        } catch (err) {
          console.warn(`Error fetching trades for ${year}:`, err);
          // Continue with other years even if one fails
        }
      }
      
      setAllYearsData(yearsData);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Request timed out. Please try again.');
      } else {
        setError(err instanceof Error ? err.message : 'Failed to fetch trades');
      }
    } finally {
      setLoading(false);
      setLoadingProgress(null);
    }
  };

  // Get all unique teams from all years data, grouped by team ID
  const getAllTeams = () => {
    const teamMap = new Map<number, {id: number, name: string}>();
    
    Object.values(allYearsData).forEach(yearData => {
      yearData.trade_packages.forEach(trade => {
        // Group by team ID but use the latest team name
        teamMap.set(trade.team1_id, {id: trade.team1_id, name: trade.team1_name});
        teamMap.set(trade.team2_id, {id: trade.team2_id, name: trade.team2_name});
      });
    });
    
    return Array.from(teamMap.values()).sort((a, b) => a.name.localeCompare(b.name));
  };

  useEffect(() => {
    if (viewMode === 'year') {
      fetchAllYearsData();
    } else if (viewMode === 'team') {
      fetchAllYearsData();
    }
  }, [viewMode]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-950 dark:to-blue-950">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Trades
          </h1>
          
          {/* View Mode Selector */}
          <div className="flex justify-center gap-2 mb-6">
            <button
              onClick={() => setViewMode('year')}
                    className={`px-6 py-2 rounded-lg font-semibold transition-all ${
                      viewMode === 'year'
                        ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                    }`}
            >
              By Year
            </button>
            <button
              onClick={() => setViewMode('team')}
                    className={`px-6 py-2 rounded-lg font-semibold transition-all ${
                      viewMode === 'team'
                        ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                    }`}
            >
              By Team
            </button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex justify-center items-center gap-5 mb-8">
                <a href="/" className="text-sm md:text-base text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white underline-offset-4 hover:underline">
            Home
          </a>
          <a href="/standings" className="text-sm md:text-base text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white underline-offset-4 hover:underline">
            Standings
          </a>
          <a href="/players" className="text-sm md:text-base text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white underline-offset-4 hover:underline">
            Players
          </a>
          <a href="/scoreboard" className="text-sm md:text-base text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white underline-offset-4 hover:underline">
            Scoreboard
          </a>
          <a href="/draft" className="text-sm md:text-base text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white underline-offset-4 hover:underline">
            Draft
          </a>
          <a href="/playoffs" className="text-sm md:text-base text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white underline-offset-4 hover:underline">
            Playoffs
          </a>
        </nav>



        {/* Loading */}
        {loading && (
          <div className="flex justify-center items-center py-20">
            <div className="flex flex-col items-center gap-4">
                     <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500"></div>
              <div className="text-center">
                <div className="text-gray-600 dark:text-gray-400 text-lg font-medium">
                  {viewMode === 'team' ? 'Loading all years...' : 'Loading trades...'}
                </div>
                <div className="text-gray-500 dark:text-gray-500 text-sm mt-1">
                  {viewMode === 'team' 
                    ? 'Fetching trade data from all years - this may take a moment'
                    : 'This may take a moment as we analyze trade data'
                  }
                </div>
                {loadingProgress && (
                  <div className="mt-3">
                           <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                      {loadingProgress.current} of {loadingProgress.total} years loaded
                    </div>
                    <div className="w-48 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${(loadingProgress.current / loadingProgress.total) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
            <p className="text-red-600 dark:text-red-400 font-medium">Error: {error}</p>
          </div>
        )}

        {/* Trade Cards */}
        {!loading && !error && (
          <>
            {/* By Year View */}
            {viewMode === 'year' && Object.keys(allYearsData).length > 0 && (
              <div className="space-y-8">
                {SUPPORTED_YEARS.map((year) => {
                  const yearData = allYearsData[year];
                  if (!yearData || yearData.trade_packages.length === 0) return null;

                  return (
                    <div key={year} className="space-y-4">
                      <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                        {year}
                      </h3>
                      <div className="overflow-x-auto pb-4">
                        <div className="flex gap-4 min-w-max">
                          {yearData.trade_packages
                            .sort((a, b) => a.week - b.week)
                            .map((trade, index) => (
                              <MiniTradeCard key={index} trade={trade} year={year} showYear={false} />
                            ))}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* By Team View */}
            {viewMode === 'team' && Object.keys(allYearsData).length > 0 && (
              <div className="space-y-8">
                {getAllTeams().map((team) => {
                  const teamTrades = Object.entries(allYearsData)
                    .flatMap(([year, yearData]) =>
                      yearData.trade_packages
                        .filter(trade => trade.team1_id === team.id || trade.team2_id === team.id)
                        .map(trade => {
                          // Reorder teams so the current team is always on the left
                          const isTeam1 = trade.team1_id === team.id;
                          return {
                            ...trade,
                            year: parseInt(year),
                            // If current team is team2, swap the teams
                            team1_id: isTeam1 ? trade.team1_id : trade.team2_id,
                            team1_name: isTeam1 ? trade.team1_name : trade.team2_name,
                            team1_players: isTeam1 ? trade.team1_players : trade.team2_players,
                            team2_id: isTeam1 ? trade.team2_id : trade.team1_id,
                            team2_name: isTeam1 ? trade.team2_name : trade.team1_name,
                            team2_players: isTeam1 ? trade.team2_players : trade.team1_players,
                          };
                        })
                    )
                    .sort((a, b) => a.year - b.year || a.week - b.week);

                  if (teamTrades.length === 0) return null;

                  const isExpanded = expandedTeams.has(team.id);

                  return (
                    <div key={team.id} className="space-y-4">
                      <button
                        onClick={() => toggleTeamExpansion(team.id)}
                        className="flex items-center justify-between w-full text-left hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg p-2 transition-colors"
                      >
                        <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                          {team.name}
                        </h3>
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            {teamTrades.length} trade{teamTrades.length !== 1 ? 's' : ''}
                          </span>
                          <svg 
                            className={`w-5 h-5 text-gray-500 dark:text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                            fill="none" 
                            stroke="currentColor" 
                            viewBox="0 0 24 24"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        </div>
                      </button>
                      {isExpanded && (
                        <div className="overflow-x-auto pb-4">
                          <div className="flex gap-4 min-w-max">
                            {teamTrades.map((trade, index) => (
                              <MiniTradeCard key={index} trade={trade} year={trade.year} showYear={true} />
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}

        {/* Empty State */}
        {!loading && !error && (
          <>
            {viewMode === 'year' && Object.keys(allYearsData).length === 0 && (
              <div className="text-center py-20">
                <div className="text-gray-500 dark:text-gray-400 text-lg mb-2">No trades found</div>
                <div className="text-gray-400 dark:text-gray-500">No trades were made in any year</div>
              </div>
            )}
            {viewMode === 'team' && Object.keys(allYearsData).length === 0 && (
              <div className="text-center py-20">
                <div className="text-gray-500 dark:text-gray-400 text-lg mb-2">No trades found</div>
                <div className="text-gray-400 dark:text-gray-500">No trades were made by any team</div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}