'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface PlayerInTrade {
  player_name: string;
  vorp_star: number | null;
  total_points: number | null;
  fantasy_pos: string | null;
}

interface TradePackage {
  week: number;
  team1_id: number;
  team1_name: string;
  team2_id: number;
  team2_name: string;
  team1_players: PlayerInTrade[];
  team2_players: PlayerInTrade[];
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

const SUPPORTED_YEARS = [2020, 2021, 2022, 2024, 2025];
const NEXT_PUBLIC_API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

// ZAV color scale using RGB - same colors as weekly stats popup
// Define your ZAV cutoff points here:
const ZAV_CUTOFFS = {
  red: -2,        // ZAV < this value
  orange: 2.5,     // ZAV < this value
  yellow: 6,    // ZAV < this value
  yellowGreen: 10, // ZAV < this value
  // ZAV >= yellowGreen uses green
};

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

// Helper function to get VORP color based on value (legacy, not used in cards)
const getVorpColor = (vorp: number | null): string => {
  if (vorp === null) return 'text-gray-400 dark:text-gray-500';
  if (vorp > 8) return 'text-green-500 dark:text-green-400';
  if (vorp > 5) return 'text-yellow-500 dark:text-yellow-400';
  if (vorp > 0) return 'text-orange-500 dark:text-orange-400';
  return 'text-red-500 dark:text-red-400';
};

const MiniTradeCard = ({ trade, year, showYear = false, onPlayerClick }: { trade: TradePackage; year: number; showYear?: boolean; onPlayerClick?: (playerName: string, year: number, tradeWeek: number, event?: React.MouseEvent<HTMLDivElement>) => void }) => {
  // Calculate winner
  const team1Total = trade.team1_players.reduce((sum, player) => sum + (player.vorp_star ?? 0), 0);
  const team2Total = trade.team2_players.reduce((sum, player) => sum + (player.vorp_star ?? 0), 0);
  const totalDiff = team1Total - team2Total;
  
  // Calculate bar: fills from middle (50%) outward based on raw ZAV difference
  // Scale: 1 ZAV difference = 4% of bar (so 12.5 ZAV difference = 50% fill, capped at 45%)
  let barStart = 50;
  let barWidth = 0;
  if (totalDiff !== 0) {
    const absDiff = Math.abs(totalDiff);
    // Scale factor: 4% per ZAV, capped at 45%
    barWidth = Math.min(45, absDiff * 4);
    
    if (totalDiff > 0) {
      // Team1 wins - bar fills from 50% to the right
      barStart = 50;
    } else {
      // Team2 wins - bar fills from 50% to the left
      barStart = 50 - barWidth;
    }
  }
  
  // Determine winner
  const winner = totalDiff > 0 ? 'team1' : totalDiff < 0 ? 'team2' : 'tie';
  
  return (
    <div className="relative flex-shrink-0 w-[28rem] h-[165px] bg-gradient-to-br from-white to-gray-50 dark:from-gray-900 dark:to-gray-800 rounded-xl border border-gray-200/60 dark:border-gray-700/60 p-1.5 pt-4 pb-1.5 hover:shadow-xl hover:shadow-blue-100/30 dark:hover:shadow-blue-900/20 transition-all duration-300 overflow-visible flex flex-col">

      {/* Week/Year Bubble - Positioned on top edge */}
      <div className="absolute -top-[20px] left-1/2 -translate-x-1/2 z-10">
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex flex-col items-center justify-center shadow-xl ring-1 ring-indigo-300/50 dark:ring-indigo-400/50">
          <div className="text-sm font-bold text-white leading-tight">W{trade.week}</div>
          {showYear && <div className="text-[10px] font-bold text-indigo-100 leading-tight">{year}</div>}
        </div>
      </div>


      {/* Two Column Layout */}
      <div className="flex gap-4 items-stretch h-full overflow-hidden">
        {/* Team 2 Side - Swapped to left */}
        <div className="flex-1 flex flex-col items-start">
          <div className="h-8 mb-1 flex items-center justify-center w-full">
            <h3 className="text-sm font-semibold text-white underline whitespace-nowrap" style={{textUnderlineOffset: '4px'}}>
              {trade.team1_name}
            </h3>
          </div>
          <div className="w-full flex-1 flex flex-col min-h-0">
            <div className="flex-1 overflow-y-auto">
              {trade.team2_players.length > 0 ? (
                trade.team2_players.map((player, index) => {
                  const vorp = player.vorp_star ?? 0;
                  const gradient = getZavGradient(vorp);
                  const textColor = getZavTextColor(vorp);
                  
                  return (
                    <div key={index} className="flex items-center justify-start py-1 border-b border-gray-300/30 dark:border-gray-600/30 last:border-b-0 w-full">
                      <div className="flex items-center gap-1.5 pl-1 w-full">
                        <div className="flex items-center gap-1.5 w-[140px]">
                        <div 
                            className="text-sm font-medium text-gray-900 dark:text-white cursor-pointer hover:scale-105 hover:text-blue-500 dark:hover:text-blue-400 transition-all duration-200 truncate" 
                          onClick={(e) => onPlayerClick?.(player.player_name, year, trade.week, e)}
                        >
                          {player.player_name}
                        </div>
                        {player.fantasy_pos && (
                            <div className="text-[10px] text-gray-500 dark:text-gray-400 opacity-70 flex-shrink-0">
                            {player.fantasy_pos}
                          </div>
                        )}
                      </div>
                      <div 
                          className="px-1.5 py-0.5 rounded text-[10px] font-bold ml-0.5 flex-shrink-0"
                        style={{ 
                          background: gradient,
                          color: textColor
                        }}
                      >
                        {vorp.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="text-[10px] text-gray-400 dark:text-gray-500 italic py-1">
                  No players
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Team 1 Side - Swapped to right */}
        <div className="flex-1 flex flex-col items-start">
          <div className="h-8 mb-1 flex items-center justify-center w-full">
            <h3 className="text-sm font-semibold text-white underline whitespace-nowrap" style={{textUnderlineOffset: '4px'}}>
              {trade.team2_name}
            </h3>
          </div>
          <div className="w-full flex-1 flex flex-col min-h-0">
            <div className="flex-1 overflow-y-auto">
              {trade.team1_players.length > 0 ? (
                trade.team1_players.map((player, index) => {
                  const vorp = player.vorp_star ?? 0;
                  const gradient = getZavGradient(vorp);
                  const textColor = getZavTextColor(vorp);
                  return (
                    <div key={index} className="flex items-center justify-start py-1 border-b border-gray-300/30 dark:border-gray-600/30 last:border-b-0 w-full">
                      <div className="flex items-center gap-1.5 pl-1 w-full">
                        <div className="flex items-center gap-1.5 w-[140px]">
                        <div 
                            className="text-sm font-medium text-gray-900 dark:text-white cursor-pointer hover:scale-105 hover:text-blue-500 dark:hover:text-blue-400 transition-all duration-200 truncate" 
                          onClick={(e) => onPlayerClick?.(player.player_name, year, trade.week, e)}
                        >
                          {player.player_name}
                        </div>
                        {player.fantasy_pos && (
                            <div className="text-[10px] text-gray-500 dark:text-gray-400 opacity-70 flex-shrink-0">
                            {player.fantasy_pos}
                          </div>
                        )}
                      </div>
                      <div 
                          className="px-1.5 py-0.5 rounded text-[10px] font-bold ml-0.5 flex-shrink-0"
                        style={{ 
                          background: gradient,
                          color: textColor
                        }}
                      >
                        {vorp.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="text-[10px] text-gray-400 dark:text-gray-500 italic py-1">
                  No players
                </div>
              )}
            </div>
          </div>
        </div>
            </div>
            
      {/* Winner Bar - Floating half on/half off bottom (using draft card style) */}
      <div className="absolute left-2 right-2 z-20" style={{ bottom: -5 }}>
        <div
          className="relative rounded-full bg-gray-200 dark:bg-gray-700 overflow-visible"
          style={{ height: 8 }}
        >
          {/* White line in the middle - always visible */}
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1 h-full bg-white dark:bg-gray-400 z-20 rounded-full" />
          
          {/* Fill bar - only shows if there's a winner */}
          {winner !== 'tie' && barWidth > 0 && (
                    <div 
              className="transition-all duration-300"
                      style={{ 
                position: 'absolute',
                left: `${barStart}%`,
                width: `${barWidth}%`,
                height: '100%',
                background: totalDiff > 0 
                  ? 'linear-gradient(to right, rgb(34, 197, 94), rgb(16, 185, 129), rgb(5, 150, 105))'
                  : 'linear-gradient(to left, rgb(34, 197, 94), rgb(16, 185, 129), rgb(5, 150, 105))',
                clipPath: totalDiff > 0
                  ? 'polygon(0 0, calc(100% - 4px) 0, 100% 50%, calc(100% - 4px) 100%, 0 100%)'
                  : 'polygon(4px 0, 100% 0, 100% 100%, 4px 100%, 0 50%)'
              }}
            />
          )}
        </div>
        
        {/* ZAV difference text - in bottom corner of winning side */}
        {winner !== 'tie' && barWidth > 0 && (
          <div 
            className="absolute bottom-2 z-30"
            style={{
              [winner === 'team1' ? 'right' : 'left']: '8px'
                      }}
                    >
            <span 
              className="text-xs font-bold"
              style={{
                background: 'linear-gradient(to right, rgb(34, 197, 94), rgb(16, 185, 129))',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text'
              }}
            >
              +{Math.abs(totalDiff).toFixed(2)}
            </span>
                    </div>
        )}
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
  const [selectedPlayers, setSelectedPlayers] = useState<Map<string, {playerName: string, year: number, selectedYear: number, tradeWeek: number, availableYears: number[], position?: {x: number, y: number}, stats?: PlayerWeeklyStatsResponse, loading?: boolean, headshotUrl?: string | null}>>(new Map());

  const hasPlayerData = (stats: PlayerWeeklyStatsResponse): boolean => {
    // Check if there's at least one week with actual data
    return stats.weekly_stats.some(stat => 
      stat.z_week_ppr !== null || stat.weekly_points_ppr !== null
    );
  };

  const handlePlayerClick = async (playerName: string, year: number, tradeWeek: number, event?: React.MouseEvent<HTMLDivElement>) => {
    const playerKey = `${playerName}_${year}_${tradeWeek}`;
    
    // Check if this player is already open
    if (selectedPlayers.has(playerKey)) {
      return; // Don't reopen if already open
    }
    
    // Always center pop-ups, stack vertically when multiple
      const existingPopups = Array.from(selectedPlayers.values());
    
    // Add player to map with loading state
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      newMap.set(playerKey, { playerName, year, selectedYear: year, tradeWeek, availableYears: [year], position: { x: 0, y: 0 }, loading: true });
      return newMap;
    });
    
    try {
      // Fetch stats and headshot in parallel
      const [statsResponse, headshotResponse] = await Promise.all([
        fetch(`${NEXT_PUBLIC_API_BASE_URL}/players/${encodeURIComponent(playerName)}/weekly-stats?year=${year}`),
        fetch(`${NEXT_PUBLIC_API_BASE_URL}/players/${encodeURIComponent(playerName)}/headshot`)
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
      const yearChecks = SUPPORTED_YEARS.map(async (checkYear) => {
        try {
          const checkResponse = await fetch(`${NEXT_PUBLIC_API_BASE_URL}/players/${encodeURIComponent(playerName)}/weekly-stats?year=${checkYear}`);
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
      setError('Failed to load player weekly stats');
      
      // Remove on error
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
      const response = await fetch(`${NEXT_PUBLIC_API_BASE_URL}/players/${encodeURIComponent(playerData.playerName)}/weekly-stats?year=${newYear}`);
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
      setError('Failed to load player weekly stats');
      
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
      
      const response = await fetch(`${NEXT_PUBLIC_API_BASE_URL}/trades/${year}`, {
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
          
          const response = await fetch(`${NEXT_PUBLIC_API_BASE_URL}/trades/${year}`, {
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
      <div className="mx-auto p-6">
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
        <nav className="flex justify-center items-center gap-3 mb-8 flex-wrap">
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
                      <div className="overflow-x-auto overflow-y-visible pb-6 pt-5 relative z-0">

                        <div className="flex gap-4 min-w-max">
                          {yearData.trade_packages
                            .sort((a, b) => a.week - b.week)
                            .map((trade, index) => (
                              <MiniTradeCard 
                                onPlayerClick={handlePlayerClick}
                                key={index} 
                                trade={trade} 
                                year={year} 
                                showYear={false}
                              />
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
                        <div className="overflow-x-auto overflow-y-visible pb-6 pt-5 relative z-0">

                          <div className="flex gap-4 min-w-max">
                            {teamTrades.map((trade, index) => (
                              <MiniTradeCard 
                                onPlayerClick={handlePlayerClick}
                                key={index} 
                                trade={trade} 
                                year={trade.year} 
                                showYear={true}
                              />
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

        {/* Player Weekly Stats Popups - Multiple can be open */}
      
        {Array.from(selectedPlayers.entries()).map(([playerKey, playerData], index) => {
          // Calculate position for this pop-up in the centered stack
          const allPopups = Array.from(selectedPlayers.entries());
          const popupHeight = 180;
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
              className="relative bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-2xl p-5 transition-all duration-300 ease-out pointer-events-auto"
              style={{
                transform: `translateY(${thisPopupY}px)`,
                maxWidth: 'calc(100vw - 32px)',
                width: 'auto',
                minWidth: '360px',
                animation: 'fadeIn 0.3s ease-out forwards',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-3 relative">
                {/* Headshot - positioned to left of name, top 1/3 above card */}
                {playerData.headshotUrl && (
                  <div className="absolute -top-14 left-0 z-10" style={{ width: '70px', height: '105px' }}>
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
                <div className="flex-1" style={{ marginLeft: playerData.headshotUrl ? '85px' : '0' }}>
                  <h2 className="text-lg font-bold text-gray-900 dark:text-white">
                    {playerData.playerName}
                  </h2>
                  <div className="flex items-center gap-2 mt-1.5">
                    <label className="text-xs text-gray-500 dark:text-gray-400">Year:</label>
                    <select
                      value={playerData.selectedYear}
                      onChange={(e) => handleYearChange(playerKey, parseInt(e.target.value))}
                      className="text-xs bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
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
                    
                    const isTradeWeek = stat.week === playerData.tradeWeek && playerData.selectedYear === playerData.year;
                    const showDivider = isTradeWeek && statIndex > 0;
                    
                    return (
                      <div key={stat.week} className="flex items-center gap-2">
                        {showDivider && (
                          <div className="w-0.5 h-16 bg-white dark:bg-white flex-shrink-0"></div>
                        )}
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
                        <div className="flex items-center gap-1.5">
                          <span className="text-white font-bold text-base">PPG:</span>
                          <span style={{ color: getZavBrightColor(playerData.stats.total_zav ?? null) }} className="font-semibold text-lg">
                            {ppg.toFixed(1)}
                          </span>
                        </div>
                      );
                    })()}
                    {playerData.stats.total_zav !== null && playerData.stats.total_zav !== undefined && (
                      <div className="flex items-center gap-1.5 ml-5">
                        <span className="text-white font-bold text-base">ZAV:</span>
                        <span style={{ color: getZavBrightColor(playerData.stats.total_zav) }} className="font-semibold text-lg">
                          {playerData.stats.total_zav.toFixed(2)}
                        </span>
                      </div>
                    )}
                    {playerData.stats.fantasy_pos && playerData.stats.pos_rank !== null && playerData.stats.pos_rank !== undefined && (
                      <div className="flex items-center gap-0 ml-5">
                        <span className="text-white font-bold text-base">{playerData.stats.fantasy_pos}</span>
                        <span style={{ color: getZavBrightColor(playerData.stats.total_zav ?? null) }} className="font-semibold text-lg">
                          {playerData.stats.pos_rank}
                        </span>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="text-center py-6 text-xs text-gray-500 dark:text-gray-400">
                  No weekly stats available
                </div>
              )}
            </div>
          </div>
          );
        })}

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