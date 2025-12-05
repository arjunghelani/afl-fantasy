'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface WaiverTransaction {
  transaction_id: number | null;
  transaction_date: string | null;
  team_name: string | null;
  action_type: string;
  player_name: string;
  player_position: string | null;
  player_zav: number | null;
}

interface WaiversResponse {
  year: number;
  transactions: WaiverTransaction[];
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

// ZAV color scale - same as other pages
const ZAV_CUTOFFS = {
  red: -2,
  orange: 2.5,
  yellow: 6,
  yellowGreen: 10,
};

function getZavGradient(zav: number | null): string {
  if (zav === null || zav === 0) {
    return 'linear-gradient(135deg, rgb(131, 131, 131), rgb(100, 100, 100))';
  }
  
  if (zav < ZAV_CUTOFFS.red) {
    return 'linear-gradient(135deg, rgb(220, 70, 70), rgb(180, 30, 30))';
  } else if (zav < ZAV_CUTOFFS.orange) {
    return 'linear-gradient(135deg, rgb(250, 130, 60), rgb(210, 90, 20))';
  } else if (zav < ZAV_CUTOFFS.yellow) {
    return 'linear-gradient(135deg, rgb(245, 210, 75), rgb(225, 170, 35))';
  } else if (zav < ZAV_CUTOFFS.yellowGreen) {
    return 'linear-gradient(135deg, rgb(160, 220, 80), rgb(120, 180, 40))';
  } else {
    return 'linear-gradient(135deg, rgb(16, 185, 129), rgb(5, 150, 105))';
  }
}

function getZavTextColor(zav: number | null): string {
  if (zav === null || zav === 0) {
    return 'rgb(40, 40, 40)';
  }
  
  // Extremely dark versions of bubble colors for text (used inside bubbles)
  if (zav < ZAV_CUTOFFS.red) {
    return 'rgb(90, 10, 10)'; // Extremely dark red
  } else if (zav < ZAV_CUTOFFS.orange) {
    return 'rgb(110, 35, 5)'; // Extremely dark orange
  } else if (zav < ZAV_CUTOFFS.yellow) {
    return 'rgb(110, 85, 5)'; // Extremely dark yellow/brown
  } else if (zav < ZAV_CUTOFFS.yellowGreen) {
    return 'rgb(50, 75, 10)'; // Extremely dark yellow-green
  } else {
    return 'rgb(1, 50, 20)'; // Extremely dark emerald green
  }
}

function getZavBrightColor(zav: number | null): string {
  if (zav === null || zav === 0) {
    return 'rgb(131, 131, 131)'; // Gray for null/zero
  }
  
  // Bright gradient colors (used for stats display)
  if (zav < ZAV_CUTOFFS.red) {
    return 'rgb(220, 70, 70)'; // Red
  } else if (zav < ZAV_CUTOFFS.orange) {
    return 'rgb(250, 130, 60)'; // Orange
  } else if (zav < ZAV_CUTOFFS.yellow) {
    return 'rgb(245, 210, 75)'; // Yellow
  } else if (zav < ZAV_CUTOFFS.yellowGreen) {
    return 'rgb(160, 220, 80)'; // Yellow-green
  } else {
    return 'rgb(16, 185, 129)'; // Emerald green
  }
}

// Fantasy points color scale - same as other pages
function getFantasyPointsGradient(points: number | null): string {
  if (points === null || points === 0) {
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
  if (points === null || points === 0) {
    return 'rgb(40, 40, 40)';
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

function formatDate(dateString: string | null): string {
  if (!dateString) return 'N/A';
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric'
    });
  } catch {
    return dateString;
  }
}

function formatTime(dateString: string | null): string {
  if (!dateString) return '';
  try {
    const date = new Date(dateString);
    return date.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true
    });
  } catch {
    return '';
  }
}

function groupTransactionsByDate(transactions: WaiverTransaction[]): { [key: string]: WaiverTransaction[] } {
  const grouped: { [key: string]: WaiverTransaction[] } = {};
  
  transactions.forEach(trans => {
    if (!trans.transaction_date) {
      const key = 'Unknown Date';
      if (!grouped[key]) grouped[key] = [];
      grouped[key].push(trans);
      return;
    }
    
    try {
      const date = new Date(trans.transaction_date);
      const dateKey = date.toLocaleDateString('en-US', { 
        month: 'long', 
        day: 'numeric', 
        year: 'numeric' 
      });
      
      if (!grouped[dateKey]) grouped[dateKey] = [];
      grouped[dateKey].push(trans);
    } catch {
      const key = 'Unknown Date';
      if (!grouped[key]) grouped[key] = [];
      grouped[key].push(trans);
    }
  });
  
  return grouped;
}

function hasPlayerData(data: PlayerWeeklyStatsResponse): boolean {
  return data.weekly_stats.some(stat => 
    stat.weekly_points_ppr !== null || stat.z_week_ppr !== null
  );
}

export default function WaiversPage() {
  const year = 2025;
  const [selectedTeam, setSelectedTeam] = useState<string>('All');
  const [selectedAction, setSelectedAction] = useState<string>('All');
  const [sortBy, setSortBy] = useState<'date' | 'zav'>('date');
  const [teamView, setTeamView] = useState<boolean>(false);
  const [transactions, setTransactions] = useState<WaiverTransaction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPlayers, setSelectedPlayers] = useState<Map<string, {
    playerName: string;
    year: number;
    position: { x: number; y: number };
    stats?: PlayerWeeklyStatsResponse;
    availableYears: number[];
    selectedYear: number;
    loading: boolean;
    headshotUrl?: string | null;
  }>>(new Map());

  // Get unique teams and action types from transactions
  const teams = Array.from(new Set(transactions.map(t => t.team_name).filter(Boolean))) as string[];
  teams.sort();

  useEffect(() => {
    async function fetchWaivers() {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`${NEXT_PUBLIC_API_BASE_URL}/waivers/${year}`, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error('Failed to fetch waiver activity');
        }
        const data: WaiversResponse = await response.json();
        setTransactions(data.transactions);
      } catch (err) {
        console.error('Error fetching waivers:', err);
        setError('Failed to load waiver activity');
      } finally {
        setLoading(false);
      }
    }
    
    fetchWaivers();
  }, []);

  const handlePlayerClick = async (playerName: string, year: number, event?: React.MouseEvent<HTMLSpanElement>) => {
    const playerKey = `${playerName}_${year}`;
    
    if (selectedPlayers.has(playerKey)) {
      return;
    }
    
    // Always center pop-ups, stack vertically when multiple
      const existingPopups = Array.from(selectedPlayers.values());
    
    // Get player position from transactions
    const transaction = transactions.find(t => t.player_name === playerName);
    const positionStr = transaction?.player_position || null;
    
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      newMap.set(playerKey, { 
        playerName, 
        year, 
        position: { x: 0, y: 0 }, 
        availableYears: [year], 
        selectedYear: year,
        loading: true 
      });
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
    const existing = selectedPlayers.get(playerKey);
    if (!existing) return;
    
    setSelectedPlayers(prev => {
      const newMap = new Map(prev);
      const existing = newMap.get(playerKey);
      if (existing) {
        newMap.set(playerKey, { ...existing, selectedYear: newYear, loading: true });
      }
      return newMap;
    });
    
    try {
      const response = await fetch(`${NEXT_PUBLIC_API_BASE_URL}/players/${encodeURIComponent(existing.playerName)}/weekly-stats?year=${newYear}`);
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
        newMap.delete(playerKey);
        return newMap;
      });
    }
  };

  // Filter transactions
  const filteredTransactions = transactions.filter(trans => {
    if (selectedTeam !== 'All' && trans.team_name !== selectedTeam) return false;
    if (selectedAction !== 'All' && trans.action_type !== selectedAction) return false;
    return true;
  });

  // Group transactions by transaction_id
  const groupByTransactionId = (transactions: WaiverTransaction[]): Map<number | null, WaiverTransaction[]> => {
    const grouped = new Map<number | null, WaiverTransaction[]>();
    transactions.forEach(trans => {
      const id = trans.transaction_id;
      if (!grouped.has(id)) {
        grouped.set(id, []);
      }
      grouped.get(id)!.push(trans);
    });
    return grouped;
  };

  const transactionGroups = groupByTransactionId(filteredTransactions);
  let groupsArray = Array.from(transactionGroups.entries()).map(([id, trans]) => ({
    id,
    transactions: trans.sort((a, b) => {
      // First, sort by action type: ADDED before DROPPED
      const aIsAdded = a.action_type === 'WAIVER ADDED' || a.action_type.includes('ADDED');
      const bIsAdded = b.action_type === 'WAIVER ADDED' || b.action_type.includes('ADDED');
      if (aIsAdded !== bIsAdded) {
        return aIsAdded ? -1 : 1; // ADDED comes first
      }
      // If same action type, sort by date
      if (!a.transaction_date || !b.transaction_date) return 0;
      return new Date(b.transaction_date).getTime() - new Date(a.transaction_date).getTime();
    })
  }));

  // Sort groups based on selected option
  if (sortBy === 'date') {
    groupsArray = groupsArray.sort((a, b) => {
      const dateA = a.transactions[0]?.transaction_date;
      const dateB = b.transactions[0]?.transaction_date;
      if (!dateA || !dateB) return 0;
      return new Date(dateB).getTime() - new Date(dateA).getTime();
  });
  } else if (sortBy === 'zav') {
    groupsArray = groupsArray.sort((a, b) => {
      // Get the highest ZAV from each transaction group
      const maxZavA = Math.max(...a.transactions.map(t => t.player_zav ?? -Infinity));
      const maxZavB = Math.max(...b.transactions.map(t => t.player_zav ?? -Infinity));
      return maxZavB - maxZavA; // Sort descending (highest ZAV first)
    });
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-950 dark:to-blue-950">
      <div className="mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Waivers & Transactions
          </h1>
          <p className="text-gray-600 dark:text-gray-400">View all add/drop activity across the league</p>

        {/* Navigation */}
          <nav className="flex justify-center items-center gap-3 mt-6 flex-wrap">
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
        </div>

        {/* Filters */}
        <div className="flex items-center justify-center gap-4 flex-wrap mb-6">
          <div className="flex items-center gap-2">
            <label className="text-gray-700 dark:text-gray-300">Team:</label>
            <select
              value={selectedTeam}
              onChange={(e) => setSelectedTeam(e.target.value)}
              className="rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-1.5 text-sm outline-none border border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-indigo-500"
            >
              <option value="All">All</option>
              {teams.map(team => (
                <option key={team} value={team}>{team}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-gray-700 dark:text-gray-300">Action:</label>
            <select
              value={selectedAction}
              onChange={(e) => setSelectedAction(e.target.value)}
              className="rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-1.5 text-sm outline-none border border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-indigo-500"
            >
              <option value="All">All</option>
              <option value="WAIVER ADDED">Add</option>
              <option value="DROPPED">Drop</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-gray-700 dark:text-gray-300">Sort by:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'date' | 'zav')}
              className="rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-1.5 text-sm outline-none border border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-indigo-500"
            >
              <option value="date">Date</option>
              <option value="zav">ZAV</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="teamView"
              checked={teamView}
              onChange={(e) => setTeamView(e.target.checked)}
              className="w-4 h-4 text-indigo-600 bg-gray-100 border-gray-300 rounded focus:ring-indigo-500 dark:focus:ring-indigo-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
            />
            <label htmlFor="teamView" className="text-gray-700 dark:text-gray-300 cursor-pointer">
              Team View
            </label>
          </div>
        </div>

        {/* Content */}
        {loading ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 text-center">
            <div className="text-gray-600 dark:text-gray-300">Loading...</div>
          </div>
        ) : error ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 text-center">
            <div className="text-red-600 dark:text-red-400">{error}</div>
          </div>
        ) : filteredTransactions.length === 0 ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 text-center">
            <div className="text-gray-600 dark:text-gray-300">No transactions found for the selected filters.</div>
          </div>
        ) : teamView ? (
          // Team View: Columns per team
          <div className="overflow-x-auto">
            <div className="inline-flex gap-4 min-w-full">
              {teams.map((team) => {
                const teamTransactions = groupsArray.filter(group => 
                  group.transactions[0]?.team_name === team
                );
                
                if (teamTransactions.length === 0) return null;
                
                return (
                  <div
                    key={team}
                    className="flex-shrink-0 w-80 bg-gray-50 dark:bg-gray-900 rounded-lg p-4 border border-gray-200 dark:border-gray-700"
                  >
                    {/* Team Header */}
                    <div className="text-gray-900 dark:text-white font-bold text-lg mb-4 pb-2 border-b border-gray-300 dark:border-gray-600">
                      {team}
                    </div>
                    
                    {/* Team Transactions */}
                    <div className="space-y-3 max-h-[600px] overflow-y-auto">
                      {teamTransactions.map((group, groupIdx) => {
                        const firstTrans = group.transactions[0];
                        
                        return (
                          <div
                            key={group.id ?? `no-id-${groupIdx}`}
                            className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700"
                          >
                            {/* Date */}
                            <div className="text-gray-500 dark:text-gray-400 text-xs font-medium mb-2">
                              {formatDate(firstTrans.transaction_date)}
                            </div>

                            {/* Transaction actions */}
                            <div className="space-y-2">
                              {group.transactions.map((trans, transIdx) => (
                                <div
                                  key={transIdx}
                                  className="flex items-center gap-2 flex-wrap"
                                >
                                  {/* Action Badge */}
                                  <div
                                    className={`px-2 py-1 rounded-full text-xs font-semibold flex-shrink-0 ${
                                      trans.action_type === 'WAIVER ADDED' || trans.action_type.includes('ADDED')
                                        ? 'bg-green-500/20 text-green-600 dark:text-green-400 border border-green-500/30'
                                        : 'bg-red-500/20 text-red-600 dark:text-red-400 border border-red-500/30'
                                    }`}
                                  >
                                    {trans.action_type === 'WAIVER ADDED' || trans.action_type.includes('ADDED') ? '+ ADD' : '- DROP'}
                                  </div>

                                  {/* Player Name */}
                                  <span
                                    onClick={(e) => handlePlayerClick(trans.player_name, year, e)}
                                    className="text-gray-900 dark:text-white font-medium cursor-pointer hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors text-sm flex-1 min-w-0"
                                  >
                                    {trans.player_name}
                                  </span>
                
                                  {/* Position and ZAV */}
                                  <div className="flex items-center gap-2 flex-shrink-0">
                                    {trans.player_position && (
                                      <span className="text-gray-500 dark:text-gray-400 text-xs">
                                        {trans.player_position}
                                      </span>
                                    )}

                                    {trans.player_zav !== null && (
                                      <div
                                        className="px-2 py-0.5 rounded text-xs font-bold"
                                        style={{
                                          background: getZavGradient(trans.player_zav),
                                          color: getZavTextColor(trans.player_zav)
                                        }}
                                      >
                                        {trans.player_zav.toFixed(1)}
                                      </div>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          // Grid View: Original layout
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {groupsArray.map((group, groupIdx) => {
                const firstTrans = group.transactions[0];
                
                return (
                    <div
                    key={group.id ?? `no-id-${groupIdx}`}
                    className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 transition-colors"
                    >
                    {/* Header with team and date */}
                    <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-300 dark:border-gray-600">
                      <div className="text-gray-900 dark:text-white font-semibold text-base tracking-tight">
                        {firstTrans.team_name || 'Unknown Team'}
                      </div>
                      <div className="text-gray-500 dark:text-gray-400 text-sm font-medium">
                        {formatDate(firstTrans.transaction_date)}
                        </div>
                        </div>

                    {/* Transaction actions */}
                    <div className="space-y-2">
                      {group.transactions.map((trans, transIdx) => (
                        <div
                          key={transIdx}
                          className="flex items-center gap-2 flex-wrap"
                        >
                        {/* Action Badge */}
                        <div
                            className={`px-2 py-1 rounded-full text-xs font-semibold flex-shrink-0 ${
                              trans.action_type === 'WAIVER ADDED' || trans.action_type.includes('ADDED')
                                ? 'bg-green-500/20 text-green-600 dark:text-green-400 border border-green-500/30'
                                : 'bg-red-500/20 text-red-600 dark:text-red-400 border border-red-500/30'
                          }`}
                        >
                            {trans.action_type === 'WAIVER ADDED' || trans.action_type.includes('ADDED') ? '+ ADD' : '- DROP'}
                        </div>

                        {/* Player Name */}
                        <span
                          onClick={(e) => handlePlayerClick(trans.player_name, year, e)}
                            className="text-gray-900 dark:text-white font-medium cursor-pointer hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors text-sm flex-1 min-w-0"
                        >
                          {trans.player_name}
                        </span>

                          {/* Position and ZAV */}
                          <div className="flex items-center gap-2 flex-shrink-0">
                        {trans.player_position && (
                              <span className="text-gray-500 dark:text-gray-400 text-xs">
                            {trans.player_position}
                          </span>
                        )}

                        {trans.player_zav !== null && (
                          <div
                                className="px-2 py-0.5 rounded text-xs font-bold"
                            style={{
                              background: getZavGradient(trans.player_zav),
                              color: getZavTextColor(trans.player_zav)
                            }}
                          >
                            {trans.player_zav.toFixed(1)}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
                );
              })}
          </div>
        )}

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


