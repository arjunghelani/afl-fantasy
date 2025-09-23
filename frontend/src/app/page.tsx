'use client';

import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <div className="max-w-6xl mx-auto p-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            Fantasy Football Dashboard
          </h1>
          <p className="text-xl text-slate-400">
            League 86952922 - Complete Analytics & Insights
          </p>
        </div>

        {/* Navigation Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-4xl mx-auto">
          
          {/* Standings */}
          <Link 
            href="/standings"
            className="group bg-gradient-to-br from-emerald-600 to-emerald-700 rounded-xl p-6 hover:from-emerald-500 hover:to-emerald-600 transition-all duration-300 hover:scale-105 hover:shadow-xl"
          >
            <div className="text-center">
              <div className="text-4xl mb-3">🏆</div>
              <h2 className="text-xl font-bold text-white mb-2">Standings</h2>
              <p className="text-emerald-100 text-sm">
                View season standings, all-time records, and team performance metrics
              </p>
            </div>
          </Link>

          {/* Players */}
          <Link 
            href="/players"
            className="group bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl p-6 hover:from-blue-500 hover:to-blue-600 transition-all duration-300 hover:scale-105 hover:shadow-xl"
          >
            <div className="text-center">
              <div className="text-4xl mb-3">👥</div>
              <h2 className="text-xl font-bold text-white mb-2">Players</h2>
              <p className="text-blue-100 text-sm">
                Player statistics, VORP/WAR analysis, and performance metrics
              </p>
            </div>
          </Link>

          {/* Scoreboard */}
          <Link 
            href="/scoreboard"
            className="group bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl p-6 hover:from-purple-500 hover:to-purple-600 transition-all duration-300 hover:scale-105 hover:shadow-xl"
          >
            <div className="text-center">
              <div className="text-4xl mb-3">📊</div>
              <h2 className="text-xl font-bold text-white mb-2">Scoreboard</h2>
              <p className="text-purple-100 text-sm">
                Historical scoreboard with detailed matchup breakdowns
              </p>
            </div>
          </Link>

          {/* Trades */}
          <Link 
            href="/trades"
            className="group bg-gradient-to-br from-orange-600 to-orange-700 rounded-xl p-6 hover:from-orange-500 hover:to-orange-600 transition-all duration-300 hover:scale-105 hover:shadow-xl"
          >
            <div className="text-center">
              <div className="text-4xl mb-3">🔄</div>
              <h2 className="text-xl font-bold text-white mb-2">Trades</h2>
              <p className="text-orange-100 text-sm">
                Trade analysis, value tracking, and transaction history
              </p>
            </div>
          </Link>

          {/* Draft */}
          <Link 
            href="/draft"
            className="group bg-gradient-to-br from-teal-600 to-teal-700 rounded-xl p-6 hover:from-teal-500 hover:to-teal-600 transition-all duration-300 hover:scale-105 hover:shadow-xl"
          >
            <div className="text-center">
              <div className="text-4xl mb-3">📋</div>
              <h2 className="text-xl font-bold text-white mb-2">Draft</h2>
              <p className="text-teal-100 text-sm">
                Draft results, pick analysis, and draft grades
              </p>
            </div>
          </Link>

          {/* Playoffs */}
          <Link 
            href="/playoffs"
            className="group bg-gradient-to-br from-red-600 to-red-700 rounded-xl p-6 hover:from-red-500 hover:to-red-600 transition-all duration-300 hover:scale-105 hover:shadow-xl"
          >
            <div className="text-center">
              <div className="text-4xl mb-3">🏈</div>
              <h2 className="text-xl font-bold text-white mb-2">Playoffs</h2>
              <p className="text-red-100 text-sm">
                Playoff brackets, championship history, and postseason stats
              </p>
            </div>
          </Link>

        </div>

      </div>
    </div>
  );
}