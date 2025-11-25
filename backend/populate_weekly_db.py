#!/usr/bin/env python3
"""
One-off script to populate database with weekly PPR points and z-score calculations
"""

import sqlite3
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime
import os
import sys
import hashlib

# Add the current directory to path so we can import from vorp.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vorp import get_weekly_fantasy_points_from_players, fill_missing_weeks
from main import _get_league
from trade_analysis import build_ownership_timeseries, guess_max_week
import requests

def create_database():
    """Create SQLite database with tables for weekly data"""
    conn = sqlite3.connect('weekly_fantasy_data.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weekly_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            fantasy_pos TEXT NOT NULL,
            week INTEGER NOT NULL,
            weekly_points_ppr REAL NOT NULL,
            year INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS z_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            fantasy_pos TEXT NOT NULL,
            week INTEGER NOT NULL,
            weekly_points_ppr REAL NOT NULL,
            log_ppr REAL NOT NULL,
            z_week_ppr REAL NOT NULL,
            year INTEGER NOT NULL,
            fantasy_team TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_totals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            fantasy_pos TEXT NOT NULL,
            total_points REAL NOT NULL,
            pos_rank INTEGER NOT NULL,
            overall_rank INTEGER NOT NULL,
            vorp_star REAL NOT NULL,
            year INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS waiver_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER,
            year INTEGER NOT NULL,
            transaction_date TIMESTAMP,
            team_id INTEGER,
            team_name TEXT,
            action_type TEXT NOT NULL,
            player_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    return conn

def clear_database(conn):
    """Clear all data from the database tables"""
    cursor = conn.cursor()
    
    print("🗑️ Clearing existing data...")
    cursor.execute("DELETE FROM weekly_points")
    cursor.execute("DELETE FROM z_scores") 
    cursor.execute("DELETE FROM player_totals")
    cursor.execute("DELETE FROM waiver_activity")
    
    # Reset auto-increment counters
    cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('weekly_points', 'z_scores', 'player_totals', 'waiver_activity')")
    
    conn.commit()
    print("✅ Database cleared")

def get_draft_data(year):
    """Get draft data from ESPN API"""
    try:
        league = _get_league(year)
        draft = league.draft
        draft_data = []
        
        for pick in draft:
            overall_pick = pick.round_num * len(league.teams) + pick.round_pick

            draft_data.append({
                'player_name': pick.playerName,
                'team_id': pick.team,
                'round': pick.round_num,
                'pick': pick.round_pick,
                'overall_pick': overall_pick
            })
        
        return draft_data
    except Exception as e:
        print(f"❌ Error getting draft data for {year}: {e}")
        return []

def check_and_add_missing_drafted_players(year):
    """Check if drafted players are in the database, add them if missing"""
    print(f"🔍 Checking for missing drafted players in {year}...")

    # Get draft data
    draft_data = get_draft_data(year)
    if not draft_data:
        print(f"❌ No draft data found for {year}")
        return

    print(f"📋 Found {len(draft_data)} drafted players")

    # Connect to database
    conn = sqlite3.connect('weekly_fantasy_data.db')
    cursor = conn.cursor()

    try:
        league = _get_league(year)

        missing_players = []
        for player in draft_data:
            player_name = player['player_name']

            # Check if player exists in player_totals
            cursor.execute('''
                SELECT COUNT(*) FROM player_totals 
                WHERE player_name = ? AND year = ?
            ''', (player_name, year))
            exists = cursor.fetchone()[0] > 0

            if not exists:
                missing_players.append(player)
                print(f"  ❌ Missing: {player_name}")
            else:
                print(f"  ✅ Found: {player_name}")

        if not missing_players:
            print("✅ All drafted players are in the database!")
            return

        print(f"\n📊 Found {len(missing_players)} missing players, recalculating VORP based on full dataset...")

        missing_names = [p['player_name'] for p in missing_players]

        # === Get full dataset for normalization ===
        weekly_df = get_weekly_fantasy_points_from_players(year, max_week=17, league=league)
        weekly_df = fill_missing_weeks(weekly_df, year, league)
        weekly_df = weekly_df.drop_duplicates(subset=['player_name', 'week'])
        weekly_df = weekly_df.loc[weekly_df.weekly_points_ppr > 0].copy()

        print(f"✅ Loaded {len(weekly_df)} total player-week records for normalization")

        # === Step 1: Compute positional totals and ranks ===
        result = duckdb.sql("""
            with totals as (
                select player_name, fantasy_pos, sum(weekly_points_ppr) as total_points,
                rank() over (partition by fantasy_pos order by sum(weekly_points_ppr) desc) as pos_rank,
                rank() over (order by sum(weekly_points_ppr) desc) as overall_rank
                from weekly_df
                group by player_name, fantasy_pos
            )
            select *
            from weekly_df
            left join totals using (player_name, fantasy_pos)
        """).df()

        # === Step 2: Apply 60% cutoff for positional VORP normalization ===
        print("✂️ Applying 60% positional cutoff (for normalization only)...")
        caps = pd.DataFrame(
            result.groupby(['fantasy_pos'])
            .agg({'pos_rank': 'max'})
            .pos_rank
            .apply(lambda x: np.ceil(0.60 * x))
        ).reset_index()

        result_cutoff = duckdb.sql("""
            select player_name, fantasy_pos, week, weekly_points_ppr,
                   result.pos_rank, overall_rank
            from result
            left join caps using (fantasy_pos)
            where result.pos_rank <= caps.pos_rank
        """).df()
        
        # Print number of players per position after cutoff
        print("📊 Players per position after 60% cutoff:")
        cutoff_counts = result_cutoff.groupby('fantasy_pos')['player_name'].nunique().sort_index()
        for pos, count in cutoff_counts.items():
            print(f"  {pos}: {count} players")

        # === Step 3: Log-transform points ===
        result_cutoff = result_cutoff[result_cutoff.weekly_points_ppr > 0].copy()
        result_cutoff['log_ppr'] = result_cutoff.weekly_points_ppr.apply(
            lambda x: np.log10(max(x, 0.1))
        )

        # === Step 4: Calculate z-scores across full positional pools ===
        print("🎯 Calculating z-scores (full positional normalization)...")
        z_scores_full = duckdb.sql("""
            with norms as (
                select avg(log_ppr) as avg_log_ppr, stddev(log_ppr) as std_log_ppr, fantasy_pos
                from result_cutoff
                group by fantasy_pos
            )
            select *,
                   (log_ppr - avg_log_ppr) / std_log_ppr as z_week_ppr
            from result_cutoff
            left join norms using (fantasy_pos)
        """).df()

        # === Step 5: Filter to missing players only ===
        z_missing = z_scores_full[z_scores_full['player_name'].isin(missing_names)].copy()
        
        # Find missing players with no data (drafted but never played)
        players_with_data = set(z_missing['player_name'].unique())
        players_without_data = [name for name in missing_names if name not in players_with_data]
        
        print(f"📊 Found {len(players_with_data)} missing players with data")
        print(f"📊 Found {len(players_without_data)} missing players with no data (drafted but never played)")
        
        if len(z_missing) == 0 and len(players_without_data) == 0:
            print("❌ No missing players to process")
            return

        # === Step 6: Insert missing players' weekly & z-score data ===
        print("💾 Inserting weekly points & z-scores for missing players...")
        for _, row in z_missing.iterrows():
            cursor.execute('''
                INSERT OR IGNORE INTO weekly_points (player_name, fantasy_pos, week, weekly_points_ppr, year)
                VALUES (?, ?, ?, ?, ?)
            ''', (row['player_name'], row['fantasy_pos'], row['week'], row['weekly_points_ppr'], year))

            cursor.execute('''
                INSERT OR IGNORE INTO z_scores (player_name, fantasy_pos, week, weekly_points_ppr, log_ppr, z_week_ppr, year)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['player_name'], row['fantasy_pos'], row['week'], 
                  row['weekly_points_ppr'], row['log_ppr'], row['z_week_ppr'], year))
        
        # === Step 7: Handle players with no data (give them 0 VORP) ===
        if len(players_without_data) > 0:
            print(f"💾 Adding {len(players_without_data)} players with no data (0 VORP)...")
            for player_name in players_without_data:
                # Get player position from draft data
                player_info = next((p for p in missing_players if p['player_name'] == player_name), None)
                if player_info:
                    # Insert a placeholder entry with 0 VORP
                    cursor.execute('''
                        INSERT OR IGNORE INTO player_totals (player_name, fantasy_pos, total_points, pos_rank, overall_rank, vorp_star, year)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (player_name, 'UNKNOWN', 0.0, 999, 999, 0.0, year))

        # === Step 7: Compute final VORP totals (same as populate_weekly_data) ===
        print("🏆 Calculating final VORP totals for missing players...")
        final_vorp = z_missing.groupby(['player_name']).agg({
            'fantasy_pos': 'first',
            'z_week_ppr': ['sum', 'std'],
            'pos_rank': 'first',
            'overall_rank': 'first'
        }).reset_index()

        final_vorp.columns = ['player_name', 'fantasy_pos', 'vorp_star', 'vorp_std', 'pos_rank', 'overall_rank']

        # === Step 8: Insert into player_totals ===
        for _, row in final_vorp.iterrows():
            total_points = result.groupby('player_name')['weekly_points_ppr'].sum().loc[row['player_name']]
            cursor.execute('''
                INSERT OR IGNORE INTO player_totals (player_name, fantasy_pos, total_points, pos_rank, overall_rank, vorp_star, year)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['player_name'], row['fantasy_pos'], total_points,
                  row['pos_rank'], row['overall_rank'], row['vorp_star'], year))

        conn.commit()

        total_added = len(final_vorp) if len(z_missing) > 0 else 0
        total_added += len(players_without_data)
        print(f"\n✅ Successfully added {total_added} missing players (normalized globally).")
        print(f"  - {len(final_vorp) if len(z_missing) > 0 else 0} players with data")
        print(f"  - {len(players_without_data)} players with no data (0 VORP)")
        
        if len(final_vorp) > 0:
            print("\n📊 Added Players with Data:")
            for _, player in final_vorp.iterrows():
                print(f"  {player['player_name']} ({player['fantasy_pos']}): {player['vorp_star']:.3f}")
        
        if len(players_without_data) > 0:
            print(f"\n📊 Added Players with No Data (0 VORP): {', '.join(players_without_data[:5])}{'...' if len(players_without_data) > 5 else ''}")

    except Exception as e:
        print(f"❌ Error adding missing players: {e}")
        import traceback; traceback.print_exc()
    finally:
        conn.close()


def populate_weekly_data(year):
    """
    Populate database with weekly fantasy data.
    
    Process:
    1. Collect player names from multiple sources (draft + weekly rosters)
    2. Get weekly stats for all those players
    3. Calculate z-scores, VORP, etc. and insert into database
    """
    print(f"🚀 Starting database population for {year}...")
    
    # Create database
    conn = create_database()
    cursor = conn.cursor()
    
    # Clear existing data
    # clear_database(conn)
    
    try:
        # Get league
        print("📡 Getting league data...")
        league = _get_league(year)
        
        # === Phase 1: Collect player names from multiple sources ===
        print("\n" + "="*60)
        print("PHASE 1: Collecting player names from multiple sources")
        print("="*60)
        player_names = collect_all_player_names(year)
        
        if not player_names:
            print("❌ No players found. Exiting.")
            return
        
        # === Phase 2: Get weekly stats for all players ===
        print("\n" + "="*60)
        print("PHASE 2: Collecting weekly stats for all players")
        print("="*60)
        weekly_df = get_weekly_stats_for_players(year, player_names)
        
        if weekly_df.empty:
            print("❌ No weekly stats found. Exiting.")
            return
        
        # Clean and deduplicate
        weekly_df = weekly_df.drop_duplicates(subset=['player_name', 'week'])
        print(f"✅ Got {len(weekly_df)} player-week records after deduplication")
        
        # === Phase 3: Insert weekly points ===
        print("\n" + "="*60)
        print("PHASE 3: Inserting weekly points into database")
        print("="*60)
        print("💾 Inserting weekly points...")
        for _, row in weekly_df.iterrows():
            cursor.execute('''
                INSERT INTO weekly_points (player_name, fantasy_pos, week, weekly_points_ppr, year)
                VALUES (?, ?, ?, ?, ?)
            ''', (row['player_name'], row['fantasy_pos'], row['week'], row['weekly_points_ppr'], year))
        
        # === Phase 4: Calculate z-scores, VORP, etc. ===
        print("\n" + "="*60)
        print("PHASE 4: Calculating z-scores, VORP, and rankings")
        print("="*60)
        
        # Calculate rankings
        print("📈 Calculating player rankings...")
        result = duckdb.sql("""
            with totals as (
                select player_name, fantasy_pos, sum(weekly_points_ppr) as total_points,
                rank() over (partition by fantasy_pos order by sum(weekly_points_ppr) desc) as pos_rank,
                rank() over (order by sum(weekly_points_ppr) desc) as overall_rank
                from weekly_df
                group by player_name, fantasy_pos
                order by total_points desc
            )
            
            SELECT *
            FROM weekly_df
            left join totals using (player_name, fantasy_pos)
        """).df()
        
        # Apply 60% cutoff
        print("✂️ Applying 60% cutoff...")
        caps = pd.DataFrame(result.groupby(['fantasy_pos']).agg({'pos_rank':'max'}).pos_rank.apply(lambda x:np.ceil(0.60 * x))).reset_index()
        
        result2 = duckdb.sql("""
            select player_name, fantasy_pos, week, weekly_points_ppr, result.pos_rank, overall_rank from result 
            left join caps using (fantasy_pos)
            where result.pos_rank <= caps.pos_rank
        """).df()
        
        # Print number of players per position after cutoff
        print("📊 Players per position after 60% cutoff:")
        cutoff_counts = result2.groupby('fantasy_pos')['player_name'].nunique().sort_index()
        for pos, count in cutoff_counts.items():
            print(f"  {pos}: {count} players")
        
        # Filter out 0 values before log transform
        print("📊 Filtering out zero values...")
        result2 = result2[result2.weekly_points_ppr > 0].copy()
        
        # Log transform (handle any remaining edge cases)
        print("📊 Log transforming points...")
        result2['log_ppr'] = result2.weekly_points_ppr.apply(lambda x: np.log10(max(x, 0.1)) if x > 0 else np.log10(0.1))
        
        # Calculate z-scores
        print("🎯 Calculating z-scores...")
        z_scores = duckdb.sql("""
            with norms as (
                select avg(log_ppr) as avg_log_ppr, stddev(log_ppr) as std_log_ppr, fantasy_pos
                from result2
                group by fantasy_pos
            )
            
            select *,
            (log_ppr - avg_log_ppr) / std_log_ppr as z_week_ppr
            from result2
            left join norms using (fantasy_pos)   
        """).df()
        
        # Insert z-scores
        print("💾 Inserting z-scores...")
        for _, row in z_scores.iterrows():
            cursor.execute('''
                INSERT INTO z_scores (player_name, fantasy_pos, week, weekly_points_ppr, log_ppr, z_week_ppr, year)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['player_name'], row['fantasy_pos'], row['week'], 
                  row['weekly_points_ppr'], row['log_ppr'], row['z_week_ppr'], year))
        
        # Calculate final VORP totals
        print("🏆 Calculating final VORP totals...")
        final_vorp = z_scores.groupby(['player_name']).agg({
            'fantasy_pos': 'first',
            'z_week_ppr': ['sum', 'std'],
            'pos_rank': 'first',
            'overall_rank': 'first'
        }).reset_index()
        
        # Flatten column names
        final_vorp.columns = ['player_name', 'fantasy_pos', 'vorp_star', 'vorp_std', 'vorp_star_rank_pos', 'vorp_star_rank_overall']
        
        # Insert player totals
        print("💾 Inserting player totals...")
        for _, row in final_vorp.iterrows():
            cursor.execute('''
                INSERT INTO player_totals (player_name, fantasy_pos, total_points, pos_rank, overall_rank, vorp_star, year)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['player_name'], row['fantasy_pos'], 
                  result.groupby('player_name')['weekly_points_ppr'].sum().loc[row['player_name']],
                  row['vorp_star_rank_pos'], row['vorp_star_rank_overall'], row['vorp_star'], year))
        
        conn.commit()
        
        # === Phase 5: Summary ===
        print("\n" + "="*60)
        print("PHASE 5: Summary")
        print("="*60)
        
        # Print summary
        print("\n📊 Database Population Summary:")
        print(f"  Total unique players: {len(player_names)}")
        print(f"  Weekly points records: {len(weekly_df)}")
        print(f"  Z-score records: {len(z_scores)}")
        print(f"  Player totals: {len(final_vorp)}")
        
        # Show top players
        print("\n🏆 Top 10 Players by VORP:")
        top_players = final_vorp.nlargest(10, 'vorp_star')
        for _, player in top_players.iterrows():
            print(f"  {player['player_name']} ({player['fantasy_pos']}): {player['vorp_star']:.3f}")
        
        print(f"\n✅ Database populated successfully!")
        print(f"📁 Database file: weekly_fantasy_data.db")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

def collect_all_player_names(year):
    """
    Collect player names from multiple sources:
    1. Draft players
    2. Players from trade analysis (players who were on rosters each week)
    
    Returns a set of unique player names
    """
    print(f"📋 Collecting player names from multiple sources for {year}...")
    league = _get_league(year)
    player_names = set()
    
    # Source 1: Draft players
    print("  📝 Getting draft players...")
    draft_data = get_draft_data(year)
    draft_player_names = set()
    for pick in draft_data:
        player_name = pick['player_name'].replace('*', '').strip()
        if player_name:
            draft_player_names.add(player_name)
            player_names.add(player_name)
    
    print(f"    ✅ Found {len(draft_player_names)} draft players")
    
    # Source 2: Players from trade analysis (players on rosters each week)
    print("  📝 Getting players from weekly rosters (trade analysis)...")
    try:
        max_week = guess_max_week(league)
        weeks = list(range(1, max_week + 1))
        owner_by_player, player_meta, team_meta = build_ownership_timeseries(league, weeks)
        
        # Extract player names from player_meta
        roster_player_names = set()
        for player_id, meta in player_meta.items():
            player_name = meta.get('name', None)
            if player_name:
                player_name = player_name.replace('*', '').strip()
                if player_name:
                    roster_player_names.add(player_name)
                    player_names.add(player_name)
        
        print(f"    ✅ Found {len(roster_player_names)} players from weekly rosters")
        
        # Show overlap
        overlap = draft_player_names & roster_player_names
        only_draft = draft_player_names - roster_player_names
        only_roster = roster_player_names - draft_player_names
        
        print(f"    📊 Overlap: {len(overlap)} players in both")
        print(f"    📊 Only in draft: {len(only_draft)} players")
        print(f"    📊 Only in rosters: {len(only_roster)} players")
        
    except Exception as e:
        print(f"    ⚠️  Error getting roster players: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"  ✅ Total unique players: {len(player_names)}")
    return player_names


def get_weekly_stats_for_players(year, player_names):
    """
    Get weekly fantasy points for a list of player names.
    Returns a DataFrame with columns: player_name, fantasy_pos, week, weekly_points_ppr
    """
    print(f"📊 Collecting weekly stats for {len(player_names)} players...")
    league = _get_league(year)
    rows = []
    
    processed = 0
    errors = 0
    
    for player_name in player_names:
        try:
            # DEBUG: Check for Ashton Jeanty
            is_ashton = 'Ashton Jeanty' in player_name or player_name == 'Ashton Jeanty'
            if is_ashton:
                print(f"  🔍 [PLAYER_INFO.STATS] Processing Ashton Jeanty...")
                print(f"      Player name in list: {player_name}")
            
            player_info = league.player_info(player_name)
            if player_info is None:
                if is_ashton:
                    print(f"  🔍 [PLAYER_INFO.STATS] Ashton Jeanty: player_info is None")
                errors += 1
                continue
            
            position = player_info.position
            if is_ashton:
                print(f"  🔍 [PLAYER_INFO.STATS] Ashton Jeanty: position = {position}")
                print(f"      Has stats attribute: {hasattr(player_info, 'stats')}")
                if hasattr(player_info, 'stats'):
                    print(f"      Stats dict: {player_info.stats}")
            
            if not hasattr(player_info, 'stats') or not player_info.stats:
                if is_ashton:
                    print(f"  🔍 [PLAYER_INFO.STATS] Ashton Jeanty: No stats available")
                continue
            
            ashton_weeks = []
            for week in player_info.stats:
                if week != 18:  # Skip week 18 (playoffs)
                    week_data = player_info.stats[week]
                    if 'points' in week_data:
                        ppr_points = week_data['points']
                        rows.append({
                            'player_name': player_name,
                            'fantasy_pos': position,
                            'week': week,
                            'weekly_points_ppr': ppr_points
                        })
                        if is_ashton:
                            ashton_weeks.append({'week': week, 'points': ppr_points})
            
            if is_ashton:
                print(f"  🔍 [PLAYER_INFO.STATS] Ashton Jeanty: Found {len(ashton_weeks)} weeks with points")
                for w in ashton_weeks:
                    print(f"      Week {w['week']}: {w['points']} points")
            
            processed += 1
            if processed % 50 == 0:
                print(f"    Processed {processed}/{len(player_names)} players...")
                
        except Exception as e:
            if is_ashton:
                print(f"  🔍 [PLAYER_INFO.STATS] Ashton Jeanty: Error - {e}")
            errors += 1
            continue
    
    print(f"  ✅ Processed {processed} players successfully")
    if errors > 0:
        print(f"  ⚠️  {errors} players had errors")
    
    df = pd.DataFrame(rows)
    print(f"  ✅ Collected {len(df)} total player-week records")
    
    # DEBUG: Print summary for Ashton Jeanty
    ashton_rows = df[df['player_name'].str.contains('Ashton Jeanty', case=False, na=False)]
    if len(ashton_rows) > 0:
        print(f"  🔍 [PLAYER_INFO.STATS SUMMARY] Ashton Jeanty found in weekly_df:")
        print(f"      Total rows: {len(ashton_rows)}")
        for _, row in ashton_rows.iterrows():
            print(f"      Week {row['week']}: {row['weekly_points_ppr']} points, Position: {row['fantasy_pos']}")
    else:
        print(f"  🔍 [PLAYER_INFO.STATS SUMMARY] Ashton Jeanty NOT found in weekly_df")
    
    return df


def get_draft_stats(year):
    """
    Legacy function - kept for backward compatibility.
    Now just calls collect_all_player_names and get_weekly_stats_for_players
    """
    player_names = collect_all_player_names(year)
    return get_weekly_stats_for_players(year, player_names)

def get_player_team_mapping(year, player_names):
    """
    Get which fantasy team each player was on each week by analyzing box scores.
    
    Args:
        year: The year/season to analyze
        player_names: Set or list of player names to map (from collect_all_player_names)
    
    Returns:
        dict: Mapping of {player_name: {week: team_name}} for each week the player was on a roster
              Example: {'Patrick Mahomes': {1: 'Team A', 2: 'Team A', 3: 'Team B'}}
    """
    print(f"📊 Getting player-to-team mapping for {year}...")
    league = _get_league(year)
    
    # Convert player_names to set for faster lookup
    player_names_set = set(player_names) if isinstance(player_names, (list, set)) else player_names
    
    # Initialize mapping: {player_name: {week: team_name}}
    player_team_map = {}
    
    # Get max week
    max_week = guess_max_week(league)
    weeks = list(range(1, max_week + 1))
    
    print(f"  📅 Processing {len(weeks)} weeks (1-{max_week})...")
    
    for week in weeks:
        try:
            box_scores = league.box_scores(week=week)
        except Exception as e:
            print(f"  ⚠️  Could not get box scores for week {week}: {e}")
            continue
        
        if not box_scores:
            continue
        
        week_count = 0
        for box in box_scores:
            # Process home lineup
            if hasattr(box, 'home_team') and hasattr(box, 'home_lineup'):
                home_team = box.home_team
                home_team_name = home_team.team_name if hasattr(home_team, 'team_name') else str(home_team)
                home_lineup = box.home_lineup or []
                
                for player in home_lineup:
                    player_name = getattr(player, 'name', None)
                    if player_name:
                        # Clean player name (remove asterisks, strip whitespace)
                        player_name = player_name.replace('*', '').strip()
                        
                        # Only process if player is in our player_names set
                        if player_name in player_names_set:
                            if player_name not in player_team_map:
                                player_team_map[player_name] = {}
                            player_team_map[player_name][week] = home_team_name
                            week_count += 1
            
            # Process away lineup
            if hasattr(box, 'away_team') and hasattr(box, 'away_lineup'):
                away_team = box.away_team
                away_team_name = away_team.team_name if hasattr(away_team, 'team_name') else str(away_team)
                away_lineup = box.away_lineup or []
                
                for player in away_lineup:
                    player_name = getattr(player, 'name', None)
                    if player_name:
                        # Clean player name (remove asterisks, strip whitespace)
                        player_name = player_name.replace('*', '').strip()
                        
                        # Only process if player is in our player_names set
                        if player_name in player_names_set:
                            if player_name not in player_team_map:
                                player_team_map[player_name] = {}
                            player_team_map[player_name][week] = away_team_name
                            week_count += 1
        
        if week_count > 0:
            print(f"    Week {week}: Mapped {week_count} player-team relationships")
    
    # Summary
    total_players = len(player_team_map)
    total_weeks_mapped = sum(len(weeks) for weeks in player_team_map.values())
    print(f"  ✅ Mapped {total_players} players across {total_weeks_mapped} player-week combinations")
    
    return player_team_map


def populate_waiver_activity(year):
    
    """
    Populate waiver_activity table with transactions from ESPN API.
    Gets all waiver adds and drops for the given year.
    """
    print(f"📋 Populating waiver activity for {year}...")
    
    conn = sqlite3.connect('weekly_fantasy_data.db')
    cursor = conn.cursor()
    
    try:
        # Clear existing data for this year
        print(f"  🗑️  Clearing existing waiver activity data for {year}...")
        cursor.execute("DELETE FROM waiver_activity WHERE year = ?", (year,))
        conn.commit()
        print(f"  ✅ Cleared existing data for {year}")
        
        # Check if transaction_id column exists, add it if not
        # cursor.execute("PRAGMA table_info(waiver_activity)")
        # columns = [row[1] for row in cursor.fetchall()]
        # if 'transaction_id' not in columns:
        #     print(f"  🔧 Adding transaction_id column to waiver_activity table...")
        #     cursor.execute("ALTER TABLE waiver_activity ADD COLUMN transaction_id INTEGER")
        #     conn.commit()
        #     print(f"  ✅ Added transaction_id column")
        
        league = _get_league(year)
        
        # Get all transactions from ESPN with pagination
        print("  📡 Fetching transactions from ESPN...")
        
        # Filter and collect waiver-related transactions
        rows_to_insert = []  # Collect all rows to insert
        skipped = 0
        batch_size = 50  # Number of transactions to fetch per batch
        offset = 0
        total_fetched = 0
        
        while True:
            try:
                # Fetch transactions with pagination
                transactions = league.recent_activity(size=batch_size, offset=offset)
                print(len(transactions))
            except Exception as e:
                if offset == 0:
                    # Only print error on first attempt
                    print(f"  ⚠️  Could not fetch transactions for {year}: {e}")
                    print(f"  ℹ️  This may be normal if the season hasn't started or transactions aren't available")
                break
            
            if not transactions:
                # No more transactions
                break
            
            batch_count = len(transactions)
            total_fetched += batch_count
            
            # Process each transaction in this batch
            for transaction in transactions:
                try:
                    # Skip trades, only get adds/drops
                    # if transaction.type and 'TRADE' in str(transaction.type).upper():
                    #     continue
                    
                    # Get transaction date (convert from milliseconds to datetime)
                    transaction_date = None
                    transaction_timestamp = None
                    if transaction.date:
                        try:
                            # Convert milliseconds to seconds and then to datetime
                            transaction_timestamp = int(transaction.date)
                            timestamp_seconds = transaction_timestamp / 1000
                            dt = datetime.fromtimestamp(timestamp_seconds)
                            transaction_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except (ValueError, TypeError, OSError) as e:
                            # If conversion fails, skip this transaction's date
                            transaction_date = None
                            transaction_timestamp = None
                    
                    # Get actions from transaction
                    actions = transaction.actions
                    if not actions:
                        skipped += 1
                        print('error in actions')
                        continue
                    
                
                    
                    # # Create a hash-based transaction_id (convert to int for database)
                    # transaction_id = int(hashlib.md5(transaction_id_str.encode()).hexdigest()[:8], 16)
                    
                    # Process actions - handle both single action and swap (nested lists)
                    action_list = []

                    # Check if actions is a list/tuple of tuples (multiple actions) or a single tuple
                    if isinstance(actions, (list, tuple)) and len(actions) > 0:
                        # Check if first element is a tuple (nested case)
                        if isinstance(actions[0], tuple):
                            # Multiple actions case: [(team, action, player), (team, action, player)]
                            action_list = list(actions)
                        else:
                            # Single action case: (team, action, player)
                            action_list = [actions]
                    else:
                        # Fallback: treat as single action
                        action_list = [actions]
                        
                    # Skip trades
                    if any(len(a) >= 2 and 'TRADED' in str(a[1]).upper() for a in action_list):
                        print('skipping trade: ', action_list)
                        continue

                    # Generate a unique transaction_id for this transaction (shared across all actions)
                    transaction_id_str = f"{year}_{transaction_timestamp or transaction.date}"
                    if action_list:
                        first_action = action_list[0]
                        if hasattr(first_action[0], 'team_name'):
                            transaction_id_str += f"_{first_action[0].team_name}"
                        if hasattr(first_action[2], 'name'):
                            transaction_id_str += f"_{first_action[2].name}"

                    transaction_id = int(hashlib.md5(transaction_id_str.encode()).hexdigest()[:8], 16)

                    # Process each action (all actions share the same transaction_id)
                    for action in action_list:
                        try:
                            # action is a tuple: (team_object, 'FA ADDED'/'DROPPED', player_object)
                            if len(action) < 3:
                                continue
                            
                            team_obj = action[0]
                            action_type_raw = action[1]
                            player_obj = action[2]
                            
                            # Get team name
                            team_name = team_obj.team_name if hasattr(team_obj, 'team_name') else str(team_obj)
                            
                            # Get player name
                            player_name = player_obj.name if hasattr(player_obj, 'name') else str(player_obj)
                            
                            # Clean player name
                            player_name = player_name.replace('*', '').strip()
                            if not player_name:
                                continue
                            
                            # Use action type as-is from ESPN
                            action_type = action_type_raw
                            
                            # Get team_id
                            team_id = None
                            if hasattr(team_obj, 'team_id'):
                                team_id = team_obj.team_id
                            elif hasattr(team_obj, 'teamId'):
                                team_id = team_obj.teamId
                            
                            # Collect row data
                            rows_to_insert.append((
                                transaction_id,
                                year,
                                transaction_date,
                                team_id,
                                team_name,
                                action_type,
                                player_name
                            ))
                            
                        except Exception as e:
                            skipped += 1
                            print(f"  ⚠️  Error processing action: {e}")
                            continue
                
                except Exception as e:
                    print(f"error in transaction: {e}")
                    skipped += 1
                    continue
            
            # Check if we've reached the last batch (fewer results than batch_size)
            if batch_count < batch_size:
                # This was the last batch
                break
            
            # Move to next batch
            offset += batch_size
        
        if total_fetched > 0:
            print(f"  ✅ Fetched {total_fetched} total transactions")
        
        # Batch insert all collected rows
        if rows_to_insert:
            print(f"  💾 Inserting {len(rows_to_insert)} waiver transaction rows...")
            cursor.executemany('''
                INSERT OR IGNORE INTO waiver_activity 
                (transaction_id, year, transaction_date, team_id, team_name, action_type, player_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', rows_to_insert)
            inserted_count = len(rows_to_insert)
        else:
            inserted_count = 0
        
        conn.commit()
        print(f"  ✅ Inserted {inserted_count} waiver transaction rows")
        if skipped > 0:
            print(f"  ⚠️  Skipped {skipped} transactions due to errors")
        
    except Exception as e:
        print(f"  ❌ Error populating waiver activity: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    # conn = create_database()
    # clear_database(conn)
    # conn.close()
    
    # Years to populate
    years = [2020, 2021, 2022, 2024, 2025]
    
    # Create database (if it doesn't exist) - just to ensure tables exist
    print("="*60)
    print("DATABASE POPULATION SCRIPT")
    print("="*60)
    print("Initializing database...")
    # conn = create_database()
    # conn.close()  # Close immediately since populate_weekly_data creates its own connection
    
    # Optionally clear existing data (uncomment to reset database)
    # print("\n⚠️  WARNING: Clearing existing data...")
    # conn = create_database()
    # clear_database(conn)
    # conn.close()
    # print("✅ Database cleared\n")
    
    # Populate database for each year
    for year in years:
        print(f"\n{'='*60}")
        print(f"PROCESSING YEAR: {year}")
        print(f"{'='*60}\n")
        try:
            populate_weekly_data(year)
            
        except Exception as e:
            print(f"\n❌ Error processing year {year}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  Continuing with next year...\n")
        
        # Populate waiver activity for this year
        try:
            populate_waiver_activity(year)
        except Exception as e:
            print(f"\n❌ Error populating waiver activity for {year}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  Continuing with next year...\n")
    
    print("\n" + "="*60)
    print("✅ DATABASE POPULATION COMPLETE")
    print("="*60)
