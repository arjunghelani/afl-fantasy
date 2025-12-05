import pandas as pd
import numpy as np
import sqlite3
import duckdb
import os
from populate_weekly_db import create_database, clear_database, _get_league
from trade_analysis import build_ownership_timeseries, guess_max_week
from espn_api.football import League
import datetime
import hashlib
import nfl_data_py as nfl
try:
    from fuzzywuzzy import fuzz, process
except ImportError:
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        fuzz = None
        process = None

def create_database(clear=False):
    """Create SQLite database with tables for weekly data"""
    conn = sqlite3.connect('weekly_fantasy_data.db')
    cursor = conn.cursor()
    
    # If clear=True, drop all existing tables
    if clear:
        print("  🗑️  Clearing existing tables...")
        tables_to_drop = [
            'weekly_points',
            'z_scores',
            'player_totals',
            'waiver_activity',
            'player_trades',
            'headshots'
        ]
        
        for table_name in tables_to_drop:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"    ✅ Dropped table: {table_name}")
            except Exception as e:
                print(f"    ⚠️  Could not drop {table_name}: {e}")
        
        conn.commit()
        print("  ✅ All tables cleared")
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weekly_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id INTEGER NOT NULL,
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
            league_id INTEGER NOT NULL,
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
            league_id INTEGER NOT NULL,
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
            league_id INTEGER NOT NULL,
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

# Draft players/data
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

def populate_waiver_activity(year, league_id=None):
    
    """
    Populate waiver_activity table with transactions from ESPN API.
    Gets all waiver adds and drops for the given year.
    """
    print(f"📋 Populating waiver activity for {year}...")
    
    conn = sqlite3.connect('weekly_fantasy_data.db')
    cursor = conn.cursor()
    
    player_names = []
    
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
        
        # Get league_id if not provided
        if league_id is None:
            league_id = getattr(league, 'league_id', None) or getattr(league, 'id', None)
            if league_id is None:
                from trade_analysis import LEAGUE_ID
                league_id = LEAGUE_ID
        
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
                    
                    # Get transaction date (convert from milliseconds to Unix timestamp in seconds)
                    transaction_date = None
                    transaction_timestamp = None
                    if transaction.date:
                        try:
                            # Convert milliseconds to seconds (Unix timestamp)
                            transaction_timestamp = int(transaction.date)
                            timestamp_seconds = transaction_timestamp / 1000
                            dt = datetime.datetime.fromtimestamp(timestamp_seconds)
                            transaction_date = datetime.datetime.strftime(dt, '%Y-%m-%d %H:%M:%S')
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
                                league_id,
                                transaction_id,
                                year,
                                transaction_date,
                                team_id,
                                team_name,
                                action_type,
                                player_name
                            ))
                            
                            player_names.append(player_name)
                            
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
                (league_id, transaction_id, year, transaction_date, team_id, team_name, action_type, player_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
        return player_names    

def get_draft_player_names(year):
    """
    Get list of player names from draft data.
    
    Returns a list of unique player names from the draft.
    """
    draft_data = get_draft_data(year)
    draft_player_names = []
    for pick in draft_data:
        player_name = pick['player_name'].replace('*', '').strip()
        if player_name:
            draft_player_names.append(player_name)
    return list(set(draft_player_names))  # Return unique names
    
def get_trade_player_names(year):
    """
    Get list of player names from trade analysis (players on rosters each week).
    
    Returns a list of unique player names from weekly rosters.
    """
    try:
        league = _get_league(year)
        max_week = guess_max_week(league)
        weeks = list(range(1, max_week + 1))
        owner_by_player, player_meta, team_meta = build_ownership_timeseries(league, weeks)
        
        # Extract player names from player_meta
        roster_player_names = []
        for player_id, meta in player_meta.items():
            player_name = meta.get('name', None)
            if player_name:
                player_name = player_name.replace('*', '').strip()
                if player_name:
                    roster_player_names.append(player_name)
        return list(set(roster_player_names))  # Return unique names
    except Exception as e:
        print(f"    ⚠️  Error getting roster players: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_waiver_player_names(year):
    """
    Get list of player names from waiver activity.
    Note: This will also populate the waiver_activity table in the database.
    
    Returns a list of unique player names from waiver transactions.
    """
    waiver_player_names = populate_waiver_activity(year)
    return list(set(waiver_player_names))  # Return unique names

def collect_all_player_names(year):
    """
    Collect player names from multiple sources:
    1. Draft players
    2. Players from trade analysis (players who were on rosters each week)
    3. Waiver activity players
    
    Returns a set of unique player names
    """
    print(f"📋 Collecting player names from multiple sources for {year}...")
    player_names = set()
    
    # Source 1: Draft players
    print("  📝 Getting draft players...")
    draft_player_names = get_draft_player_names(year)
    player_names.update(draft_player_names)
    print(f"    ✅ Found {len(draft_player_names)} draft players")
    
    # Source 2: Players from trade analysis (players on rosters each week)
    print("  📝 Getting players from weekly rosters (trade analysis)...")
    roster_player_names = get_trade_player_names(year)
    player_names.update(roster_player_names)
    print(f"    ✅ Found {len(roster_player_names)} players from weekly rosters")

    # Source 3: Waiver activity
    print("  📝 Getting players from waiver activity...")
    waiver_player_names = get_waiver_player_names(year)
    player_names.update(waiver_player_names)
    print(f"    ✅ Found {len(waiver_player_names)} players from waiver activity")
    
    print(f"  ✅ Total unique players: {len(player_names)}")
    return player_names

def get_player_team_mapping(year, player_names):
    """
    Get which fantasy team each player was on each week by analyzing box scores.
    Also captures PPR points from box scores as a backup data source.
    
    Args:
        year: The year/season to analyze
        player_names: Set or list of player names to map (from collect_all_player_names)
    
    Returns:
        tuple: (team_mapping, points_mapping)
        - team_mapping: {player_name: {week: team_name}}
        - points_mapping: {player_name: {week: points}} (from box scores)
    """
    print(f"📊 Getting player-to-team mapping and points from box scores for {year}...")
    league = _get_league(year)
    
    # Convert player_names to set for faster lookup
    player_names_set = set(player_names) if isinstance(player_names, (list, set)) else player_names
    
    # Initialize mappings
    player_team_map = {}  # {player_name: {week: team_name}}
    player_points_map = {}  # {player_name: {week: points}} (from box scores)
    
    # Get max week - cap at 12 for 2025
    guessed_max_week = guess_max_week(league)
    if year == 2025:
        max_week = 12  # Hard cap at week 12 for 2025
    else:
        max_week = min(guessed_max_week, 17)
        weeks = list(range(1, max_week + 1))
    
    print(f"  📅 Processing {len(weeks)} weeks (1-{max_week})")
    
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
                        
                        # DEBUG: Print data for Ashton Jeanty
                        if 'Ashton Jeanty' in player_name or player_name == 'Ashton Jeanty':
                            print(f"  🔍 [BOX SCORES] Found Ashton Jeanty in week {week} home lineup:")
                            print(f"      Player name: {player_name}")
                            print(f"      Team: {home_team_name}")
                            player_points = getattr(player, 'points', None)
                            print(f"      Points: {player_points}")
                        
                        # Only process if player is in our player_names set
                        if player_name in player_names_set:
                            if player_name not in player_team_map:
                                player_team_map[player_name] = {}
                            if player_name not in player_points_map:
                                player_points_map[player_name] = {}
                            
                            # Get team name
                            player_team_map[player_name][week] = home_team_name
                            
                            # Get points from box score
                            player_points = getattr(player, 'points', None)
                            if player_points is not None:
                                player_points_map[player_name][week] = float(player_points)
                            
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
                        
                        # DEBUG: Print data for Ashton Jeanty
                        if player_name == 'Ashton Jeanty':
                            print(f"  🔍 [BOX SCORES] Found Ashton Jeanty in week {week} away lineup:")
                            print(f"      Player name: {player_name}")
                            print(f"      Team: {away_team_name}")
                            player_points = getattr(player, 'points', None)
                            print(f"      Points: {player_points}")
                        
                        # Only process if player is in our player_names set
                        if player_name in player_names_set:
                            if player_name not in player_team_map:
                                player_team_map[player_name] = {}
                            if player_name not in player_points_map:
                                player_points_map[player_name] = {}
                            
                            # Get team name
                            player_team_map[player_name][week] = away_team_name
                            
                            # Get points from box score
                            player_points = getattr(player, 'points', None)
                            if player_points is not None:
                                player_points_map[player_name][week] = float(player_points)
                            
                            week_count += 1
        
        if week_count > 0:
            print(f"    Week {week}: Mapped {week_count} player-team relationships")
    
    # Summary
    total_players = len(player_team_map)
    total_weeks_mapped = sum(len(weeks) for weeks in player_team_map.values())
    total_points_captured = sum(len(weeks) for weeks in player_points_map.values())
    print(f"  ✅ Mapped {total_players} players across {total_weeks_mapped} player-week combinations")
    print(f"  ✅ Captured {total_points_captured} player-week points from box scores")
    
    # DEBUG: Print summary for Ashton Jeanty
    if 'Ashton Jeanty' in player_team_map:
        print(f"  🔍 [BOX SCORES SUMMARY] Ashton Jeanty found in team_mapping:")
        print(f"      Weeks: {list(player_team_map['Ashton Jeanty'].keys())}")
        print(f"      Teams: {list(player_team_map['Ashton Jeanty'].values())}")
        if 'Ashton Jeanty' in player_points_map:
            print(f"      Points by week: {player_points_map['Ashton Jeanty']}")
    else:
        print(f"  🔍 [BOX SCORES SUMMARY] Ashton Jeanty NOT found in team_mapping")
        print(f"      Is in player_names_set: {'Ashton Jeanty' in player_names_set}")
    
    return player_team_map, player_points_map

def calculate_z_scores_for_players(year, player_names):
    """
    Calculate z-scores for all players in the list for the given year.
    Uses the same method as populate_weekly_db.py:
    1. Get weekly stats for all players
    2. Calculate rankings (total points, pos_rank, overall_rank)
    3. Log transform points
    4. Calculate z-scores by position
    
    Args:
        year: The year/season to analyze
        player_names: Set or list of player names to calculate z-scores for
    
    Returns:
        DataFrame: z_scores DataFrame with columns: player_name, fantasy_pos, week, 
                   weekly_points_ppr, log_ppr, z_week_ppr, pos_rank, overall_rank
    """
    print(f"📊 Calculating z-scores for {len(player_names)} players for {year}...")
    
    # Import get_weekly_stats_for_players from populate_weekly_db
    from populate_weekly_db import get_weekly_stats_for_players
    
    # Get team mapping and points from box scores
    print("  📊 Getting team mapping and points from box scores...")
    team_mapping, box_score_points = get_player_team_mapping(year, player_names)
    
    # Get weekly stats for all players (from player_info.stats)
    print("  📈 Getting weekly stats for players (from player_info.stats)...")
    weekly_df = get_weekly_stats_for_players(year, player_names)
    
    if weekly_df.empty:
        print("  ⚠️  No weekly stats found from player_info. Using box score data only.")
        weekly_df = pd.DataFrame(columns=['player_name', 'fantasy_pos', 'week', 'weekly_points_ppr'])
    
    # Clean and deduplicate
    weekly_df = weekly_df.drop_duplicates(subset=['player_name', 'week'])
    print(f"  ✅ Got {len(weekly_df)} player-week records from player_info.stats")
    
    # Convert box score points to DataFrame
    print("  🔄 Converting box score points to DataFrame...")
    box_score_rows = []
    
    # Get player positions from weekly_df to add to box_score data
    player_positions_from_stats = weekly_df.groupby('player_name')['fantasy_pos'].first().to_dict() if not weekly_df.empty else {}
    
    for player_name, weeks_dict in box_score_points.items():
        # Get position from weekly_df if available, otherwise will be filled later
        position = player_positions_from_stats.get(player_name, None)
        for week, points in weeks_dict.items():
            box_score_rows.append({
                'player_name': player_name,
                'week': week,
                'weekly_points_ppr': points,
                'fantasy_pos': position,  # May be None, will be filled later
                'source': 'box_score'  # Track source
            })
    box_score_df = pd.DataFrame(box_score_rows)
    
    if not box_score_df.empty:
        print(f"  ✅ Got {len(box_score_df)} player-week records from box scores")
    
    # Combine both sources: start with box_score (left side), merge player_info to fill gaps
    print("  🔀 Combining data from both sources...")
    
    # DEBUG: Check Ashton Jeanty in both DataFrames before merge
    if not weekly_df.empty:
        ashton_weekly = weekly_df[weekly_df['player_name'].str.contains('Ashton Jeanty', case=False, na=False)]
        if len(ashton_weekly) > 0:
            print(f"  🔍 [COMBINE] Ashton Jeanty in weekly_df (player_info.stats):")
            for _, row in ashton_weekly.iterrows():
                print(f"      Week {row['week']}: {row['weekly_points_ppr']} points, Position: {row['fantasy_pos']}")
        else:
            print(f"  🔍 [COMBINE] Ashton Jeanty NOT in weekly_df")
    
    if not box_score_df.empty:
        ashton_box = box_score_df[box_score_df['player_name'].str.contains('Ashton Jeanty', case=False, na=False)]
        if len(ashton_box) > 0:
            print(f"  🔍 [COMBINE] Ashton Jeanty in box_score_df:")
            for _, row in ashton_box.iterrows():
                print(f"      Week {row['week']}: {row['weekly_points_ppr']} points, Position: {row['fantasy_pos']}")
        else:
            print(f"  🔍 [COMBINE] Ashton Jeanty NOT in box_score_df")
    
    if not box_score_df.empty and not weekly_df.empty:
        # Start with box_score as the base (left join)
        # Merge player_info data to fill in any gaps where box_score doesn't have data
        combined_df = box_score_df.merge(
            weekly_df[['player_name', 'week', 'weekly_points_ppr', 'fantasy_pos']],
            on=['player_name', 'week'],
            how='outer',
            suffixes=('_box', '_player')
        )
        
        # Prioritize box_score data: use box_score points if available, otherwise use player_info points
        combined_df['weekly_points_ppr'] = combined_df['weekly_points_ppr_box'].fillna(combined_df['weekly_points_ppr_player'])
        
        # Prioritize box_score position if available, otherwise use player_info position
        combined_df['fantasy_pos'] = combined_df['fantasy_pos_box'].fillna(combined_df['fantasy_pos_player'])
        
        # Drop the intermediate columns
        combined_df = combined_df[['player_name', 'week', 'weekly_points_ppr', 'fantasy_pos']].copy()
        
        print(f"  ✅ Combined to {len(combined_df)} unique player-week records (box_score base with player_info fill)")
        
        # DEBUG: Check Ashton Jeanty after merge
        ashton_combined = combined_df[combined_df['player_name'].str.contains('Ashton Jeanty', case=False, na=False)]
        if len(ashton_combined) > 0:
            print(f"  🔍 [COMBINE] Ashton Jeanty in combined_df after merge:")
            for _, row in ashton_combined.iterrows():
                print(f"      Week {row['week']}: {row['weekly_points_ppr']} points, Position: {row['fantasy_pos']}")
        else:
            print(f"  🔍 [COMBINE] Ashton Jeanty NOT in combined_df after merge")
    elif not box_score_df.empty:
        # Only box score data available
        combined_df = box_score_df.drop(columns=['source'])
        print(f"  ✅ Using box score data only ({len(combined_df)} records)")
    else:
        # Only player_info data available
        combined_df = weekly_df.copy()
        print(f"  ✅ Using player_info data only ({len(combined_df)} records)")
    
    # Fill missing weeks with NULL values for all players
    print("  📅 Filling missing weeks with NULL values...")
    league = _get_league(year)
    
    # Determine max week - cap at 12 for 2025
    guessed_max_week = guess_max_week(league)
    if year == 2025:
        max_week = 12  # Hard cap at week 12 for 2025
    else:
        max_week = min(guessed_max_week, 17)
    
    print(f"  📅 Using max_week: {max_week} (guessed: {guessed_max_week})")
    
    # Get all unique players and their positions
    player_positions = combined_df.groupby('player_name')['fantasy_pos'].first().to_dict()
    
    # If we have players from box scores but not in combined_df yet, add them
    for player_name in team_mapping.keys():
        if player_name not in player_positions:
            # Try to get position from player_info or set as unknown
            player_positions[player_name] = 'UNKNOWN'
    
    # Create a complete index of all player-week combinations (only up to max_week)
    all_weeks = list(range(1, max_week + 1))
    all_players = list(player_positions.keys())
    
    # Create MultiIndex for all combinations
    complete_index = pd.MultiIndex.from_product(
        [all_players, all_weeks],
        names=['player_name', 'week']
    )
        
    # Set index on combined_df
    combined_df_indexed = combined_df.set_index(['player_name', 'week'])
    
    # Reindex to include all weeks, filling missing with NaN
    weekly_df_filled = combined_df_indexed.reindex(complete_index)
    
    # Fill in fantasy_pos for missing weeks (same position for all weeks per player)
    # Convert Index to Series so fillna can use it
    player_positions_series = pd.Series(
        weekly_df_filled.index.get_level_values('player_name').map(player_positions),
        index=weekly_df_filled.index
    )
    weekly_df_filled['fantasy_pos'] = weekly_df_filled['fantasy_pos'].fillna(player_positions_series)
    
    # Reset index
    weekly_df = weekly_df_filled.reset_index()
    
    # Fill weekly_points_ppr with NaN (NULL) for missing weeks
    weekly_df['weekly_points_ppr'] = weekly_df['weekly_points_ppr'].fillna(np.nan)
    
    print(f"  ✅ Filled to {len(weekly_df)} player-week records (all players × all weeks)")
    
    # Calculate rankings (sum ignores NULL values by default in SQL)
    print("  📈 Calculating player rankings...")
    result = duckdb.sql("""
        with totals as (
            select player_name, fantasy_pos, 
                   sum(weekly_points_ppr) as total_points,
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
    
    # Skip 60% cutoff - use all players
    print("  📊 Using all players (no cutoff applied)...")
    result2 = result.copy()
    
    # Print number of players per position
    print("  📊 Players per position:")
    pos_counts = result2.groupby('fantasy_pos')['player_name'].nunique().sort_index()
    for pos, count in pos_counts.items():
        print(f"    {pos}: {count} players")
    
    # Filter out NULL and 0 values before log transform (keep NULL rows for later)
    print("  📊 Filtering out zero values (keeping NULL rows)...")
    # Store NULL rows separately
    null_rows = result2[result2['weekly_points_ppr'].isna()].copy()
    # Keep only rows with actual points > 0 for z-score calculation
    result2 = result2[(result2['weekly_points_ppr'].notna()) & (result2['weekly_points_ppr'] > 0)].copy()
    
    # Log transform (handle any remaining edge cases)
    print("  📊 Log transforming points...")
    result2['log_ppr'] = result2.weekly_points_ppr.apply(lambda x: np.log10(max(x, 0.1)) if x > 0 else np.log10(0.1))
    
    # Calculate z-scores
    print("  🎯 Calculating z-scores...")
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
    
    # Add back NULL rows with NULL z-scores
    if len(null_rows) > 0:
        # Add NULL columns for log_ppr and z_week_ppr
        null_rows['log_ppr'] = np.nan
        null_rows['z_week_ppr'] = np.nan
        # Combine with z_scores
        z_scores = pd.concat([z_scores, null_rows], ignore_index=True)
    
    # Add fantasy_team from team_mapping
    print("  🏈 Adding fantasy team information...")
    team_rows = []
    for player_name, weeks_dict in team_mapping.items():
        for week, team_name in weeks_dict.items():
            team_rows.append({
                'player_name': player_name,
                'week': week,
                'fantasy_team': team_name
            })
    
    if team_rows:
        team_df = pd.DataFrame(team_rows)
        
        # DEBUG: Check Ashton Jeanty in team_df before merge
        ashton_team = team_df[team_df['player_name'].str.contains('Ashton Jeanty', case=False, na=False)]
        if len(ashton_team) > 0:
            print(f"  🔍 [FANTASY_TEAM] Ashton Jeanty in team_df before merge:")
            for _, row in ashton_team.iterrows():
                print(f"      Week {row['week']}: {row['fantasy_team']}")
        else:
            print(f"  🔍 [FANTASY_TEAM] Ashton Jeanty NOT in team_df")
        
        # Merge fantasy_team into z_scores
        z_scores = z_scores.merge(team_df, on=['player_name', 'week'], how='left')
        print(f"  ✅ Added fantasy team for {team_df['player_name'].nunique()} players")
        
        # DEBUG: Check Ashton Jeanty in z_scores after merge
        ashton_z = z_scores[z_scores['player_name'].str.contains('Ashton Jeanty', case=False, na=False)]
        if len(ashton_z) > 0:
            print(f"  🔍 [FANTASY_TEAM] Ashton Jeanty in z_scores after merge:")
            for _, row in ashton_z.iterrows():
                print(f"      Week {row['week']}: Points={row.get('weekly_points_ppr')}, ZAV={row.get('z_week_ppr')}, Team={row.get('fantasy_team')}")
        else:
            print(f"  🔍 [FANTASY_TEAM] Ashton Jeanty NOT in z_scores after merge")
    else:
        # Add empty fantasy_team column if no team mapping data
        z_scores['fantasy_team'] = None
        print("  ⚠️  No team mapping data available")
    
    print(f"  ✅ Calculated z-scores for {len(z_scores)} player-week combinations")
    print(f"  ✅ Unique players with z-scores: {z_scores['player_name'].nunique()}")
    print(f"  ✅ Rows with NULL values (bye weeks): {z_scores['weekly_points_ppr'].isna().sum()}")
    
    return z_scores

def update_player_totals_from_z_scores(year, league_id=None, db_path='weekly_fantasy_data.db'):
    """
    Update the player_totals table by aggregating data from z_scores table.
    
    Calculates:
    - total_points: Sum of weekly_points_ppr (excluding week 0)
    - vorp_star: Sum of z_week_ppr (excluding week 0)
    - pos_rank: Rank within position based on vorp_star
    - overall_rank: Overall rank based on vorp_star
    
    Args:
        year: The year/season
        league_id: The league ID (if None, will use default from trade_analysis)
        db_path: Path to the SQLite database
    
    Returns:
        int: Number of rows updated/inserted
    """
    print(f"📊 Updating player_totals from z_scores for {year}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get league_id if not provided
        if league_id is None:
            from trade_analysis import LEAGUE_ID
            league_id = LEAGUE_ID
        
        # Query z_scores to aggregate by player
        query = """
            SELECT 
                player_name,
                fantasy_pos,
                SUM(COALESCE(weekly_points_ppr, 0)) as total_points,
                SUM(COALESCE(z_week_ppr, 0)) as vorp_star
            FROM z_scores
            WHERE year = ? AND league_id = ? AND week != 0
            GROUP BY player_name, fantasy_pos
        """
        
        df = pd.read_sql_query(query, conn, params=[year, league_id])
        
        if df.empty:
            print(f"  ⚠️  No z_scores data found for {year}")
            return 0
        
        # Calculate ranks
        # Positional rank: rank within each position by vorp_star
        df['pos_rank'] = df.groupby('fantasy_pos')['vorp_star'].rank(method='dense', ascending=False).astype(int)
        
        # Overall rank: rank across all players by vorp_star
        df['overall_rank'] = df['vorp_star'].rank(method='dense', ascending=False).astype(int)
        
        # Clear existing data for this year and league
        print(f"  🗑️  Clearing existing player_totals data for {year} (league_id: {league_id})...")
        cursor.execute("DELETE FROM player_totals WHERE year = ? AND league_id = ?", (year, league_id))
        conn.commit()
        print(f"  ✅ Cleared existing data for {year}")
        
        # Prepare rows for insertion
        rows_to_insert = []
        for _, row in df.iterrows():
            rows_to_insert.append((
                league_id,
                str(row['player_name']),
                str(row['fantasy_pos']),
                float(row['total_points']),
                int(row['pos_rank']),
                int(row['overall_rank']),
                float(row['vorp_star']),
                year
            ))
        
        # Batch insert
        print(f"  💾 Inserting {len(rows_to_insert)} player totals records...")
        cursor.executemany('''
            INSERT INTO player_totals 
            (league_id, player_name, fantasy_pos, total_points, pos_rank, overall_rank, vorp_star, year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows_to_insert)
        
        conn.commit()
        inserted_count = len(rows_to_insert)
        print(f"  ✅ Successfully inserted {inserted_count} player totals records")
        
        # # Print top 10 players
        # print(f"\n  🏆 Top 10 Players by VORP*:")
        # top_10 = df.nlargest(10, 'vorp_star')
        # for _, player in top_10.iterrows():
        #     print(f"    {player['player_name']} ({player['fantasy_pos']}): {player['vorp_star']:.3f}")
        
        return inserted_count
        
    except Exception as e:
        print(f"  ❌ Error updating player_totals: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        conn.close()

def write_z_scores_to_db(z_scores_df, year, league_id=None, db_path='weekly_fantasy_data.db'):
    """
    Write z-scores DataFrame to the z_scores table in the database.
    
    Args:
        z_scores_df: DataFrame with columns: player_name, fantasy_pos, week, 
                     weekly_points_ppr, log_ppr, z_week_ppr, pos_rank, overall_rank
        year: The year/season
        league_id: The league ID (if None, will try to extract from DataFrame or use default)
        db_path: Path to the SQLite database
    
    Returns:
        int: Number of rows inserted
    """
    print(f"💾 Writing z-scores to database for {year}...")
    
    if z_scores_df.empty:
        print("  ⚠️  No z-scores data to write")
        return 0
    
    # Get league_id if not provided
    if league_id is None:
        # Try to get from z_scores_df if it has league_id column
        if 'league_id' in z_scores_df.columns:
            league_id = z_scores_df['league_id'].iloc[0] if len(z_scores_df) > 0 else None
        if league_id is None:
            from trade_analysis import LEAGUE_ID
            league_id = LEAGUE_ID
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if table allows NULL values - if not, we may need to update schema
        cursor.execute("PRAGMA table_info(z_scores)")
        columns = cursor.fetchall()
        # Check if weekly_points_ppr, log_ppr, z_week_ppr allow NULL
        # SQLite schema: [cid, name, type, notnull, dflt_value, pk]
        
        # Clear existing data for this year and league
        print(f"  🗑️  Clearing existing z_scores data for {year} (league_id: {league_id})...")
        cursor.execute("DELETE FROM z_scores WHERE year = ? AND league_id = ?", (year, league_id))
        conn.commit()
        print(f"  ✅ Cleared existing data for {year}")
        
        # Prepare rows for insertion
        rows_to_insert = []
        null_count = 0
        
        for _, row in z_scores_df.iterrows():
            # Handle NULL values - convert NaN to None for SQLite
            # Use None for NULL values (SQLite will handle it)
            weekly_points = None if pd.isna(row.get('weekly_points_ppr')) else float(row['weekly_points_ppr'])
            log_ppr = None if pd.isna(row.get('log_ppr')) else float(row['log_ppr'])
            z_week_ppr = None if pd.isna(row.get('z_week_ppr')) else float(row['z_week_ppr'])
            
            if weekly_points is None:
                null_count += 1
            
            # Handle fantasy_team (can be None/NULL)
            fantasy_team = None if pd.isna(row.get('fantasy_team')) else str(row['fantasy_team'])
            
            rows_to_insert.append((
                league_id,  # Add league_id as first value
                str(row['player_name']),
                str(row['fantasy_pos']) if pd.notna(row.get('fantasy_pos')) else 'UNKNOWN',
                int(row['week']),
                weekly_points,  # Can be None for NULL
                log_ppr,  # Can be None for NULL
                z_week_ppr,  # Can be None for NULL
                year,
                fantasy_team  # Can be None for NULL
            ))
        
        # Batch insert
        print(f"  💾 Inserting {len(rows_to_insert)} z-score records ({null_count} with NULL points)...")
        cursor.executemany('''
            INSERT INTO z_scores 
            (league_id, player_name, fantasy_pos, week, weekly_points_ppr, log_ppr, z_week_ppr, year, fantasy_team)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows_to_insert)
        
        conn.commit()
        inserted_count = len(rows_to_insert)
        print(f"  ✅ Successfully inserted {inserted_count} z-score records")
        
        return inserted_count
        
    except sqlite3.IntegrityError as e:
        # If NOT NULL constraint fails, we need to handle it differently
        if "NOT NULL" in str(e):
            print(f"  ⚠️  NOT NULL constraint issue. Updating schema to allow NULL values...")
            try:
                # Try to alter the table to allow NULL (SQLite doesn't support ALTER COLUMN easily)
                # Instead, we'll use 0.0 for NULL values
                conn.rollback()
                rows_to_insert = []
                for _, row in z_scores_df.iterrows():
                    # Use 0.0 for NULL values to satisfy NOT NULL constraint
                    weekly_points = 0.0 if pd.isna(row.get('weekly_points_ppr')) else float(row['weekly_points_ppr'])
                    log_ppr = 0.0 if pd.isna(row.get('log_ppr')) else float(row['log_ppr'])
                    z_week_ppr = 0.0 if pd.isna(row.get('z_week_ppr')) else float(row['z_week_ppr'])
                    fantasy_team = None if pd.isna(row.get('fantasy_team')) else str(row['fantasy_team'])
                    
                    # Get league_id for this row
                    row_league_id = row.get('league_id') if 'league_id' in z_scores_df.columns else league_id
                    
                    rows_to_insert.append((
                        row_league_id,
                        str(row['player_name']),
                        str(row['fantasy_pos']) if pd.notna(row.get('fantasy_pos')) else 'UNKNOWN',
                        int(row['week']),
                        weekly_points,
                        log_ppr,
                        z_week_ppr,
                        year,
                        fantasy_team
                    ))
                
                cursor.executemany('''
                    INSERT INTO z_scores 
                    (league_id, player_name, fantasy_pos, week, weekly_points_ppr, log_ppr, z_week_ppr, year, fantasy_team)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', rows_to_insert)
                conn.commit()
                print(f"  ✅ Successfully inserted {len(rows_to_insert)} z-score records (using 0.0 for NULL values)")
                return len(rows_to_insert)
            except Exception as e2:
                print(f"  ❌ Error after retry: {e2}")
                raise
        else:
            raise
    except Exception as e:
        print(f"  ❌ Error writing z-scores to database: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        conn.close()

def write_trades_to_db(year, league_id=None, db_path='weekly_fantasy_data.db'):
    """
    Write trade information to the database.
    Creates a row for each player traded, with a trade_id to link trades together.
    Calculates the ZAV (z_week_ppr sum) that each player gave to their new team.
    
    Args:
        year: The year/season to analyze
        league_id: The league ID (if None, will use default from trade_analysis)
        db_path: Path to the SQLite database
    
    Returns:
        int: Number of trade records inserted
    """
    print(f"📊 Writing trade data to database for {year}...")
    
    # Import trade_analysis functions
    from trade_analysis import get_league, build_trade_dataframe, guess_max_week, LEAGUE_ID
    
    # Get league_id if not provided
    if league_id is None:
        league_id = LEAGUE_ID
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get trade data using trade_analysis
        print(f"  🔍 Detecting trades for {year}...")
        league = get_league(year)
        trade_df = build_trade_dataframe(league, start_week=1, end_week=None, only_trade_like=True)
        
        if trade_df.empty:
            print(f"  ⚠️  No trades detected for {year}")
            return 0
        
        print(f"  ✅ Detected {len(trade_df)} player movements in trades")
        
        # Generate trade_id for each trade
        # Group by week and team pairs to create unique trade_ids
        print(f"  🔗 Generating trade_ids...")
        
        # Create a unique trade_id for each (week, team_pair) combination
        # Sort team pairs to ensure consistent trade_id regardless of direction
        trade_df['team_pair'] = trade_df.apply(
            lambda row: tuple(sorted([row['from_team_id'], row['to_team_id']])), 
            axis=1
        )
        
        # Create trade_id based on week and team pair
        trade_df['trade_id'] = trade_df.apply(
            lambda row: f"{year}_{row['week']}_{row['team_pair'][0]}_{row['team_pair'][1]}",
            axis=1
        )
        
        # Calculate ZAV for each player to their new team
        print(f"  📈 Calculating ZAV for each player to their new team...")
        
        rows_to_insert = []
        for _, row in trade_df.iterrows():
            player_name = row['player_name']
            week = row['week']
            to_team_name = row['to_team_name']
            to_team_id = row['to_team_id']
            
            # Query z_scores to get ZAV player gave to new team
            # Sum z_week_ppr where player was on the new team FROM the trade week onwards
            zav_query = """
                SELECT SUM(COALESCE(z_week_ppr, 0)) as total_zav
                FROM z_scores
                WHERE player_name = ? 
                  AND year = ? 
                  AND fantasy_team = ? 
                  AND week >= ?
            """
            cursor.execute(zav_query, (player_name, year, to_team_name, week))
            result = cursor.fetchone()
            total_zav = float(result[0]) if result and result[0] is not None else 0.0
            
            rows_to_insert.append((
                league_id,
                year,
                row['week'],
                player_name,
                row['from_team_id'],
                row['from_team_name'],
                to_team_id,
                to_team_name,
                row['trade_id'],
                total_zav
            ))
        
        # Create table if it doesn't exist
        table_name = 'player_trades'
        print(f"  🗄️  Creating/updating table {table_name}...")
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                league_id INTEGER NOT NULL,
                year INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                from_team_id INTEGER NOT NULL,
                from_team_name TEXT NOT NULL,
                to_team_id INTEGER NOT NULL,
                to_team_name TEXT NOT NULL,
                trade_id TEXT NOT NULL,
                zav_to_new_team REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for player_trades table
        cursor.execute(f'''
            CREATE INDEX IF NOT EXISTS idx_player_trades_league_year 
            ON {table_name}(league_id, year, player_name)
        ''')
        
        # Check if zav_to_new_team column exists, add it if not
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        if 'zav_to_new_team' not in columns:
            print(f"  🔧 Adding zav_to_new_team column to {table_name}...")
            cursor.execute(f'''
                ALTER TABLE {table_name} 
                ADD COLUMN zav_to_new_team REAL DEFAULT 0.0
            ''')
            conn.commit()
            print(f"  ✅ Added zav_to_new_team column")
        
        # Clear existing data for this year and league
        print(f"  🗑️  Clearing existing trade data for {year} (league_id: {league_id})...")
        cursor.execute(f"DELETE FROM {table_name} WHERE year = ? AND league_id = ?", (year, league_id))
        conn.commit()
        print(f"  ✅ Cleared existing data for {year}")
        
        # Insert trade data
        print(f"  💾 Inserting {len(rows_to_insert)} trade records...")
        cursor.executemany(f'''
            INSERT INTO {table_name} 
            (league_id, year, week, player_name, from_team_id, from_team_name, to_team_id, to_team_name, trade_id, zav_to_new_team)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows_to_insert)
        
        conn.commit()
        inserted_count = len(rows_to_insert)
        print(f"  ✅ Successfully inserted {inserted_count} trade records")
        
        # Print summary
        unique_trades = trade_df['trade_id'].nunique()
        print(f"  📊 Summary: {unique_trades} unique trades, {inserted_count} players traded")
        
        return inserted_count
        
    except Exception as e:
        print(f"  ❌ Error writing trades to database: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        conn.close()


def clean_player_name(name):
    """
    Clean player name to match the format used in the database.
    Removes asterisks, plus signs, periods, and strips whitespace.
    
    Args:
        name: Player name string
    
    Returns:
        str: Cleaned player name
    """
    if not name:
        return ""
    return str(name).replace("*", "").replace("+", "").replace(".", "").strip()

def populate_headshots(league_id, db_path='weekly_fantasy_data.db'):
    """
    Populate headshots table with player headshot URLs from nfl_data_py.
    
    This function:
    1. Creates the headshots table if it doesn't exist
    2. Imports players from nfl_data_py using import_players()
    3. Gets all unique player names from the database (z_scores table) for the given league_id
    4. Matches player names between nfl_data_py (full names) and database (cleaned names)
    5. Inserts headshot URLs into the headshots table
    
    Args:
        league_id: League ID to filter players by
        db_path: Path to the SQLite database
    
    Returns:
        int: Number of headshots inserted/updated
    """
    print("📸 Populating headshots table...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create headshots table if it doesn't exist
        print("  🗄️  Creating headshots table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS headshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                league_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                headshot_url TEXT,
                nfl_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(league_id, player_name)
            )
        ''')
        conn.commit()
        print("  ✅ Headshots table created/verified")
        
        # Get all unique player names from the database
        print("  📋 Getting unique player names from database...")
        cursor.execute('''
            SELECT DISTINCT player_name 
            FROM z_scores
            WHERE league_id = ?
            ORDER BY player_name
        ''', (league_id,))
        db_player_names = [row[0] for row in cursor.fetchall()]
        print(f"  ✅ Found {len(db_player_names)} unique players in database")
        
        if not db_player_names:
            print("  ⚠️  No players found in database. Skipping headshot population.")
            return 0
        
        # Import players from nfl_data_py
        print("  📡 Importing players from nfl_data_py...")
        try:
            nfl_players_df = nfl.import_players()
            print(f"  ✅ Imported {len(nfl_players_df)} players from nfl_data_py")
        except Exception as e:
            print(f"  ❌ Error importing players from nfl_data_py: {e}")
            import traceback
            traceback.print_exc()
            return 0
        
        if nfl_players_df.empty:
            print("  ⚠️  No players found in nfl_data_py. Skipping headshot population.")
            return 0
        
        # Check if 'headshot' column exists
        if 'headshot' not in nfl_players_df.columns:
            print("  ⚠️  'headshot' column not found in nfl_data_py data. Available columns:")
            print(f"      {list(nfl_players_df.columns)}")
            return 0
        
        # Check if 'name' column exists (for matching)
        if 'display_name' not in nfl_players_df.columns:
            print("  ⚠️  'display_name' column not found in nfl_data_py data. Available columns:")
            print(f"      {list(nfl_players_df.columns)}")
            return 0
        
        # Create a mapping of cleaned nfl names to nfl data
        print("  🔍 Creating name mapping for matching...")
        nfl_name_map = {}
        for _, row in nfl_players_df.iterrows():
            nfl_full_name = str(row.get('display_name', '')).strip()
            if not nfl_full_name:
                continue
            
            # Clean the nfl name to match database format
            cleaned_nfl_name = clean_player_name(nfl_full_name)
            
            # Store mapping: cleaned_name -> (full_name, headshot_url)
            headshot_url = row.get('headshot', None)
            if cleaned_nfl_name:
                # If multiple players have the same cleaned name, keep the one with a headshot
                if cleaned_nfl_name not in nfl_name_map:
                    nfl_name_map[cleaned_nfl_name] = (nfl_full_name, headshot_url)
                elif headshot_url and not nfl_name_map[cleaned_nfl_name][1]:
                    # Update if we found a headshot for this name
                    nfl_name_map[cleaned_nfl_name] = (nfl_full_name, headshot_url)
        
        print(f"  ✅ Created mapping for {len(nfl_name_map)} nfl players")
        
        # Get team logos for D/ST players
        print("  🏈 Getting team logos for D/ST players...")
        team_logo_map = {}
        try:
            team_desc_df = nfl.import_team_desc()
            if not team_desc_df.empty and 'team_name' in team_desc_df.columns:
                for _, row in team_desc_df.iterrows():
                    team_full_name = str(row.get('team_name', '')).strip()
                    if not team_full_name:
                        continue
                    
                    # Extract the last part of the team name (e.g., "Buffalo Bills" -> "Bills")
                    team_name_parts = team_full_name.split()
                    if len(team_name_parts) > 0:
                        # Take the last part (team name, not city)
                        team_name_last = team_name_parts[-1]
                        
                        # Get logo (prefer wikipedia, fallback to espn)
                        logo_url = None
                        if 'team_logo_wikipedia' in team_desc_df.columns:
                            logo_url = row.get('team_logo_wikipedia')
                        if not logo_url or pd.isna(logo_url):
                            if 'team_logo_espn' in team_desc_df.columns:
                                logo_url = row.get('team_logo_espn')
                        
                        if logo_url and pd.notna(logo_url) and logo_url:
                            # Store mapping: last part of team name -> logo URL
                            team_logo_map[team_name_last.lower()] = str(logo_url)
                            # Also store full team name mapping
                            team_logo_map[team_full_name.lower()] = str(logo_url)
                
                print(f"  ✅ Created team logo mapping for {len(team_logo_map)} teams")
            else:
                print("  ⚠️  Could not load team descriptions or team_name column missing")
        except Exception as e:
            print(f"  ⚠️  Error loading team descriptions: {e}")
            import traceback
            traceback.print_exc()
        
        # Match database players with nfl_data_py players
        print("  🔗 Matching database players with nfl_data_py players...")
        rows_to_insert = []
        matched_count = 0
        unmatched_count = 0
        unmatched_players = []  # Track unmatched players for debugging
        dst_matched_count = 0
        
        for db_player_name in db_player_names:
            cleaned_db_name = clean_player_name(db_player_name)
            
            # Check if this is a D/ST player
            is_dst = False
            logo_url = None
            
            # Check for D/ST pattern (e.g., "Buffalo Bills D/ST", "Bills D/ST")
            if 'd/st' in cleaned_db_name.lower() or 'dst' in cleaned_db_name.lower():
                is_dst = True
                # Extract team name from D/ST string
                # Remove "D/ST", "DST", "D/ST", etc. and clean
                dst_name_clean = cleaned_db_name.lower()
                dst_name_clean = dst_name_clean.replace('d/st', '').replace('dst', '').replace('def', '').strip()
                
                # Try to match with team logo map
                # First try exact match with cleaned name
                if dst_name_clean in team_logo_map:
                    logo_url = team_logo_map[dst_name_clean]
                else:
                    # Try matching with last word(s) of the D/ST name
                    dst_parts = dst_name_clean.split()
                    if dst_parts:
                        # Try last word
                        if dst_parts[-1] in team_logo_map:
                            logo_url = team_logo_map[dst_parts[-1]]
                        # Try last two words (e.g., "New York" -> "Giants")
                        elif len(dst_parts) >= 2:
                            last_two = ' '.join(dst_parts[-2:])
                            if last_two in team_logo_map:
                                logo_url = team_logo_map[last_two]
            
            if is_dst and logo_url:
                # D/ST player with logo found
                rows_to_insert.append((
                    league_id,
                    db_player_name,
                    logo_url,
                    f"{db_player_name} (Team Logo)"
                ))
                dst_matched_count += 1
                matched_count += 1
            elif cleaned_db_name in nfl_name_map:
                # Regular player with headshot
                nfl_full_name, headshot_url = nfl_name_map[cleaned_db_name]
                rows_to_insert.append((
                    league_id,
                    db_player_name,  # Use original database name (not cleaned)
                    headshot_url if pd.notna(headshot_url) and headshot_url else None,
                    nfl_full_name
                ))
                matched_count += 1
            else:
                unmatched_count += 1
                unmatched_players.append(db_player_name)
                # Still insert a row with NULL headshot for tracking
                rows_to_insert.append((
                    league_id,
                    db_player_name,
                    None,
                    None
                ))
        
        print(f"  ✅ Matched {matched_count} players total ({matched_count - dst_matched_count} headshots, {dst_matched_count} D/ST logos)")
        if unmatched_count > 0:
            print(f"  ⚠️  {unmatched_count} players could not be matched")
            
            # Use fuzzy matching to find closest matches
            if fuzz and process and nfl_name_map:
                print(f"\n  🔍 DEBUG: Unmatched players with closest fuzzy matches:")
                # Get all cleaned nfl names for fuzzy matching
                nfl_cleaned_names = list(nfl_name_map.keys())
                
                for i, player in enumerate(unmatched_players[:20], 1):
                    cleaned = clean_player_name(player)
                    # Skip D/ST players for fuzzy matching (they should use team logos)
                    if 'd/st' in cleaned.lower() or 'dst' in cleaned.lower() or 'def' in cleaned.lower():
                        print(f"      {i}. '{player}' (cleaned: '{cleaned}') [D/ST - skipped fuzzy match]")
                        continue
                    
                    # Find closest match using fuzzywuzzy
                    try:
                        closest_match, score = process.extractOne(cleaned, nfl_cleaned_names, scorer=fuzz.ratio)
                        if closest_match:
                            # Get the full nfl name for the closest match
                            nfl_full_name, _ = nfl_name_map[closest_match]
                            print(f"      {i}. '{player}' (cleaned: '{cleaned}')")
                            print(f"         → Closest match: '{nfl_full_name}' (cleaned: '{closest_match}') - Score: {score}%")
                        else:
                            print(f"      {i}. '{player}' (cleaned: '{cleaned}') - No close match found")
                    except Exception as e:
                        print(f"      {i}. '{player}' (cleaned: '{cleaned}') - Error in fuzzy match: {e}")
                
                if len(unmatched_players) > 20:
                    print(f"      ... and {len(unmatched_players) - 20} more")
            else:
                # Fallback if fuzzywuzzy is not available
                print(f"\n  🔍 DEBUG: Unmatched players (first 20):")
                for i, player in enumerate(unmatched_players[:20], 1):
                    cleaned = clean_player_name(player)
                    print(f"      {i}. '{player}' (cleaned: '{cleaned}')")
                if len(unmatched_players) > 20:
                    print(f"      ... and {len(unmatched_players) - 20} more")
                if not fuzz:
                    print(f"\n  💡 Tip: Install fuzzywuzzy for better matching: pip install fuzzywuzzy[speedup]")
        
        # Insert or update headshots
        print(f"  💾 Inserting/updating {len(rows_to_insert)} headshot records...")
        cursor.executemany('''
            INSERT OR REPLACE INTO headshots 
            (league_id, player_name, headshot_url, nfl_name, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', rows_to_insert)
        
        conn.commit()
        inserted_count = len(rows_to_insert)
        print(f"  ✅ Successfully inserted/updated {inserted_count} headshot records")
        
        # Print some examples of matched players
        if matched_count > 0:
            print(f"\n  📊 Sample matched players (first 5):")
            cursor.execute('''
                SELECT player_name, nfl_name, headshot_url IS NOT NULL as has_headshot
                FROM headshots
                WHERE headshot_url IS NOT NULL AND league_id = ?
                LIMIT 5
            ''', (league_id,))
            for row in cursor.fetchall():
                print(f"      {row[0]} -> {row[1]} (headshot: {'Yes' if row[2] else 'No'})")
        
        return inserted_count
        
    except Exception as e:
        print(f"  ❌ Error populating headshots: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        conn.close()


def populate_league_data(league_id: int, years: list = [2020, 2021, 2022, 2024, 2025], clear_db: bool = False, status_callback=None):
    """
    Populate database with data for a given league_id across multiple years.
    
    Args:
        league_id: The league ID to populate
        years: List of years to populate (default: [2020, 2021, 2022, 2024, 2025])
        clear_db: Whether to clear database before populating (default: False)
        status_callback: Optional callback function(status_message) for progress updates
    
    Returns:
        dict: Summary of population results
    """
    def update_status(msg):
        if status_callback:
            status_callback(msg)
        print(msg)
    
    update_status("Creating/Initializing Database...")
    create_database(clear=clear_db)
    update_status("✅ Database initialized")
    
    results = {
        'league_id': league_id,
        'years_processed': [],
        'years_failed': [],
        'total_players': 0,
        'total_z_scores': 0,
        'total_player_totals': 0,
        'headshots_count': 0,
        'errors': []
    }
    
    # Populate headshots once (not per year)
    headshots_populated = False
    
    for year in years:
        update_status(f"Starting population for {year}...")
        
        try:
            # Step 2: Collect all players from all sources
            update_status(f"Collecting players for {year}...")
            player_names = collect_all_player_names(year)
            if not player_names:
                update_status(f"⚠️  No players found for {year}. Skipping.")
                results['years_failed'].append({'year': year, 'error': 'No players found'})
                continue
            update_status(f"✅ Collected {len(player_names)} unique players for {year}")
            results['total_players'] += len(player_names)
            
            # Step 3: Calculate z-scores for all players
            update_status(f"Calculating z-scores for {year}...")
            z_scores_df = calculate_z_scores_for_players(year, player_names)
            if z_scores_df.empty:
                update_status(f"⚠️  No z-scores calculated for {year}. Skipping.")
                results['years_failed'].append({'year': year, 'error': 'No z-scores calculated'})
                continue
            update_status(f"✅ Calculated z-scores for {year}")
            
            # Step 4: Write z-scores to database
            update_status(f"Writing z-scores to database for {year}...")
            z_scores_count = write_z_scores_to_db(z_scores_df, year, league_id=league_id)
            update_status(f"✅ Wrote {z_scores_count} z-score records for {year}")
            results['total_z_scores'] += z_scores_count
            
            # Step 5: Update player_totals from z_scores
            update_status(f"Updating player totals for {year}...")
            player_totals_count = update_player_totals_from_z_scores(year, league_id=league_id)
            update_status(f"✅ Updated {player_totals_count} player totals for {year}")
            results['total_player_totals'] += player_totals_count
            
            # Step 6: Populate headshots (once, not per year)
            if not headshots_populated:
                update_status("Populating headshots...")
                headshots_count = populate_headshots(league_id)
                update_status(f"✅ Populated {headshots_count} headshots")
                results['headshots_count'] = headshots_count
                headshots_populated = True
            
            # Step 7: Write trades to database
            update_status(f"Writing trades for {year}...")
            try:
                trades_count = write_trades_to_db(year, league_id=league_id)
                update_status(f"✅ Wrote {trades_count} trade records for {year}")
            except Exception as e:
                error_msg = f"⚠️  Error writing trades for {year}: {e}"
                update_status(error_msg)
                results['errors'].append(error_msg)
                import traceback
                traceback.print_exc()
            
            # Step 8: Populate waiver activity
            update_status(f"Populating waiver activity for {year}...")
            try:
                waiver_player_names = populate_waiver_activity(year, league_id=league_id)
                update_status(f"✅ Populated waiver activity for {year}")
            except Exception as e:
                error_msg = f"⚠️  Error populating waivers for {year}: {e}"
                update_status(error_msg)
                results['errors'].append(error_msg)
                import traceback
                traceback.print_exc()
            
            results['years_processed'].append(year)
            update_status(f"✅ Completed population for {year}")
            
        except Exception as e:
            error_msg = f"❌ Fatal error for {year}: {e}"
            update_status(error_msg)
            results['years_failed'].append({'year': year, 'error': str(e)})
            results['errors'].append(error_msg)
            import traceback
            traceback.print_exc()
            continue  # Continue with next year
    
    update_status("✅ Database population complete!")
    return results


if __name__ == "__main__":
    from trade_analysis import LEAGUE_ID
    
    print("\n" + "="*60)
    print("STEP 1: Creating/Initializing Database")
    print("="*60)
    create_database(clear=True)
    print("✅ Database initialized")
    
    results = populate_league_data(LEAGUE_ID, clear_db=False)
    
    print("\n" + "="*60)
    print("✅ FINAL SUMMARY")
    print("="*60)
    print(f"Years processed: {results['years_processed']}")
    if results['years_failed']:
        print(f"Years failed: {[f['year'] for f in results['years_failed']]}")
    print(f"Total players: {results['total_players']}")
    print(f"Total z-scores: {results['total_z_scores']}")
    print(f"Total player totals: {results['total_player_totals']}")
    print(f"Headshots: {results['headshots_count']}")
