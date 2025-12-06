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

def create_database(clear=False, db_path='weekly_fantasy_data_optimized.db'):
    """Create SQLite database with optimized tables for weekly data"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # If clear=True, drop all existing tables
    if clear:
        print("  🗑️  Clearing existing tables...")
        tables_to_drop = [
            'weekly_points',  # Keep for now (can migrate later)
            'nfl_z_scores',
            'nfl_player_totals',
            'league_waivers',
            'league_trades',
            'league_team_mapping',
            'player_headshots'
        ]
        
        for table_name in tables_to_drop:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"    ✅ Dropped table: {table_name}")
            except Exception as e:
                print(f"    ⚠️  Could not drop {table_name}: {e}")
        
        conn.commit()
        print("  ✅ All tables cleared")
    
    # Create NFL-wide tables (league-agnostic)
    # nfl_z_scores: No league_id, no fantasy_team
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nfl_z_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            fantasy_pos TEXT NOT NULL,
            week INTEGER NOT NULL,
            weekly_points_ppr REAL,
            log_ppr REAL,
            z_week_ppr REAL,
            year INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_name, week, year)
        )
    ''')
    
    # nfl_player_totals: No league_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nfl_player_totals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            fantasy_pos TEXT NOT NULL,
            total_points REAL NOT NULL,
            pos_rank INTEGER NOT NULL,
            overall_rank INTEGER NOT NULL,
            vorp_star REAL NOT NULL,
            year INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_name, year)
        )
    ''')
    
    # Create league-specific tables
    # league_waivers: Has league_id, week, zav_value
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS league_waivers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id INTEGER NOT NULL,
            transaction_id INTEGER,
            year INTEGER NOT NULL,
            week INTEGER,
            transaction_date TIMESTAMP,
            team_id INTEGER,
            team_name TEXT,
            action_type TEXT NOT NULL,
            player_name TEXT NOT NULL,
            zav_value REAL DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(league_id, transaction_id)
        )
    ''')
    
    # league_trades: Has league_id, UNIQUE constraint
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS league_trades (
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(league_id, trade_id, player_name)
        )
    ''')
    
    # league_team_mapping: NEW table to replace z_scores.fantasy_team
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS league_team_mapping (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            week INTEGER NOT NULL,
            year INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(league_id, player_name, week, year)
        )
    ''')
    
    # Create indexes for league_team_mapping
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_team_mapping_league_year 
        ON league_team_mapping(league_id, year, week)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_team_mapping_player 
        ON league_team_mapping(league_id, player_name, year)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_team_mapping_team 
        ON league_team_mapping(league_id, team_id, year, week)
    ''')
    
    # player_headshots: league_id is nullable (NULL = global)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_headshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id INTEGER,
            player_name TEXT NOT NULL,
            headshot_url TEXT,
            nfl_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(league_id, player_name)
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

def populate_waiver_activity(year, league_id=None, db_path='weekly_fantasy_data_optimized.db'):
    """
    Populate league_waivers table with transactions from ESPN API.
    Gets all waiver adds and drops for the given year.
    Now includes week field and optional zav_value.
    """
    print(f"📋 Populating waiver activity for {year}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    player_names = []
    
    try:
        # Clear existing data for this year and league
        print(f"  🗑️  Clearing existing waiver activity data for {year}...")
        if league_id:
            cursor.execute("DELETE FROM league_waivers WHERE year = ? AND league_id = ?", (year, league_id))
        else:
            cursor.execute("DELETE FROM league_waivers WHERE year = ?", (year,))
        conn.commit()
        print(f"  ✅ Cleared existing data for {year}")
        
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
                    # Get transaction date (convert from milliseconds to Unix timestamp in seconds)
                    transaction_date = None
                    transaction_timestamp = None
                    week = None  # Extract week from transaction_date
                    
                    if transaction.date:
                        try:
                            # Convert milliseconds to seconds (Unix timestamp)
                            transaction_timestamp = int(transaction.date)
                            timestamp_seconds = transaction_timestamp / 1000
                            dt = datetime.datetime.fromtimestamp(timestamp_seconds)
                            transaction_date = datetime.datetime.strftime(dt, '%Y-%m-%d %H:%M:%S')
                            
                            # Calculate week from transaction date
                            # Get season start date (typically first Tuesday of September)
                            season_start = datetime.datetime(year, 9, 1)
                            # Find first Tuesday
                            days_until_tuesday = (1 - season_start.weekday()) % 7
                            if days_until_tuesday == 0:
                                days_until_tuesday = 7
                            first_tuesday = season_start + datetime.timedelta(days=days_until_tuesday)
                            
                            # Calculate week number (each week is 7 days)
                            days_diff = (dt - first_tuesday).days
                            if days_diff >= 0:
                                week = (days_diff // 7) + 1
                                # Cap at 18 weeks
                                if week > 18:
                                    week = None
                        except (ValueError, TypeError, OSError) as e:
                            # If conversion fails, skip this transaction's date
                            transaction_date = None
                            transaction_timestamp = None
                            week = None
                    
                    # Get actions from transaction
                    actions = transaction.actions
                    if not actions:
                        skipped += 1
                        print('error in actions')
                        continue
                    
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
                            
                            # Collect row data (zav_value is NULL for now, can be calculated later)
                            rows_to_insert.append((
                                league_id,
                                transaction_id,
                                year,
                                week,  # NEW: week field
                                transaction_date,
                                team_id,
                                team_name,
                                action_type,
                                player_name,
                                None  # zav_value (optional, can be calculated later)
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
                INSERT OR IGNORE INTO league_waivers 
                (league_id, transaction_id, year, week, transaction_date, team_id, team_name, action_type, player_name, zav_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    Note: This will also populate the league_waivers table in the database.
    
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

def get_player_team_mapping(year, player_names, league_id=None):
    """
    Get which fantasy team each player was on each week by analyzing box scores.
    Also captures PPR points from box scores as a backup data source.
    
    Args:
        year: The year/season to analyze
        player_names: Set or list of player names to map (from collect_all_player_names)
        league_id: League ID for storing in league_team_mapping table
    
    Returns:
        tuple: (team_mapping, points_mapping)
        - team_mapping: {player_name: {week: (team_name, team_id)}}
        - points_mapping: {player_name: {week: points}} (from box scores)
    """
    print(f"📊 Getting player-to-team mapping and points from box scores for {year}...")
    league = _get_league(year)
    
    # Get league_id if not provided
    if league_id is None:
        league_id = getattr(league, 'league_id', None) or getattr(league, 'id', None)
        if league_id is None:
            from trade_analysis import LEAGUE_ID
            league_id = LEAGUE_ID
    
    # Convert player_names to set for faster lookup
    player_names_set = set(player_names) if isinstance(player_names, (list, set)) else player_names
    
    # Initialize mappings
    player_team_map = {}  # {player_name: {week: (team_name, team_id)}}
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
                home_team_id = getattr(home_team, 'team_id', None) or getattr(home_team, 'teamId', None)
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
                            if player_name not in player_points_map:
                                player_points_map[player_name] = {}
                            
                            # Get team name and ID
                            player_team_map[player_name][week] = (home_team_name, home_team_id)
                            
                            # Get points from box score
                            player_points = getattr(player, 'points', None)
                            if player_points is not None:
                                player_points_map[player_name][week] = float(player_points)
                            
                            week_count += 1
            
            # Process away lineup
            if hasattr(box, 'away_team') and hasattr(box, 'away_lineup'):
                away_team = box.away_team
                away_team_name = away_team.team_name if hasattr(away_team, 'team_name') else str(away_team)
                away_team_id = getattr(away_team, 'team_id', None) or getattr(away_team, 'teamId', None)
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
                            if player_name not in player_points_map:
                                player_points_map[player_name] = {}
                            
                            # Get team name and ID
                            player_team_map[player_name][week] = (away_team_name, away_team_id)
                            
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
    
    return player_team_map, player_points_map

def write_team_mapping_to_db(team_mapping, year, league_id, db_path='weekly_fantasy_data_optimized.db'):
    """
    Write team mapping data to league_team_mapping table.
    
    Args:
        team_mapping: {player_name: {week: (team_name, team_id)}}
        year: The year/season
        league_id: The league ID
        db_path: Path to the SQLite database
    
    Returns:
        int: Number of rows inserted
    """
    print(f"💾 Writing team mapping to database for {year}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Clear existing data for this year and league
        print(f"  🗑️  Clearing existing team mapping data for {year} (league_id: {league_id})...")
        cursor.execute("DELETE FROM league_team_mapping WHERE year = ? AND league_id = ?", (year, league_id))
        conn.commit()
        print(f"  ✅ Cleared existing data for {year}")
        
        # Prepare rows for insertion
        rows_to_insert = []
        for player_name, weeks_dict in team_mapping.items():
            for week, (team_name, team_id) in weeks_dict.items():
                rows_to_insert.append((
                    league_id,
                    player_name,
                    week,
                    year,
                    team_id if team_id is not None else 0,
                    team_name
                ))
        
        # Batch insert
        print(f"  💾 Inserting {len(rows_to_insert)} team mapping records...")
        cursor.executemany('''
            INSERT OR IGNORE INTO league_team_mapping 
            (league_id, player_name, week, year, team_id, team_name)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', rows_to_insert)
        
        conn.commit()
        inserted_count = len(rows_to_insert)
        print(f"  ✅ Successfully inserted {inserted_count} team mapping records")
        
        return inserted_count
        
    except Exception as e:
        print(f"  ❌ Error writing team mapping to database: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        conn.close()

def calculate_z_scores_for_players(year, player_names):
    """
    Calculate z-scores for all players in the list for the given year.
    Uses the same method as populate_weekly_db.py:
    1. Get weekly stats for all players
    2. Calculate rankings (total points, pos_rank, overall_rank)
    3. Log transform points
    4. Calculate z-scores by position
    
    NOTE: This function no longer includes fantasy_team in the z_scores DataFrame.
    Team mapping is stored separately in league_team_mapping table.
    
    Args:
        year: The year/season to analyze
        player_names: Set or list of player names to calculate z-scores for
    
    Returns:
        DataFrame: z_scores DataFrame with columns: player_name, fantasy_pos, week, 
                   weekly_points_ppr, log_ppr, z_week_ppr, pos_rank, overall_rank
                   (NO fantasy_team column)
    """
    print(f"📊 Calculating z-scores for {len(player_names)} players for {year}...")
    
    # Import get_weekly_stats_for_players from populate_weekly_db
    from populate_weekly_db import get_weekly_stats_for_players
    
    # Get team mapping and points from box scores (for points only, not for z_scores)
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
    
    # NOTE: We no longer add fantasy_team to z_scores DataFrame
    # Team mapping is stored separately in league_team_mapping table
    
    print(f"  ✅ Calculated z-scores for {len(z_scores)} player-week combinations")
    print(f"  ✅ Unique players with z-scores: {z_scores['player_name'].nunique()}")
    print(f"  ✅ Rows with NULL values (bye weeks): {z_scores['weekly_points_ppr'].isna().sum()}")
    
    return z_scores

def update_player_totals_from_z_scores(year, league_id=None, db_path='weekly_fantasy_data_optimized.db'):
    """
    Update the nfl_player_totals table by aggregating data from nfl_z_scores table.
    
    Calculates:
    - total_points: Sum of weekly_points_ppr (excluding week 0)
    - vorp_star: Sum of z_week_ppr (excluding week 0)
    - pos_rank: Rank within position based on vorp_star
    - overall_rank: Overall rank based on vorp_star
    
    NOTE: This now writes to nfl_player_totals (no league_id) and reads from nfl_z_scores (no league_id).
    
    Args:
        year: The year/season
        league_id: The league ID (ignored, kept for compatibility)
        db_path: Path to the SQLite database
    
    Returns:
        int: Number of rows updated/inserted
    """
    print(f"📊 Updating nfl_player_totals from nfl_z_scores for {year}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Query nfl_z_scores to aggregate by player (no league_id filter)
        query = """
            SELECT 
                player_name,
                fantasy_pos,
                SUM(COALESCE(weekly_points_ppr, 0)) as total_points,
                SUM(COALESCE(z_week_ppr, 0)) as vorp_star
            FROM nfl_z_scores
            WHERE year = ? AND week != 0
            GROUP BY player_name, fantasy_pos
        """
        
        df = pd.read_sql_query(query, conn, params=[year])
        
        if df.empty:
            print(f"  ⚠️  No z_scores data found for {year}")
            return 0
        
        # Calculate ranks
        # Positional rank: rank within each position by vorp_star
        df['pos_rank'] = df.groupby('fantasy_pos')['vorp_star'].rank(method='dense', ascending=False).astype(int)
        
        # Overall rank: rank across all players by vorp_star
        df['overall_rank'] = df['vorp_star'].rank(method='dense', ascending=False).astype(int)
        
        # Clear existing data for this year (no league_id filter)
        print(f"  🗑️  Clearing existing nfl_player_totals data for {year}...")
        cursor.execute("DELETE FROM nfl_player_totals WHERE year = ?", (year,))
        conn.commit()
        print(f"  ✅ Cleared existing data for {year}")
        
        # Prepare rows for insertion (no league_id)
        rows_to_insert = []
        for _, row in df.iterrows():
            rows_to_insert.append((
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
            INSERT OR REPLACE INTO nfl_player_totals 
            (player_name, fantasy_pos, total_points, pos_rank, overall_rank, vorp_star, year, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', rows_to_insert)
        
        conn.commit()
        inserted_count = len(rows_to_insert)
        print(f"  ✅ Successfully inserted {inserted_count} player totals records")
        
        return inserted_count
        
    except Exception as e:
        print(f"  ❌ Error updating nfl_player_totals: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        conn.close()

def write_z_scores_to_db(z_scores_df, year, league_id=None, db_path='weekly_fantasy_data_optimized.db'):
    """
    Write z-scores DataFrame to the nfl_z_scores table in the database.
    
    NOTE: This now writes to nfl_z_scores (no league_id, no fantasy_team).
    
    Args:
        z_scores_df: DataFrame with columns: player_name, fantasy_pos, week, 
                     weekly_points_ppr, log_ppr, z_week_ppr, pos_rank, overall_rank
                     (NO league_id, NO fantasy_team)
        year: The year/season
        league_id: The league ID (ignored, kept for compatibility)
        db_path: Path to the SQLite database
    
    Returns:
        int: Number of rows inserted
    """
    print(f"💾 Writing z-scores to database for {year}...")
    
    if z_scores_df.empty:
        print("  ⚠️  No z-scores data to write")
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Clear existing data for this year (no league_id filter)
        print(f"  🗑️  Clearing existing nfl_z_scores data for {year}...")
        cursor.execute("DELETE FROM nfl_z_scores WHERE year = ?", (year,))
        conn.commit()
        print(f"  ✅ Cleared existing data for {year}")
        
        # Prepare rows for insertion (no league_id, no fantasy_team)
        rows_to_insert = []
        null_count = 0
        
        for _, row in z_scores_df.iterrows():
            # Handle NULL values - convert NaN to None for SQLite
            weekly_points = None if pd.isna(row.get('weekly_points_ppr')) else float(row['weekly_points_ppr'])
            log_ppr = None if pd.isna(row.get('log_ppr')) else float(row['log_ppr'])
            z_week_ppr = None if pd.isna(row.get('z_week_ppr')) else float(row['z_week_ppr'])
            
            if weekly_points is None:
                null_count += 1
            
            rows_to_insert.append((
                str(row['player_name']),
                str(row['fantasy_pos']) if pd.notna(row.get('fantasy_pos')) else 'UNKNOWN',
                int(row['week']),
                weekly_points,  # Can be None for NULL
                log_ppr,  # Can be None for NULL
                z_week_ppr,  # Can be None for NULL
                year
            ))
        
        # Batch insert
        print(f"  💾 Inserting {len(rows_to_insert)} z-score records ({null_count} with NULL points)...")
        cursor.executemany('''
            INSERT OR IGNORE INTO nfl_z_scores 
            (player_name, fantasy_pos, week, weekly_points_ppr, log_ppr, z_week_ppr, year)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', rows_to_insert)
        
        conn.commit()
        inserted_count = len(rows_to_insert)
        print(f"  ✅ Successfully inserted {inserted_count} z-score records")
        
        return inserted_count
        
    except Exception as e:
        print(f"  ❌ Error writing z-scores to database: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        conn.close()

def write_trades_to_db(year, league_id=None, db_path='weekly_fantasy_data_optimized.db'):
    """
    Write trade information to the database.
    Creates a row for each player traded, with a trade_id to link trades together.
    Calculates the ZAV (z_week_ppr sum) that each player gave to their new team.
    
    NOTE: This now uses league_team_mapping JOIN with nfl_z_scores instead of z_scores.fantasy_team.
    
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
        # NEW: Use league_team_mapping JOIN with nfl_z_scores
        print(f"  📈 Calculating ZAV for each player to their new team...")
        
        rows_to_insert = []
        for _, row in trade_df.iterrows():
            player_name = row['player_name']
            week = row['week']
            to_team_id = row['to_team_id']
            to_team_name = row['to_team_name']
            
            # Query using JOIN: nfl_z_scores + league_team_mapping
            # Sum z_week_ppr where player was on the new team FROM the trade week onwards
            zav_query = """
                SELECT SUM(COALESCE(zs.z_week_ppr, 0)) as total_zav
                FROM nfl_z_scores zs
                INNER JOIN league_team_mapping ltm 
                    ON zs.player_name = ltm.player_name 
                    AND zs.week = ltm.week 
                    AND zs.year = ltm.year
                WHERE zs.player_name = ? 
                  AND zs.year = ? 
                  AND ltm.league_id = ?
                  AND ltm.team_id = ?
                  AND zs.week >= ?
            """
            cursor.execute(zav_query, (player_name, year, league_id, to_team_id, week))
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
        table_name = 'league_trades'
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(league_id, trade_id, player_name)
            )
        ''')
        
        # Create index for league_trades table
        cursor.execute(f'''
            CREATE INDEX IF NOT EXISTS idx_league_trades_league_year 
            ON {table_name}(league_id, year, player_name)
        ''')
        
        # Clear existing data for this year and league
        print(f"  🗑️  Clearing existing trade data for {year} (league_id: {league_id})...")
        cursor.execute(f"DELETE FROM {table_name} WHERE year = ? AND league_id = ?", (year, league_id))
        conn.commit()
        print(f"  ✅ Cleared existing data for {year}")
        
        # Insert trade data
        print(f"  💾 Inserting {len(rows_to_insert)} trade records...")
        cursor.executemany(f'''
            INSERT OR IGNORE INTO {table_name} 
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

def populate_headshots(league_id, db_path='weekly_fantasy_data_optimized.db'):
    """
    Populate player_headshots table with player headshot URLs from nfl_data_py.
    
    This function:
    1. Creates the player_headshots table if it doesn't exist
    2. Imports players from nfl_data_py using import_players()
    3. Gets all unique player names from the database (nfl_z_scores table)
    4. Matches player names between nfl_data_py (full names) and database (cleaned names)
    5. Inserts headshot URLs into the player_headshots table
    
    NOTE: league_id can be NULL for global headshots, or set for league-specific.
    
    Args:
        league_id: League ID (can be None for global headshots)
        db_path: Path to the SQLite database
    
    Returns:
        int: Number of headshots inserted/updated
    """
    print("📸 Populating headshots table...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create player_headshots table if it doesn't exist
        print("  🗄️  Creating player_headshots table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_headshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                league_id INTEGER,
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
        
        # Get all unique player names from the database (nfl_z_scores, no league_id filter)
        print("  📋 Getting unique player names from database...")
        cursor.execute('''
            SELECT DISTINCT player_name 
            FROM nfl_z_scores
            ORDER BY player_name
        ''')
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
                    league_id,  # Can be None for global
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
                    league_id,  # Can be None for global
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
                    league_id,  # Can be None for global
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
            INSERT OR REPLACE INTO player_headshots 
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
                FROM player_headshots
                WHERE headshot_url IS NOT NULL AND (league_id = ? OR league_id IS NULL)
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


def populate_league_data(league_id: int, years: list = [2020, 2021, 2022, 2024, 2025], clear_db: bool = False, status_callback=None, db_path='weekly_fantasy_data_optimized.db'):
    """
    Populate database with data for a given league_id across multiple years.
    
    NOTE: This now uses the optimized schema:
    - nfl_z_scores (no league_id)
    - nfl_player_totals (no league_id)
    - league_team_mapping (league-specific)
    - league_waivers (league-specific)
    - league_trades (league-specific)
    - player_headshots (league_id can be NULL)
    
    Args:
        league_id: The league ID to populate
        years: List of years to populate (default: [2020, 2021, 2022, 2024, 2025])
        clear_db: Whether to clear database before populating (default: False)
        status_callback: Optional callback function(status_message) for progress updates
        db_path: Path to the database file (default: 'weekly_fantasy_data_optimized.db')
    
    Returns:
        dict: Summary of population results
    """
    def update_status(msg):
        if status_callback:
            status_callback(msg)
        print(msg)
    
    update_status("Creating/Initializing Database...")
    create_database(clear=clear_db, db_path=db_path)
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
    
    # Populate headshots once (not per year) - use NULL league_id for global
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
            
            # Step 3: Calculate z-scores for all players (no league_id, no fantasy_team)
            update_status(f"Calculating z-scores for {year}...")
            z_scores_df = calculate_z_scores_for_players(year, player_names)
            if z_scores_df.empty:
                update_status(f"⚠️  No z-scores calculated for {year}. Skipping.")
                results['years_failed'].append({'year': year, 'error': 'No z-scores calculated'})
                continue
            update_status(f"✅ Calculated z-scores for {year}")
            
            # Step 4: Write z-scores to database (nfl_z_scores, no league_id)
            update_status(f"Writing z-scores to database for {year}...")
            z_scores_count = write_z_scores_to_db(z_scores_df, year, league_id=league_id, db_path=db_path)
            update_status(f"✅ Wrote {z_scores_count} z-score records for {year}")
            results['total_z_scores'] += z_scores_count
            
            # Step 5: Write team mapping to database (league_team_mapping)
            update_status(f"Writing team mapping for {year}...")
            team_mapping, _ = get_player_team_mapping(year, player_names, league_id=league_id)
            team_mapping_count = write_team_mapping_to_db(team_mapping, year, league_id, db_path=db_path)
            update_status(f"✅ Wrote {team_mapping_count} team mapping records for {year}")
            
            # Step 6: Update nfl_player_totals from nfl_z_scores (no league_id)
            update_status(f"Updating player totals for {year}...")
            player_totals_count = update_player_totals_from_z_scores(year, league_id=league_id, db_path=db_path)
            update_status(f"✅ Updated {player_totals_count} player totals for {year}")
            results['total_player_totals'] += player_totals_count
            
            # Step 7: Populate headshots (once, not per year) - use NULL for global
            if not headshots_populated:
                update_status("Populating headshots...")
                headshots_count = populate_headshots(None, db_path=db_path)  # NULL league_id for global headshots
                update_status(f"✅ Populated {headshots_count} headshots")
                results['headshots_count'] = headshots_count
                headshots_populated = True
            
            # Step 8: Write trades to database (league_trades, uses league_team_mapping)
            update_status(f"Writing trades for {year}...")
            try:
                trades_count = write_trades_to_db(year, league_id=league_id, db_path=db_path)
                update_status(f"✅ Wrote {trades_count} trade records for {year}")
            except Exception as e:
                error_msg = f"⚠️  Error writing trades for {year}: {e}"
                update_status(error_msg)
                results['errors'].append(error_msg)
                import traceback
                traceback.print_exc()
            
            # Step 9: Populate waiver activity (league_waivers, with week field)
            update_status(f"Populating waiver activity for {year}...")
            try:
                waiver_player_names = populate_waiver_activity(year, league_id=league_id, db_path=db_path)
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
    
    # Use a separate database file to avoid affecting old data
    OPTIMIZED_DB_PATH = 'weekly_fantasy_data_optimized.db'
    
    print("\n" + "="*60)
    print("STEP 1: Creating/Initializing Database")
    print(f"Using database: {OPTIMIZED_DB_PATH}")
    print("="*60)
    create_database(clear=True, db_path=OPTIMIZED_DB_PATH)
    print("✅ Database initialized")
    
    results = populate_league_data(LEAGUE_ID, clear_db=False, db_path=OPTIMIZED_DB_PATH)
    
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

