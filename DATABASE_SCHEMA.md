# Database Schema Overview

## Database: `weekly_fantasy_data.db` (SQLite)

### Tables Overview

The database contains **6 main tables** that store all fantasy football data:

---

## 1. `weekly_points`
**Purpose**: Stores raw weekly PPR fantasy points for each player

**Schema**:
```sql
CREATE TABLE weekly_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    fantasy_pos TEXT NOT NULL,
    week INTEGER NOT NULL,
    weekly_points_ppr REAL NOT NULL,
    year INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Data Flow**:
- Populated by `populate_weekly_db.py` → `populate_weekly_data()`
- Gets data from ESPN API: `league.box_scores(week=week)` → extracts player points
- One row per player per week per year per league

**Example Data**:
```
league_id: 86952922
player_name: "Josh Allen"
fantasy_pos: "QB"
week: 1
weekly_points_ppr: 24.5
year: 2024
```

---

## 2. `z_scores`
**Purpose**: Stores calculated z-scores (normalized performance) for each player per week

**Schema**:
```sql
CREATE TABLE z_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    fantasy_pos TEXT NOT NULL,
    week INTEGER NOT NULL,
    weekly_points_ppr REAL NOT NULL,
    log_ppr REAL NOT NULL,              -- log10 transform of points
    z_week_ppr REAL NOT NULL,           -- z-score for this week
    year INTEGER NOT NULL,
    fantasy_team TEXT,                  -- which fantasy team owned the player
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Data Flow**:
- Populated by `populate_weekly_db.py` → `populate_weekly_data()`
- Calculates z-scores using:
  1. Log-transform of points: `log10(points)`
  2. Position-based normalization: `(log_ppr - mean) / stddev` per position
- Includes `fantasy_team` which tracks which fantasy team owned the player each week

**Example Data**:
```
league_id: 86952922
player_name: "Josh Allen"
fantasy_pos: "QB"
week: 1
weekly_points_ppr: 24.5
log_ppr: 1.389
z_week_ppr: 1.23
year: 2024
fantasy_team: "Team A"
```

---

## 3. `player_totals`
**Purpose**: Stores aggregated season totals and rankings for each player

**Schema**:
```sql
CREATE TABLE player_totals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    fantasy_pos TEXT NOT NULL,
    total_points REAL NOT NULL,         -- sum of all weekly points
    pos_rank INTEGER NOT NULL,          -- rank within position
    overall_rank INTEGER NOT NULL,       -- rank across all positions
    vorp_star REAL NOT NULL,            -- sum of z_week_ppr (ZAV)
    year INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Data Flow**:
- Populated by `source_players.py` → `update_player_totals_from_z_scores()`
- Aggregates data from `z_scores` table:
  - `total_points` = sum of `weekly_points_ppr`
  - `vorp_star` = sum of `z_week_ppr` (this is the ZAV metric)
  - Ranks calculated per position and overall

**Example Data**:
```
league_id: 86952922
player_name: "Josh Allen"
fantasy_pos: "QB"
total_points: 392.5
pos_rank: 1
overall_rank: 5
vorp_star: 12.45
year: 2024
```

---

## 4. `waiver_activity`
**Purpose**: Stores all waiver wire transactions (adds/drops)

**Schema**:
```sql
CREATE TABLE waiver_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    transaction_id INTEGER,              -- ESPN transaction ID
    year INTEGER NOT NULL,
    transaction_date TIMESTAMP,
    team_id INTEGER,
    team_name TEXT,
    action_type TEXT NOT NULL,           -- "ADD" or "DROP"
    player_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Data Flow**:
- Populated by `source_players.py` → `populate_waiver_activity()`
- Gets data from ESPN API: `league.recent_activity()`
- Filters for waiver transactions (adds/drops)
- Uses `INSERT OR IGNORE` to avoid duplicates

**Example Data**:
```
league_id: 86952922
transaction_id: 12345
year: 2024
transaction_date: "2024-09-15 10:30:00"
team_id: 1
team_name: "Team A"
action_type: "ADD"
player_name: "Gus Edwards"
```

---

## 5. `player_trades`
**Purpose**: Stores all trade transactions between teams

**Schema**:
```sql
CREATE TABLE player_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    week INTEGER NOT NULL,               -- week the trade occurred
    player_name TEXT NOT NULL,
    from_team_id INTEGER NOT NULL,
    from_team_name TEXT NOT NULL,
    to_team_id INTEGER NOT NULL,
    to_team_name TEXT NOT NULL,
    trade_id TEXT NOT NULL,              -- unique trade identifier
    zav_to_new_team REAL DEFAULT 0.0,    -- ZAV after trade to new team
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Index**:
```sql
CREATE INDEX idx_player_trades_league_year 
ON player_trades(league_id, year, player_name)
```

**Data Flow**:
- Populated by `source_players.py` → `write_trades_to_db()`
- Analyzes ESPN transaction history to detect trades
- Calculates `zav_to_new_team` = sum of z-scores after trade

**Example Data**:
```
league_id: 86952922
year: 2024
week: 5
player_name: "Josh Allen"
from_team_id: 1
from_team_name: "Team A"
to_team_id: 2
to_team_name: "Team B"
trade_id: "trade_2024_week5_123"
zav_to_new_team: 8.5
```

---

## 6. `headshots`
**Purpose**: Stores player headshot URLs from NFL data

**Schema**:
```sql
CREATE TABLE headshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    headshot_url TEXT,
    nfl_name TEXT,                       -- matched NFL name
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, player_name)
)
```

**Data Flow**:
- Populated by `source_players.py` → `populate_headshots()`
- Uses fuzzy matching to match fantasy player names to NFL player database
- Gets headshot URLs from `nfl_data_py` library
- One headshot per player per league (UNIQUE constraint)

**Example Data**:
```
league_id: 86952922
player_name: "Josh Allen"
headshot_url: "https://a.espncdn.com/i/headshots/nfl/players/full/3918297.png"
nfl_name: "Josh Allen"
```

---

## Data Relationships

```
weekly_points (raw data)
    ↓
z_scores (calculated metrics)
    ↓
player_totals (aggregated season stats)

waiver_activity (independent transaction log)
player_trades (independent transaction log)
headshots (independent player metadata)
```

## Key Features

1. **Multi-League Support**: All tables include `league_id` to support multiple leagues
2. **Multi-Year Support**: All tables include `year` to support multiple seasons
3. **Indexes**: Created on `(league_id, year, player_name)` for fast queries
4. **Timestamps**: `created_at` tracks when data was inserted

## Data Population Flow

1. **Initial Setup**: `create_database(clear=True)` creates all tables
2. **Weekly Points**: `populate_weekly_data()` → `weekly_points` table
3. **Z-Scores**: Calculated from weekly_points → `z_scores` table
4. **Player Totals**: Aggregated from z_scores → `player_totals` table
5. **Waivers**: `populate_waiver_activity()` → `waiver_activity` table
6. **Trades**: `write_trades_to_db()` → `player_trades` table
7. **Headshots**: `populate_headshots()` → `headshots` table

All orchestrated by `populate_league_data()` in `source_players.py`

