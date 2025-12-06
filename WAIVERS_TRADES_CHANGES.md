# Changes Needed for Waivers and Trades Tables

## Current vs. Optimized Schema

---

## 1. `league_trades` Table

### Current Structure
```sql
CREATE TABLE player_trades (
    league_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    from_team_id INTEGER NOT NULL,
    from_team_name TEXT NOT NULL,
    to_team_id INTEGER NOT NULL,
    to_team_name TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    zav_to_new_team REAL DEFAULT 0.0,  -- Calculated from z_scores.fantasy_team
    ...
)
```

### Changes Needed

#### ✅ **Keep As-Is:**
- All existing fields (league_id, year, week, player_name, teams, trade_id)
- `zav_to_new_team` field (but calculation method changes)

#### 🔄 **Calculation Method Change:**

**Current Calculation** (in `write_trades_to_db`):
```python
# OLD: Queries z_scores table directly
zav_query = """
    SELECT SUM(COALESCE(z_week_ppr, 0)) as total_zav
    FROM z_scores
    WHERE player_name = ? 
      AND year = ? 
      AND fantasy_team = ?  -- Uses fantasy_team column
      AND week >= ?
"""
cursor.execute(zav_query, (player_name, year, to_team_name, week))
```

**New Calculation** (optimized schema):
```python
# NEW: Joins nfl_z_scores with league_team_mapping
zav_query = """
    SELECT SUM(COALESCE(nfl.z_week_ppr, 0)) as total_zav
    FROM nfl_z_scores nfl
    INNER JOIN league_team_mapping mapping
        ON nfl.player_name = mapping.player_name
        AND nfl.week = mapping.week
        AND nfl.year = mapping.year
    WHERE nfl.player_name = ? 
      AND nfl.year = ? 
      AND mapping.league_id = ?
      AND mapping.team_id = ?  -- Use team_id instead of team_name
      AND nfl.week >= ?
"""
cursor.execute(zav_query, (player_name, year, league_id, to_team_id, week))
```

#### 📝 **Recommended Schema:**
```sql
CREATE TABLE league_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    from_team_id INTEGER NOT NULL,
    from_team_name TEXT NOT NULL,
    to_team_id INTEGER NOT NULL,  -- ✅ Use team_id for joins
    to_team_name TEXT NOT NULL,  -- Keep for display
    trade_id TEXT NOT NULL,
    zav_to_new_team REAL DEFAULT 0.0,  -- ✅ Keep as cached value
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, trade_id, player_name)
)
```

**Indexes**:
```sql
CREATE INDEX idx_trades_league_year ON league_trades(league_id, year, week);
CREATE INDEX idx_trades_player ON league_trades(league_id, player_name, year);
CREATE INDEX idx_trades_team ON league_trades(league_id, to_team_id, year);  -- NEW: For zav calculation
```

#### 💡 **Options for `zav_to_new_team`:**

**Option A: Keep as Cached Field** (Recommended)
- ✅ Pros: Fast queries, no JOIN needed for trade analysis
- ✅ Pros: Pre-calculated during data population
- ❌ Cons: Needs recalculation if team_mapping changes
- **Best for**: Most use cases, better performance

**Option B: Calculate On-the-Fly**
- ✅ Pros: Always accurate, no stale data
- ❌ Cons: Requires JOIN on every query
- **Best for**: When team mappings change frequently

**Recommendation**: **Keep as cached field**, recalculate when team_mapping is updated.

---

## 2. `league_waivers` Table

### Current Structure
```sql
CREATE TABLE waiver_activity (
    league_id INTEGER NOT NULL,
    transaction_id INTEGER,
    year INTEGER NOT NULL,
    transaction_date TIMESTAMP,
    team_id INTEGER,
    team_name TEXT,
    action_type TEXT NOT NULL,  -- "ADD" or "DROP"
    player_name TEXT NOT NULL,
    ...
)
```

### Changes Needed

#### ➕ **Add `week` Field** (Recommended)

**Why?**
- Useful for filtering waivers by week
- Enables joins with `league_team_mapping` and `nfl_z_scores`
- Better for time-series analysis
- Can calculate "ZAV value added" for waiver pickups

**New Schema**:
```sql
CREATE TABLE league_waivers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    transaction_id INTEGER,
    year INTEGER NOT NULL,
    week INTEGER,  -- ✅ NEW: Week the transaction occurred
    transaction_date TIMESTAMP,
    team_id INTEGER,
    team_name TEXT,
    action_type TEXT NOT NULL,  -- "ADD" or "DROP"
    player_name TEXT NOT NULL,
    zav_value REAL DEFAULT NULL,  -- ✅ NEW (optional): ZAV value at time of transaction
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, transaction_id)
)
```

**How to get `week`?**
- Extract from `transaction_date` (ESPN provides this)
- Or calculate: `week = (transaction_date - season_start_date) / 7 + 1`
- Or query `league_team_mapping` to find first week player appears on team

#### ➕ **Add `zav_value` Field** (Optional but Useful)

**Purpose**: Track the ZAV value of players when they were added/dropped

**Calculation**:
```python
# For ADD transactions: Get player's ZAV up to that week
zav_query = """
    SELECT SUM(COALESCE(nfl.z_week_ppr, 0)) as total_zav
    FROM nfl_z_scores nfl
    WHERE nfl.player_name = ? 
      AND nfl.year = ? 
      AND nfl.week <= ?  -- Up to transaction week
"""
```

**Use Cases**:
- "Best waiver pickups" (highest ZAV after being added)
- "Worst drops" (high ZAV players that were dropped)
- Trade analysis (compare waiver ZAV vs trade ZAV)

#### 📝 **Recommended Schema:**
```sql
CREATE TABLE league_waivers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    transaction_id INTEGER,
    year INTEGER NOT NULL,
    week INTEGER,  -- ✅ NEW: Week of transaction
    transaction_date TIMESTAMP,
    team_id INTEGER,
    team_name TEXT,
    action_type TEXT NOT NULL,  -- "ADD" or "DROP"
    player_name TEXT NOT NULL,
    zav_value REAL DEFAULT NULL,  -- ✅ NEW (optional): ZAV at transaction time
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, transaction_id)
)
```

**Indexes**:
```sql
CREATE INDEX idx_waivers_league_year ON league_waivers(league_id, year, transaction_date DESC);
CREATE INDEX idx_waivers_player ON league_waivers(league_id, player_name, year);
CREATE INDEX idx_waivers_week ON league_waivers(league_id, year, week);  -- ✅ NEW: For week-based queries
CREATE INDEX idx_waivers_team ON league_waivers(league_id, team_id, year);  -- ✅ NEW: For team analysis
```

---

## 3. Updated Calculation Functions

### Trades: `zav_to_new_team` Calculation

**Current** (`source_players.py` line 1180-1190):
```python
zav_query = """
    SELECT SUM(COALESCE(z_week_ppr, 0)) as total_zav
    FROM z_scores
    WHERE player_name = ? 
      AND year = ? 
      AND fantasy_team = ?  -- ❌ OLD: Uses fantasy_team column
      AND week >= ?
"""
cursor.execute(zav_query, (player_name, year, to_team_name, week))
```

**New** (optimized schema):
```python
zav_query = """
    SELECT SUM(COALESCE(nfl.z_week_ppr, 0)) as total_zav
    FROM nfl_z_scores nfl
    INNER JOIN league_team_mapping mapping
        ON nfl.player_name = mapping.player_name
        AND nfl.week = mapping.week
        AND nfl.year = mapping.year
    WHERE nfl.player_name = ? 
      AND nfl.year = ? 
      AND mapping.league_id = ?
      AND mapping.team_id = ?  -- ✅ NEW: Use team_id from mapping
      AND nfl.week >= ?
"""
cursor.execute(zav_query, (player_name, year, league_id, to_team_id, week))
```

### Waivers: `zav_value` Calculation (if added)

**For ADD transactions**:
```python
# Get player's ZAV up to the transaction week
zav_query = """
    SELECT SUM(COALESCE(nfl.z_week_ppr, 0)) as total_zav
    FROM nfl_z_scores nfl
    WHERE nfl.player_name = ? 
      AND nfl.year = ? 
      AND nfl.week <= ?
"""
cursor.execute(zav_query, (player_name, year, transaction_week))
```

**For DROP transactions**:
```python
# Get player's ZAV after being dropped (what they lost)
zav_query = """
    SELECT SUM(COALESCE(nfl.z_week_ppr, 0)) as total_zav
    FROM nfl_z_scores nfl
    WHERE nfl.player_name = ? 
      AND nfl.year = ? 
      AND nfl.week > ?
"""
cursor.execute(zav_query, (player_name, year, transaction_week))
```

---

## 4. Migration Steps

### Step 1: Add New Fields
```sql
-- Add week to waivers (if not exists)
ALTER TABLE league_waivers ADD COLUMN week INTEGER;

-- Add zav_value to waivers (optional)
ALTER TABLE league_waivers ADD COLUMN zav_value REAL DEFAULT NULL;

-- Update week from transaction_date (if possible)
UPDATE league_waivers 
SET week = CAST((julianday(transaction_date) - julianday(year || '-09-01')) / 7 + 1 AS INTEGER)
WHERE week IS NULL;
```

### Step 2: Update Calculation Functions
- Modify `write_trades_to_db()` to use JOIN with `league_team_mapping`
- Modify `populate_waiver_activity()` to extract `week` and optionally calculate `zav_value`

### Step 3: Add Indexes
```sql
CREATE INDEX idx_trades_team ON league_trades(league_id, to_team_id, year);
CREATE INDEX idx_waivers_week ON league_waivers(league_id, year, week);
CREATE INDEX idx_waivers_team ON league_waivers(league_id, team_id, year);
```

---

## Summary of Changes

### Trades Table:
- ✅ **Keep all existing fields**
- 🔄 **Change `zav_to_new_team` calculation** to use JOIN with `league_team_mapping`
- ✅ **Use `team_id` instead of `team_name`** for joins (more reliable)
- ✅ **Keep `zav_to_new_team` as cached field** (better performance)

### Waivers Table:
- ➕ **Add `week` field** (extract from transaction_date)
- ➕ **Add `zav_value` field** (optional, for analysis)
- ➕ **Add indexes** on `week` and `team_id`

### Benefits:
- ✅ Better query performance with proper indexes
- ✅ Enables week-based analysis
- ✅ Can track ZAV value of waiver transactions
- ✅ Consistent with optimized schema (uses team_mapping for joins)

