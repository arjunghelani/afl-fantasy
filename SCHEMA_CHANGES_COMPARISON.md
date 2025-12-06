# Database Schema Changes: Current vs. Optimized

## Current Schema (6 Tables)

---

### 1. `weekly_points` - **NO CHANGES NEEDED** ✅
**Current Schema:**
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

**Status**: Can be migrated to `nfl_weekly_points` (remove `league_id`) OR kept as-is for backup.

---

### 2. `z_scores` - **MAJOR CHANGE** 🔄
**Current Schema:**
```sql
CREATE TABLE z_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,        -- ❌ REMOVE
    player_name TEXT NOT NULL,
    fantasy_pos TEXT NOT NULL,
    week INTEGER NOT NULL,
    weekly_points_ppr REAL NOT NULL,
    log_ppr REAL NOT NULL,
    z_week_ppr REAL NOT NULL,
    year INTEGER NOT NULL,
    fantasy_team TEXT,                 -- ❌ REMOVE (moved to league_team_mapping)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**New Schema** (`nfl_z_scores`):
```sql
CREATE TABLE nfl_z_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,          -- ✅ KEEP
    fantasy_pos TEXT NOT NULL,          -- ✅ KEEP
    week INTEGER NOT NULL,              -- ✅ KEEP
    weekly_points_ppr REAL NOT NULL,    -- ✅ KEEP
    log_ppr REAL NOT NULL,              -- ✅ KEEP
    z_week_ppr REAL NOT NULL,           -- ✅ KEEP
    year INTEGER NOT NULL,              -- ✅ KEEP
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_name, week, year)     -- ✅ NEW: Prevent duplicates
)
```

**Changes:**
- ❌ **REMOVE**: `league_id` (no longer needed - same z-scores for all leagues)
- ❌ **REMOVE**: `fantasy_team` (moved to `league_team_mapping` table)
- ✅ **ADD**: `UNIQUE(player_name, week, year)` constraint

---

### 3. `player_totals` - **MAJOR CHANGE** 🔄
**Current Schema:**
```sql
CREATE TABLE player_totals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,        -- ❌ REMOVE
    player_name TEXT NOT NULL,
    fantasy_pos TEXT NOT NULL,
    total_points REAL NOT NULL,
    pos_rank INTEGER NOT NULL,
    overall_rank INTEGER NOT NULL,
    vorp_star REAL NOT NULL,
    year INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**New Schema** (`nfl_player_totals`):
```sql
CREATE TABLE nfl_player_totals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,          -- ✅ KEEP
    fantasy_pos TEXT NOT NULL,          -- ✅ KEEP
    total_points REAL NOT NULL,         -- ✅ KEEP
    pos_rank INTEGER NOT NULL,          -- ✅ KEEP
    overall_rank INTEGER NOT NULL,      -- ✅ KEEP
    vorp_star REAL NOT NULL,           -- ✅ KEEP
    year INTEGER NOT NULL,             -- ✅ KEEP
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- ✅ NEW: Track updates
    UNIQUE(player_name, year)           -- ✅ NEW: One row per player per year
)
```

**Changes:**
- ❌ **REMOVE**: `league_id` (no longer needed - same totals for all leagues)
- ✅ **ADD**: `updated_at` timestamp
- ✅ **ADD**: `UNIQUE(player_name, year)` constraint

---

### 4. `waiver_activity` - **MINOR CHANGES** ➕
**Current Schema:**
```sql
CREATE TABLE waiver_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,        -- ✅ KEEP
    transaction_id INTEGER,            -- ✅ KEEP
    year INTEGER NOT NULL,              -- ✅ KEEP
    transaction_date TIMESTAMP,        -- ✅ KEEP
    team_id INTEGER,                   -- ✅ KEEP
    team_name TEXT,                    -- ✅ KEEP
    action_type TEXT NOT NULL,         -- ✅ KEEP
    player_name TEXT NOT NULL,         -- ✅ KEEP
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**New Schema** (`league_waivers`):
```sql
CREATE TABLE league_waivers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,        -- ✅ KEEP
    transaction_id INTEGER,            -- ✅ KEEP
    year INTEGER NOT NULL,              -- ✅ KEEP
    week INTEGER,                       -- ✅ NEW: Week of transaction
    transaction_date TIMESTAMP,        -- ✅ KEEP
    team_id INTEGER,                   -- ✅ KEEP
    team_name TEXT,                    -- ✅ KEEP
    action_type TEXT NOT NULL,         -- ✅ KEEP
    player_name TEXT NOT NULL,         -- ✅ KEEP
    zav_value REAL DEFAULT NULL,       -- ✅ NEW (optional): ZAV at transaction time
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, transaction_id)  -- ✅ NEW: Prevent duplicate transactions
)
```

**Changes:**
- ✅ **ADD**: `week` field (extract from transaction_date)
- ✅ **ADD**: `zav_value` field (optional, for analysis)
- ✅ **ADD**: `UNIQUE(league_id, transaction_id)` constraint
- 📝 **RENAME**: `waiver_activity` → `league_waivers` (for consistency)

---

### 5. `player_trades` - **NO SCHEMA CHANGES** ✅ (but calculation changes)
**Current Schema:**
```sql
CREATE TABLE player_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,        -- ✅ KEEP
    year INTEGER NOT NULL,              -- ✅ KEEP
    week INTEGER NOT NULL,              -- ✅ KEEP
    player_name TEXT NOT NULL,         -- ✅ KEEP
    from_team_id INTEGER NOT NULL,    -- ✅ KEEP
    from_team_name TEXT NOT NULL,     -- ✅ KEEP
    to_team_id INTEGER NOT NULL,      -- ✅ KEEP
    to_team_name TEXT NOT NULL,       -- ✅ KEEP
    trade_id TEXT NOT NULL,           -- ✅ KEEP
    zav_to_new_team REAL DEFAULT 0.0, -- ✅ KEEP (but calculation changes)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**New Schema** (`league_trades`):
```sql
CREATE TABLE league_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,        -- ✅ KEEP
    year INTEGER NOT NULL,              -- ✅ KEEP
    week INTEGER NOT NULL,              -- ✅ KEEP
    player_name TEXT NOT NULL,         -- ✅ KEEP
    from_team_id INTEGER NOT NULL,    -- ✅ KEEP
    from_team_name TEXT NOT NULL,     -- ✅ KEEP
    to_team_id INTEGER NOT NULL,      -- ✅ KEEP
    to_team_name TEXT NOT NULL,       -- ✅ KEEP
    trade_id TEXT NOT NULL,           -- ✅ KEEP
    zav_to_new_team REAL DEFAULT 0.0, -- ✅ KEEP (calculation method changes)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, trade_id, player_name)  -- ✅ NEW: Prevent duplicates
)
```

**Changes:**
- ✅ **ADD**: `UNIQUE(league_id, trade_id, player_name)` constraint
- 📝 **RENAME**: `player_trades` → `league_trades` (for consistency)
- 🔄 **CALCULATION CHANGE**: `zav_to_new_team` now calculated via JOIN with `league_team_mapping` instead of `z_scores.fantasy_team`

---

### 6. `headshots` - **NO CHANGES** ✅ (optional: make league_id nullable)
**Current Schema:**
```sql
CREATE TABLE headshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,        -- ⚠️  Could be NULL for global
    player_name TEXT NOT NULL,
    headshot_url TEXT,
    nfl_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, player_name)
)
```

**New Schema** (`player_headshots`):
```sql
CREATE TABLE player_headshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER,                  -- ⚠️  OPTIONAL: NULL = global, non-NULL = league-specific
    player_name TEXT NOT NULL,
    headshot_url TEXT,
    nfl_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, player_name)     -- ✅ KEEP
)
```

**Changes:**
- ⚠️  **OPTIONAL**: Make `league_id` nullable (NULL = global headshots, non-NULL = league-specific)
- 📝 **RENAME**: `headshots` → `player_headshots` (for consistency)

---

## NEW TABLE: `league_team_mapping` ➕

**Purpose**: Maps which fantasy team owned each player each week (replaces `z_scores.fantasy_team`)

**Schema:**
```sql
CREATE TABLE league_team_mapping (
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
```

**Indexes:**
```sql
CREATE INDEX idx_team_mapping_league_year ON league_team_mapping(league_id, year, week);
CREATE INDEX idx_team_mapping_player ON league_team_mapping(league_id, player_name, year);
CREATE INDEX idx_team_mapping_team ON league_team_mapping(league_id, team_id, year, week);
```

---

## Summary Table

| Table | Schema Changes? | Changes |
|-------|----------------|---------|
| `weekly_points` | ❌ No | Can migrate to `nfl_weekly_points` (remove `league_id`) or keep as backup |
| `z_scores` | ✅ **YES** | Remove `league_id`, remove `fantasy_team`, add `UNIQUE` constraint → becomes `nfl_z_scores` |
| `player_totals` | ✅ **YES** | Remove `league_id`, add `updated_at`, add `UNIQUE` constraint → becomes `nfl_player_totals` |
| `waiver_activity` | ✅ **YES** | Add `week`, add `zav_value` (optional), add `UNIQUE` constraint → becomes `league_waivers` |
| `player_trades` | ⚠️  **MINOR** | Add `UNIQUE` constraint, calculation method changes → becomes `league_trades` |
| `headshots` | ⚠️  **OPTIONAL** | Make `league_id` nullable (optional) → becomes `player_headshots` |
| **NEW** `league_team_mapping` | ➕ **NEW TABLE** | New table to replace `z_scores.fantasy_team` |

---

## Migration Impact

### Tables Requiring Data Migration:
1. **`z_scores`** → `nfl_z_scores`
   - Extract unique `(player_name, week, year)` combinations
   - Remove `league_id` and `fantasy_team` columns
   - Deduplicate (one row per player/week/year)

2. **`player_totals`** → `nfl_player_totals`
   - Extract unique `(player_name, year)` combinations
   - Remove `league_id` column
   - Deduplicate (one row per player/year)

3. **`z_scores.fantasy_team`** → `league_team_mapping`
   - Extract all `(league_id, player_name, week, year, fantasy_team)` combinations
   - Map `fantasy_team` name to `team_id` and `team_name`
   - Insert into new `league_team_mapping` table

4. **`waiver_activity`** → `league_waivers`
   - Add `week` column (calculate from `transaction_date`)
   - Optionally calculate `zav_value`
   - Add `UNIQUE` constraint

5. **`player_trades`** → `league_trades`
   - Recalculate `zav_to_new_team` using new JOIN method
   - Add `UNIQUE` constraint

### Tables with No Data Changes:
- `headshots` → `player_headshots` (optional rename, optional nullable `league_id`)

---

## Key Takeaways

1. **3 tables get major changes**: `z_scores`, `player_totals`, `waiver_activity`
2. **1 new table created**: `league_team_mapping` (replaces `z_scores.fantasy_team`)
3. **1 table calculation changes**: `player_trades.zav_to_new_team` (same schema, different calculation)
4. **2 tables mostly unchanged**: `weekly_points`, `headshots`

The main optimization is **separating NFL performance data (league-agnostic) from league-specific ownership data**.

