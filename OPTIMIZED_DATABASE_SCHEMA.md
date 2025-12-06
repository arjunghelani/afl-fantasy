# Optimized Database Schema for Multi-League Scalability

## Core Concept

**Separate NFL performance data (league-agnostic) from fantasy league-specific data**

- **NFL Performance** (same across all leagues): z-scores, player_totals, weekly_points
- **League-Specific**: team ownership, draft, trades, waivers

---

## Proposed Schema Structure

### **Tier 1: NFL Performance Data (No `league_id`)**

These tables store calculated metrics that are **identical across all leagues** since they're based on NFL performance.

#### 1. `nfl_weekly_points`
**Purpose**: Raw weekly PPR fantasy points (NFL performance, not league-specific)

```sql
CREATE TABLE nfl_weekly_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,
    fantasy_pos TEXT NOT NULL,
    week INTEGER NOT NULL,
    weekly_points_ppr REAL NOT NULL,
    year INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_name, week, year)
)
```

**Indexes**:
```sql
CREATE INDEX idx_nfl_weekly_player_year ON nfl_weekly_points(player_name, year, week);
CREATE INDEX idx_nfl_weekly_year_week ON nfl_weekly_points(year, week, fantasy_pos);
```

**Data**: One row per player per week per year (regardless of league)
- Josh Allen Week 1 2024 = 24.5 points (same for all leagues)

---

#### 2. `nfl_z_scores`
**Purpose**: Calculated z-scores per week (NFL performance, not league-specific)

```sql
CREATE TABLE nfl_z_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,
    fantasy_pos TEXT NOT NULL,
    week INTEGER NOT NULL,
    weekly_points_ppr REAL NOT NULL,
    log_ppr REAL NOT NULL,
    z_week_ppr REAL NOT NULL,
    year INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_name, week, year)
)
```

**Indexes**:
```sql
CREATE INDEX idx_nfl_z_player_year ON nfl_z_scores(player_name, year, week);
CREATE INDEX idx_nfl_z_year_pos ON nfl_z_scores(year, fantasy_pos, z_week_ppr);
```

**Data**: One row per player per week per year
- Josh Allen Week 1 2024 z-score = 1.23 (same for all leagues)

---

#### 3. `nfl_player_totals`
**Purpose**: Season aggregates (NFL performance, not league-specific)

```sql
CREATE TABLE nfl_player_totals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,
    fantasy_pos TEXT NOT NULL,
    total_points REAL NOT NULL,
    pos_rank INTEGER NOT NULL,
    overall_rank INTEGER NOT NULL,
    vorp_star REAL NOT NULL,  -- ZAV (sum of z_week_ppr)
    year INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_name, year)
)
```

**Indexes**:
```sql
CREATE INDEX idx_nfl_totals_year_vorp ON nfl_player_totals(year, vorp_star DESC);
CREATE INDEX idx_nfl_totals_year_pos ON nfl_player_totals(year, fantasy_pos, pos_rank);
```

**Data**: One row per player per year
- Josh Allen 2024: total_points=392.5, vorp_star=12.45 (same for all leagues)

---

### **Tier 2: League-Specific Data (With `league_id`)**

These tables store data that **varies by league**.

#### 4. `league_team_mapping`
**Purpose**: Maps which fantasy team owned each player each week

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

**Indexes**:
```sql
CREATE INDEX idx_team_mapping_league_year ON league_team_mapping(league_id, year, week);
CREATE INDEX idx_team_mapping_player ON league_team_mapping(league_id, player_name, year);
CREATE INDEX idx_team_mapping_team ON league_team_mapping(league_id, team_id, year, week);
```

**Data**: One row per player per week per league per year
- League 86952922: Josh Allen Week 1 2024 → Team A
- League 12345678: Josh Allen Week 1 2024 → Team B (different owner!)

**This is the key join table** that links NFL performance to league-specific ownership.

---

#### 5. `league_draft`
**Purpose**: Draft picks per league

```sql
CREATE TABLE league_draft (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    player_name TEXT NOT NULL,
    round_num INTEGER,
    pick_num INTEGER,
    overall_pick INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, year, player_name)  -- Player can only be drafted once per league per year
)
```

**Indexes**:
```sql
CREATE INDEX idx_draft_league_year ON league_draft(league_id, year, overall_pick);
CREATE INDEX idx_draft_player ON league_draft(league_id, player_name, year);
```

---

#### 6. `league_trades`
**Purpose**: Trade transactions per league

```sql
CREATE TABLE league_trades (
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
    zav_to_new_team REAL DEFAULT 0.0,  -- Calculated from nfl_z_scores
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, trade_id, player_name)
)
```

**Indexes**:
```sql
CREATE INDEX idx_trades_league_year ON league_trades(league_id, year, week);
CREATE INDEX idx_trades_player ON league_trades(league_id, player_name, year);
```

---

#### 7. `league_waivers`
**Purpose**: Waiver transactions per league

```sql
CREATE TABLE league_waivers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    transaction_id INTEGER,
    year INTEGER NOT NULL,
    transaction_date TIMESTAMP,
    team_id INTEGER,
    team_name TEXT,
    action_type TEXT NOT NULL,  -- "ADD" or "DROP"
    player_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, transaction_id)  -- Prevent duplicate transactions
)
```

**Indexes**:
```sql
CREATE INDEX idx_waivers_league_year ON league_waivers(league_id, year, transaction_date DESC);
CREATE INDEX idx_waivers_player ON league_waivers(league_id, player_name, year);
```

---

#### 8. `player_headshots`
**Purpose**: Player images (could be league-agnostic, but keeping league_id for flexibility)

```sql
CREATE TABLE player_headshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,  -- Could be NULL for global, but keeping for now
    player_name TEXT NOT NULL,
    headshot_url TEXT,
    nfl_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, player_name)
)
```

---

## Query Examples

### Get Player ZAV for a Specific League

```sql
SELECT 
    nfl.player_name,
    nfl.fantasy_pos,
    nfl.vorp_star,
    nfl.total_points,
    nfl.pos_rank,
    nfl.overall_rank
FROM nfl_player_totals nfl
WHERE nfl.year = 2024
ORDER BY nfl.vorp_star DESC;
```

### Get Player Weekly Stats with Team Ownership

```sql
SELECT 
    nfl.player_name,
    nfl.week,
    nfl.weekly_points_ppr,
    nfl.z_week_ppr,
    mapping.team_name,
    mapping.team_id
FROM nfl_z_scores nfl
LEFT JOIN league_team_mapping mapping
    ON nfl.player_name = mapping.player_name
    AND nfl.week = mapping.week
    AND nfl.year = mapping.year
    AND mapping.league_id = 86952922
WHERE nfl.year = 2024
    AND nfl.player_name = 'Josh Allen'
ORDER BY nfl.week;
```

### Get Trade Analysis with ZAV

```sql
SELECT 
    t.player_name,
    t.from_team_name,
    t.to_team_name,
    t.week,
    t.zav_to_new_team,
    nfl.vorp_star as season_zav
FROM league_trades t
JOIN nfl_player_totals nfl
    ON t.player_name = nfl.player_name
    AND t.year = nfl.year
WHERE t.league_id = 86952922
    AND t.year = 2024
ORDER BY t.week;
```

---

## Benefits of This Structure

### 1. **Massive Storage Reduction**
- **Before**: 100 leagues × 500 players × 17 weeks = 850,000 z-score rows
- **After**: 500 players × 17 weeks = 8,500 z-score rows + 100 leagues × 500 players × 17 weeks = 850,000 team_mapping rows
- **Savings**: ~99% reduction in z-score storage (only store once, not per league)

### 2. **Faster Updates**
- Calculate z-scores **once per year** (not per league)
- Only update `league_team_mapping` when new leagues are added
- Parallel processing: Calculate NFL metrics once, then map to all leagues

### 3. **Easier Maintenance**
- Single source of truth for player performance
- Fix calculation bugs once, affects all leagues
- Historical data preserved (can add new leagues without recalculating)

### 4. **Better Query Performance**
- Smaller tables = faster queries
- Indexes on both sides of JOINs
- Can cache NFL performance data (rarely changes)

---

## Migration Strategy

### Phase 1: Create New Tables
1. Create `nfl_*` tables (no league_id)
2. Create `league_*` tables (with league_id)
3. Create `league_team_mapping` table

### Phase 2: Populate NFL Tables
1. Extract unique player/week/year combinations from current `z_scores`
2. Insert into `nfl_z_scores` (deduplicated)
3. Aggregate into `nfl_player_totals`

### Phase 3: Populate League Tables
1. For each league:
   - Extract team ownership from current `z_scores.fantasy_team`
   - Insert into `league_team_mapping`
   - Copy draft, trades, waivers to new `league_*` tables

### Phase 4: Update Application Code
1. Update queries to JOIN `nfl_*` with `league_team_mapping`
2. Update data population scripts
3. Test with existing leagues

### Phase 5: Drop Old Tables
1. After verification, drop old `z_scores`, `player_totals` tables
2. Keep `weekly_points` as backup or migrate to `nfl_weekly_points`

---

## Alternative: Hybrid Approach

If you want to keep some league-specific calculations (e.g., different scoring systems):

```sql
-- Keep league_id in z_scores but make it optional
CREATE TABLE nfl_z_scores (
    ...
    league_id INTEGER NULL,  -- NULL = global, non-NULL = league-specific override
    ...
)
```

This allows:
- Default: Use global z-scores (NULL league_id)
- Override: League-specific z-scores (if scoring differs)

---

## Recommendations

1. **Start with the separated structure** (nfl_* + league_*)
2. **Use `league_team_mapping` as the join key** for all queries
3. **Calculate NFL metrics once per year** (background job)
4. **Update team mappings incrementally** (only when leagues change)
5. **Consider partitioning by year** if database gets very large

This structure scales to **thousands of leagues** without duplicating calculated metrics!

