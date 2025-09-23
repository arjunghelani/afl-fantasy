from typing import List, Optional
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from espn_api.football import League
from fastapi import Query
import pandas as pd
import re
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta


from vorp import build_vorp_table, build_linear_extrapolated_table


LEAGUE_ID = int(os.getenv("LEAGUE_ID", "86952922"))
SUPPORTED_YEARS = {2020, 2021, 2022, 2024}

app = FastAPI(title="Fantasy League API", version="0.1.0")

# Allow CORS from Next.js dev server
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Models
# ======================

class TeamStanding(BaseModel):
    team_id: int
    team_name: str
    wins: int
    losses: int
    ties: int
    points_for: float
    points_against: float
    win_percentage: float
    streak_length: Optional[int] = None
    streak_type: Optional[str] = None

    # NEW: all-play expected wins (sum of weekly P(win))
    expected_wins: Optional[float] = None

class StandingsResponse(BaseModel):
    year: int
    league_id: int
    num_teams: int
    teams: List[TeamStanding]

class PlayoffGame(BaseModel):
    home_team: str
    away_team: str
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    winner: Optional[str] = None
    round_name: str
    week: Optional[int] = None
    matchup_type: Optional[str] = None

class PlayoffBracket(BaseModel):
    year: int
    league_id: int
    games: List[PlayoffGame]

# --- Draft models ---
class DraftPick(BaseModel):
    year: int
    team_id: int
    team_name: str
    owner: Optional[str] = None
    round_num: Optional[int] = None
    pick_num: Optional[int] = None
    overall_pick: Optional[int] = None
    player_name: str
    position: Optional[str] = None
    pro_team: Optional[str] = None
    is_keeper: bool = False
    auction_price: Optional[float] = None

class DraftResponse(BaseModel):
    year: int
    league_id: int
    picks: List[DraftPick]

# --- Season VORP* models ---
class PlayerVorp(BaseModel):
    player_name: str
    team: Optional[str] = None
    fantasy_pos: str
    g: Optional[int] = None
    fantasy_points_ppr: float
    vorp_star: float
    vorp_star_rank_overall: int
    vorp_star_rank_pos: int
    partial_season: Optional[bool] = None
    vorp_star_extrap: Optional[float] = None

class VorpResponse(BaseModel):
    year: int
    players: List[PlayerVorp]
    count: int
    used_ppg: bool = False

# NEW: Injury extrapolation response models
class ExtrapolatedRow(BaseModel):
    player_name: str
    team: Optional[str] = None
    fantasy_pos: str
    # NEW:
    fantasy_points_ppr: float
    ppr_per_game: Optional[float] = None
    # existing:
    true_vorp_star: float
    delta_vorp_star_mean: float
    delta_vorp_star_p10: float
    delta_vorp_star_p90: float
    adj_vorp_star: float
    weeks_played: Optional[int] = None
    missed_weeks: Optional[int] = None

class ExtrapolatedResponse(BaseModel):
    year: int
    sims: int
    weeks_in_season: int
    count: int
    rows: List[ExtrapolatedRow]

# ======================
# Local cache for ESPN player info (unchanged)
# ======================

PLAYER_CACHE_FILE = Path("./player_cache.json")
PLAYER_CACHE = {}

def load_player_cache() -> None:
    global PLAYER_CACHE
    if PLAYER_CACHE_FILE.exists():
        try:
            data = json.loads(PLAYER_CACHE_FILE.read_text())
            PLAYER_CACHE = {int(k): v for k, v in data.items()}
            print(f"[cache] loaded {len(PLAYER_CACHE)} players")
        except Exception as e:
            print(f"[cache] load failed: {e}")

def save_player_cache() -> None:
    try:
        to_dump = {str(k): v for k, v in PLAYER_CACHE.items()}
        PLAYER_CACHE_FILE.write_text(json.dumps(to_dump))
    except Exception as e:
        print(f"[cache] save failed: {e}")

@app.on_event("startup")
def _startup_cache():
    load_player_cache()

@app.on_event("shutdown")
def _shutdown_cache():
    save_player_cache()

def get_player_info_cached(league: League, player_id: int):
    if not player_id:
        return {"position": None, "proTeam": None, "name": None}
    cached = PLAYER_CACHE.get(player_id)
    if cached is not None:
        return cached
    try:
        pl = league.player_info(playerId=player_id)
        info = {
            "position": getattr(pl, "position", None),
            "proTeam": getattr(pl, "proTeam", None),
            "name": getattr(pl, "name", None),
        }
    except Exception as e:
        print(f"[cache] player_info failed for {player_id}: {e}")
        info = {"position": None, "proTeam": None, "name": None}
    PLAYER_CACHE[player_id] = info
    save_player_cache()
    return info

# ======================
# Health & league helpers
# ======================

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

def _get_league(year: int) -> League:
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")
    espn_s2 = "AEC20e998honXS4Wi0Z8qzlJdam4%2F%2BaApa7apspnhKR0Npb%2FMsF5DuQsFUcHW%2FhPihQun9U6PGITOi2CkbdfDCkUc8druBVhAwT08Lzrvv8oZli8YAuTi9mIWg7YqtorCNOEKPxHpYswnT3q7b885tRDKBJpLCH0T2h4h1p%2B02SfdlDhjEB2gHqFk1xl6tJRNMBiCkZ8i5RttLW6ER9ZvLTmmAdb5ceZhQ27NEMiMf%2BjWSSvwBdnf2roxwt9baw33BVnnITqYVb8FXsaUwm7%2Bm0m9GLQ%2B66%2BU%2Brg%2BQngXm1ekA%3D%3D"
    swid = "{B431504E-F779-4C49-B3E8-28DDF7409957}"
    kwargs = {"league_id": LEAGUE_ID, "year": year, "swid": swid, "espn_s2": espn_s2}
    if espn_s2 and swid:
        kwargs.update({"espn_s2": espn_s2, "swid": swid})
    return League(**kwargs)

def _compute_expected_wins_map(year: int) -> dict[int, float]:
    """
    All-play expected wins:
      For each week, for each team:
        P(win) = (teams with strictly lower score + 0.5 * tied-others) / (N - 1)
      Season Expected Wins = sum of weekly P(win).
    Returns: { team_id -> expected_wins_float }
    """
    league = _get_league(year)

    exp_sum: dict[int, float] = defaultdict(float)

    # Walk weeks; include any week that returns a scoreboard (reg+post if available)
    for wk in range(1, 19):  # 1..18 to cover years with wk18
        try:
            sb = league.scoreboard(week=wk)
        except Exception:
            continue
        if not sb:
            continue

        # Collect (team_id, score) for the week
        week_scores: list[tuple[int, float]] = []
        for m in sb:
            home = getattr(m, "home_team", None)
            away = getattr(m, "away_team", None)
            hs = getattr(m, "home_score", None)
            as_ = getattr(m, "away_score", None)
            if not home or not away or hs is None or as_ is None:
                continue
            try:
                hid = int(getattr(home, "team_id", 0) or 0)
                aid = int(getattr(away, "team_id", 0) or 0)
                hpts = float(hs)
                apts = float(as_)
            except Exception:
                continue
            if hid <= 0 or aid <= 0:
                continue
            week_scores.append((hid, hpts))
            week_scores.append((aid, apts))

        if len(week_scores) < 2:
            continue

        # Build distribution for all-play probabilities
        scores_only = [s for (_, s) in week_scores]
        N = len(scores_only)
        c = Counter(scores_only)
        uniq = sorted(c.keys())
        lower_prefix = {}
        running = 0
        for v in uniq:
            lower_prefix[v] = running
            running += c[v]

        # Assign weekly P(win)
        for tid, s in week_scores:
            lower = lower_prefix.get(s, 0)
            tie_others = c.get(s, 0) - 1
            p_win = (lower + 0.5 * max(tie_others, 0)) / (N - 1)
            exp_sum[tid] += float(p_win)

    # Round lightly for stability in UI
    return {tid: round(w, 3) for tid, w in exp_sum.items()}

# ======================
# Standings / Playoffs / Draft (unchanged)
# ======================

@app.get("/standings/{year}", response_model=StandingsResponse)
def get_standings(year: int) -> StandingsResponse:
    try:
        league = _get_league(year)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # NEW: compute all-play expected wins for this season
    try:
        ew_map = _compute_expected_wins_map(year)  # {team_id -> expected_wins}
    except Exception as e:
        # fail-soft: keep standings working even if ew calc fails
        print(f"[expected_wins] failed for {year}: {e}")
        ew_map = {}

    teams_out: List[TeamStanding] = []
    for t in league.teams:
        wins = getattr(t, "wins", 0)
        losses = getattr(t, "losses", 0)
        ties = getattr(t, "ties", 0)
        total_games = wins + losses + ties
        win_percentage = (wins / total_games * 100) if total_games > 0 else 0.0

        tid = int(getattr(t, "team_id", 0) or 0)
        teams_out.append(
            TeamStanding(
                team_id=tid,
                team_name=getattr(t, "team_name", "Team"),
                wins=wins,
                losses=losses,
                ties=ties,
                points_for=float(getattr(t, "points_for", 0.0)),
                points_against=float(getattr(t, "points_against", 0.0)),
                win_percentage=round(win_percentage, 1),
                streak_length=getattr(t, "streak_length", None),
                streak_type=getattr(t, "streak_type", None),
                expected_wins=ew_map.get(tid),   # NEW
            )
        )

    # Same sort as before
    teams_out.sort(key=lambda x: (x.win_percentage, x.points_for), reverse=True)
    return StandingsResponse(year=year, league_id=LEAGUE_ID, num_teams=len(teams_out), teams=teams_out)


@app.get("/playoffs/{year}", response_model=PlayoffBracket)
def get_playoffs(year: int) -> PlayoffBracket:
    try:
        league = _get_league(year)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    games = []
    try:
        for week in range(1, 18):
            try:
                scoreboard = league.scoreboard(week=week)
                if not scoreboard:
                    continue
                for matchup in scoreboard:
                    matchup_type = getattr(matchup, 'matchup_type', 'NONE')
                    if matchup_type in ['WINNERS_BRACKET', 'LOSERS_CONSOLATION_LADDER', 'WINNERS_CONSOLATION_LADDER']:
                        home_team = getattr(matchup, 'home_team', None)
                        away_team = getattr(matchup, 'away_team', None)
                        home_name = getattr(home_team, 'team_name', 'TBD') if home_team else 'TBD'
                        away_name = getattr(away_team, 'team_name', 'TBD') if away_team else 'TBD'
                        home_score = getattr(matchup, 'home_score', None)
                        away_score = getattr(matchup, 'away_score', None)
                        winner = None
                        if home_score is not None and away_score is not None:
                            winner = home_name if home_score > away_score else away_name
                        round_name = 'Playoff'
                        if matchup_type == 'WINNERS_BRACKET':
                            round_name = 'Winners Bracket'
                        elif matchup_type == 'LOSERS_CONSOLATION_LADDER':
                            round_name = 'Consolation'
                        elif matchup_type == 'WINNERS_CONSOLATION_LADDER':
                            round_name = 'Winners Consolation'

                        games.append(PlayoffGame(
                            home_team=home_name,
                            away_team=away_name,
                            home_score=home_score,
                            away_score=away_score,
                            winner=winner,
                            round_name=round_name,
                            week=week,
                            matchup_type=matchup_type
                        ))
            except Exception:
                continue

        def _label_rounds_from_weeks(games: List[PlayoffGame]) -> None:
            wb_weeks = sorted({g.week for g in games if g.matchup_type == 'WINNERS_BRACKET' and g.week is not None})
            idx_by_week = {wk: i for i, wk in enumerate(wb_weeks)}
            for g in games:
                if g.matchup_type == 'WINNERS_BRACKET' and g.week in idx_by_week:
                    idx = idx_by_week[g.week]
                    if idx == 0: g.round_name = 'Quarterfinals'
                    elif idx == 1: g.round_name = 'Semifinals'
                    elif idx == 2: g.round_name = 'Championship'
                    else: g.round_name = f'Playoffs (Round {idx+1})'
                elif g.matchup_type == 'LOSERS_CONSOLATION_LADDER':
                    g.round_name = 'Consolation'
                elif g.matchup_type == 'WINNERS_CONSOLATION_LADDER':
                    g.round_name = 'Winners Consolation'
                else:
                    g.round_name = g.round_name or 'Playoff'

        if games:
            _label_rounds_from_weeks(games)

        if not games:
            teams = league.teams
            if len(teams) >= 8:
                for j in range(0, len(teams), 2):
                    if j + 1 < len(teams):
                        games.append(PlayoffGame(home_team=teams[j].team_name, away_team=teams[j+1].team_name, round_name='Quarterfinals'))
                games.append(PlayoffGame(home_team="Winner Q1", away_team="Winner Q2", round_name='Semifinals'))
                games.append(PlayoffGame(home_team="Winner Q3", away_team="Winner Q4", round_name='Semifinals'))
                games.append(PlayoffGame(home_team="Winner SF1", away_team="Winner SF2", round_name='Championship'))
            elif len(teams) >= 4:
                for j in range(0, len(teams), 2):
                    if j + 1 < len(teams):
                        games.append(PlayoffGame(home_team=teams[j].team_name, away_team=teams[j+1].team_name, round_name='Semifinals'))
                games.append(PlayoffGame(home_team="Winner SF1", away_team="Winner SF2", round_name='Championship'))
            else:
                if len(teams) >= 2:
                    games.append(PlayoffGame(home_team=teams[0].team_name, away_team=teams[1].team_name, round_name='Championship'))
    except Exception:
        games = [PlayoffGame(home_team="TBD", away_team="TBD", round_name="Playoff")]

    ROUND_ORDER = {
        'Quarterfinals': 1,
        'Semifinals': 2,
        'Championship': 3,
        'Winners Bracket': 50,
        'Winners Consolation': 60,
        'Consolation': 70,
        'Playoff': 90,
    }
    games.sort(key=lambda g: (ROUND_ORDER.get(g.round_name, 999), g.week or 0))
    return PlayoffBracket(year=year, league_id=LEAGUE_ID, games=games)

@app.get("/draft/{year}", response_model=DraftResponse)
def get_draft(year: int) -> DraftResponse:
    try:
        league = _get_league(year)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        raw = getattr(league, "draft", []) or []
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Draft unavailable: {e}")

    team_count = len(league.teams)
    picks_out: List[DraftPick] = []

    for p in raw:
        team_obj = getattr(p, "team", None)
        team_id = int(getattr(team_obj, "team_id", 0) or 0)
        team_name = getattr(team_obj, "team_name", None) or "Team"

        round_num = getattr(p, "round_num", None)
        round_pick = getattr(p, "round_pick", None)
        pick_num = round_pick

        overall_pick = None
        if isinstance(round_num, int) and isinstance(round_pick, int) and team_count:
            overall_pick = (round_num - 1) * team_count + round_pick

        player_id = getattr(p, "playerId", None)
        player_name = (
            getattr(p, "playerName", None)
            or getattr(getattr(p, "player", None), "name", None)
            or "TBD"
        )
        player_name = re.sub(r"[*+.]", "", str(player_name)).strip()
        player_name = re.sub(r"\s+", " ", player_name)

        position = None
        pro_team = None
        if player_id:
            info = get_player_info_cached(league, int(player_id))
            if info:
                position = position or info["position"]
                pro_team = pro_team or info["proTeam"]
                if (not player_name or player_name == "TBD") and info["name"]:
                    player_name = info["name"]

        is_keeper = bool(getattr(p, "keeper", False) or getattr(p, "keeper_status", False))
        auction_price = getattr(p, "auction_price", None)

        picks_out.append(
            DraftPick(
                year=year,
                team_id=team_id,
                team_name=team_name,
                round_num=round_num,
                pick_num=pick_num,
                overall_pick=overall_pick,
                player_name=player_name,
                position=position,
                pro_team=pro_team,
                is_keeper=is_keeper,
                auction_price=auction_price,
            )
        )

    picks_out.sort(key=lambda x: (x.round_num if x.round_num is not None else 999,
                                  x.pick_num if x.pick_num is not None else 999))
    return DraftResponse(year=year, league_id=LEAGUE_ID, picks=picks_out)

# ======================
# Helpers for metrics
# ======================

# NEW: load weekly points for simulation (expects standard columns)
def load_weekly_points(year: int) -> pd.DataFrame:
    
    """
    Try file first, then fallback to ESPN Player.stats if not found.
    """
    import os
    from pathlib import Path

    # 1) file paths (unchanged)
    pattern = os.getenv("WEEKLY_POINTS_PATH")
    candidate_paths = []
    if pattern:
        try:
            candidate_paths.append(Path(pattern.format(year=year)))
        except Exception:
            pass

    candidate_paths += [
        Path(f"./data/weekly_points_{year}.parquet"),
        Path(f"./data/weekly_points_{year}.csv"),
    ]

    df = None
    for p in candidate_paths:
        if isinstance(p, Path) and p.exists():
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
            elif p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            break

    if df is None and pattern and not Path(pattern.format(year=year)).exists():
        # try direct pandas read_* on non-local path (e.g., s3://)
        try:
            path_like = pattern.format(year=year)
            if path_like.lower().endswith(".parquet"):
                df = pd.read_parquet(path_like)
            elif path_like.lower().endswith(".csv"):
                df = pd.read_csv(path_like)
        except Exception:
            df = None

    # 2) ESPN fallback
    if df is None or df.empty or os.getenv("USE_ESPN_WEEKLIES", "0") == "1":
        try:
            df = build_weekly_points_from_espn(year)
            # optional: persist to speed up subsequent requests
            out_path = Path(f"./data/weekly_points_{year}.parquet")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                df.to_parquet(out_path)
            except Exception:
                # parquet may not be available; ignore
                pass
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=501,
                detail=f"Weekly points not found via files or ESPN: {e}"
            )

    # Normalize columns & filter (same as before)
    cols = {c.lower(): c for c in df.columns}
    def _first_present(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    name_col = _first_present("player_name", "player")
    team_col = _first_present("team", "tm")
    pos_col  = _first_present("fantasy_pos", "pos")
    week_col = _first_present("week", "wk")
    pts_col  = _first_present("weekly_points_ppr", "ppr", "fantasy_points_ppr_week")

    missing = [("player_name", name_col), ("team", team_col), ("fantasy_pos", pos_col),
               ("week", week_col), ("weekly_points_ppr", pts_col)]
    missing = [want for (want, got) in missing if got is None]
    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"Weekly points file missing columns: {missing}",
        )

    out = pd.DataFrame({
        "player_name": pd.Series(df[name_col]).astype(str).str.replace(r"[*+.]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip(),
        "team": df[team_col],
        "fantasy_pos": df[pos_col],
        "week": pd.to_numeric(df[week_col], errors="coerce").astype("Int64"),
        "weekly_points_ppr": pd.to_numeric(df[pts_col], errors="coerce").fillna(0.0),
    })

    out = out[out["fantasy_pos"].isin(ALLOWED_POS)].copy()
    out["week"] = out["week"].astype(int)
    return out


# --- NEW: build weekly points straight from ESPN league data ---
ALLOWED_POS = {"QB","RB","WR","TE"}

def _clean_name(n: str) -> str:
    return (
        str(n or "")
        .replace("*","").replace("+","").replace(".","")
        .strip()
    )

def _map_pos(p: str) -> str:
    p = (p or "").upper().strip()
    # espn_api typically uses QB/RB/WR/TE already, but normalize anyway
    if p in ALLOWED_POS: return p
    if p.startswith("QB"): return "QB"
    if p.startswith("RB"): return "RB"
    if p.startswith("WR"): return "WR"
    if p.startswith("TE"): return "TE"
    return ""  # filtered out later



def build_weekly_points_from_espn(year: int) -> pd.DataFrame:
    """
    Assemble per-player per-week PPR points from ESPN.
    We use the league draft to get stable playerIds, then fetch each player's weekly stats.
    Output columns:
      ['player_name','team','fantasy_pos','week','weekly_points_ppr']
    """
    league = _get_league(year)

    # collect drafted playerIds (stable set that covers the guys you care about for WAR)
    draft = getattr(league, "draft", []) or []
    player_ids = {int(getattr(p, "playerId", 0) or 0) for p in draft if getattr(p, "playerId", None)}
    if not player_ids:
        # fallback: walk team rosters
        for t in getattr(league, "teams", []):
            for pl in getattr(t, "roster", []) or []:
                pid = int(getattr(pl, "playerId", 0) or 0)
                if pid: player_ids.add(pid)

    rows = []
    for pid in player_ids:
        try:
            pl = league.player_info(playerId=pid)  # espn_api Player with 'stats'
        except Exception:
            continue

        name = _clean_name(getattr(pl, "name", "") or "")
        pos  = _map_pos(getattr(pl, "position", "") or "")
        team = getattr(pl, "proTeam", None)

        if not name or pos not in ALLOWED_POS:
            continue

        stats = getattr(pl, "stats", {}) or {}
        # stats keys can be ints or strings ('1','2',...); values contain {'points': float, ...}
        for wk_key, wk_blob in stats.items():
            try:
                wk = int(wk_key)
            except Exception:
                continue
            pts = 0.0
            try:
                pts = float((wk_blob or {}).get("points", 0.0) or 0.0)
            except Exception:
                pts = 0.0

            rows.append({
                "player_name": name,
                "team": team,
                "fantasy_pos": pos,
                "week": wk,
                "weekly_points_ppr": pts,
            })

    if not rows:
        raise HTTPException(
            status_code=501,
            detail="ESPN weekly points could not be assembled (no rows). Check league credentials or draft data."
        )

    df = pd.DataFrame(rows)
    # Basic hygiene
    df = df[df["fantasy_pos"].isin(ALLOWED_POS)].copy()
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    df["weekly_points_ppr"] = pd.to_numeric(df["weekly_points_ppr"], errors="coerce").fillna(0.0)
    df = df[df["week"] > 0]
    return df


# ======================
# Season VORP* endpoint (CHANGED: pass league size)
# ======================

@app.get("/metrics/vorp/{year}", response_model=VorpResponse)
def get_vorp(
    year: int,
    use_ppg: bool = Query(False, description="Use points per game"),
    top: int = Query(500, ge=1, le=2000, description="Limit rows"),
):
    try:
        # CHANGED: use real league size for replacement baseline
        try:
            league = _get_league(year)
            team_count = len(league.teams) if getattr(league, "teams", None) else 12
        except Exception:
            team_count = 12

        table = build_vorp_table(
            year=year,
            use_ppg=use_ppg,
            teams=team_count,                                   # CHANGED
            starters_per_team={"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25},  # CHANGED (consistent)
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to build VORP*: {e}")

    if table is None or len(table) == 0:
        return VorpResponse(year=year, players=[], count=0, used_ppg=use_ppg)

    table = table.head(top).copy()

    required = {
        "player_name", "fantasy_pos", "fantasy_points_ppr",
        "vorp_star", "vorp_star_rank_overall", "vorp_star_rank_pos",
    }
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"VORP table missing columns {missing}; available={list(table.columns)}",
        )

    import numpy as np
    for col in ["fantasy_points_ppr", "vorp_star"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)
    for col in ["vorp_star_rank_overall", "vorp_star_rank_pos"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if "g" in table.columns:
        table["g"] = pd.to_numeric(table["g"], errors="coerce")
    else:
        table["g"] = np.nan

    table = table[
        table["player_name"].notna()
        & table["fantasy_pos"].notna()
        & table["vorp_star"].notna()
        & table["vorp_star_rank_overall"].notna()
        & table["vorp_star_rank_pos"].notna()
    ]

    players: List[PlayerVorp] = []
    for row in table.itertuples(index=False):
        try:
            players.append(
                PlayerVorp(
                    player_name=str(getattr(row, "player_name")),
                    team=(None if pd.isna(getattr(row, "team", None)) else str(getattr(row, "team", None))),
                    fantasy_pos=str(getattr(row, "fantasy_pos")),
                    g=(None if pd.isna(getattr(row, "g")) else int(getattr(row, "g"))),
                    fantasy_points_ppr=float(getattr(row, "fantasy_points_ppr")),
                    vorp_star=float(getattr(row, "vorp_star")),
                    vorp_star_rank_overall=int(getattr(row, "vorp_star_rank_overall")),
                    vorp_star_rank_pos=int(getattr(row, "vorp_star_rank_pos")),
                    partial_season=bool(getattr(row, "partial_season", False)),
                    vorp_star_extrap=(
                        None if pd.isna(getattr(row, "vorp_star_extrap", np.nan))
                        else float(getattr(row, "vorp_star_extrap"))
                    ),
                )
            )
        except Exception as e:
            print(f"[vorp] skipped row due to serialization error: {e}")
            continue

    return VorpResponse(year=year, players=players, count=len(players), used_ppg=use_ppg)

# ======================
# NEW: Injury-extrapolated WAR endpoint
# ======================

# @app.get("/metrics/war-extrapolated/{year}", response_model=ExtrapolatedResponse)
@app.get("/metrics/war-extrapolated/{year}", response_model=ExtrapolatedResponse)
def get_war_extrapolated(
    year: int,
    weeks_in_season: int = Query(17, ge=1, le=18, description="Scoring weeks to allocate replacement across."),
    # sims is ignored in linear mode, kept for backward compat with the frontend.
    sims: int = Query(1000, ge=100, le=20000, description="(ignored in linear mode)"),
    pos: Optional[str] = Query(None, description="Filter positions, comma-separated (e.g. 'QB,RB,WR,TE')"),
    limit: int = Query(5000, ge=1, le=200000, description="Row cap for response, applied after compute"),
    # Optional explicit caps; if omitted we apply defaults for selected positions.
    preselect_per_pos: Optional[str] = Query(
        None,
        description="CSV mapping like 'QB=30,RB=75,WR=75,TE=30' to preselect BEFORE compute"
    ),
):
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")

    # League size for baseline/scale
    try:
        league = _get_league(year)
        team_count = len(league.teams) if getattr(league, "teams", None) else 12
    except Exception:
        team_count = 12

    # Parse pos filter
    pos_set = None
    if pos:
        pos_set = {p.strip().upper() for p in pos.split(",") if p.strip()}

    # Build per-position caps
    DEFAULT_CAPS = {"QB": 30, "RB": 75, "WR": 75, "TE": 30}
    caps: dict[str, int]
    if preselect_per_pos:
        try:
            caps = {}
            for part in preselect_per_pos.split(","):
                if not part.strip():
                    continue
                k, v = part.split("=")
                k = k.strip().upper()
                caps[k] = int(v)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid preselect_per_pos format: {e}")
    else:
        # If no explicit caps provided, apply defaults to the selected positions only
        caps = {k: v for k, v in DEFAULT_CAPS.items() if (pos_set is None or k in pos_set)}

    # Linear compute on a PRESELECTED subset
    try:
        table = build_linear_extrapolated_table(
            year=year,
            weeks_in_season=weeks_in_season,
            teams=team_count,
            starters_per_team={"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25},
            pos_filter=pos_set,
            per_pos_caps=caps,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to build linear extrapolated WAR: {e}")

    # Order & apply global response cap
    table = table.sort_values("adj_vorp_star", ascending=False).head(limit)

    # Serialize
    rows: List[ExtrapolatedRow] = []
    for r in table.itertuples(index=False):
        try:
            rows.append(
                ExtrapolatedRow(
                    player_name=str(getattr(r, "player_name")),
                    team=None if pd.isna(getattr(r, "team", None)) else str(getattr(r, "team")),
                    fantasy_pos=str(getattr(r, "fantasy_pos")),
                    fantasy_points_ppr=float(getattr(r, "fantasy_points_ppr", 0.0)),   # NEW
                    ppr_per_game=(None if pd.isna(getattr(r, "ppr_per_game", None)) else float(getattr(r, "ppr_per_game"))),  # NEW
                    true_vorp_star=float(getattr(r, "true_vorp_star", 0.0)),
                    delta_vorp_star_mean=float(getattr(r, "delta_vorp_star_mean", 0.0)),
                    delta_vorp_star_p10=float(getattr(r, "delta_vorp_star_p10", 0.0)),
                    delta_vorp_star_p90=float(getattr(r, "delta_vorp_star_p90", 0.0)),
                    adj_vorp_star=float(getattr(r, "adj_vorp_star", 0.0)),
                    weeks_played=(None if pd.isna(getattr(r, "weeks_played", None)) else int(getattr(r, "weeks_played"))),
                    missed_weeks=(None if pd.isna(getattr(r, "missed_weeks", None)) else int(getattr(r, "missed_weeks"))),
                )
            )

        except Exception:
            continue

    return ExtrapolatedResponse(
        year=year,
        sims=sims,  # kept for schema compatibility; not used in linear mode
        weeks_in_season=weeks_in_season,
        count=len(rows),
        rows=rows,
    )

# ======================
# NEW: Trades endpoint
# ======================

class TradePackage(BaseModel):
    week: int
    team1_id: int
    team1_name: str
    team2_id: int
    team2_name: str
    team1_players: List[str]
    team2_players: List[str]
    total_players: int
    is_trade_like: bool

class TradeSummary(BaseModel):
    trade_week: int
    team_a: str
    team_b: str
    team_a_vorp_received: float
    team_b_vorp_received: float
    net_advantage: float
    winner: str
    players_to_a: str
    players_to_b: str

class TradesResponse(BaseModel):
    year: int
    trade_packages: List[TradePackage]
    trade_summary: List[TradeSummary]
    count: int

# NEW: Scoreboard models
class GameResult(BaseModel):
    week: int
    opponent: str
    score: float
    opponent_score: float
    result: str  # "W" or "L"
    margin: float
    is_playoff: bool = False
    matchup_type: str = "NONE"

class TeamScoreboard(BaseModel):
    team_name: str
    wins: int
    losses: int
    total_points: float
    win_percentage: float
    games: List[GameResult]

class ScoreboardResponse(BaseModel):
    year: int
    teams: List[TeamScoreboard]

# NEW: Matchup detail models
class PlayerScore(BaseModel):
    player_name: str
    position: str
    points: float
    projected_points: float

class TeamRoster(BaseModel):
    team_name: str
    total_score: float
    players: List[PlayerScore]

class MatchupDetail(BaseModel):
    year: int
    week: int
    home_team: TeamRoster
    away_team: TeamRoster
    is_playoff: bool

# Simple in-memory cache for trades data
_trades_cache = {}
_cache_timestamps = {}

@app.get("/trades/{year}", response_model=TradesResponse)
def get_trades(year: int):
    """
    Get trade analysis for a given year.
    This endpoint runs the trade analysis logic from your test.ipynb
    """
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")
    
    # Check cache first (cache for 1 hour)
    cache_key = f"trades_{year}"
    if cache_key in _trades_cache:
        cache_time = _cache_timestamps.get(cache_key)
        if cache_time and datetime.now() - cache_time < timedelta(hours=1):
            return _trades_cache[cache_key]
    
    try:
        # Import the trade analysis functions
        from trade_analysis import run_trade_analysis, get_league, get_team_map
        
        # Try to get basic trade data first (faster)
        try:
            from trade_analysis import get_league, get_team_map, build_ownership_timeseries, detect_transfers, cluster_trades, guess_max_week
            
            league = get_league(year)
            team_meta = get_team_map(league)
            
            # Use the proper trade clustering logic from test.ipynb
            weeks = range(1, guess_max_week(league) + 1)
            owner_by_player, player_meta, team_meta = build_ownership_timeseries(league, weeks)
            changes = detect_transfers(owner_by_player, weeks)
            packages_by_week = cluster_trades(changes, min_players_total=2)
            
            if not packages_by_week:
                return TradesResponse(
                    year=year,
                    trade_packages=[],
                    trade_summary=[],
                    count=0
                )
            
            # Convert packages to trade packages
            trade_packages = []
            trade_summary = []
            
            for week, packages in packages_by_week.items():
                for pkg in packages:
                    if not pkg["is_trade_like"]:
                        continue
                        
                    team_a, team_b = pkg["teams"]
                    team_a_name = team_meta.get(team_a, {}).get("name", f"Team {team_a}")
                    team_b_name = team_meta.get(team_b, {}).get("name", f"Team {team_b}")
                    
                    # Get player names for each direction
                    team_a_players = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["a_to_b"]]
                    team_b_players = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["b_to_a"]]
                    
                    trade_packages.append(TradePackage(
                        week=week,
                        team1_id=team_a,
                        team1_name=team_a_name,
                        team2_id=team_b,
                        team2_name=team_b_name,
                        team1_players=team_a_players,
                        team2_players=team_b_players,
                        total_players=len(team_a_players) + len(team_b_players),
                        is_trade_like=pkg["is_trade_like"]
                    ))
                    
                    trade_summary.append(TradeSummary(
                        trade_week=week,
                        team_a=team_a_name,
                        team_b=team_b_name,
                        team_a_vorp_received=0.0,
                        team_b_vorp_received=0.0,
                        net_advantage=0.0,
                        winner="Tie",
                        players_to_a=", ".join(team_b_players),
                        players_to_b=", ".join(team_a_players)
                    ))
            
            result = TradesResponse(
                year=year,
                trade_packages=trade_packages,
                trade_summary=trade_summary,
                count=len(trade_packages)
            )
            
            # Cache the result
            _trades_cache[cache_key] = result
            _cache_timestamps[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            # Fallback to full analysis if basic method fails
            print(f"Basic trade analysis failed: {e}, falling back to full analysis")
            pass
        
        # Run the full trade analysis as fallback
        trade_df, packages_by_week, player_meta, team_meta = run_trade_analysis(year)
        
        if trade_df.empty:
            return TradesResponse(
                year=year,
                trade_packages=[],
                trade_summary=[],
                count=0
            )
        
        # Convert packages to TradePackage objects
        trade_packages = []
        for week, packages in packages_by_week.items():
            for pkg in packages:
                if not pkg["is_trade_like"]:
                    continue
                    
                team_a, team_b = pkg["teams"]
                team_a_name = team_meta.get(team_a, {}).get("name", f"Team {team_a}")
                team_b_name = team_meta.get(team_b, {}).get("name", f"Team {team_b}")
                
                # Get player names for each team
                team_a_players = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["a_to_b"]]
                team_b_players = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["b_to_a"]]
                
                trade_packages.append(TradePackage(
                    week=week,
                    team1_id=team_a,
                    team1_name=team_a_name,
                    team2_id=team_b,
                    team2_name=team_b_name,
                    team1_players=team_a_players,
                    team2_players=team_b_players,
                    total_players=len(team_a_players) + len(team_b_players),
                    is_trade_like=pkg["is_trade_like"]
                ))
        
        # Create trade summary from packages
        trade_summary = []
        for week, packages in packages_by_week.items():
            for pkg in packages:
                if not pkg["is_trade_like"]:
                    continue
                    
                team_a, team_b = pkg["teams"]
                team_a_name = team_meta.get(team_a, {}).get("name", f"Team {team_a}")
                team_b_name = team_meta.get(team_b, {}).get("name", f"Team {team_b}")
                
                # Get player names for each direction
                players_to_a = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["b_to_a"]]
                players_to_b = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["a_to_b"]]
                
                trade_summary.append(TradeSummary(
                    trade_week=week,
                    team_a=team_a_name,
                    team_b=team_b_name,
                    team_a_vorp_received=0.0,  # We'll calculate this if needed
                    team_b_vorp_received=0.0,  # We'll calculate this if needed
                    net_advantage=0.0,  # We'll calculate this if needed
                    winner="Tie",  # We'll calculate this if needed
                    players_to_a=", ".join(players_to_a) if players_to_a else "None",
                    players_to_b=", ".join(players_to_b) if players_to_b else "None"
                ))
        
        result = TradesResponse(
            year=year,
            trade_packages=trade_packages,
            trade_summary=trade_summary,
            count=len(trade_packages)
        )
        
        # Cache the result
        _trades_cache[cache_key] = result
        _cache_timestamps[cache_key] = datetime.now()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to analyze trades: {e}")

@app.get("/scoreboard/{year}", response_model=ScoreboardResponse)
def get_scoreboard(year: int):
    """
    Get historical scoreboard data for a given year.
    Returns team records and game-by-game results.
    """
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")
    
    try:
        from trade_analysis import get_league, guess_max_week
        
        league = get_league(year)
        max_week = guess_max_week(league)
        
        # Get team data
        teams_data = {}
        for team in league.teams:
            if type(team) != int:
                teams_data[team.team_id] = {
                    'name': team.team_name,
                    'games': [],
                    'wins': 0,
                    'losses': 0,
                    'total_points': 0.0
                }
        
        # Process each week's games
        for week in range(1, max_week + 1):
            try:
                box_scores = league.box_scores(week=week)
                for box in box_scores:
                    # Include both regular season and playoff games
                    # if hasattr(box, 'is_playoff') and box.is_playoff:
                    #     continue
                    home_team = box.home_team
                    away_team = box.away_team
                    home_score = box.home_score
                    away_score = box.away_score
                    
                    # Determine winner and loser
                    if type(home_team) != int and type(away_team) != int:
                        if home_score > away_score:
                            winner_id = home_team.team_id
                            loser_id = away_team.team_id
                            home_result = "W"
                            away_result = "L"
                        else:
                            winner_id = away_team.team_id
                            loser_id = home_team.team_id
                            home_result = "L"
                            away_result = "W"
                        
                        # Add game data for home team
                        if home_team.team_id in teams_data:
                            is_playoff = getattr(box, 'is_playoff', False)
                            game_data = {
                                'week': week,
                                'opponent': away_team.team_name,
                                'score': home_score,
                                'opponent_score': away_score,
                                'result': home_result,
                                'margin': abs(home_score - away_score),
                                'is_playoff': is_playoff,
                                'matchup_type': getattr(box, 'matchup_type', 'NONE')
                            }
                            teams_data[home_team.team_id]['games'].append(game_data)
                            
                            # Only count wins/losses and points for regular season games (week 14 and before)
                            if not is_playoff and week <= 14:
                                teams_data[home_team.team_id]['total_points'] += home_score
                                if home_result == "W":
                                    teams_data[home_team.team_id]['wins'] += 1
                                else:
                                    teams_data[home_team.team_id]['losses'] += 1
                        
                        # Add game data for away team
                        if away_team.team_id in teams_data:
                            is_playoff = getattr(box, 'is_playoff', False)
                            game_data = {
                                'week': week,
                                'opponent': home_team.team_name,
                                'score': away_score,
                                'opponent_score': home_score,
                                'result': away_result,
                                'margin': abs(away_score - home_score),
                                'is_playoff': is_playoff,
                                'matchup_type': getattr(box, 'matchup_type', 'NONE')
                            }
                            teams_data[away_team.team_id]['games'].append(game_data)
                            
                            # Only count wins/losses and points for regular season games (week 14 and before)
                            if not is_playoff and week <= 14:
                                teams_data[away_team.team_id]['total_points'] += away_score
                                if away_result == "W":
                                    teams_data[away_team.team_id]['wins'] += 1
                                else:
                                    teams_data[away_team.team_id]['losses'] += 1
                            
            except Exception as e:
                print(f"Error processing week {week}: {e}")
                continue
        
        # Convert to response format
        teams = []
        for team_id, data in teams_data.items():
            total_games = data['wins'] + data['losses']
            win_percentage = (data['wins'] / total_games * 100) if total_games > 0 else 0.0
            
            # Sort games by week
            games_sorted = sorted(data['games'], key=lambda x: x['week'])
            
            teams.append(TeamScoreboard(
                team_name=data['name'],
                wins=data['wins'],
                losses=data['losses'],
                total_points=round(data['total_points'], 1),
                win_percentage=round(win_percentage, 1),
                games=[GameResult(**game) for game in games_sorted]
            ))
        
        # Sort teams by win percentage (descending)
        teams.sort(key=lambda x: x.win_percentage, reverse=True)
        
        return ScoreboardResponse(year=year, teams=teams)
        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to get scoreboard: {e}")

@app.get("/matchup/{year}/{week}", response_model=MatchupDetail)
def get_matchup_detail(year: int, week: int, team1: str = None, team2: str = None):
    """
    Get detailed matchup information for a specific week and year.
    Returns roster details, player scores, and matchup info.
    
    Args:
        year: The season year
        week: The week number
        team1: First team name (optional, for specific matchup)
        team2: Second team name (optional, for specific matchup)
    """
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")
    
    try:
        from trade_analysis import get_league
        
        league = get_league(year)
        
        # Get box scores for the specific week
        box_scores = league.box_scores(week=week)
        
        if not box_scores:
            raise HTTPException(status_code=404, detail=f"No matchup found for week {week} in {year}")
        
        # Find the specific matchup if team names are provided
        box = None
        if team1 and team2:
            # Look for matchup between the two specific teams
            for b in box_scores:
                home_name = b.home_team.team_name
                away_name = b.away_team.team_name
                if ((home_name == team1 and away_name == team2) or 
                    (home_name == team2 and away_name == team1)):
                    box = b
                    break
        else:
            # Fall back to first matchup if no team names specified
            box = box_scores[0]
        
        if not box:
            raise HTTPException(status_code=404, detail=f"Matchup between {team1} and {team2} not found for week {week} in {year}")
        
        # Skip if it's a playoff game
        if hasattr(box, 'is_playoff') and box.is_playoff:
            raise HTTPException(status_code=404, detail=f"Playoff game not available")
        
        def process_team_roster(lineup):
            """Process team roster and separate starters from bench"""
            starters = []
            bench = []
            
            if not lineup:
                return starters, bench
                
            for player in lineup:
                player_points = getattr(player, 'points', 0.0)
                player_slot = getattr(player, 'slot_position', '')
                player_name = getattr(player, 'name', 'Unknown')
                player_position = getattr(player, 'position', 'Unknown')
                projected_points = getattr(player, 'projected_points', 0.0)
                
                # Use "FLEX" for flex position instead of actual position
                display_position = "FLEX" if player_slot == "FLEX" else player_position
                
                player_data = PlayerScore(
                    player_name=player_name,
                    position=player_slot,
                    points=player_points,
                    projected_points=projected_points
                )
                
                # Check if it's a starter or bench player
                if player_slot in ['QB', 'RB', 'WR', 'TE', 'FLEX', 'K', 'D/ST']:
                    starters.append((player_slot, player_data))
                elif player_slot in ['BE', 'IR']:
                    bench.append(player_data)
                # If slot_position is empty or unclear, check if they scored points
                elif player_points > 0:
                    # Likely a starter if they scored points
                    starters.append((player_slot or 'UNKNOWN', player_data))
                else:
                    # Likely bench if no points
                    bench.append(player_data)
            
            # Sort starters by position order
            position_order = ['QB', 'RB', 'WR', 'TE', 'FLEX', 'K', 'D/ST']
            def sort_key(item):
                slot, player = item
                try:
                    return position_order.index(slot)
                except ValueError:
                    return 999  # Put unknown positions at the end
            
            starters.sort(key=sort_key)
            
            # Extract just the player data in order
            ordered_starters = [player for _, player in starters]
            
            return ordered_starters, bench
        
        # Process both team rosters
        home_starters, home_bench = process_team_roster(box.home_lineup if hasattr(box, 'home_lineup') else None)
        away_starters, away_bench = process_team_roster(box.away_lineup if hasattr(box, 'away_lineup') else None)
        
        # Combine starters and bench for each team
        home_players = home_starters# + home_bench
        away_players = away_starters# + away_bench
        
        # Create team rosters
        home_team = TeamRoster(
            team_name=box.home_team.team_name,
            total_score=box.home_score,
            players=home_players
        )
        
        away_team = TeamRoster(
            team_name=box.away_team.team_name,
            total_score=box.away_score,
            players=away_players
        )
        
        return MatchupDetail(
            year=year,
            week=week,
            home_team=home_team,
            away_team=away_team,
            is_playoff=getattr(box, 'is_playoff', False)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to get matchup details: {e}")


# Uvicorn
# ======================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
