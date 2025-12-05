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
from trade_analysis import get_league
import numpy as np
import threading
import sqlite3

from vorp import build_vorp_table, build_linear_extrapolated_table


LEAGUE_ID = int(os.getenv("LEAGUE_ID", "86952922"))
SUPPORTED_YEARS = {2020, 2021, 2022, 2024, 2025}

# Status tracking for league initialization
league_status = {}  # {league_id: {'status': 'idle'|'initializing'|'ready'|'error', 'message': str, 'progress': str}}
status_lock = threading.Lock()

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
def get_standings(year: int, league_id: Optional[int] = Query(None, description="League ID")) -> StandingsResponse:
    # Use provided league_id or fall back to default
    effective_league_id = league_id or LEAGUE_ID
    
    try:
        # For now, standings uses ESPN API directly, so we still use _get_league
        # TODO: Update _get_league to accept league_id parameter
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
    return StandingsResponse(year=year, league_id=effective_league_id, num_teams=len(teams_out), teams=teams_out)


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
        player_name = re.sub(r"[*+]", "", str(player_name)).strip()  # Only remove asterisks and plus signs, keep periods
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
    league_id: Optional[int] = Query(None, description="League ID"),
):
    # Use provided league_id or fall back to default
    effective_league_id = league_id or LEAGUE_ID
    
    try:
        # Query VORP data from database
        import sqlite3
        conn = sqlite3.connect('weekly_fantasy_data.db')
        cursor = conn.cursor()
        
        # Check if player_totals table exists and has league_id column, add if missing
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='player_totals'")
        player_totals_exists = cursor.fetchone() is not None
        
        if player_totals_exists:
            cursor.execute("PRAGMA table_info(player_totals)")
            player_totals_columns = [col[1] for col in cursor.fetchall()]
            if 'league_id' not in player_totals_columns:
                try:
                    cursor.execute("ALTER TABLE player_totals ADD COLUMN league_id INTEGER")
                    conn.commit()
                    cursor.execute("UPDATE player_totals SET league_id = ? WHERE league_id IS NULL", (LEAGUE_ID,))
                    conn.commit()
                except Exception as e:
                    print(f"Warning: Could not add league_id to player_totals: {e}")
        
        # Check if z_scores table exists and has league_id column, add if missing
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='z_scores'")
        z_scores_exists = cursor.fetchone() is not None
        
        has_league_id = False
        if z_scores_exists:
            cursor.execute("PRAGMA table_info(z_scores)")
            z_scores_columns = [col[1] for col in cursor.fetchall()]
            has_league_id = 'league_id' in z_scores_columns
            
            # If table exists but doesn't have league_id, add it
            if not has_league_id:
                try:
                    cursor.execute("ALTER TABLE z_scores ADD COLUMN league_id INTEGER")
                    conn.commit()
                    has_league_id = True
                    # Set default league_id for existing rows
                    cursor.execute("UPDATE z_scores SET league_id = ? WHERE league_id IS NULL", (LEAGUE_ID,))
                    conn.commit()
                except Exception as e:
                    print(f"Warning: Could not add league_id to z_scores: {e}")
                    has_league_id = False
        
        if has_league_id:
            query = """
                SELECT pt.player_name, pt.fantasy_pos, pt.total_points, pt.pos_rank, 
                       pt.overall_rank, pt.vorp_star,
                       COUNT(CASE WHEN zs.week != 0 AND zs.weekly_points_ppr IS NOT NULL THEN 1 END) as games_played,
                        (SELECT zs2.fantasy_team 
                        FROM z_scores zs2 
                        WHERE zs2.player_name = pt.player_name 
                          AND zs2.year = pt.year 
                          AND (zs2.league_id = pt.league_id OR (zs2.league_id IS NULL AND pt.league_id IS NULL))
                          AND zs2.fantasy_team IS NOT NULL
                        ORDER BY zs2.week DESC, zs2.id DESC
                        LIMIT 1) as fantasy_team
                FROM player_totals pt
                LEFT JOIN z_scores zs ON pt.player_name = zs.player_name 
                    AND pt.year = zs.year 
                    AND (pt.league_id = zs.league_id OR (pt.league_id IS NULL AND zs.league_id IS NULL))
                WHERE pt.year = ? AND (pt.league_id = ? OR pt.league_id IS NULL)
                GROUP BY pt.player_name, pt.fantasy_pos, pt.total_points, pt.pos_rank, 
                         pt.overall_rank, pt.vorp_star
                ORDER BY pt.vorp_star DESC
            """
            table = pd.read_sql_query(query, conn, params=[year, effective_league_id])
        else:
            # Fallback for old schema without league_id
            query = """
                SELECT pt.player_name, pt.fantasy_pos, pt.total_points, pt.pos_rank, 
                       pt.overall_rank, pt.vorp_star,
                       COUNT(CASE WHEN zs.week != 0 AND zs.weekly_points_ppr IS NOT NULL THEN 1 END) as games_played,
                        (SELECT zs2.fantasy_team 
                        FROM z_scores zs2 
                        WHERE zs2.player_name = pt.player_name 
                          AND zs2.year = pt.year 
                          AND zs2.fantasy_team IS NOT NULL
                        ORDER BY zs2.week DESC, zs2.id DESC
                        LIMIT 1) as fantasy_team
                FROM player_totals pt
                LEFT JOIN z_scores zs ON pt.player_name = zs.player_name 
                    AND pt.year = zs.year
                WHERE pt.year = ?
                GROUP BY pt.player_name, pt.fantasy_pos, pt.total_points, pt.pos_rank, 
                         pt.overall_rank, pt.vorp_star
                ORDER BY pt.vorp_star DESC
            """
            table = pd.read_sql_query(query, conn, params=[year, LEAGUE_ID])
        conn.close()
        
        # Calculate extrapolated VORP (handle division by zero)
        # Use 16 weeks for 2020 (COVID season), 17 weeks for other years
        extrapolation_weeks = 16 if year == 2020 else 17
        table['vorp_star_extrap'] = table['vorp_star'] * (extrapolation_weeks / table['games_played'].clip(lower=1))
        table['partial_season'] = table['games_played'] < extrapolation_weeks
        
        print(f"📊 Retrieved {len(table)} players from database for {year}")
        
    except Exception as e:
        print(f"❌ Database query failed: {e}")
        return VorpResponse(year=year, players=[], count=0, used_ppg=use_ppg)

    if table is None or len(table) == 0:
        return VorpResponse(year=year, players=[], count=0, used_ppg=use_ppg)

    # Apply top limit
    table = table.head(top).copy()

    players: List[PlayerVorp] = []
    for row in table.itertuples(index=False):
        try:
            fantasy_team = getattr(row, "fantasy_team", None)
            fantasy_team_str = None if pd.isna(fantasy_team) or fantasy_team is None else str(fantasy_team)
            players.append(
                PlayerVorp(
                    player_name=str(getattr(row, "player_name")),
                    team=fantasy_team_str,  # Fantasy team from z_scores
                    fantasy_pos=str(getattr(row, "fantasy_pos")),
                    g=(None if pd.isna(getattr(row, "games_played")) else int(getattr(row, "games_played"))),
                    fantasy_points_ppr=float(getattr(row, "total_points")),
                    vorp_star=float(getattr(row, "vorp_star")),
                    vorp_star_rank_overall=int(getattr(row, "overall_rank")),
                    vorp_star_rank_pos=int(getattr(row, "pos_rank")),
                    partial_season=bool(getattr(row, "partial_season")),
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

class PlayerInTrade(BaseModel):
    """Player information in a trade, including VORP data"""
    player_name: str
    vorp_star: Optional[float] = None  # VORP* value from player_totals
    total_points: Optional[float] = None  # Total fantasy points
    fantasy_pos: Optional[str] = None  # Position (QB, RB, WR, TE, etc.)

class TradePackage(BaseModel):
    week: int
    team1_id: int
    team1_name: str
    team2_id: int
    team2_name: str
    team1_players: List[PlayerInTrade]  # Changed from List[str]
    team2_players: List[PlayerInTrade]  # Changed from List[str]
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

class TopScoringWeek(BaseModel):
    team_name: str
    points: float
    week: int

class ScoreboardResponse(BaseModel):
    year: int
    teams: List[TeamScoreboard]
    top_scoring_week: Optional[TopScoringWeek] = None

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
        # Try to get trade data from database first (much faster)
        try:
            import sqlite3
            import pandas as pd
            
            conn = sqlite3.connect('weekly_fantasy_data.db')
            
            # Check if player_trades table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", ('player_trades',))
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                print(f"Table player_trades not found, falling back to calculation")
                conn.close()
                raise Exception("Database table not found")
            
            # Get trade data from database with VORP information
            query = """
                SELECT 
                    t.week, 
                    t.player_name, 
                    t.from_team_id, 
                    t.from_team_name, 
                    t.to_team_id, 
                    t.to_team_name, 
                    t.trade_id,
                    pt.vorp_star,
                    pt.total_points,
                    pt.fantasy_pos
                FROM player_trades t
                LEFT JOIN player_totals pt 
                    ON t.player_name = pt.player_name 
                    AND pt.year = ?
                WHERE t.year = ?
                ORDER BY t.week, t.trade_id, t.player_name
            """
            
            df = pd.read_sql_query(query, conn, params=[year, year])
            conn.close()
            
            if len(df) == 0:
                return TradesResponse(
                    year=year,
                    trade_packages=[],
                    trade_summary=[],
                    count=0
                )
            
            # Group by trade_id to reconstruct trade packages
            trade_packages = []
            trade_summary = []
            
            # Query z_scores for all players to get post-trade ZAV
            conn = sqlite3.connect('weekly_fantasy_data.db')
            z_scores_query = """
                SELECT player_name, week, z_week_ppr, fantasy_team, year
                FROM z_scores
                WHERE year = ? AND league_id = ?
            """
            z_scores_df = pd.read_sql_query(z_scores_query, conn, params=[year, LEAGUE_ID])
            conn.close()
            
            for trade_id, group in df.groupby('trade_id'):
                # Get trade info from first row
                first_row = group.iloc[0]
                week = first_row['week']
                
                # Group players by direction with VORP data
                team_a_players = []
                team_b_players = []
                team_a_id = None
                team_a_name = None
                team_b_id = None
                team_b_name = None
                
                for _, row in group.iterrows():
                    if team_a_id is None:
                        # Determine team order based on first player
                        team_a_id = row['from_team_id']
                        team_a_name = row['from_team_name']
                        team_b_id = row['to_team_id']
                        team_b_name = row['to_team_name']
                    
                    # Calculate post-trade ZAV from z_scores
                    # Filter: player_name, fantasy_team = to_team_name, week >= trade_week
                    player_name = row['player_name']
                    to_team_name = row['to_team_name']
                    
                    post_trade_zav = None
                    if not z_scores_df.empty:
                        player_z_scores = z_scores_df[
                            (z_scores_df['player_name'] == player_name) &
                            (z_scores_df['fantasy_team'] == to_team_name) &
                            (z_scores_df['week'] >= week)
                        ]
                        
                        if len(player_z_scores) > 0:
                            # Sum z_week_ppr, handling NULL values
                            post_trade_zav = player_z_scores['z_week_ppr'].fillna(0).sum()
                    
                    # Create PlayerInTrade object with post-trade ZAV
                    player_data = PlayerInTrade(
                        player_name=player_name,
                        vorp_star=post_trade_zav,  # Use post-trade ZAV instead of seasonal vorp_star
                        total_points=float(row['total_points']) if pd.notna(row.get('total_points')) else None,
                        fantasy_pos=str(row['fantasy_pos']) if pd.notna(row.get('fantasy_pos')) else None
                    )
                    
                    # Add player to appropriate list based on direction
                    if row['from_team_id'] == team_a_id:
                        team_a_players.append(player_data)
                    else:
                        team_b_players.append(player_data)
                
                # Create TradePackage (same format as before)
                trade_packages.append(TradePackage(
                    week=week,
                    team1_id=team_a_id,
                    team1_name=team_a_name,
                    team2_id=team_b_id,
                    team2_name=team_b_name,
                    team1_players=team_a_players,
                    team2_players=team_b_players,
                    total_players=len(team_a_players) + len(team_b_players),
                    is_trade_like=True
                ))
                
                # Create TradeSummary (same format as before)
                trade_summary.append(TradeSummary(
                    trade_week=week,
                    team_a=team_a_name,
                    team_b=team_b_name,
                    team_a_vorp_received=0.0,
                    team_b_vorp_received=0.0,
                    net_advantage=0.0,
                    winner="Tie",
                    players_to_a=", ".join([p.player_name for p in team_b_players]),
                    players_to_b=", ".join([p.player_name for p in team_a_players])
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
            
            print(f"✅ Loaded {len(trade_packages)} trades from database for {year}")
            return result
            
        except Exception as e:
            # Fallback to calculation if database method fails
            print(f"Database trade loading failed: {e}, falling back to calculation")
            pass
        
        # Fallback: Import the trade analysis functions
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
            
            # Step 1: Collect ALL unique player names from ALL trades first
            all_unique_player_names = set()
            for week, packages in packages_by_week.items():
                for pkg in packages:
                    if not pkg["is_trade_like"]:
                        continue
                    team_a_player_names = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["a_to_b"]]
                    team_b_player_names = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["b_to_a"]]
                    all_unique_player_names.update(team_a_player_names)
                    all_unique_player_names.update(team_b_player_names)
            
            # Step 2: Query z_scores for post-trade ZAV calculation
            # We'll calculate this per-trade since we need to know which team each player was traded to
            import sqlite3
            conn_z = sqlite3.connect('weekly_fantasy_data.db')
            z_scores_query = """
                SELECT player_name, week, z_week_ppr, fantasy_team, year
                FROM z_scores
                WHERE year = ? AND league_id = ?
            """
            z_scores_df = pd.read_sql_query(z_scores_query, conn_z, params=[year, LEAGUE_ID])
            conn_z.close()
            
            # Also get player_totals for total_points and fantasy_pos
            vorp_map = {}
            if all_unique_player_names:
                try:
                    conn_vorp = sqlite3.connect('weekly_fantasy_data.db')
                    placeholders = ','.join(['?'] * len(all_unique_player_names))
                    vorp_query = f"""
                        SELECT player_name, total_points, fantasy_pos
                        FROM player_totals
                        WHERE year = ? AND league_id = ? AND player_name IN ({placeholders})
                    """
                    vorp_df = pd.read_sql_query(vorp_query, conn_vorp, params=[year, LEAGUE_ID] + list(all_unique_player_names))
                    for _, vorp_row in vorp_df.iterrows():
                        vorp_map[vorp_row['player_name']] = {
                            'total_points': float(vorp_row['total_points']) if pd.notna(vorp_row['total_points']) else None,
                            'fantasy_pos': str(vorp_row['fantasy_pos']) if pd.notna(vorp_row['fantasy_pos']) else None
                        }
                    conn_vorp.close()
                except Exception as e:
                    print(f"Warning: Could not fetch player_totals data for fallback: {e}")
            
            # Step 3: Convert packages to trade packages using the lookup map
            trade_packages = []
            trade_summary = []
            
            for week, packages in packages_by_week.items():
                for pkg in packages:
                    if not pkg["is_trade_like"]:
                        continue
                        
                    team_a, team_b = pkg["teams"]
                    team_a_name = team_meta.get(team_a, {}).get("name", f"Team {team_a}")
                    team_b_name = team_meta.get(team_b, {}).get("name", f"Team {team_b}")
                    
                    # Get player names for this trade
                    team_a_player_names = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["a_to_b"]]
                    team_b_player_names = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["b_to_a"]]
                    
                    # Calculate post-trade ZAV for each player
                    # Players in team_a_player_names were traded TO team_a (from team_b)
                    # Players in team_b_player_names were traded TO team_b (from team_a)
                    team_a_players = []
                    for name in team_a_player_names:
                        # Player was traded TO team_a, so query z_scores where fantasy_team = team_a_name and week >= trade_week
                        team_a_name = team_meta.get(team_a, {}).get('name', f'Team {team_a}')
                        post_trade_zav = None
                        if not z_scores_df.empty:
                            player_z_scores = z_scores_df[
                                (z_scores_df['player_name'] == name) &
                                (z_scores_df['fantasy_team'] == team_a_name) &
                                (z_scores_df['week'] >= week)
                            ]
                            if len(player_z_scores) > 0:
                                post_trade_zav = player_z_scores['z_week_ppr'].fillna(0).sum()
                        
                        team_a_players.append(PlayerInTrade(
                            player_name=name,
                            vorp_star=post_trade_zav,
                            total_points=vorp_map.get(name, {}).get('total_points'),
                            fantasy_pos=vorp_map.get(name, {}).get('fantasy_pos')
                        ))
                    
                    team_b_players = []
                    for name in team_b_player_names:
                        # Player was traded TO team_b, so query z_scores where fantasy_team = team_b_name and week >= trade_week
                        team_b_name = team_meta.get(team_b, {}).get('name', f'Team {team_b}')
                        post_trade_zav = None
                        if not z_scores_df.empty:
                            player_z_scores = z_scores_df[
                                (z_scores_df['player_name'] == name) &
                                (z_scores_df['fantasy_team'] == team_b_name) &
                                (z_scores_df['week'] >= week)
                            ]
                            if len(player_z_scores) > 0:
                                post_trade_zav = player_z_scores['z_week_ppr'].fillna(0).sum()
                        
                        team_b_players.append(PlayerInTrade(
                            player_name=name,
                            vorp_star=post_trade_zav,
                            total_points=vorp_map.get(name, {}).get('total_points'),
                            fantasy_pos=vorp_map.get(name, {}).get('fantasy_pos')
                        ))
                    
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
                        players_to_a=", ".join(team_b_player_names),
                        players_to_b=", ".join(team_a_player_names)
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
        
        # Step 1: Collect ALL unique player names from ALL trades first
        all_unique_player_names = set()
        for week, packages in packages_by_week.items():
            for pkg in packages:
                if not pkg["is_trade_like"]:
                    continue
                team_a_player_names = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["a_to_b"]]
                team_b_player_names = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["b_to_a"]]
                all_unique_player_names.update(team_a_player_names)
                all_unique_player_names.update(team_b_player_names)
        
        # Step 2: Query z_scores for post-trade ZAV calculation
        import sqlite3
        conn_z = sqlite3.connect('weekly_fantasy_data.db')
        z_scores_query = """
            SELECT player_name, week, z_week_ppr, fantasy_team, year
            FROM z_scores
            WHERE year = ? AND league_id = ?
        """
        z_scores_df = pd.read_sql_query(z_scores_query, conn_z, params=[year, LEAGUE_ID])
        conn_z.close()
        
        # Also get player_totals for total_points and fantasy_pos
        vorp_map = {}
        if all_unique_player_names:
            try:
                conn_vorp = sqlite3.connect('weekly_fantasy_data.db')
                placeholders = ','.join(['?'] * len(all_unique_player_names))
                vorp_query = f"""
                    SELECT player_name, total_points, fantasy_pos
                    FROM player_totals
                    WHERE year = ? AND league_id = ? AND player_name IN ({placeholders})
                """
                vorp_df = pd.read_sql_query(vorp_query, conn_vorp, params=[year, LEAGUE_ID] + list(all_unique_player_names))
                for _, vorp_row in vorp_df.iterrows():
                    vorp_map[vorp_row['player_name']] = {
                        'total_points': float(vorp_row['total_points']) if pd.notna(vorp_row['total_points']) else None,
                        'fantasy_pos': str(vorp_row['fantasy_pos']) if pd.notna(vorp_row['fantasy_pos']) else None
                    }
                conn_vorp.close()
            except Exception as e:
                print(f"Warning: Could not fetch player_totals data for fallback: {e}")
        
        # Step 3: Convert packages to TradePackage objects using the lookup map
        trade_packages = []
        for week, packages in packages_by_week.items():
            for pkg in packages:
                if not pkg["is_trade_like"]:
                    continue
                    
                team_a, team_b = pkg["teams"]
                team_a_name = team_meta.get(team_a, {}).get("name", f"Team {team_a}")
                team_b_name = team_meta.get(team_b, {}).get("name", f"Team {team_b}")
                
                # Get player names for each team
                team_a_player_names = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["a_to_b"]]
                team_b_player_names = [player_meta.get(pid, {}).get("name", str(pid)) for pid in pkg["b_to_a"]]
                
                # Calculate post-trade ZAV for each player
                # Players in team_a_player_names were traded TO team_a (from team_b)
                # Players in team_b_player_names were traded TO team_b (from team_a)
                team_a_players = []
                for name in team_a_player_names:
                    # Player was traded TO team_a, so query z_scores where fantasy_team = team_a_name and week >= trade_week
                    post_trade_zav = None
                    if not z_scores_df.empty:
                        player_z_scores = z_scores_df[
                            (z_scores_df['player_name'] == name) &
                            (z_scores_df['fantasy_team'] == team_a_name) &
                            (z_scores_df['week'] >= week)
                        ]
                        if len(player_z_scores) > 0:
                            post_trade_zav = player_z_scores['z_week_ppr'].fillna(0).sum()
                    
                    team_a_players.append(PlayerInTrade(
                        player_name=name,
                        vorp_star=post_trade_zav,
                        total_points=vorp_map.get(name, {}).get('total_points'),
                        fantasy_pos=vorp_map.get(name, {}).get('fantasy_pos')
                    ))
                
                team_b_players = []
                for name in team_b_player_names:
                    # Player was traded TO team_b, so query z_scores where fantasy_team = team_b_name and week >= trade_week
                    post_trade_zav = None
                    if not z_scores_df.empty:
                        player_z_scores = z_scores_df[
                            (z_scores_df['player_name'] == name) &
                            (z_scores_df['fantasy_team'] == team_b_name) &
                            (z_scores_df['week'] >= week)
                        ]
                        if len(player_z_scores) > 0:
                            post_trade_zav = player_z_scores['z_week_ppr'].fillna(0).sum()
                    
                    team_b_players.append(PlayerInTrade(
                        player_name=name,
                        vorp_star=post_trade_zav,
                        total_points=vorp_map.get(name, {}).get('total_points'),
                        fantasy_pos=vorp_map.get(name, {}).get('fantasy_pos')
                    ))
                
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
        
        # Get top scoring week for the year
        top_scoring_week_data = None
        try:
            top_scoring_week = league.top_scoring_week()
            if top_scoring_week:
                # Handle different return formats from top_scoring_week()
                if isinstance(top_scoring_week, tuple):
                    # If it returns (team_name, points)
                    top_scoring_week_data = TopScoringWeek(
                        team_name=top_scoring_week[0],
                        points=top_scoring_week[1],
                        week=0  # We'll find the week below
                    )
                elif hasattr(top_scoring_week, 'team_name'):
                    top_scoring_week_data = TopScoringWeek(
                        team_name=top_scoring_week.team_name,
                        points=getattr(top_scoring_week, 'points', 0),
                        week=getattr(top_scoring_week, 'week', 0)
                    )
        except Exception as e:
            print(f"Error getting top scoring week: {e}")
        
        # If we have team_name and points but not week, find the week manually
        if top_scoring_week_data and top_scoring_week_data.week == 0:
            max_points = top_scoring_week_data.points
            top_team_name = top_scoring_week_data.team_name
            top_week = None
            for team_id, data in teams_data.items():
                if data['name'] == top_team_name:
                    for game in data['games']:
                        if abs(game['score'] - max_points) < 0.1 and not game['is_playoff']:
                            top_week = game['week']
                            break
                    if top_week:
                        break
            if top_week:
                top_scoring_week_data.week = top_week
        
        # Fallback: find it manually if top_scoring_week() didn't work
        if not top_scoring_week_data:
            max_points = 0
            top_team_name = None
            top_week = None
            for team_id, data in teams_data.items():
                for game in data['games']:
                    if game['score'] > max_points and not game['is_playoff']:
                        max_points = game['score']
                        top_team_name = data['name']
                        top_week = game['week']
            if top_team_name:
                top_scoring_week_data = TopScoringWeek(
                    team_name=top_team_name,
                    points=max_points,
                    week=top_week
                )
        
        return ScoreboardResponse(year=year, teams=teams, top_scoring_week=top_scoring_week_data)
        
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
        
        
        league = get_league(year)
        
        # Get box scores for the specific week
        box_scores = league.box_scores(week=week)
        
        if not box_scores:
            raise HTTPException(status_code=404, detail=f"No matchup found for week {week} in {year}")
        
        # Find the specific matchup if team names are provided
        box = None
        if team1 and team2:
            # Look for matchup between the two specific teams
            for i, b in enumerate(box_scores):
                if type(b.home_team) != int and type(b.away_team) != int:
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
        
        # Allow playoff games for matchup details
        
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
        try:
            home_starters, home_bench = process_team_roster(box.home_lineup if hasattr(box, 'home_lineup') else None)
            away_starters, away_bench = process_team_roster(box.away_lineup if hasattr(box, 'away_lineup') else None)
        except Exception as e:
            # Fallback to empty rosters if processing fails
            home_starters, home_bench = [], []
            away_starters, away_bench = [], []
        
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


# ======================
# Player Weekly Stats Endpoint
# ======================

class WeeklyStat(BaseModel):
    week: int
    z_week_ppr: Optional[float] = None
    weekly_points_ppr: Optional[float] = None

class PlayerWeeklyStatsResponse(BaseModel):
    player_name: str
    year: int
    max_week: int
    weekly_stats: List[WeeklyStat]
    total_points: Optional[float] = None
    total_zav: Optional[float] = None
    fantasy_pos: Optional[str] = None
    pos_rank: Optional[int] = None

class TeamZAV(BaseModel):
    team_id: int
    team_name: str
    total_zav: float

class TeamZAVResponse(BaseModel):
    year: int
    teams: List[TeamZAV]

@app.get("/teams/{year}/zav-totals", response_model=TeamZAVResponse)
def get_team_zav_totals(year: int):
    """
    Get total ZAV for each team in a given year.
    Uses the fantasy_team column in z_scores table (populated by populate_weekly_db.py).
    """
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")
    
    import sqlite3
    
    try:
        # Connect to database
        conn = sqlite3.connect('weekly_fantasy_data.db')
        cursor = conn.cursor()
        
        # Query z_scores table, grouping by fantasy_team and summing z_week_ppr
        cursor.execute("""
            SELECT 
                fantasy_team as team_name,
                SUM(z_week_ppr) as total_zav
            FROM z_scores
            WHERE year = ? AND league_id = ? AND fantasy_team IS NOT NULL
            GROUP BY fantasy_team
            ORDER BY total_zav DESC
        """, (year, LEAGUE_ID))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to response format
        team_list = []
        for team_name, total_zav in results:
            # Get team_id from league (for consistency with response model)
            # If we can't find it, use 0
            team_id = 0
            try:
                league = get_league(year)
                for team in league.teams:
                    if team.team_name == team_name:
                        team_id = team.team_id
                        break
            except:
                pass
            
            team_list.append(TeamZAV(
                team_id=team_id,
                team_name=team_name,
                total_zav=float(total_zav) if total_zav is not None else 0.0
            ))
        
        return TeamZAVResponse(year=year, teams=team_list)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get team ZAV totals: {e}")

class PlayerRoster(BaseModel):
    player_name: str
    position: str
    pos_rank: Optional[int] = None
    vorp_star: Optional[float] = None  # Seasonal ZAV

class TeamRosterDetail(BaseModel):
    team_id: int
    team_name: str
    players: List[PlayerRoster]

class TeamRostersResponse(BaseModel):
    year: int
    teams: List[TeamRosterDetail]

@app.get("/teams/{year}/rosters", response_model=TeamRostersResponse)
def get_team_rosters(year: int):
    """
    Get current rosters for all teams with positional rankings.
    Uses box scores from the most recent week to get all players on rosters.
    """
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")
    
    import sqlite3
    from collections import defaultdict
    
    try:
        from trade_analysis import get_league, guess_max_week
        
        league = get_league(year)
        teams = league.teams
        
        # Get max_week - try database first, fallback to guess_max_week
        conn = sqlite3.connect('weekly_fantasy_data.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MAX(week) as max_week
            FROM z_scores
            WHERE year = ? AND league_id = ?
        """, (year, LEAGUE_ID))
        
        result = cursor.fetchone()
        max_week = result[0] if result and result[0] else None
        
        # If no database data, use guess_max_week
        if not max_week:
            max_week = guess_max_week(league)
        
        # Cap max_week at 17 for previous years (before 2025)
        if year < 2025:
            max_week = min(max_week, 17) if max_week else 17
        
        if not max_week:
            conn.close()
            return TeamRostersResponse(year=year, teams=[])
        
        # Get box scores for max_week
        try:
            box_scores = league.box_scores(week=max_week)
        except Exception as e:
            conn.close()
            raise HTTPException(status_code=500, detail=f"Failed to get box scores for week {max_week}: {e}")
        
        # Collect all players from box scores
        # Structure: {team_name: {player_name: {'position': pos, 'team_id': id}}}
        roster_data = defaultdict(dict)
        
        for box in box_scores:
            # Process home lineup
            if hasattr(box, 'home_team') and hasattr(box, 'home_lineup'):
                home_team = box.home_team
                if home_team and hasattr(home_team, 'team_name') and hasattr(home_team, 'team_id'):
                    home_lineup = box.home_lineup or []
                    for player in home_lineup:
                        player_name = getattr(player, 'name', None)
                        if not player_name:
                            continue
                        # Clean player name (remove asterisks, keep periods)
                        player_name = player_name.replace('*', '').strip()
                        position = getattr(player, 'position', None)
                        roster_data[home_team.team_name][player_name] = {
                            'position': position,
                            'team_id': home_team.team_id
                        }
            
            # Process away lineup
            if hasattr(box, 'away_team') and hasattr(box, 'away_lineup'):
                away_team = box.away_team
                if away_team and hasattr(away_team, 'team_name') and hasattr(away_team, 'team_id'):
                    away_lineup = box.away_lineup or []
                    for player in away_lineup:
                        player_name = getattr(player, 'name', None)
                        if not player_name:
                            continue
                        # Clean player name (remove asterisks, keep periods)
                        player_name = player_name.replace('*', '').strip()
                        position = getattr(player, 'position', None)
                        roster_data[away_team.team_name][player_name] = {
                            'position': position,
                            'team_id': away_team.team_id
                        }
        
        # Get all unique player names
        all_player_names = set()
        for team_players in roster_data.values():
            all_player_names.update(team_players.keys())
        
        if not all_player_names:
            conn.close()
            return TeamRostersResponse(year=year, teams=[])
        
        # Query player_totals for ZAV and rankings for all players
        placeholders = ','.join(['?'] * len(all_player_names))
        cursor.execute(f"""
            SELECT 
                player_name,
                fantasy_pos as position,
                pos_rank,
                vorp_star
            FROM player_totals
            WHERE player_name IN ({placeholders}) AND year = ? AND league_id = ?
        """, list(all_player_names) + [year, LEAGUE_ID])
        
        player_stats = {}
        for row in cursor.fetchall():
            player_name, position, pos_rank, vorp_star = row
            # Handle NULL values from database - convert to None
            player_stats[player_name] = {
                'position': position if position else None,
                'pos_rank': int(pos_rank) if pos_rank is not None else None,
                'vorp_star': float(vorp_star) if vorp_star is not None and vorp_star != '' else None
            }
        
        conn.close()
        
        # Build team rosters
        teams_dict = {}
        for team in teams:
            teams_dict[team.team_name] = {
                'team_id': team.team_id,
                'team_name': team.team_name,
                'players': []
            }
        
        # Add players to their respective teams
        for team_name, players_dict in roster_data.items():
            if team_name not in teams_dict:
                continue
            
            for player_name, player_info in players_dict.items():
                # Get stats from player_totals if available
                stats = player_stats.get(player_name, {})
                position = stats.get('position') or player_info.get('position')
                pos_rank = stats.get('pos_rank')
                vorp_star = stats.get('vorp_star')
                
                teams_dict[team_name]['players'].append(PlayerRoster(
                    player_name=player_name,
                    position=position,
                    pos_rank=pos_rank,
                    vorp_star=vorp_star
                ))
        
        # Convert to list and sort by team name
        team_list = [TeamRosterDetail(**data) for data in teams_dict.values()]
        team_list.sort(key=lambda x: x.team_name)
        
        return TeamRostersResponse(year=year, teams=team_list)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get team rosters: {e}")

@app.get("/players/{player_name}/weekly-stats", response_model=PlayerWeeklyStatsResponse)
def get_player_weekly_stats(player_name: str, year: int = Query(...)):
    """
    Get weekly stats (z-scores and PPR points) for a player in a given year.
    Returns stats for all weeks 1-max_week, with null values for missing weeks.
    """
    import sqlite3
    from urllib.parse import unquote
    
    # FastAPI automatically URL-decodes path parameters, so player_name should already be decoded
    # However, if it was double-encoded, we need to decode again
    # Check if it looks URL-encoded (contains %)
    if '%' in player_name:
        player_name = unquote(player_name)
    
    try:
        conn = sqlite3.connect('weekly_fantasy_data.db')
        
        # First, get the max week for this year from the database
        max_week_query = """
            SELECT MAX(week) as max_week
            FROM z_scores
            WHERE year = ? AND league_id = ?
        """
        max_week_result = pd.read_sql_query(max_week_query, conn, params=[year, LEAGUE_ID])
        max_week = int(max_week_result['max_week'].iloc[0]) if not max_week_result['max_week'].isna().iloc[0] else 17
        # Cap max_week at 17 for previous years (before 2025)
        if year < 2025:
            max_week = min(max_week, 17)
        
        # Get all weekly stats for this player and year, plus position
        query = """
            SELECT week, z_week_ppr, weekly_points_ppr, fantasy_pos
            FROM z_scores
            WHERE player_name = ? AND year = ? AND league_id = ?
            ORDER BY week
        """
        
        stats_df = pd.read_sql_query(query, conn, params=[player_name, year, LEAGUE_ID])
        
        # Get totals and positional ranking from player_totals
        totals_query = """
            SELECT total_points, vorp_star, fantasy_pos, pos_rank
            FROM player_totals
            WHERE player_name = ? AND year = ? AND league_id = ?
        """
        totals_df = pd.read_sql_query(totals_query, conn, params=[player_name, year, LEAGUE_ID])
        
        total_points = None
        total_zav = None
        fantasy_pos = None
        pos_rank = None
        
        if not totals_df.empty:
            total_points = float(totals_df['total_points'].iloc[0]) if pd.notna(totals_df['total_points'].iloc[0]) else None
            total_zav = float(totals_df['vorp_star'].iloc[0]) if pd.notna(totals_df['vorp_star'].iloc[0]) else None
            fantasy_pos = str(totals_df['fantasy_pos'].iloc[0]) if pd.notna(totals_df['fantasy_pos'].iloc[0]) else None
            pos_rank = int(totals_df['pos_rank'].iloc[0]) if pd.notna(totals_df['pos_rank'].iloc[0]) else None
        elif not stats_df.empty:
            # Fallback: get position from z_scores if not in player_totals
            fantasy_pos = str(stats_df['fantasy_pos'].iloc[0]) if pd.notna(stats_df['fantasy_pos'].iloc[0]) else None
            # Calculate totals from weekly stats
            total_points = float(stats_df['weekly_points_ppr'].sum()) if pd.notna(stats_df['weekly_points_ppr']).any() else None
            total_zav = float(stats_df['z_week_ppr'].sum()) if pd.notna(stats_df['z_week_ppr']).any() else None
        
        # If no results found, try a case-insensitive search or check for D/ST variations
        if stats_df.empty and ('/' in player_name or 'D/ST' in player_name.upper() or 'DST' in player_name.upper()):
            # Try case-insensitive search for defense players
            query_ci = """
                SELECT week, z_week_ppr, weekly_points_ppr
                FROM z_scores
                WHERE UPPER(player_name) LIKE UPPER(?) AND year = ? AND league_id = ?
                ORDER BY week
            """
            # Try matching with wildcard for team name variations
            search_pattern = f"%{player_name}%"
            stats_df = pd.read_sql_query(query_ci, conn, params=[search_pattern, year, LEAGUE_ID])
            # If still no results, try exact case-insensitive match
            if stats_df.empty:
                query_exact = """
                    SELECT week, z_week_ppr, weekly_points_ppr
                    FROM z_scores
                    WHERE UPPER(player_name) = UPPER(?) AND year = ? AND league_id = ?
                    ORDER BY week
                """
                stats_df = pd.read_sql_query(query_exact, conn, params=[player_name, year, LEAGUE_ID])
        
        conn.close()
        
        # Create a map of week -> stats
        stats_map = {}
        for _, row in stats_df.iterrows():
            week = int(row['week'])
            stats_map[week] = {
                'z_week_ppr': float(row['z_week_ppr']) if pd.notna(row['z_week_ppr']) else None,
                'weekly_points_ppr': float(row['weekly_points_ppr']) if pd.notna(row['weekly_points_ppr']) else None
            }
        
        # Build response with all weeks 1 to max_week
        weekly_stats = []
        for week in range(1, max_week + 1):
            if week in stats_map:
                weekly_stats.append(WeeklyStat(
                    week=week,
                    z_week_ppr=stats_map[week]['z_week_ppr'],
                    weekly_points_ppr=stats_map[week]['weekly_points_ppr']
                ))
            else:
                weekly_stats.append(WeeklyStat(
                    week=week,
                    z_week_ppr=None,
                    weekly_points_ppr=None
                ))
        
        return PlayerWeeklyStatsResponse(
            player_name=player_name,
            year=year,
            max_week=max_week,
            weekly_stats=weekly_stats,
            total_points=total_points,
            total_zav=total_zav,
            fantasy_pos=fantasy_pos,
            pos_rank=pos_rank
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get player weekly stats: {e}")

@app.get("/players/{player_name}/headshot")
def get_player_headshot(player_name: str):
    """
    Get headshot URL for a player from the headshots table.
    Returns the headshot_url if available, or null if not found.
    """
    import sqlite3
    from urllib.parse import unquote
    
    # FastAPI automatically URL-decodes path parameters, so player_name should already be decoded
    # However, if it was double-encoded, we need to decode again
    if '%' in player_name:
        player_name = unquote(player_name)
    
    try:
        conn = sqlite3.connect('weekly_fantasy_data.db')
        cursor = conn.cursor()
        
        query = """
            SELECT headshot_url
            FROM headshots
            WHERE player_name = ? AND league_id = ?
        """
        cursor.execute(query, (player_name, LEAGUE_ID))
        result = cursor.fetchone()
        
        conn.close()
        
        if result and result[0]:
            return {"headshot_url": result[0]}
        else:
            # Try case-insensitive search for D/ST players
            conn = sqlite3.connect('weekly_fantasy_data.db')
            cursor = conn.cursor()
            query_ci = """
                SELECT headshot_url
                FROM headshots
                WHERE UPPER(player_name) = UPPER(?) AND league_id = ?
            """
            cursor.execute(query_ci, (player_name, LEAGUE_ID))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return {"headshot_url": result[0]}
            else:
                return {"headshot_url": None}
                
    except Exception as e:
        return {"headshot_url": None}

# ======================
# Waiver Activity
# ======================

class WaiverTransaction(BaseModel):
    transaction_id: Optional[int] = None
    transaction_date: Optional[str]
    team_name: Optional[str]
    action_type: str
    player_name: str
    player_position: Optional[str] = None
    player_zav: Optional[float] = None

class WaiversResponse(BaseModel):
    year: int
    transactions: List[WaiverTransaction]
    count: int

@app.get("/waivers/{year}", response_model=WaiversResponse)
def get_waiver_activity(year: int):
    """
    Get waiver activity (adds/drops) for a given year.
    Includes player ZAV and position from player_totals.
    """
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")
    
    import sqlite3
    
    try:
        conn = sqlite3.connect('weekly_fantasy_data.db')
        
        # Query waiver_activity with LEFT JOIN to player_totals for ZAV and position
        query = """
            SELECT 
                wa.transaction_id,
                wa.transaction_date,
                wa.team_name,
                wa.action_type,
                wa.player_name,
                pt.fantasy_pos as player_position,
                pt.vorp_star as player_zav
            FROM waiver_activity wa
            LEFT JOIN player_totals pt 
                ON wa.player_name = pt.player_name 
                AND wa.year = pt.year
                AND wa.league_id = pt.league_id
            WHERE wa.year = ? AND wa.league_id = ?
            ORDER BY wa.transaction_date DESC, wa.transaction_id
        """
        
        df = pd.read_sql_query(query, conn, params=[year, LEAGUE_ID])
        conn.close()
        
        transactions = []
        for _, row in df.iterrows():
            transactions.append(WaiverTransaction(
                transaction_id=int(row['transaction_id']) if pd.notna(row['transaction_id']) else None,
                transaction_date=row['transaction_date'] if pd.notna(row['transaction_date']) else None,
                team_name=row['team_name'] if pd.notna(row['team_name']) else None,
                action_type=row['action_type'],
                player_name=row['player_name'],
                player_position=row['player_position'] if pd.notna(row['player_position']) else None,
                player_zav=float(row['player_zav']) if pd.notna(row['player_zav']) and row['player_zav'] != '' else None
            ))
        
        return WaiversResponse(
            year=year,
            transactions=transactions,
            count=len(transactions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get waiver activity: {e}")


# ======================
# Recent Trades
# ======================

class PlayerInRecentTrade(BaseModel):
    player_name: str
    vorp_star: Optional[float] = None
    fantasy_pos: Optional[str] = None

class RecentTradeItem(BaseModel):
    week: int
    trade_id: str
    team1_name: str
    team2_name: str
    team1_players: List[PlayerInRecentTrade]
    team2_players: List[PlayerInRecentTrade]

class RecentTradesResponse(BaseModel):
    year: int
    trades: List[RecentTradeItem]
    count: int

@app.get("/recent-trades/{year}", response_model=RecentTradesResponse)
def get_recent_trades(year: int):
    """
    Get recent trades for a given year with player details and ZAV.
    Returns the 5 most recent trades sorted by week descending.
    """
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")
    
    import sqlite3
    
    try:
        conn = sqlite3.connect('weekly_fantasy_data.db')
        cursor = conn.cursor()
        
        # Check if trades table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", ('player_trades',))
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            return RecentTradesResponse(
                year=year,
                trades=[],
                count=0
            )
        
        # Get recent trades - get distinct trade_ids first
        trade_query = """
            SELECT DISTINCT
                t.week,
                t.trade_id
            FROM player_trades t
            WHERE t.year = ?
            ORDER BY t.week DESC, t.trade_id DESC
            LIMIT 5
        """
        
        trade_df = pd.read_sql_query(trade_query, conn, params=(year, LEAGUE_ID))
        
        # If no trades found for requested year, try 2024 as fallback
        if len(trade_df) == 0 and year == 2025:
            trade_query_2024 = """
                SELECT DISTINCT
                    t.week,
                    t.trade_id
                FROM player_trades t
                WHERE t.year = 2024
                ORDER BY t.week DESC, t.trade_id DESC
                LIMIT 5
            """
            trade_df = pd.read_sql_query(trade_query_2024, conn)
            year = 2024  # Use 2024 data
        
        trades = []
        for _, row in trade_df.iterrows():
            week = int(row['week'])
            trade_id = str(row['trade_id'])
            
            # Get players for this trade with ZAV
            players_query = """
                SELECT 
                    t.player_name,
                    t.from_team_id,
                    t.from_team_name,
                    t.to_team_id,
                    t.to_team_name,
                    pt.vorp_star,
                    pt.fantasy_pos
                FROM player_trades t
                LEFT JOIN player_totals pt 
                    ON t.player_name = pt.player_name 
                    AND pt.year = ?
                    AND pt.league_id = t.league_id
                WHERE t.year = ? AND t.league_id = ? AND t.week = ? AND t.trade_id = ?
                ORDER BY t.player_name
            """
            
            players_df = pd.read_sql_query(players_query, conn, params=[year, LEAGUE_ID, year, LEAGUE_ID, week, trade_id])
            
            if len(players_df) == 0:
                continue
            
            # Determine team names from the players (get unique team names involved)
            all_team_names = set()
            for _, player_row in players_df.iterrows():
                all_team_names.add(player_row['from_team_name'])
                all_team_names.add(player_row['to_team_name'])
            
            team_names_list = sorted(list(all_team_names))
            if len(team_names_list) < 2:
                continue
            
            team1_name = team_names_list[0]
            team2_name = team_names_list[1]
            
            team1_players = []
            team2_players = []
            
            for _, player_row in players_df.iterrows():
                player_data = PlayerInRecentTrade(
                    player_name=player_row['player_name'],
                    vorp_star=float(player_row['vorp_star']) if pd.notna(player_row.get('vorp_star')) else None,
                    fantasy_pos=str(player_row['fantasy_pos']) if pd.notna(player_row.get('fantasy_pos')) else None
                )
                
                # Group by receiving team (to_team_name)
                # Players going TO team1 go in team1_players
                # Players going TO team2 go in team2_players
                if player_row['to_team_name'] == team1_name:
                    team1_players.append(player_data)
                elif player_row['to_team_name'] == team2_name:
                    team2_players.append(player_data)
            
            trades.append(RecentTradeItem(
                week=week,
                trade_id=trade_id,
                team1_name=team1_name,
                team2_name=team2_name,
                team1_players=team1_players,
                team2_players=team2_players
            ))
        
        conn.close()
        
        return RecentTradesResponse(
            year=year,
            trades=trades,
            count=len(trades)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent trades: {e}")

class RecentWaiverItem(BaseModel):
    transaction_id: Optional[int]
    transaction_date: str
    team_name: str
    added_players: List[PlayerInRecentTrade]
    dropped_players: List[PlayerInRecentTrade]

class RecentWaiversResponse(BaseModel):
    year: int
    transactions: List[RecentWaiverItem]
    count: int

@app.get("/recent-waivers/{year}", response_model=RecentWaiversResponse)
def get_recent_waivers(year: int, league_id: Optional[int] = Query(None, description="League ID")):
    """
    Get recent waiver activity for a given year with player details and ZAV.
    Returns the 5 most recent waiver transactions sorted by date descending.
    """
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")
    
    import sqlite3
    
    try:
        conn = sqlite3.connect('weekly_fantasy_data.db')
        cursor = conn.cursor()
        
        # Check if waiver_activity table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", ('waiver_activity',))
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            return RecentWaiversResponse(
                year=year,
                transactions=[],
                count=0
            )
        
        # Get recent waiver transactions - get distinct transaction_ids first
        # Handle both NULL league_id (old data) and specific league_id
        waiver_query = """
            SELECT DISTINCT
                wa.transaction_id,
                wa.transaction_date,
                wa.team_name
            FROM waiver_activity wa
            WHERE wa.year = ? AND (wa.league_id = ? OR wa.league_id IS NULL)
            ORDER BY wa.transaction_date DESC, wa.transaction_id DESC
            LIMIT 5
        """
        
        # Use provided league_id or fall back to default
        effective_league_id = league_id or LEAGUE_ID
        
        waiver_df = pd.read_sql_query(waiver_query, conn, params=[year, effective_league_id])
        
        # If no waivers found for requested year, try 2024 as fallback
        if len(waiver_df) == 0 and year == 2025:
            waiver_query_2024 = """
                SELECT DISTINCT
                    wa.transaction_id,
                    wa.transaction_date,
                    wa.team_name
                FROM waiver_activity wa
                WHERE wa.year = 2024 AND (wa.league_id = ? OR wa.league_id IS NULL)
                ORDER BY wa.transaction_date DESC, wa.transaction_id DESC
                LIMIT 5
            """
            waiver_df = pd.read_sql_query(waiver_query_2024, conn, params=[effective_league_id])
            year = 2024  # Use 2024 data
        
        transactions = []
        for _, row in waiver_df.iterrows():
            transaction_id = int(row['transaction_id']) if pd.notna(row['transaction_id']) else None
            transaction_date = str(row['transaction_date']) if pd.notna(row['transaction_date']) else ''
            team_name = str(row['team_name']) if pd.notna(row['team_name']) else ''
            
            # Get players for this transaction with ZAV
            if transaction_id is not None:
                players_query = """
                    SELECT 
                        wa.player_name,
                        wa.action_type,
                        pt.vorp_star,
                        pt.fantasy_pos
                    FROM waiver_activity wa
                    LEFT JOIN player_totals pt 
                        ON wa.player_name = pt.player_name 
                        AND wa.year = pt.year
                        AND (wa.league_id = pt.league_id OR (wa.league_id IS NULL AND pt.league_id IS NULL))
                    WHERE wa.transaction_id = ? AND wa.year = ? AND (wa.league_id = ? OR wa.league_id IS NULL)
                    ORDER BY wa.action_type DESC, wa.player_name
                """
                players_df = pd.read_sql_query(players_query, conn, params=[transaction_id, year, LEAGUE_ID])
            else:
                # If no transaction_id, match by date and team
                players_query = """
                    SELECT 
                        wa.player_name,
                        wa.action_type,
                        pt.vorp_star,
                        pt.fantasy_pos
                    FROM waiver_activity wa
                    LEFT JOIN player_totals pt 
                        ON wa.player_name = pt.player_name 
                        AND wa.year = pt.year
                        AND wa.league_id = pt.league_id
                    WHERE wa.transaction_date = ? AND wa.team_name = ? AND wa.year = ? AND wa.league_id = ?
                    ORDER BY wa.action_type DESC, wa.player_name
                """
                players_df = pd.read_sql_query(players_query, conn, params=[transaction_date, team_name, year, LEAGUE_ID])
            
            if len(players_df) == 0:
                continue
            
            added_players = []
            dropped_players = []
            
            for _, player_row in players_df.iterrows():
                player_data = PlayerInRecentTrade(
                    player_name=player_row['player_name'],
                    vorp_star=float(player_row['vorp_star']) if pd.notna(player_row.get('vorp_star')) else None,
                    fantasy_pos=str(player_row['fantasy_pos']) if pd.notna(player_row.get('fantasy_pos')) else None
                )
                
                action_type = str(player_row['action_type']).upper()
                if 'ADDED' in action_type or 'ADD' in action_type:
                    added_players.append(player_data)
                elif 'DROPPED' in action_type or 'DROP' in action_type:
                    dropped_players.append(player_data)
            
            transactions.append(RecentWaiverItem(
                transaction_id=transaction_id,
                transaction_date=transaction_date,
                team_name=team_name,
                added_players=added_players,
                dropped_players=dropped_players
            ))
        
        conn.close()
        
        return RecentWaiversResponse(
            year=year,
            transactions=transactions,
            count=len(transactions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent waivers: {e}")


# ======================
# League Initialization Endpoints
# ======================

class LeagueStatusResponse(BaseModel):
    league_id: int
    status: str  # 'idle', 'initializing', 'ready', 'error'
    message: str
    progress: Optional[str] = None

class InitializeLeagueResponse(BaseModel):
    league_id: int
    status: str
    message: str

def check_league_data_exists(league_id: int) -> bool:
    """Check if data exists for a given league_id in the database"""
    try:
        conn = sqlite3.connect('weekly_fantasy_data.db')
        cursor = conn.cursor()
        
        # Check if any tables have data for this league_id
        tables_to_check = ['player_totals', 'z_scores', 'waiver_activity', 'player_trades']
        
        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE league_id = ? LIMIT 1", (league_id,))
                result = cursor.fetchone()
                if result and result[0] > 0:
                    conn.close()
                    return True
            except sqlite3.OperationalError:
                # Table might not exist or might not have league_id column
                continue
        
        conn.close()
        return False
    except Exception as e:
        print(f"Error checking league data: {e}")
        return False

def validate_league_id(league_id: int, year: int = 2024) -> bool:
    """Validate that league_id is accessible via ESPN API"""
    try:
        # Use the same credentials as _get_league
        espn_s2 = "AEC20e998honXS4Wi0Z8qzlJdam4%2F%2BaApa7apspnhKR0Npb%2FMsF5DuQsFUcHW%2FhPihQun9U6PGITOi2CkbdfDCkUc8druBVhAwT08Lzrvv8oZli8YAuTi9mIWg7YqtorCNOEKPxHpYswnT3q7b885tRDKBJpLCH0T2h4h1p%2B02SfdlDhjEB2gHqFk1xl6tJRNMBiCkZ8i5RttLW6ER9ZvLTmmAdb5ceZhQ27NEMiMf%2BjWSSvwBdnf2roxwt9baw33BVnnITqYVb8FXsaUwm7%2Bm0m9GLQ%2B66%2BU%2Brg%2BQngXm1ekA%3D%3D"
        swid = "{B431504E-F779-4C49-B3E8-28DDF7409957}"
        league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
        # Try to access a property to validate
        _ = league.teams
        return True
    except Exception as e:
        print(f"Error validating league_id {league_id}: {e}")
        return False

@app.get("/api/league-status/{league_id}", response_model=LeagueStatusResponse)
def get_league_status(league_id: int):
    """Get the initialization status for a league"""
    with status_lock:
        status_info = league_status.get(league_id, {
            'status': 'idle',
            'message': 'Not initialized',
            'progress': None
        })
        
        # Also check if data exists in database
        if status_info['status'] == 'idle' and check_league_data_exists(league_id):
            status_info = {
                'status': 'ready',
                'message': 'Data already exists',
                'progress': None
            }
    
    return LeagueStatusResponse(
        league_id=league_id,
        status=status_info['status'],
        message=status_info['message'],
        progress=status_info.get('progress')
    )

def run_initialization(league_id: int, force: bool = False):
    """Run league initialization in background thread"""
    def status_callback(msg: str):
        with status_lock:
            league_status[league_id] = {
                'status': 'initializing',
                'message': msg,
                'progress': msg
            }
    
    try:
        with status_lock:
            league_status[league_id] = {
                'status': 'initializing',
                'message': 'Starting initialization...',
                'progress': 'Initializing...'
            }
        
        # Validate league_id first
        status_callback("Validating league ID...")
        if not validate_league_id(league_id):
            with status_lock:
                league_status[league_id] = {
                    'status': 'error',
                    'message': f'Invalid league ID: {league_id}. Could not access league via ESPN API.',
                    'progress': None
                }
            return
        
        # Check if data exists
        if not force and check_league_data_exists(league_id):
            with status_lock:
                league_status[league_id] = {
                    'status': 'ready',
                    'message': 'Data already exists. Use force=true to re-populate.',
                    'progress': None
                }
            return
        
        # Import and run population
        from source_players import populate_league_data
        
        status_callback("Populating database...")
        results = populate_league_data(
            league_id=league_id,
            years=[2020, 2021, 2022, 2024, 2025],
            clear_db=False,
            status_callback=status_callback
        )
        
        # Update status
        with status_lock:
            if results['years_failed']:
                league_status[league_id] = {
                    'status': 'ready',
                    'message': f"Initialization complete with warnings. Processed: {results['years_processed']}, Failed: {[f['year'] for f in results['years_failed']]}",
                    'progress': None
                }
            else:
                league_status[league_id] = {
                    'status': 'ready',
                    'message': f"Initialization complete! Processed {len(results['years_processed'])} years.",
                    'progress': None
                }
    
    except Exception as e:
        with status_lock:
            league_status[league_id] = {
                'status': 'error',
                'message': f'Initialization failed: {str(e)}',
                'progress': None
            }
        import traceback
        traceback.print_exc()

@app.post("/api/initialize-league/{league_id}", response_model=InitializeLeagueResponse)
def initialize_league(league_id: int, force: bool = Query(False, description="Force re-population even if data exists")):
    """Initialize/populate database for a given league_id"""
    
    # Check if already initializing
    with status_lock:
        current_status = league_status.get(league_id, {}).get('status', 'idle')
        if current_status == 'initializing':
            raise HTTPException(
                status_code=409,
                detail=f"League {league_id} is already being initialized"
            )
    
    # Check if data exists and not forcing
    if not force and check_league_data_exists(league_id):
        return InitializeLeagueResponse(
            league_id=league_id,
            status='exists',
            message='Data already exists for this league. Use force=true to re-populate.'
        )
    
    # Start initialization in background thread
    thread = threading.Thread(target=run_initialization, args=(league_id, force))
    thread.daemon = True
    thread.start()
    
    return InitializeLeagueResponse(
        league_id=league_id,
        status='started',
        message='Initialization started in background. Check /api/league-status/{league_id} for progress.'
    )


# Uvicorn
# ======================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
