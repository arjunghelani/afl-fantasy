from typing import List, Optional
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from espn_api.football import League
from fastapi import Query
import pandas as pd
import re

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
    week: Optional[int] = None           # <— NEW
    matchup_type: Optional[str] = None   # <— NEW



class PlayoffBracket(BaseModel):
    year: int
    league_id: int
    games: List[PlayoffGame]
    
# --- Draft models (3.9-friendly typing) ---
from typing import List, Optional

class DraftPick(BaseModel):
    year: int
    team_id: int                      # <-- NEW (stable key)
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
    

class PlayerVorp(BaseModel):
    player_name: str
    team: Optional[str] = None
    fantasy_pos: str
    g: Optional[int] = None
    fantasy_points_ppr: float
    vorp_star: float
    vorp_star_rank_overall: int
    vorp_star_rank_pos: int
    # NEW
    partial_season: Optional[bool] = None
    vorp_star_extrap: Optional[float] = None



class VorpResponse(BaseModel):
    year: int
    players: List[PlayerVorp]
    count: int
    used_ppg: bool = False
    
import json
from pathlib import Path

PLAYER_CACHE_FILE = Path("./player_cache.json")
# { int(playerId): {"position": str|None, "proTeam": str|None, "name": str|None} }
PLAYER_CACHE = {}

def load_player_cache() -> None:
    """Load cache from disk at startup."""
    global PLAYER_CACHE
    if PLAYER_CACHE_FILE.exists():
        try:
            data = json.loads(PLAYER_CACHE_FILE.read_text())
            # JSON keys come back as strings; coerce to int
            PLAYER_CACHE = {int(k): v for k, v in data.items()}
            print(f"[cache] loaded {len(PLAYER_CACHE)} players")
        except Exception as e:
            print(f"[cache] load failed: {e}")

def save_player_cache() -> None:
    """Persist cache to disk."""
    try:
        # dump with string keys for JSON
        to_dump = {str(k): v for k, v in PLAYER_CACHE.items()}
        PLAYER_CACHE_FILE.write_text(json.dumps(to_dump))
        # print("[cache] saved player cache")
    except Exception as e:
        print(f"[cache] save failed: {e}")

@app.on_event("startup")
def _startup_cache():
    load_player_cache()

@app.on_event("shutdown")
def _shutdown_cache():
    save_player_cache()
    
def get_player_info_cached(league: League, player_id: int):
    """Return dict with position/proTeam/name using cache; fetch & persist on miss."""
    if not player_id:
        return {"position": None, "proTeam": None, "name": None}

    cached = PLAYER_CACHE.get(player_id)
    if cached is not None:
        return cached

    # cache miss -> fetch once
    try:
        pl = league.player_info(playerId=player_id)  # Player object
        info = {
            "position": getattr(pl, "position", None),
            "proTeam": getattr(pl, "proTeam", None),
            "name": getattr(pl, "name", None),
        }
    except Exception as e:
        print(f"[cache] player_info failed for {player_id}: {e}")
        info = {"position": None, "proTeam": None, "name": None}

    PLAYER_CACHE[player_id] = info
    # Persist immediately so subsequent server restarts are warm
    save_player_cache()
    return info




@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _get_league(year: int) -> League:
    if year not in SUPPORTED_YEARS:
        raise HTTPException(status_code=400, detail=f"Year {year} not supported. Supported: {sorted(SUPPORTED_YEARS)}")

    espn_s2 = "AEC20e998honXS4Wi0Z8qzlJdam4%2F%2BaApa7apspnhKR0Npb%2FMsF5DuQsFUcHW%2FhPihQun9U6PGITOi2CkbdfDCkUc8druBVhAwT08Lzrvv8oZli8YAuTi9mIWg7YqtorCNOEKPxHpYswnT3q7b885tRDKBJpLCH0T2h4h1p%2B02SfdlDhjEB2gHqFk1xl6tJRNMBiCkZ8i5RttLW6ER9ZvLTmmAdb5ceZhQ27NEMiMf%2BjWSSvwBdnf2roxwt9baw33BVnnITqYVb8FXsaUwm7%2Bm0m9GLQ%2B66%2BU%2Brg%2BQngXm1ekA%3D%3D"
    swid = "{B431504E-F779-4C49-B3E8-28DDF7409957}"
    kwargs = {"league_id": LEAGUE_ID, "year": year, "swid":swid, "espn_s2":espn_s2}
    if espn_s2 and swid:
        kwargs.update({"espn_s2": espn_s2, "swid": swid})
    return League(**kwargs)


@app.get("/standings/{year}", response_model=StandingsResponse)
def get_standings(year: int) -> StandingsResponse:
    try:
        league = _get_league(year)
    except Exception as e:
        # espn_api throws generic exceptions on missing/unauthorized
        raise HTTPException(status_code=502, detail=str(e))

    teams_out: List[TeamStanding] = []
    for t in league.teams:
        wins = getattr(t, "wins", 0)
        losses = getattr(t, "losses", 0)
        ties = getattr(t, "ties", 0)
        total_games = wins + losses + ties
        
        # Calculate win percentage (wins / total games, default to 0.0 if no games)
        win_percentage = (wins / total_games * 100) if total_games > 0 else 0.0
        
        # espn_api team fields: wins, losses, ties, points_for, points_against, streak_length, streak_type
        teams_out.append(
            TeamStanding(
                team_id=t.team_id,
                team_name=t.team_name,
                wins=wins,
                losses=losses,
                ties=ties,
                points_for=float(getattr(t, "points_for", 0.0)),
                points_against=float(getattr(t, "points_against", 0.0)),
                win_percentage=round(win_percentage, 1),
                streak_length=getattr(t, "streak_length", None),
                streak_type=getattr(t, "streak_type", None),
            )
        )
    
    # Sort teams by win percentage (descending), then by points for as tiebreaker
    teams_out.sort(key=lambda x: (x.win_percentage, x.points_for), reverse=True)

    return StandingsResponse(
        year=year,
        league_id=LEAGUE_ID,
        num_teams=len(teams_out),
        teams=teams_out,
    )


@app.get("/playoffs/{year}", response_model=PlayoffBracket)
def get_playoffs(year: int) -> PlayoffBracket:
    try:
        league = _get_league(year)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    games = []
    
    # Get playoff data from ESPN API by scanning all weeks
    try:
        print(f"=== DEBUG: Starting playoff data collection for year {year} ===")
        
        # Scan all weeks to find playoff matchups
        for week in range(1, 18):  # NFL regular season + playoffs
            try:
                print(f"--- Checking week {week} ---")
                scoreboard = league.scoreboard(week=week)
                
                if scoreboard and len(scoreboard) > 0:
                    print(f"  Week {week}: Found {len(scoreboard)} matchups")
                    
                    for i, matchup in enumerate(scoreboard):
                        # Check if this is a playoff matchup
                        matchup_type = getattr(matchup, 'matchup_type', 'NONE')
                        print(f"    Matchup {i}: type = '{matchup_type}'")
                        
                        # Only process playoff matchups
                        if matchup_type in ['WINNERS_BRACKET', 'LOSERS_CONSOLATION_LADDER', 'WINNERS_CONSOLATION_LADDER']:
                            print(f"    *** PLAYOFF MATCHUP FOUND! ***")
                            try:
                                home_team = getattr(matchup, 'home_team', None)
                                away_team = getattr(matchup, 'away_team', None)
                                
                                if home_team and away_team:
                                    home_name = getattr(home_team, 'team_name', 'TBD')
                                    away_name = getattr(away_team, 'team_name', 'TBD')
                                else:
                                    home_name = 'TBD'
                                    away_name = 'TBD'
                                
                                home_score = getattr(matchup, 'home_score', None)
                                away_score = getattr(matchup, 'away_score', None)
                                
                                print(f"    Adding playoff game: {home_name} vs {away_name} ({home_score} to {away_score})")
                                
                                # Determine winner
                                winner = None
                                if home_score is not None and away_score is not None:
                                    winner = home_name if home_score > away_score else away_name
                                
                                # Determine round name based on matchup type
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
                                    week=week,                           # <— NEW
                                    matchup_type=matchup_type            # <— NEW
                                ))

                                print(f"    *** SUCCESS: Added playoff game to list. Total games now: {len(games)} ***")
                            except Exception as e:
                                print(f"    ERROR processing playoff matchup: {e}")
                                continue
                        else:
                            print(f"    Regular season matchup, skipping")
                            
            except Exception as e:
                print(f"  Week {week} error: {e}")
                continue
        
        print(f"=== DEBUG: Finished scanning all weeks ===")
        print(f"=== Total playoff games found: {len(games)} ===")
        
        def _label_rounds_from_weeks(games: list[PlayoffGame]) -> None:
            """Mutates `games`: sets consistent round_name based on playoff week order."""
            # Order unique winners-bracket weeks: QF -> SF -> Final
            wb_weeks = sorted({
                g.week for g in games
                if g.matchup_type == 'WINNERS_BRACKET' and g.week is not None
            })
            idx_by_week = {wk: i for i, wk in enumerate(wb_weeks)}

            for g in games:
                if g.matchup_type == 'WINNERS_BRACKET' and g.week in idx_by_week:
                    idx = idx_by_week[g.week]
                    if idx == 0:
                        g.round_name = 'Quarterfinals'
                    elif idx == 1:
                        g.round_name = 'Semifinals'
                    elif idx == 2:
                        g.round_name = 'Championship'
                    else:
                        g.round_name = f'Playoffs (Round {idx+1})'
                elif g.matchup_type == 'LOSERS_CONSOLATION_LADDER':
                    g.round_name = 'Consolation'
                elif g.matchup_type == 'WINNERS_CONSOLATION_LADDER':
                    g.round_name = 'Winners Consolation'
                else:
                    g.round_name = g.round_name or 'Playoff'

        # Apply labels if we found any games
        if games:
            _label_rounds_from_weeks(games)
            
        # print(games)

        
        # If no playoff data found, create a basic structure
        if not games:
            print(f"=== NO PLAYOFF GAMES FOUND - Creating fallback structure ===")
            teams = league.teams
            if len(teams) >= 8:
                # 8+ teams: 3 rounds
                # Quarterfinals
                for j in range(0, len(teams), 2):
                    if j + 1 < len(teams):
                        games.append(PlayoffGame(
                            home_team=teams[j].team_name,
                            away_team=teams[j+1].team_name,
                            round_name='Quarterfinals'
                        ))
                
                # Semifinals
                games.append(PlayoffGame(
                    home_team="Winner Q1",
                    away_team="Winner Q2",
                    round_name='Semifinals'
                ))
                games.append(PlayoffGame(
                    home_team="Winner Q3",
                    away_team="Winner Q4",
                    round_name='Semifinals'
                ))
                
                # Championship
                games.append(PlayoffGame(
                    home_team="Winner SF1",
                    away_team="Winner SF2",
                    round_name='Championship'
                ))
                    
            elif len(teams) >= 4:
                # 4-7 teams: 2 rounds
                # Semifinals
                for j in range(0, len(teams), 2):
                    if j + 1 < len(teams):
                        games.append(PlayoffGame(
                            home_team=teams[j].team_name,
                            away_team=teams[j+1].team_name,
                            round_name='Semifinals'
                        ))
                
                # Championship
                games.append(PlayoffGame(
                    home_team="Winner SF1",
                    away_team="Winner SF2",
                    round_name='Championship'
                ))
            else:
                # 2-3 teams: single championship game
                if len(teams) >= 2:
                    games.append(PlayoffGame(
                        home_team=teams[0].team_name,
                        away_team=teams[1].team_name,
                        round_name='Championship'
                    ))
        else:
            print(f"=== PLAYOFF GAMES FOUND - Using real data ===")
    
    except Exception as e:
        print(f"=== MAIN ERROR: {e} ===")
        import traceback
        traceback.print_exc()
        # If playoff data extraction fails, create basic structure
        games = [PlayoffGame(
            home_team="TBD",
            away_team="TBD",
            round_name="Playoff"
        )]
        
    ROUND_ORDER = {
    'Quarterfinals': 1,
    'Semifinals': 2,
    'Championship': 3,
    'Winners Bracket': 50,
    'Winners Consolation': 60,
    'Consolation': 70,
    'Playoff': 90,
    }

    games.sort(
        key=lambda g: (
            ROUND_ORDER.get(g.round_name, 999),
            g.week or 0
        )
    )


    print(f"=== FINAL: Returning {len(games)} games ===")
    return PlayoffBracket(
        year=year,
        league_id=LEAGUE_ID,
        games=games
    )
    
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

        team_count = len(league.teams)
        overall_pick = None
        if isinstance(round_num, int) and isinstance(round_pick, int) and team_count:
            overall_pick = (round_num - 1) * team_count + round_pick

        player_id = getattr(p, "playerId", None)

        # Define player_name FIRST (from the pick), then let cache override if missing
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
            info = get_player_info_cached(league, int(player_id))  # dict from cache
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
                pick_num=pick_num,          # round pick (P:…)
                overall_pick=overall_pick,  # computed
                player_name=player_name,
                position=position,
                pro_team=pro_team,
                is_keeper=is_keeper,
                auction_price=auction_price,
            )
        )


    picks_out.sort(
        key=lambda x: (
            x.round_num if x.round_num is not None else 999,
            x.pick_num if x.pick_num is not None else 999,
        )
    )

    return DraftResponse(year=year, league_id=LEAGUE_ID, picks=picks_out)



@app.get("/metrics/vorp/{year}", response_model=VorpResponse)
def get_vorp(
    year: int,
    use_ppg: bool = Query(False, description="Use points per game"),
    top: int = Query(500, ge=1, le=2000, description="Limit rows"),
):
    # 1) Build the table
    try:
        from vorp import build_vorp_table
        table = build_vorp_table(year=year, use_ppg=use_ppg)
    except Exception as e:
        # Surface the root cause — this is what produced your 502
        raise HTTPException(status_code=502, detail=f"Failed to build VORP*: {e}")

    if table is None or len(table) == 0:
        # Return a valid (empty) response so the UI can fall back gracefully
        return VorpResponse(year=year, players=[], count=0, used_ppg=use_ppg)

    table = table.head(top).copy()

    # 2) Ensure required columns exist
    required = {
        "player_name",
        "fantasy_pos",
        "fantasy_points_ppr",
        "vorp_star",
        "vorp_star_rank_overall",
        "vorp_star_rank_pos",
    }
    missing = [c for c in required if c not in table.columns]
    if missing:
        # Helpful error; you can also log table.columns here
        raise HTTPException(
            status_code=502,
            detail=f"VORP table missing columns {missing}; available={list(table.columns)}",
        )

    # 3) Coerce dtypes and replace NaNs with safe defaults
    import numpy as np
    for col in ["fantasy_points_ppr", "vorp_star"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)

    # ranks must be ints; coerce and drop rows that still fail
    for col in ["vorp_star_rank_overall", "vorp_star_rank_pos"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")

    # optional games played
    if "g" in table.columns:
        table["g"] = pd.to_numeric(table["g"], errors="coerce")
    else:
        table["g"] = np.nan

    # 4) Drop rows that lack essential fields after coercion
    table = table[
        table["player_name"].notna()
        & table["fantasy_pos"].notna()
        & table["vorp_star"].notna()
        & table["vorp_star_rank_overall"].notna()
        & table["vorp_star_rank_pos"].notna()
    ]

    players: list[PlayerVorp] = []
    # Use itertuples for speed and simple attribute access
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
                    # NEW
                    partial_season=bool(getattr(row, "partial_season", False)),
                    vorp_star_extrap=(
                        None if pd.isna(getattr(row, "vorp_star_extrap", np.nan))
                        else float(getattr(row, "vorp_star_extrap"))
                    ),
                )
            )


        except Exception as e:
            # Skip any bad row rather than nuking the whole response
            print(f"[vorp] skipped row due to serialization error: {e}")
            continue

    return VorpResponse(
        year=year,
        players=players,
        count=len(players),
        used_ppg=use_ppg,
    )
    
    
    

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


