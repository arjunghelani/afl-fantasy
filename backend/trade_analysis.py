"""
Trade analysis functions extracted from test.ipynb
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from espn_api.football import League
from urllib.parse import unquote
import requests
from bs4 import BeautifulSoup, Comment
from typing import Dict, List, Tuple, Optional
import os

# Configuration
LEAGUE_ID = 86952922
ESPN_S2 = "AEC20e998honXS4Wi0Z8qzlJdam4%2F%2BaApa7apspnhKR0Npb%2FMsF5DuQsFUcHW%2FhPihQun9U6PGITOi2CkbdfDCkUc8druBVhAwT08Lzrvv8oZli8YAuTi9mIWg7YqtorCNOEKPxHpYswnT3q7b885tRDKBJpLCH0T2h4h1p%2B02SfdlDhjEB2gHqFk1xl6tJRNMBiCkZ8i5RttLW6ER9ZvLTmmAdb5ceZhQ27NEMiMf%2BjWSSvwBdnf2roxwt9baw33BVnnITqYVb8FXsaUwm7%2Bm0m9GLQ%2B66%2BU%2Brg%2BQngXm1ekA%3D%3D"
SWID = "{B431504E-F779-4C49-B3E8-28DDF7409957}"

def get_league(year: int) -> League:
    """Get ESPN league instance for a given year"""
    espn_s2 = unquote(ESPN_S2)
    return League(league_id=LEAGUE_ID, year=year, swid=SWID, espn_s2=espn_s2)

def guess_max_week(league: League) -> int:
    """Guess the maximum week for a league"""
    teams = league.teams
    if not teams:
        return 17
    return max(len(t.scores) for t in teams) or 17

def get_team_map(league: League) -> dict:
    """Get team ID to team name mapping"""
    return {t.team_id: t.team_name for t in league.teams}

def get_weekly_rosters(league: League, week: int) -> Tuple[Dict, Dict, Dict]:
    """
    Returns:
      roster_by_team: dict[team_id] -> set[playerId]
      player_meta: dict[playerId] -> {'name': str}
      team_meta: dict[team_id] -> {'name': str}
    """
    roster_by_team = defaultdict(set)
    player_meta = {}
    team_meta = get_team_map(league)

    try:
        box_scores = league.box_scores(week=week)
    except Exception:
        return {}, {}, team_meta

    for box in box_scores:
        for side in ("home", "away"):
            team = getattr(box, f"{side}_team", None)
            lineup = getattr(box, f"{side}_lineup", []) or []
            
            if team is None:
                continue
            
            if type(team) != int:
                tid = team.team_id
            for p in lineup:
                pid = getattr(p, "playerId", None)
                if pid is None:
                    continue
                roster_by_team[tid].add(pid)
                # Cache simple player metadata for printing later
                if pid not in player_meta:
                    player_meta[pid] = {"name": getattr(p, "name", f"Player {pid}")}

    return roster_by_team, player_meta, {tid: {"name": name} for tid, name in team_meta.items()}

def build_ownership_timeseries(league: League, weeks: List[int]) -> Tuple[Dict, Dict, Dict]:
    """
    Returns:
      owner_by_player: dict[playerId] -> dict[week] -> team_id (or None if not rostered we didn't see them)
      player_meta, team_meta
    """
    owner_by_player = defaultdict(dict)
    player_meta_all = {}
    team_meta = get_team_map(league)

    for wk in weeks:
        r_by_team, player_meta, _team_meta = get_weekly_rosters(league, wk)
        # merge player meta
        for k, v in player_meta.items():
            if k not in player_meta_all:
                player_meta_all[k] = v

        # invert to player->team
        for tid, pset in r_by_team.items():
            for pid in pset:
                owner_by_player[pid][wk] = tid

    return owner_by_player, player_meta_all, {tid: {"name": name} for tid, name in team_meta.items()}

def detect_transfers(owner_by_player: Dict, weeks: List[int]) -> List[Dict]:
    """
    Returns list of dicts:
     {'week': int, 'playerId': int, 'from_team': int, 'to_team': int}
    """
    week_list = sorted(weeks)
    changes = []
    for pid, w2owner in owner_by_player.items():
        last_team = None
        for w in week_list:
            team = w2owner.get(w, None)
            if last_team is None:
                last_team = team
                continue
            if team != last_team:
                # only flag if both sides are not None (ignore FA/unknown -> team)
                if last_team is not None and team is not None:
                    changes.append({
                        "week": w,
                        "playerId": pid,
                        "from_team": last_team,
                        "to_team": team,
                    })
                last_team = team
    return changes

def cluster_trades(changes: List[Dict], min_players_total: int = 2) -> Dict:
    """
    Group changes by week and team pair; classify candidate packages.
    Returns dict:
      week -> list of packages, each:
        {
          'teams': (a, b),             # team_ids sorted
          'a_to_b': [playerId, ...],
          'b_to_a': [playerId, ...],
          'is_trade_like': bool
        }
    """
    by_week_pair = defaultdict(lambda: defaultdict(lambda: {"a_to_b": [], "b_to_a": []}))

    for ch in changes:
        w = ch["week"]
        a, b = ch["from_team"], ch["to_team"]
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        # Normalize direction with respect to sorted pair
        if (a, b) == pair:
            by_week_pair[w][pair]["a_to_b"].append(ch["playerId"])
        else:
            by_week_pair[w][pair]["b_to_a"].append(ch["playerId"])

    # finalize + classify
    result = defaultdict(list)
    for w, pairs in by_week_pair.items():
        for pair, payload in pairs.items():
            a_to_b = payload["a_to_b"]
            b_to_a = payload["b_to_a"]
            is_trade_like = (len(a_to_b) > 0 and len(b_to_a) > 0) or (len(a_to_b) + len(b_to_a) >= min_players_total)
            result[w].append({
                "teams": pair,
                "a_to_b": a_to_b,
                "b_to_a": b_to_a,
                "is_trade_like": is_trade_like
            })
    return result

def build_trade_dataframe(league: League, start_week: int = 1, end_week: Optional[int] = None, only_trade_like: bool = False) -> pd.DataFrame:
    """
    Returns a pandas DataFrame with one row per detected ownership change.
    Columns:
      week, player_id, player_name,
      from_team_id, from_team_name,
      to_team_id, to_team_name,
      trade_like (bool)
    Set only_trade_like=True to keep only symmetric (both directions) moves.
    """
    if end_week is None:
        end_week = guess_max_week(league)
    weeks = range(start_week, end_week + 1)

    # reuse prior pipeline
    owner_by_player, player_meta, team_meta = build_ownership_timeseries(league, weeks)
    changes = detect_transfers(owner_by_player, weeks)
    packages_by_week = cluster_trades(changes, min_players_total=2)

    # fast lookup: (week, fro, to, playerId) -> trade_like?
    trade_like_lookup = {}
    for w, packages in packages_by_week.items():
        for pkg in packages:
            a, b = pkg["teams"]
            tl = pkg["is_trade_like"]
            # players that went a->b
            for pid in pkg["a_to_b"]:
                trade_like_lookup[(w, a, b, pid)] = tl
            # players that went b->a
            for pid in pkg["b_to_a"]:
                trade_like_lookup[(w, b, a, pid)] = tl

    rows = []
    for ch in changes:
        w = ch["week"]
        fro = ch["from_team"]
        to = ch["to_team"]
        pid = ch["playerId"]
        tl = trade_like_lookup.get((w, fro, to, pid), False)

        if only_trade_like and not tl:
            continue

        rows.append({
            "week": w,
            "player_id": pid,
            "player_name": player_meta.get(pid, {}).get("name", str(pid)),
            "from_team_id": fro,
            "from_team_name": team_meta.get(fro, {}).get("name", f"Team {fro}"),
            "to_team_id": to,
            "to_team_name": team_meta.get(to, {}).get("name", f"Team {to}"),
            "trade_like": tl,
        })

    df = pd.DataFrame(rows).sort_values(["week", "player_name"]).reset_index(drop=True)
    return df

def run_trade_analysis(year: int) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """
    Run the complete trade analysis for a given year.
    Returns:
    - trade_df: DataFrame with detected trades
    - packages_by_week: Trade packages grouped by week
    - player_meta: Player metadata
    - team_meta: Team metadata
    """
    try:
        league = get_league(year)
        trade_df = build_trade_dataframe(league, start_week=1, end_week=None, only_trade_like=True)
        
        # Get additional data for analysis
        weeks = range(1, guess_max_week(league) + 1)
        owner_by_player, player_meta, team_meta = build_ownership_timeseries(league, weeks)
        changes = detect_transfers(owner_by_player, weeks)
        packages_by_week = cluster_trades(changes, min_players_total=2)
        
        return trade_df, packages_by_week, player_meta, team_meta
        
    except Exception as e:
        print(f"Error in trade analysis for {year}: {e}")
        return pd.DataFrame(), {}, {}, {}
