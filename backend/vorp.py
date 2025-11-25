import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import re
import numpy as np
from typing import Optional, List, Dict, Any
import duckdb


# espn_s2 = "AEC20e998honXS4Wi0Z8qzlJdam4%2F%2BaApa7apspnhKR0Npb%2FMsF5DuQsFUcHW%2FhPihQun9U6PGITOi2CkbdfDCkUc8druBVhAwT08Lzrvv8oZli8YAuTi9mIWg7YqtorCNOEKPxHpYswnT3q7b885tRDKBJpLCH0T2h4h1p%2B02SfdlDhjEB2gHqFk1xl6tJRNMBiCkZ8i5RttLW6ER9ZvLTmmAdb5ceZhQ27NEMiMf%2BjWSSvwBdnf2roxwt9baw33BVnnITqYVb8FXsaUwm7%2Bm0m9GLQ%2B66%2BU%2Brg%2BQngXm1ekA%3D%3D"
# swid = "{B431504E-F779-4C49-B3E8-28DDF7409957}"
# LEAGUE_ID = 86952922


# def get_league(year):
#     league = League(league_id=LEAGUE_ID, year=year, swid=swid, espn_s2=espn_s2)
#     return league
    
    

# -----------------------------
# Scraper (unchanged)
# -----------------------------
def scrape_pfr_fantasy(year: int) -> pd.DataFrame:
    url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    def find_fantasy_table(soup: BeautifulSoup):
        t = soup.find("table", {"id": "fantasy"})
        if t:
            return t
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if 'id="fantasy"' in c:
                frag = BeautifulSoup(c, "html.parser")
                t = frag.find("table", {"id": "fantasy"})
                if t:
                    return t
        return None

    table = find_fantasy_table(soup)
    if table is None:
        raise ValueError(f"Fantasy table not found for {year}")

    thead = table.find("thead")
    header_rows = [tr for tr in thead.find_all("tr") if "over_header" not in tr.get("class", [])]
    last_hdr = header_rows[-1]

    cols = []
    for th in last_hdr.find_all("th"):
        key = th.get("data-stat", "").strip()
        if key:
            cols.append(key)

    data = []
    tbody = table.find("tbody")
    for tr in tbody.find_all("tr"):
        if "thead" in tr.get("class", []):
            continue
        row = {}
        th = tr.find("th", {"scope": "row"})
        if th is not None:
            k = th.get("data-stat", "").strip() or "rk"
            row[k] = th.get_text(strip=True)
        for td in tr.find_all("td"):
            k = td.get("data-stat", "").strip()
            if not k:
                continue
            row[k] = td.get_text(strip=True)
        if row:
            data.append(row)

    df = pd.DataFrame(data)
    extras = [c for c in df.columns if c not in cols]
    df = df[[c for c in cols if c in df.columns] + extras]

    if "player" in df.columns:
        df = df[df["player"].str.lower() != "player"]

    # Coerce numeric-ish columns
    for c in df.columns:
        if c in ("player", "team", "pos", "tm"):
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")

    return df


# -----------------------------
# Your VORP* formula (season totals)
# -----------------------------
# def compute_vorp_star(
#     df: pd.DataFrame,
#     teams: int = 12,
#     starters_per_team: dict = None,
#     use_ppg: bool = False,
#     min_games_for_ppg: int = 14,     # threshold for “full season”
#     pool_factor: float = 2.0,
#     winsor_limits: tuple = (0.02, 0.98),
# ):
#     """
#     Season VORP* using season totals for replacement & scaling.
#     Adds:
#       - partial_season (True for 3..min_games_for_ppg-1)
#       - vorp_star_extrap (17-game pace VORP*, only meaningful for partial seasons)
#     """
#     df = df.copy()

#     # 1) Metric for current-season VORP* display
#     if use_ppg:
#         df["points_used"] = df["fantasy_points_ppr"] / df["g"].replace(0, np.nan)
#     else:
#         df["points_used"] = df["fantasy_points_ppr"]
#     df["points_used"] = df["points_used"].fillna(0.0)

#     # --- Flags ---
#     FULL_SEASON_MIN = int(min_games_for_ppg)  # e.g., 12
#     df["partial_season"] = (df["g"] >= 3) & (df["g"] < FULL_SEASON_MIN)
#     df["injury_flag"] = df["g"] < FULL_SEASON_MIN  # keep if you still want a general flag

#     # 2) Replacement level per position (from **season totals**)
#     if starters_per_team is None:
#         starters_per_team = {"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25}

#     rep_index = {
#         pos: int(teams * starters_per_team.get(pos, 0))
#         for pos in df["fantasy_pos"].unique()
#     }

#     def winsorize(s, lo=0.02, hi=0.98):
#         ql, qh = np.quantile(s, [lo, hi])
#         return np.clip(s, ql, qh)

#     out_frames = []
#     for pos, grp in df.groupby("fantasy_pos", sort=False):
#         # --- Replacement from TOTALS ---
#         totals_sorted = grp.sort_values("fantasy_points_ppr", ascending=False)
#         R = rep_index.get(pos, 0)
#         if R <= 0 or R > len(totals_sorted):
#             rep_points_total = 0.0
#         else:
#             rep_points_total = float(totals_sorted.iloc[R - 1]["fantasy_points_ppr"])

#         # For the current (display) metric:
#         gsort = grp.sort_values("points_used", ascending=False)

#         # VORP_raw (display metric) must be in same space as replacement.
#         if use_ppg:
#             # estimate season-total from PPG by *games played*
#             points_used_total_like = gsort["points_used"] * gsort["g"]
#         else:
#             points_used_total_like = gsort["points_used"]

#         gsort = gsort.assign(rep_points=rep_points_total)
#         gsort = gsort.assign(vorp_raw=points_used_total_like - rep_points_total)

#         # Robust scale from top-of-pool VORP_raw
#         pool_size = int(max(R * pool_factor, min(len(gsort), R))) if R > 0 else min(len(gsort), 24)
#         pool = gsort.iloc[:pool_size]["vorp_raw"].values
#         pool_w = winsorize(pool, *winsor_limits) if len(pool) else np.array([0.0])

#         scale = float(np.std(pool_w, ddof=0))
#         if not np.isfinite(scale) or scale == 0:
#             scale = 1.0

#         gsort = gsort.assign(pos_scale_robust=scale)
#         gsort = gsort.assign(vorp_star=gsort["vorp_raw"] / scale)

#         # --- 17-game extrapolation (for partial seasons only) ---
#         ppg = gsort["fantasy_points_ppr"] / gsort["g"].replace(0, np.nan)
#         proj_points_17 = (ppg * 17).fillna(gsort["fantasy_points_ppr"])  # fallback if g==0

#         vorp_raw_proj = proj_points_17 - rep_points_total
#         vorp_star_extrap = vorp_raw_proj / scale

#         # only meaningful for partial rows; keep original otherwise
#         gsort = gsort.assign(
#             vorp_star_extrap=np.where(gsort["partial_season"], vorp_star_extrap, gsort["vorp_star"])
#         )

#         out_frames.append(gsort)

#     result = pd.concat(out_frames, axis=0).sort_index()

#     # Ranks on the actual (non-extrapolated) vorp_star
#     result["vorp_star_rank_overall"] = result["vorp_star"].rank(method="dense", ascending=False).astype(int)
#     result["vorp_star_rank_pos"] = (
#         result.groupby("fantasy_pos")["vorp_star"].rank(method="dense", ascending=False).astype(int)
#     )

#     return result


# -----------------------------
# Public builders used by API
# -----------------------------
ALLOWED_POS = {"QB", "RB", "WR", "TE"}

# def build_vorp_table(year:int, use_ppg:bool=False,
#                      teams:int=12,
#                      starters_per_team:dict=None,
#                      pool_factor:float=1.0,
#                      winsor_limits:tuple=(0.02,0.98)) -> pd.DataFrame:
#     if starters_per_team is None:
#         starters_per_team = {"QB":1.25,"RB":2.5,"WR":2.5,"TE":1.25}
#     # 1) scrape
#     df = scrape_pfr_fantasy(year)

#     # 2) clean
#     df["player_name"] = (
#         df["player"]
#         .astype(str)
#         .str.replace(r"[*+.]", "", regex=True)  # remove *, +, and .
#         .str.replace(r"\s+", " ", regex=True)
#         .str.strip()
#     )

#     cols_keep = ["player_name", "team", "fantasy_pos", "g", "gs", "fantasy_points_ppr"]
#     stats = df.loc[:, [c for c in cols_keep if c in df.columns]].copy()
#     stats = stats[stats["fantasy_pos"].isin(ALLOWED_POS)].copy()

#     for c in ["g", "gs", "fantasy_points_ppr"]:
#         if c in stats.columns:
#             stats[c] = pd.to_numeric(stats[c], errors="coerce").fillna(0.0)

#     # 3) compute VORP* (season formula)
#     df_v = compute_vorp_star(
#         stats,
#         teams=teams,
#         starters_per_team=starters_per_team,
#         use_ppg=use_ppg,           # your choice
#         min_games_for_ppg=14,
#         pool_factor=1,
#         winsor_limits=(0.02, 0.98),
#     )

#     out_cols = [
#         "player_name", "team", "fantasy_pos", "g",
#         "fantasy_points_ppr", "vorp_star",
#         "vorp_star_rank_overall", "vorp_star_rank_pos",
#         "partial_season", "vorp_star_extrap",
#     ]
#     out = df_v[out_cols].sort_values("vorp_star", ascending=False).reset_index(drop=True)
#     return out


'''
____________________________________________________________________________________________
____________________________________________________________________________________________
____________________________________________________________________________________________
____________________________________________________________________________________________
____________________________________________________________________________________________
____________________________________________________________________________________________

'''
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

ALLOWED_POS = {"QB", "RB", "WR", "TE"}

# ---------------------------------------------------------------------
# Season VORP★ (your original logic, kept intact)
# ---------------------------------------------------------------------
def compute_vorp_star(
    df: pd.DataFrame,
    teams: int = 12,
    starters_per_team: dict = None,
    use_ppg: bool = False,
    min_games_for_ppg: int = 14,     # threshold for “full season”
    pool_factor: float = 2.0,
    winsor_limits: tuple = (0.02, 0.98),
) -> pd.DataFrame:
    """
    Season VORP* using season totals for replacement & scaling.
    Adds:
      - partial_season (True for 3..min_games_for_ppg-1)
      - vorp_star_extrap (17-game pace VORP*, only meaningful for partial seasons)
    Expected columns in df:
      ["player_name","team","fantasy_pos","g","gs","fantasy_points_ppr"]
    """
    df = df.copy()

    # 1) Metric for current-season VORP* display
    if use_ppg:
        df["points_used"] = df["fantasy_points_ppr"] / df["g"].replace(0, np.nan)
    else:
        df["points_used"] = df["fantasy_points_ppr"]
    df["points_used"] = df["points_used"].fillna(0.0)

    # --- Flags ---
    FULL_SEASON_MIN = int(min_games_for_ppg)
    df["partial_season"] = (df["g"] >= 3) & (df["g"] < FULL_SEASON_MIN)
    df["injury_flag"] = df["g"] < FULL_SEASON_MIN

    # 2) Replacement level per position (from **season totals**)
    if starters_per_team is None:
        starters_per_team = {"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25}

    rep_index = {
        pos: int(teams * starters_per_team.get(pos, 0))
        for pos in df["fantasy_pos"].unique()
    }

    def winsorize(s, lo=0.02, hi=0.98):
        ql, qh = np.quantile(s, [lo, hi]) if len(s) else (0.0, 0.0)
        return np.clip(s, ql, qh)

    out_frames = []
    for pos, grp in df.groupby("fantasy_pos", sort=False):
        # --- Replacement from TOTALS ---
        totals_sorted = grp.sort_values("fantasy_points_ppr", ascending=False)
        R = rep_index.get(pos, 0)
        if R <= 0 or R > len(totals_sorted):
            rep_points_total = 0.0
        else:
            rep_points_total = float(totals_sorted.iloc[R - 1]["fantasy_points_ppr"])

        # For the display metric:
        gsort = grp.sort_values("points_used", ascending=False)

        # VORP_raw must be in same space as replacement (season totals)
        if use_ppg:
            points_used_total_like = gsort["points_used"] * gsort["g"]
        else:
            points_used_total_like = gsort["points_used"]

        gsort = gsort.assign(rep_points=rep_points_total)
        gsort = gsort.assign(vorp_raw=points_used_total_like - rep_points_total)

        # Robust pos scale from top-of-pool VORP_raw
        pool_size = int(max(R * pool_factor, min(len(gsort), R))) if R > 0 else min(len(gsort), 24)
        pool = gsort.iloc[:pool_size]["vorp_raw"].values
        pool_w = winsorize(pool, *winsor_limits) if len(pool) else np.array([0.0])

        scale = float(np.std(pool_w, ddof=0))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0

        gsort = gsort.assign(pos_scale_robust=scale)
        gsort = gsort.assign(vorp_star=gsort["vorp_raw"] / scale)

        # --- 17-game extrapolation (for partial seasons only) ---
        ppg = gsort["fantasy_points_ppr"] / gsort["g"].replace(0, np.nan)
        proj_points_17 = (ppg * 17).fillna(gsort["fantasy_points_ppr"])  # fallback if g==0

        vorp_raw_proj = proj_points_17 - rep_points_total
        vorp_star_extrap = vorp_raw_proj / scale

        # only meaningful for partial rows; keep original otherwise
        gsort = gsort.assign(
            vorp_star_extrap=np.where(gsort["partial_season"], vorp_star_extrap, gsort["vorp_star"])
        )

        out_frames.append(gsort)

    result = pd.concat(out_frames, axis=0).sort_index()

    # Ranks on the actual (non-extrapolated) vorp_star
    result["vorp_star_rank_overall"] = result["vorp_star"].rank(method="dense", ascending=False).astype(int)
    result["vorp_star_rank_pos"] = (
        result.groupby("fantasy_pos")["vorp_star"].rank(method="dense", ascending=False).astype(int)
    )

    return result


# ---------------------------------------------------------------------
# NEW: Weekly VORP★ with rescaled weekly cutoffs (sums match season VORP★)
# ---------------------------------------------------------------------
# UNUSED FUNCTION - Commented out as not imported in main.py
# def compute_weekly_vorp_star_rescaled(
#     season_df: pd.DataFrame,
#     weekly_df: pd.DataFrame,
#     teams: int = 12,
#     starters_per_team: Dict[str, float] = None,
#     pool_factor: float = 2.0,
#     winsor_limits: tuple = (0.02, 0.98),
#     weeks_to_use: Optional[list] = None,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Build weekly VORP and VORP★ so that weekly sums equal your season VORP and VORP★.

#     Inputs:
#       season_df columns (season totals): ["player_name","team","fantasy_pos","g","gs","fantasy_points_ppr"]
#       weekly_df columns (game logs):     ["player_name","fantasy_pos","week","fantasy_points_ppr_week"]

#     Returns:
#       weekly_out:
#         ["player_name","fantasy_pos","week","fantasy_points_ppr_week",
#          "r_week_rescaled","vorp_week_raw","vorp_week_star"]
#       season_check:
#         aggregate by player with:
#          ["player_name","fantasy_pos","season_points",
#           "season_vorp_raw_from_weekly","season_vorp_star_from_weekly",
#           "season_vorp_raw_reference","season_vorp_star_reference"]
#       (You can compare *_from_weekly vs *_reference to verify equality.)
#     """
#     if starters_per_team is None:
#         starters_per_team = {"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25}

#     # --- 0) Sanity & prep
#     season_df = season_df.copy()
#     weekly_df = weekly_df.copy()
#     season_df = season_df[season_df["fantasy_pos"].isin(ALLOWED_POS)].copy()
#     weekly_df = weekly_df[weekly_df["fantasy_pos"].isin(ALLOWED_POS)].copy()

#     # Ensure numeric
#     season_df["fantasy_points_ppr"] = pd.to_numeric(season_df["fantasy_points_ppr"], errors="coerce").fillna(0.0)
#     weekly_df["fantasy_points_ppr_week"] = pd.to_numeric(weekly_df["fantasy_points_ppr_week"], errors="coerce").fillna(0.0)

#     # Which weeks count?
#     if weeks_to_use is None:
#         weeks_to_use = sorted(weekly_df["week"].dropna().unique().tolist())

#     # --- 1) REPLACEMENT season totals per position (same as season method)
#     rep_index = {
#         pos: int(teams * starters_per_team.get(pos, 0))
#         for pos in season_df["fantasy_pos"].unique()
#     }

#     rep_points_total_by_pos: Dict[str, float] = {}
#     for pos, grp in season_df.groupby("fantasy_pos", sort=False):
#         totals_sorted = grp.sort_values("fantasy_points_ppr", ascending=False)
#         R = rep_index.get(pos, 0)
#         if R <= 0 or R > len(totals_sorted):
#             rep_points_total_by_pos[pos] = 0.0
#         else:
#             rep_points_total_by_pos[pos] = float(totals_sorted.iloc[R - 1]["fantasy_points_ppr"])

#     # --- 2) POS SCALE (same winsorized-top-of-pool logic)
#     def winsorize(s, lo=0.02, hi=0.98):
#         ql, qh = np.quantile(s, [lo, hi]) if len(s) else (0.0, 0.0)
#         return np.clip(s, ql, qh)

#     pos_scale_by_pos: Dict[str, float] = {}
#     for pos, grp in season_df.groupby("fantasy_pos", sort=False):
#         totals_sorted = grp.sort_values("fantasy_points_ppr", ascending=False)
#         R = rep_index.get(pos, 0)
#         rep_points_total = rep_points_total_by_pos.get(pos, 0.0)
#         vorp_raw_total = totals_sorted["fantasy_points_ppr"] - rep_points_total

#         pool_size = int(max(R * pool_factor, min(len(totals_sorted), R))) if R > 0 else min(len(totals_sorted), 24)
#         pool = vorp_raw_total.iloc[:pool_size].values
#         pool_w = winsorize(pool, *winsor_limits) if len(pool) else np.array([0.0])

#         scale = float(np.std(pool_w, ddof=0))
#         if not np.isfinite(scale) or scale == 0:
#             scale = 1.0
#         pos_scale_by_pos[pos] = scale

#     # --- 3) Weekly K-th cutoffs per position (environment), then rescale
#     # K = round(teams * starters_per_team[pos])
#     K_by_pos = {pos: max(1, int(round(teams * starters_per_team.get(pos, 0.0)))) for pos in ALLOWED_POS}

#     # raw weekly cutoffs \tilde r_{pos, t}
#     r_tilde_rows = []
#     by_pos_week = weekly_df.groupby(["fantasy_pos", "week"])
#     for (pos, wk), g in by_pos_week:
#         if wk not in weeks_to_use: 
#             continue
#         scores = g["fantasy_points_ppr_week"].dropna().sort_values(ascending=False)
#         n = len(scores)
#         K = min(K_by_pos.get(pos, 0), n) if n > 0 else 0
#         R_week = float(scores.iloc[K-1]) if K > 0 else 0.0
#         r_tilde_rows.append({"fantasy_pos": pos, "week": wk, "r_tilde": R_week})
#     r_tilde = pd.DataFrame(r_tilde_rows)

#     # Rescale so sum_t r_{pos,t} == season replacement total P_rep,pos
#     r_scaled_rows = []
#     for pos, g in r_tilde.groupby("fantasy_pos"):
#         rep_total = rep_points_total_by_pos.get(pos, 0.0)
#         denom = g["r_tilde"].sum()
#         c_pos = (rep_total / denom) if denom > 0 else (rep_total / max(len(weeks_to_use), 1))
#         tmp = g.copy()
#         tmp["r_week_rescaled"] = c_pos * tmp["r_tilde"]
#         r_scaled_rows.append(tmp)
#     r_scaled = pd.concat(r_scaled_rows, ignore_index=True) if r_scaled_rows else pd.DataFrame(columns=["fantasy_pos","week","r_tilde","r_week_rescaled"])

#     # --- 4) Weekly VORP and VORP★
#     weekly_out = weekly_df.merge(r_scaled[["fantasy_pos","week","r_week_rescaled"]],
#                                  on=["fantasy_pos","week"], how="left")
#     weekly_out["r_week_rescaled"] = weekly_out["r_week_rescaled"].fillna(0.0)

#     weekly_out["vorp_week_raw"] = weekly_out["fantasy_points_ppr_week"] - weekly_out["r_week_rescaled"]
#     weekly_out["pos_scale"] = weekly_out["fantasy_pos"].map(pos_scale_by_pos).fillna(1.0)
#     weekly_out["vorp_week_star"] = weekly_out["vorp_week_raw"] / weekly_out["pos_scale"]

#     # --- 5) Season check: do weekly sums match season VORP & VORP★?
#     season_from_weekly = (weekly_out
#         .groupby(["player_name","fantasy_pos"], as_index=False)
#         .agg(season_points=("fantasy_points_ppr_week","sum"),
#              season_vorp_raw_from_weekly=("vorp_week_raw","sum"),
#              season_vorp_star_from_weekly=("vorp_week_star","sum"))
#     )

#     # Reference season VORP & VORP★ from totals method
#     season_ref = season_df.copy()
#     season_ref["rep_points_total"] = season_ref["fantasy_pos"].map(rep_points_total_by_pos).fillna(0.0)
#     season_ref["season_vorp_raw_reference"] = season_ref["fantasy_points_ppr"] - season_ref["rep_points_total"]
#     season_ref["pos_scale"] = season_ref["fantasy_pos"].map(pos_scale_by_pos).fillna(1.0)
#     season_ref["season_vorp_star_reference"] = season_ref["season_vorp_raw_reference"] / season_ref["pos_scale"]

#     season_check = (season_from_weekly
#         .merge(season_ref[["player_name","fantasy_pos","fantasy_points_ppr",
#                            "season_vorp_raw_reference","season_vorp_star_reference"]],
#                on=["player_name","fantasy_pos"], how="left")
#         .rename(columns={"fantasy_points_ppr":"season_points_reference"})
#     )

#     return weekly_out, season_check
def get_weekly_fantasy_points_from_players(year, max_week, league) -> pd.DataFrame:
    """Get weekly fantasy points from box scores (starters + bench) and free agents"""
    # league = get_league(year)
    rows = []
    b = 0
    f = 0
    
    for week in range(1, max_week + 1):
        try:
            # Get box scores for starters and bench
            box_scores = league.box_scores(week=week)
            for box in box_scores:
                for side in ("home_lineup", "away_lineup", "home_bench", "away_bench"):
                    lineup = getattr(box, side, []) or []
                    for player in lineup:
                        player_name = getattr(player, "name", None)
                        position = getattr(player, "position", None)
                        points = getattr(player, "points", None)
                        
                        if player_name and position and points is not None:
                            rows.append({
                                "player_name": str(player_name),
                                "fantasy_pos": str(position),
                                "week": week,
                                "weekly_points_ppr": float(points)
                            })
                            b+=1
            
            # Get free agents
            free_agents = league.free_agents()
            for player in free_agents:
                if hasattr(player, 'stats') and player.stats and str(week) in player.stats:
                    week_data = player.stats[str(week)]
                    if 'points' in week_data:
                        rows.append({
                            "player_name": str(player.name),
                            "fantasy_pos": str(player.position),
                            "week": week,
                            "weekly_points_ppr": float(week_data['points'])
                        })
                        f+=1
        except Exception as e:
            continue
        
    print(f"Box scores: {b}, Free agents: {f}")
    
    return pd.DataFrame(rows)


def fill_missing_weeks(df, year, league) -> pd.DataFrame:
    """Fill missing weeks for players who don't have 17 weeks of data"""
    
    # Find players with less than 17 weeks
    player_week_counts = df.groupby('player_name')['week'].count()
    incomplete_players = player_week_counts[player_week_counts < 17].index.tolist()
    
    if not incomplete_players:
        return df
    
    # Get stats for incomplete players
    additional_rows = []
    for player_name in incomplete_players:
        try:
            player_info = league.player_info(player_name)
            team = player_info.proTeam
            
            if hasattr(player_info, 'stats') and player_info.stats:
                for week in range(1, 18):
        
                    
                    week_data = player_info.stats[week]

                    if 'points' in week_data:
                        additional_rows.append({
                            "player_name": str(player_name),
                            "fantasy_pos": str(player_info.position),
                            "week": week,
                            "weekly_points_ppr": float(week_data['points']),
                            'team':str(team)
                        })
        except Exception as e:
            continue
    
    # Combine original data with additional data
    if additional_rows:
        additional_df = pd.DataFrame(additional_rows)
        return pd.concat([df, additional_df], ignore_index=True)
    return df


# ---------------------------------------------------------------------
# Example wrapper (optional)
# ---------------------------------------------------------------------
def build_vorp_table(year, use_ppg=False,
                     teams=12,
                     starters_per_team=None,
                     pool_factor=1.0,
                     winsor_limits=(0.02,0.98),
                     league=None) -> pd.DataFrame:
    """
    Keeps your original builder (season VORP★). You can separately call
    compute_weekly_vorp_star_rescaled(...) once you have weekly logs.
    """
    if starters_per_team is None:
        starters_per_team = {"QB":1.25,"RB":2.5,"WR":2.5,"TE":1.25}
        
    # 1) scrape (you provide these)
    # df = scrape_pfr_fantasy(year)  # season totals only
    
    df = get_weekly_fantasy_points_from_players(year, max_week=17, league=league)
    df = fill_missing_weeks(df, year, league)
    df = df.drop_duplicates(subset=['player_name', 'week'])
    df = df.loc[df.weekly_points_ppr != 0]
        
    
    result = duckdb.sql("""
                    with totals as (
                        select player_name, fantasy_pos, team, sum(weekly_points_ppr) as total_points,
                        rank() over (partition by fantasy_pos order by sum(weekly_points_ppr) desc) as pos_rank,
                        rank() over (order by sum(weekly_points_ppr) desc) as overall_rank
                        from df
                        group by player_name, fantasy_pos, team 
                        order by total_points desc
                    )
                    
                    SELECT *
                    FROM df
                    left join totals using (player_name, fantasy_pos)
                    """).df()
    
    
    caps = pd.DataFrame(result.groupby(['fantasy_pos']).agg({'pos_rank':'max'}).\
        pos_rank.apply(lambda x:np.ceil(0.60 * x))).reset_index()
    
    ## filter top 60%
    result2 = duckdb.sql("""
           select player_name, fantasy_pos, team, week, weekly_points_ppr, result.pos_rank, overall_rank from result 
           left join caps using (fantasy_pos)
           where result.pos_rank <= caps.pos_rank
           """).df()
    
    result2['log_ppr'] = result2.weekly_points_ppr.apply(lambda x:np.log10(x))
    
    ## Weekly z-scores
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
    
    
    ## Season z-scores (mean and sd)
    z_season = z_scores.groupby(['player_name', 'team']).agg({'fantasy_pos':'first', 'z_week_ppr':'sum', 
            'log_ppr':'count', 'weekly_points_ppr':'sum'}).reset_index().sort_values(['z_week_ppr'], ascending=False)
    
    z_season['inj_flag'] = (z_season['log_ppr'] < 14).astype(int)
    
    # Extrapolate injury z-score
    z_season = duckdb.sql("""
           select *,
           case when inj_flag = 1 then z_week_ppr * 16 / log_ppr else z_week_ppr end as adj_z_week_ppr
           from z_season
           """).df()
    
    z_season = z_season.rename(columns={'log_ppr':'g', 'weekly_points_ppr':'fantasy_points_ppr'})
    
    
    z_season["player_name"] = (
        z_season["player_name"]
        .astype(str)
        .str.replace(r"[*+.]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    
    # 2) clean
    # try:
    #     df["player_name"] = (
    #         df["player"]
    #         .astype(str)
    #         .str.replace(r"[*+.]", "", regex=True)
    #         .str.replace(r"\s+", " ", regex=True)
    #         .str.strip()
    #     )
    # except Exception as e:
    #     continue

    # cols_keep = ["player_name", "team", "fantasy_pos", "g", "gs", "fantasy_points_ppr"]
    # stats = df.loc[:, [c for c in cols_keep if c in df.columns]].copy()
    # stats = stats[stats["fantasy_pos"].isin(ALLOWED_POS)].copy()

    # for c in ["g", "gs", "fantasy_points_ppr"]:
    #     if c in stats.columns:
    #         stats[c] = pd.to_numeric(stats[c], errors="coerce").fillna(0.0)

    # # 3) compute VORP* (season formula)
    # df_v = compute_vorp_star(
    #     stats,
    #     teams=teams,
    #     starters_per_team=starters_per_team,
    #     use_ppg=use_ppg,           # your choice
    #     min_games_for_ppg=14,
    #     pool_factor=pool_factor,
    #     winsor_limits=winsor_limits,
    # )

    out_cols = [
        "player_name", "team", "fantasy_pos", "g",
        "fantasy_points_ppr", "z_week_ppr",
        "inj_flag", "adj_z_week_ppr",
    ]
    out = z_season[out_cols]
    return out

'''
____________________________________________________________________________________________
____________________________________________________________________________________________
____________________________________________________________________________________________
____________________________________________________________________________________________
____________________________________________________________________________________________
'''





# =============================================================================
# NEW: Season baseline & scale helpers (for weekly + simulation, Option B)
# =============================================================================
# UNUSED FUNCTION - Commented out as not imported in main.py
# def compute_season_baseline_and_scale(
#     season_df: pd.DataFrame,
#     teams: int = 12,
#     starters_per_team: Optional[Dict[str, float]] = None,
#     pool_factor: float = 1.0,
#     winsor_limits: tuple = (0.02, 0.98),
# ) -> tuple[Dict[str, float], Dict[str, float]]:
#     """
#     From season totals, compute:
#       - rep_points_total_by_pos: {pos -> season replacement total points}
#       - scale_by_pos: {pos -> robust season SD of (points - replacement) among top-of-pool}
#     """
#     if starters_per_team is None:
#         starters_per_team = {"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25}

#     rep_index = {pos: int(teams * starters_per_team.get(pos, 0))
#                  for pos in season_df["fantasy_pos"].unique()}

#     def winsorize(s, lo=0.02, hi=0.98):
#         ql, qh = np.quantile(s, [lo, hi]) if len(s) else (0.0, 0.0)
#         return np.clip(s, ql, qh)

#     rep_points_total_by_pos: Dict[str, float] = {}
#     scale_by_pos: Dict[str, float] = {}

#     for pos, grp in season_df.groupby("fantasy_pos", sort=False):
#         totals_sorted = grp.sort_values("fantasy_points_ppr", ascending=False)
#         R = rep_index.get(pos, 0)
#         if R <= 0 or R > len(totals_sorted):
#             rep_points_total = 0.0
#         else:
#             rep_points_total = float(totals_sorted.iloc[R - 1]["fantasy_points_ppr"])
#         rep_points_total_by_pos[pos] = rep_points_total

#         vorp_raw = totals_sorted["fantasy_points_ppr"] - rep_points_total
#         pool_size = int(max(R * pool_factor, min(len(totals_sorted), R))) if R > 0 else min(len(totals_sorted), 24)
#         pool = vorp_raw.iloc[:pool_size].values
#         pool_w = winsorize(pool, *winsor_limits) if len(pool) else np.array([0.0])

#         scale = float(np.std(pool_w, ddof=0))
#         scale_by_pos[pos] = scale if np.isfinite(scale) and scale != 0 else 1.0

#     return rep_points_total_by_pos, scale_by_pos


# =============================================================================
# NEW: Weekly VORP* using season baseline & scale (Option B)
# =============================================================================
# UNUSED FUNCTION - Commented out as not imported in main.py
# def compute_weekly_vorp_star(
#     weekly_df: pd.DataFrame,
#     rep_points_total_by_pos: Dict[str, float],
#     scale_by_pos: Dict[str, float],
#     weeks_in_season: int = 17,
#     weekly_points_col: str = "weekly_points_ppr",
# ) -> pd.DataFrame:
#     """
#     Input weekly_df columns (minimum):
#       ['player_name','team','fantasy_pos','week', weekly_points_col]
#     Output adds:
#       ['rep_points_week','vorp_raw_week','vorp_star_week','scale_pos_season']
#     Weekly baseline = (season_rep_total / weeks_in_season), scale = season scale.
#     """
#     wdf = weekly_df.copy()

#     wdf["rep_points_week"] = wdf["fantasy_pos"].map(
#         lambda p: rep_points_total_by_pos.get(p, 0.0) / float(weeks_in_season)
#     )
#     wdf["scale_pos_season"] = wdf["fantasy_pos"].map(lambda p: scale_by_pos.get(p, 1.0)).replace(0.0, 1.0)

#     wdf["vorp_raw_week"] = wdf[weekly_points_col].fillna(0.0) - wdf["rep_points_week"]
#     wdf["vorp_star_week"] = wdf["vorp_raw_week"] / wdf["scale_pos_season"]

#     return wdf


# UNUSED FUNCTION - Commented out as not imported in main.py
# def build_weekly_vorp_table(
#     year: int,
#     weekly_points_df: pd.DataFrame,
#     weeks_in_season: int = 17,
#     teams: int = 12,
#     starters_per_team: Optional[Dict[str, float]] = None,
#     pool_factor: float = 1.0,
#     winsor_limits: tuple = (0.02, 0.98),
#     weekly_points_col: str = "weekly_points_ppr",
# ) -> pd.DataFrame:
#     """
#     Option B weekly VORP* table.
#     """
#     if starters_per_team is None:
#         starters_per_team = {"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25}

#     # Season scrape -> compute season baselines & scales
#     season_df_raw = scrape_pfr_fantasy(year)
#     season_df_raw["player_name"] = (
#         season_df_raw["player"].astype(str)
#         .str.replace(r"[*+.]", "", regex=True)
#         .str.replace(r"\s+", " ", regex=True)
#         .str.strip()
#     )
#     season_stats = season_df_raw.loc[:, ["player_name","team","fantasy_pos","fantasy_points_ppr"]].copy()
#     season_stats = season_stats[season_stats["fantasy_pos"].isin(ALLOWED_POS)].copy()
#     season_stats["fantasy_points_ppr"] = pd.to_numeric(season_stats["fantasy_points_ppr"], errors="coerce").fillna(0.0)

#     rep_by_pos, scale_by_pos = compute_season_baseline_and_scale(
#         season_stats,
#         teams=teams,
#         starters_per_team=starters_per_team,
#         pool_factor=pool_factor,
#         winsor_limits=winsor_limits,
#     )

#     # Prepare weekly df
#     wdf = weekly_points_df.copy()
#     required = {"player_name","team","fantasy_pos","week", weekly_points_col}
#     missing = required - set(wdf.columns)
#     if missing:
#         raise ValueError(f"weekly_points_df missing required columns: {missing}")
#     wdf = wdf[wdf["fantasy_pos"].isin(ALLOWED_POS)].copy()

#     wdf["player_name"] = (
#         wdf["player_name"].astype(str)
#         .str.replace(r"[*+.]", "", regex=True)
#         .str.replace(r"\s+", " ", regex=True)
#         .str.strip()
#     )
#     wdf["week"] = pd.to_numeric(wdf["week"], errors="coerce").astype(int)
#     wdf[weekly_points_col] = pd.to_numeric(wdf[weekly_points_col], errors="coerce").fillna(0.0)

#     return compute_weekly_vorp_star(
#         wdf,
#         rep_points_total_by_pos=rep_by_pos,
#         scale_by_pos=scale_by_pos,
#         weeks_in_season=weeks_in_season,
#         weekly_points_col=weekly_points_col,
#     )


# =============================================================================
# NEW: Monte Carlo injury extrapolation in VORP* units (Option B-consistent)
# =============================================================================
# UNUSED FUNCTION - Commented out as not imported in main.py
# def simulate_injury_extrapolation(
#     weekly_df: pd.DataFrame,
#     rep_points_total_by_pos: Dict[str, float],
#     scale_by_pos: Dict[str, float],
#     *,
#     weeks_in_season: int = 17,
#     sims: int = 1000,
#     min_weeks_for_player_bootstrap: int = 4,
#     played_points_threshold: float = 0.1,
#     positional_pool_clip: tuple = (0.01, 0.99),
#     random_state: Optional[int] = None,
#     weekly_points_col: str = "weekly_points_ppr",
# ) -> pd.DataFrame:
#     """
#     Monte Carlo estimate of expected *additional* VORP* a player would have produced
#     in *missed* weeks. Uses player's own weekly distribution with shrinkage to the
#     positional weekly distribution when sample is small.

#     Returns per-player DataFrame with columns:
#       ['player_name','fantasy_pos','weeks_played','missed_weeks',
#        'rep_points_week','scale_pos_season',
#        'delta_vorp_star_mean','delta_vorp_star_p10','delta_vorp_star_p90',
#        'delta_vorp_raw_mean','delta_vorp_raw_p10','delta_vorp_raw_p90']
#     """
#     rng = np.random.default_rng(random_state)

#     df = weekly_df.copy()
#     df[weekly_points_col] = pd.to_numeric(df[weekly_points_col], errors="coerce").fillna(0.0)

#     # Build positional weekly pools (clip tails for robustness)
#     pos_pools: Dict[str, np.ndarray] = {}
#     for pos, grp in df.groupby("fantasy_pos", sort=False):
#         pts = grp[weekly_points_col].values.astype(float)
#         if len(pts) == 0:
#             pos_pools[pos] = np.array([0.0])
#             continue
#         lo, hi = np.quantile(pts, positional_pool_clip)
#         pts = np.clip(pts, lo, hi)
#         pos_pools[pos] = pts

#     # Precompute per-position weekly replacement & scale
#     rep_week_by_pos = {p: rep_points_total_by_pos.get(p, 0.0) / float(weeks_in_season) for p in rep_points_total_by_pos}
#     scale_by_pos_safe = {p: (s if np.isfinite(s) and s > 0 else 1.0) for p, s in scale_by_pos.items()}

#     out_rows = []

#     for (player, pos), grp in df.groupby(["player_name", "fantasy_pos"], sort=False):
#         pos_pool = pos_pools.get(pos, np.array([0.0]))
#         rep_w = rep_week_by_pos.get(pos, 0.0)
#         scale = scale_by_pos_safe.get(pos, 1.0)

#         # Played weeks (we treat > threshold as "played"; zeros count as missed)
#         pts_all = grp[weekly_points_col].values.astype(float)
#         played_mask = pts_all > float(played_points_threshold)
#         played_pts = pts_all[played_mask]
#         weeks_played = int(played_mask.sum())
#         missed_weeks = int(max(0, int(weeks_in_season) - weeks_played))

#         if missed_weeks == 0:
#             out_rows.append({
#                 "player_name": player,
#                 "fantasy_pos": pos,
#                 "weeks_played": weeks_played,
#                 "missed_weeks": 0,
#                 "rep_points_week": rep_w,
#                 "scale_pos_season": scale,
#                 "delta_vorp_star_mean": 0.0,
#                 "delta_vorp_star_p10": 0.0,
#                 "delta_vorp_star_p90": 0.0,
#                 "delta_vorp_raw_mean": 0.0,
#                 "delta_vorp_raw_p10": 0.0,
#                 "delta_vorp_raw_p90": 0.0,
#             })
#             continue

#         # Determine shrinkage λ based on own sample size
#         n = len(played_pts)
#         if n >= min_weeks_for_player_bootstrap:
#             lam = 0.0
#         elif n >= 3:
#             lam = 0.25
#         elif n >= 1:
#             lam = 0.50
#         else:
#             lam = 1.0  # no player data, fall back entirely to positional pool

#         # Prepare draws
#         # If we have player data, draw from it; otherwise draw zeros (will be overridden by mask)
#         if n > 0:
#             draws_player = rng.choice(played_pts, size=(sims, missed_weeks), replace=True)
#         else:
#             draws_player = np.zeros((sims, missed_weeks), dtype=float)

#         draws_pos = rng.choice(pos_pool, size=(sims, missed_weeks), replace=True)

#         # Mixture: with prob λ use positional draw, else player's draw
#         if lam >= 1.0 or n == 0:
#             draws = draws_pos
#         elif lam <= 0.0:
#             draws = draws_player
#         else:
#             mask = rng.random((sims, missed_weeks)) < lam
#             draws = np.where(mask, draws_pos, draws_player)

#         # Convert to VORP* units against season baselines
#         vorp_raw = draws - rep_w            # (sims, missed_weeks)
#         vorp_star = vorp_raw / scale

#         sums_raw = vorp_raw.sum(axis=1)
#         sums_star = vorp_star.sum(axis=1)

#         out_rows.append({
#             "player_name": player,
#             "fantasy_pos": pos,
#             "weeks_played": weeks_played,
#             "missed_weeks": missed_weeks,
#             "rep_points_week": rep_w,
#             "scale_pos_season": scale,
#             "delta_vorp_star_mean": float(np.mean(sums_star)),
#             "delta_vorp_star_p10": float(np.quantile(sums_star, 0.10)),
#             "delta_vorp_star_p90": float(np.quantile(sums_star, 0.90)),
#             "delta_vorp_raw_mean": float(np.mean(sums_raw)),
#             "delta_vorp_raw_p10": float(np.quantile(sums_raw, 0.10)),
#             "delta_vorp_raw_p90": float(np.quantile(sums_raw, 0.90)),
#         })

#     return pd.DataFrame(out_rows)


# =============================================================================
# NEW: Convenience builder that returns True WAR + Monte Carlo Δ + Adjusted
# =============================================================================
# =============================================================================
# REPLACEMENT: Linear (PPG*17) extrapolated WAR — matches draft board logic
# =============================================================================
# UNUSED FUNCTION - Commented out as not imported in main.py
# def build_extrapolated_vorp_table(
#     year: int,
#     weekly_points_df: pd.DataFrame,   # kept for signature compatibility; not used
#     *,
#     weeks_in_season: int = 17,
#     teams: int = 12,
#     starters_per_team: Optional[Dict[str, float]] = None,
#     pool_factor: float = 1.0,         # unused here; kept for API stability
#     winsor_limits: tuple = (0.02, 0.98),  # unused here; kept for API stability
#     sims: int = 1000,                 # unused here; kept for API stability
#     min_weeks_for_player_bootstrap: int = 4,  # unused here; kept for API stability
#     played_points_threshold: float = 0.1,     # unused here; kept for API stability
#     positional_pool_clip: tuple = (0.01, 0.99),  # unused here; kept for API stability
#     random_state: Optional[int] = None,          # unused here; kept for API stability
#     weekly_points_col: str = "weekly_points_ppr",# unused here; kept for API stability
# ) -> pd.DataFrame:
#     """
#     Returns per-player table with:
#       ['player_name','fantasy_pos','team',
#        'true_vorp_star','delta_vorp_star_mean','delta_vorp_star_p10','delta_vorp_star_p90',
#        'adj_vorp_star','weeks_played','missed_weeks']

#     Linear method to match the draft board:
#       - true_vorp_star = season VORP* from totals (same as /metrics/vorp)
#       - vorp_star_extrap = 17-game pace VORP* for partials (already computed in compute_vorp_star)
#       - delta_vorp_star_mean = vorp_star_extrap - true_vorp_star  (0 for full seasons)
#       - adj_vorp_star = vorp_star_extrap  (equals true_vorp_star for full seasons)
#       - weeks_played = g; missed_weeks = max(0, weeks_in_season - g)
#     """
#     import numpy as np
#     if starters_per_team is None:
#         starters_per_team = {"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25}

#     # Build season table (includes vorp_star, g, partial_season, vorp_star_extrap)
#     season_table = build_vorp_table(
#         year=year,
#         use_ppg=False,
#         teams=teams,
#         starters_per_team=starters_per_team,
#     )

#     # Ensure required columns exist
#     needed = {"player_name", "fantasy_pos", "team", "vorp_star", "g"}
#     missing = [c for c in needed if c not in season_table.columns]
#     if missing:
#         raise ValueError(f"build_extrapolated_vorp_table (linear): missing columns {missing}")

#     # Some older builds may not have vorp_star_extrap; fall back to vorp_star in that case
#     if "vorp_star_extrap" not in season_table.columns:
#         season_table["vorp_star_extrap"] = season_table["vorp_star"]

#     out = season_table.loc[:, ["player_name", "fantasy_pos", "team", "vorp_star", "vorp_star_extrap", "g"]].copy()

#     # Numeric hygiene
#     for c in ["vorp_star", "vorp_star_extrap", "g"]:
#         out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

#     # Linear deltas & adjusted
#     out["true_vorp_star"] = out["vorp_star"]
#     out["delta_vorp_star_mean"] = out["vorp_star_extrap"] - out["vorp_star"]

#     # No distribution in linear method; set p10/p90 == mean
#     out["delta_vorp_star_p10"] = out["delta_vorp_star_mean"]
#     out["delta_vorp_star_p90"] = out["delta_vorp_star_mean"]

#     out["adj_vorp_star"] = out["vorp_star_extrap"]

#     # Weeks played / missed
#     out["weeks_played"] = out["g"].round().astype(int).clip(lower=0)
#     out["missed_weeks"] = (int(weeks_in_season) - out["weeks_played"]).clip(lower=0)

#     # Final column order expected by /metrics/war-extrapolated
#     cols = [
#         "player_name", "team", "fantasy_pos",
#         "true_vorp_star",
#         "delta_vorp_star_mean", "delta_vorp_star_p10", "delta_vorp_star_p90",
#         "adj_vorp_star",
#         "weeks_played", "missed_weeks",
#     ]
#     out = out.loc[:, cols]

#     return out

from typing import Optional, Dict, Set

def build_linear_extrapolated_table(
    year: int,
    *,
    weeks_in_season: int = 17,
    teams: int = 12,
    starters_per_team: Optional[Dict[str, float]] = None,
    pos_filter: Optional[Set[str]] = None,
    per_pos_caps: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Linear (no Monte Carlo) extrapolation:
      - true_vorp_star   = season-total VORP* (earned in games played)
      - adj_vorp_star    = 17-game linear pace VORP* (ppg * weeks - replacement) / scale
      - delta_*          = adj - true (p10/p90 == mean for linear mode)
      - weeks_played     = 'g'
      - missed_weeks     = weeks_in_season - g

    Preselects top N per position (per_pos_caps) BEFORE computing, and
    respects an optional pos_filter (e.g., {"QB","RB"}).
    """
    if starters_per_team is None:
        starters_per_team = {"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25}

    # --- Full season scrape & clean (once) ---
    season_df_raw = scrape_pfr_fantasy(year)
    season_df_raw["player_name"] = (
        season_df_raw["player"].astype(str)
        .str.replace(r"[*+.]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    cols_keep = ["player_name", "team", "fantasy_pos", "g", "fantasy_points_ppr"]
    stats = season_df_raw.loc[:, [c for c in cols_keep if c in season_df_raw.columns]].copy()
    stats = stats[stats["fantasy_pos"].isin(ALLOWED_POS)].copy()

    # coerce numerics
    for c in ["g", "fantasy_points_ppr"]:
        if c in stats.columns:
            stats[c] = pd.to_numeric(stats[c], errors="coerce").fillna(0.0)

    # --- Baselines from the full (unfiltered) table for stability ---
    season_baseline_input = season_df_raw.loc[:, ["player_name","team","fantasy_pos","fantasy_points_ppr"]].copy()
    season_baseline_input = season_baseline_input[season_baseline_input["fantasy_pos"].isin(ALLOWED_POS)].copy()
    season_baseline_input["fantasy_points_ppr"] = pd.to_numeric(
        season_baseline_input["fantasy_points_ppr"], errors="coerce"
    ).fillna(0.0)

    rep_by_pos, scale_by_pos = compute_season_baseline_and_scale(
        season_baseline_input,
        teams=teams,
        starters_per_team=starters_per_team,
        pool_factor=1.0,
        winsor_limits=(0.02, 0.98),
    )

    # --- Apply pos filter (if any) ---
    if pos_filter:
        want = {p.upper().strip() for p in pos_filter}
        stats = stats[stats["fantasy_pos"].isin(want)]

    if stats.empty:
        return pd.DataFrame(columns=[
            "player_name","team","fantasy_pos",
            "true_vorp_star","delta_vorp_star_mean","delta_vorp_star_p10","delta_vorp_star_p90",
            "adj_vorp_star","weeks_played","missed_weeks",
        ])

    # --- Preselect top N per position by total season points ---
    if per_pos_caps:
        caps = {k.upper(): int(v) for k, v in per_pos_caps.items() if k.upper() in ALLOWED_POS}
        stats = stats.sort_values(["fantasy_pos", "fantasy_points_ppr"], ascending=[True, False]).copy()
        stats["rank_in_pos"] = stats.groupby("fantasy_pos").cumcount() + 1
        mask = pd.Series(False, index=stats.index)
        for pos_name, topn in caps.items():
            mask |= (stats["fantasy_pos"] == pos_name) & (stats["rank_in_pos"] <= int(topn))
        stats = stats[mask].copy()
        stats.drop(columns=["rank_in_pos"], inplace=True)

    if stats.empty:
        return pd.DataFrame(columns=[
            "player_name","team","fantasy_pos",
            "true_vorp_star","delta_vorp_star_mean","delta_vorp_star_p10","delta_vorp_star_p90",
            "adj_vorp_star","weeks_played","missed_weeks",
        ])

    # --- Linear VORP* for the subset ---
    stats["rep_points_total"] = stats["fantasy_pos"].map(rep_by_pos).fillna(0.0)
    stats["scale_pos"] = stats["fantasy_pos"].map(scale_by_pos).replace(0.0, 1.0)

    # true (earned) WAR*
    stats["vorp_raw_total"] = stats["fantasy_points_ppr"] - stats["rep_points_total"]
    stats["true_vorp_star"] = stats["vorp_raw_total"] / stats["scale_pos"]

    # linear 17-game pace for partials (same logic as the draft board)
    g = stats["g"].replace(0.0, np.nan)
    ppg = stats["fantasy_points_ppr"] / g
    proj_points = (ppg * float(weeks_in_season)).fillna(stats["fantasy_points_ppr"])
    stats["adj_vorp_star"] = (proj_points - stats["rep_points_total"]) / stats["scale_pos"]

    # delta = adj - true (p10/p90 collapse to mean for linear mode)
    stats["delta_vorp_star_mean"] = stats["adj_vorp_star"] - stats["true_vorp_star"]
    stats["delta_vorp_star_p10"] = stats["delta_vorp_star_mean"]
    stats["delta_vorp_star_p90"] = stats["delta_vorp_star_mean"]

    # weeks
    stats["weeks_played"] = pd.to_numeric(stats["g"], errors="coerce").fillna(0).astype(int)
    stats["missed_weeks"] = (int(weeks_in_season) - stats["weeks_played"]).clip(lower=0).astype(int)

    stats["fantasy_points_ppr"] = pd.to_numeric(stats["fantasy_points_ppr"], errors="coerce").fillna(0.0)
    stats["ppr_per_game"] = stats["fantasy_points_ppr"] / stats["weeks_played"].replace(0, np.nan)

    out = stats.loc[:, [
        "player_name","team","fantasy_pos",
        # NEW:
        "fantasy_points_ppr","ppr_per_game",
        # existing:
        "true_vorp_star",
        "delta_vorp_star_mean","delta_vorp_star_p10","delta_vorp_star_p90",
        "adj_vorp_star",
        "weeks_played","missed_weeks",
    ]].sort_values("adj_vorp_star", ascending=False).reset_index(drop=True)

    return out
