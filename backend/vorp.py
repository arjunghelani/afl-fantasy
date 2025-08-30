import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import re
import numpy as np
from typing import Optional, List, Dict, Any

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
# Your VORP* formula (plugged)
# -----------------------------
def compute_vorp_star(
    df: pd.DataFrame,
    teams: int = 12,
    starters_per_team: dict = None,
    use_ppg: bool = False,
    min_games_for_ppg: int = 12,
    pool_factor: float = 2.0,         # pool ~ 2× starters per position
    winsor_limits: tuple = (0.02, 0.98)  # clamp extremes for robust scale
):
    """
    Returns a copy of df with columns:
      - points_used         (PPG or season total)
      - rep_points          (replacement level for the position)
      - vorp_raw            (points_used - rep_points)
      - pos_scale_robust    (1.4826 * MAD of (points_used - rep_points) in pool)
      - vorp_star           (vorp_raw / pos_scale_robust)
      - injury_flag         (True if < min_games_for_ppg when use_ppg)
    """

    df = df.copy()

    # 1) Decide points metric (total or PPG)
    if use_ppg:
        df["points_used"] = df["fantasy_points_ppr"] / df["g"].replace(0, np.nan)
        # flag small-sample seasons
        df["injury_flag"] = (df["g"] < min_games_for_ppg) | (~np.isfinite(df["points_used"]))
    else:
        df["points_used"] = df["fantasy_points_ppr"]
        df["injury_flag"] = (df["g"] < min_games_for_ppg) | (~np.isfinite(df["points_used"]))
    # Fill any NaNs created (e.g., 0 games) with 0 so later math works;
    # they’ll still be flagged as injury_flag=True if use_ppg
    df["points_used"] = df["points_used"].fillna(0.0)

    # 2) Replacement level per position
    # Default lineup for 12-team, 1QB/2RB/2WR/1TE (adjust if your league differs)
    if starters_per_team is None:
        starters_per_team = {"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25}

    # How many starters in the league per position = teams * starters_per_team[pos]
    rep_index = {
        pos: int(teams * starters_per_team.get(pos, 0))
        for pos in df["fantasy_pos"].unique()
    }
    
    # Helper: robust MAD scale
    def mad(x):
        med = np.median(x)
        return 1.4826 * np.median(np.abs(x - med))

    # Helper: winsorize
    def winsorize(s, lo=0.02, hi=0.98):
        ql, qh = np.quantile(s, [lo, hi])
        return np.clip(s, ql, qh)

    out_frames = []
    for pos, grp in df.groupby("fantasy_pos", sort=False):
        # sort high→low by points_used
        gsort = grp.sort_values("points_used", ascending=False)

        # replacement rank (e.g., RB24). If none defined, skip normalization gracefully.
        # R = rep_index.get(pos, 0)
        # if R <= 0 or R > len(gsort):
        #     # no valid replacement index; fall back to zero baseline/scale=1 to avoid div-by-0
        #     rep_points = 0.0
        # else:
        #     rep_points = gsort.iloc[R-1]["points_used"]
        
    
        R = rep_index.get(pos, 0)
        if R <= 0 or R > len(gsort):
            rep_points = 0.0
        else:
            rep_points = gsort.iloc[R-1]["points_used"]


        gsort = gsort.assign(rep_points=rep_points)
        gsort = gsort.assign(vorp_raw=gsort["points_used"] - rep_points)

        # build pool ~ top (pool_factor × starters) to estimate robust spread
        pool_size = int(max(R * pool_factor, min(len(gsort), R))) if R > 0 else min(len(gsort), 24)
        pool = gsort.iloc[:pool_size]["vorp_raw"].values

        # winsorize pool then MAD
        pool_w = winsorize(pool, *winsor_limits) if len(pool) else np.array([0.0])
        scale = np.std(pool_w, ddof=0)   # population std dev
        if scale == 0 or not np.isfinite(scale):
            scale = 1.0   # avoid divide by 0

        gsort = gsort.assign(pos_scale_robust=scale)
        gsort = gsort.assign(vorp_star=gsort["vorp_raw"] / scale)

        out_frames.append(gsort)

    result = pd.concat(out_frames, axis=0).sort_index()

    # Optional: ranks
    result["vorp_star_rank_overall"] = result["vorp_star"].rank(method="dense", ascending=False).astype(int)
    result["vorp_star_rank_pos"] = (
        result.groupby("fantasy_pos")["vorp_star"].rank(method="dense", ascending=False).astype(int)
    )

    return result


# -----------------------------
# Public builder used by API
# -----------------------------
ALLOWED_POS = {"QB", "RB", "WR", "TE"}

def build_vorp_table(year: int, use_ppg: bool = False) -> Optional[pd.DataFrame]:
    """
    Returns DataFrame:
    ['player_name','team','fantasy_pos','g','fantasy_points_ppr',
     'vorp_star','vorp_star_rank_overall','vorp_star_rank_pos']
    sorted by vorp_star desc.
    """
    starters_per_team = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

    # 1) scrape
    df = scrape_pfr_fantasy(year)

    # 2) clean
    df["player_name"] = (
    df["player"]
    .astype(str)
    .str.replace(r"[*+.]", "", regex=True)  # remove *, +, and .
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    )


    cols_keep = ["player_name", "team", "fantasy_pos", "g", "gs", "fantasy_points_ppr"]
    stats = df.loc[:, [c for c in cols_keep if c in df.columns]].copy()
    stats = stats[stats["fantasy_pos"].isin(ALLOWED_POS)].copy()

    for c in ["g", "gs", "fantasy_points_ppr"]:
        if c in stats.columns:
            stats[c] = pd.to_numeric(stats[c], errors="coerce").fillna(0.0)

    # 3) compute VORP* (your formula)
    df_v = compute_vorp_star(
        stats,
        teams=12,
        starters_per_team={"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25},
        use_ppg=False,           # set True if you prefer per-game
        min_games_for_ppg=12,
        pool_factor=1,
        winsor_limits=(0.02, 0.98),
    )

    out_cols = [
        "player_name", "team", "fantasy_pos", "g",
        "fantasy_points_ppr", "vorp_star",
        "vorp_star_rank_overall", "vorp_star_rank_pos",
    ]
    out = df_v[out_cols].sort_values("vorp_star", ascending=False).reset_index(drop=True)
    return out
