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
    min_games_for_ppg: int = 12,     # threshold for “full season”
    pool_factor: float = 2.0,
    winsor_limits: tuple = (0.02, 0.98),
):
    """
    Adds:
      - partial_season (True for 3..min_games_for_ppg-1)
      - vorp_star_extrap (17-game pace VORP*, only meaningful for partial seasons)
    Uses replacement baseline computed from **season totals**.
    """
    df = df.copy()

    # 1) Metric for current-season VORP* display
    if use_ppg:
        df["points_used"] = df["fantasy_points_ppr"] / df["g"].replace(0, np.nan)
    else:
        df["points_used"] = df["fantasy_points_ppr"]
    df["points_used"] = df["points_used"].fillna(0.0)

    # --- Flags ---
    FULL_SEASON_MIN = int(min_games_for_ppg)  # e.g., 12
    df["partial_season"] = (df["g"] >= 3) & (df["g"] < FULL_SEASON_MIN)
    df["injury_flag"] = df["g"] < FULL_SEASON_MIN  # keep if you still want a general flag

    # 2) Replacement level per position (from **season totals**)
    if starters_per_team is None:
        starters_per_team = {"QB": 1.25, "RB": 2.5, "WR": 2.5, "TE": 1.25}

    rep_index = {
        pos: int(teams * starters_per_team.get(pos, 0))
        for pos in df["fantasy_pos"].unique()
    }

    def winsorize(s, lo=0.02, hi=0.98):
        ql, qh = np.quantile(s, [lo, hi])
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

        # For the current (display) metric:
        gsort = grp.sort_values("points_used", ascending=False)

        # VORP_raw (display metric) must be in same space as replacement.
        # If use_ppg=True, points_used is PPG; convert to a total-like space
        # so subtraction against rep_points_total is apples-to-apples.
        if use_ppg:
            # estimate season-total from current points_used (PPG) by *games played*
            # (this makes the displayed vorp_star reflect actual played total vs replacement)
            points_used_total_like = gsort["points_used"] * gsort["g"]
        else:
            points_used_total_like = gsort["points_used"]

        gsort = gsort.assign(rep_points=rep_points_total)
        gsort = gsort.assign(vorp_raw=points_used_total_like - rep_points_total)

        # Robust scale from top-of-pool VORP_raw
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
    # NEW
    "partial_season", "vorp_star_extrap",
    ]
    out = df_v[out_cols].sort_values("vorp_star", ascending=False).reset_index(drop=True)


    return out
