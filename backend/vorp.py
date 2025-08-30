import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import re

def scrape_pfr_fantasy(year: int) -> pd.DataFrame:
    url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    # Helper: return the fantasy table element whether it's in DOM or inside comments
    def find_fantasy_table(soup: BeautifulSoup):
        # 1) try live
        t = soup.find("table", {"id": "fantasy"})
        if t:
            return t
        # 2) scan comments
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

    # Build column order from the LAST header row in thead (skip over_header rows)
    thead = table.find("thead")
    header_rows = [tr for tr in thead.find_all("tr") if "over_header" not in tr.get("class", [])]
    # last header row is the one with the actual columns
    last_hdr = header_rows[-1]
    cols = []
    for th in last_hdr.find_all("th"):
        key = th.get("data-stat", "").strip()
        if key:  # skip corner blanks
            cols.append(key)

    # Parse body rows using data-stat keys
    data = []
    tbody = table.find("tbody")
    for tr in tbody.find_all("tr"):
        # PFR sometimes repeats a header row inside tbody with class="thead"
        if "thead" in tr.get("class", []):
            continue

        row = {}
        # some tables have a row header in <th scope="row">
        th = tr.find("th", {"scope": "row"})
        if th is not None:
            k = th.get("data-stat", "").strip() or "rk"
            row[k] = th.get_text(strip=True)

        for td in tr.find_all("td"):
            k = td.get("data-stat", "").strip()
            if not k:
                continue
            row[k] = td.get_text(strip=True)

        # skip completely empty rows
        if not row:
            continue
        data.append(row)

    # Create DataFrame and align to column order (keep any extra keys too)
    df = pd.DataFrame(data)

    # Ensure the final column order starts with `cols` then any extras
    extras = [c for c in df.columns if c not in cols]
    df = df[[c for c in cols if c in df.columns] + extras]

    # Clean: remove repeated header rows if any sneaked in
    if "player" in df.columns:
        df = df[df["player"].str.lower() != "player"]

    # Coerce numeric columns
    for c in df.columns:
        if c in ("player", "team", "pos", "tm"):
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")

    return df

# Example usage
year = None
df_year = scrape_pfr_fantasy(year)

df_year['player_name'] = df_year['player'].str.replace(r"[*+]", "", regex=True).str.strip()
stats = df_year[['player_name', 'team', 'fantasy_pos', 'g', 'gs', 'fantasy_points_ppr']]
stats = stats.loc[stats['fantasy_pos'].isin(['QB', 'RB', 'WR', 'TE'])]
stats['fantasy_points_ppr'].fillna(0, inplace=True)
stats["pos_rank"] = (
    stats.groupby("fantasy_pos")["fantasy_points_ppr"]
      .rank(method="dense", ascending=False)
      .astype(int)
)

stats["full_rank"] = (
    stats["fantasy_points_ppr"]
      .rank(method="dense", ascending=False)
      .astype(int)
)


def compute_vorp_star(
    df: pd.DataFrame,
    teams: int = 12,
    starters_per_team: dict = None,
    use_ppg: bool = False,
    min_games_for_ppg: int = 8,
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
        df["injury_flag"] = False

    # Fill any NaNs created (e.g., 0 games) with 0 so later math works;
    # they’ll still be flagged as injury_flag=True if use_ppg
    df["points_used"] = df["points_used"].fillna(0.0)

    # 2) Replacement level per position
    # Default lineup for 12-team, 1QB/2RB/2WR/1TE (adjust if your league differs)
    if starters_per_team is None:
        starters_per_team = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

    # How many starters in the league per position = teams * starters_per_team[pos]
    rep_index = {
        pos: teams * starters_per_team.get(pos, 0)
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
        
        if pos == "TE":
            # use average of TE13–TE20 as replacement
            tail = gsort.iloc[8:12]["points_used"]  # 0-based index → TE13 is row 12
            rep_points = tail.mean() if len(tail) else 0.0
        else:
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
        scale = mad(pool_w)
        if scale == 0 or not np.isfinite(scale):
            # fallback: use IQR/1.349 (approximately SD under normality) or 1 to avoid divide-by-0
            q1, q3 = np.percentile(pool_w, [25, 75]) if len(pool_w) else (0.0, 0.0)
            iqr = q3 - q1
            scale = (iqr / 1.349) if iqr > 0 else 1.0

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

# -------------------------
# Example usage on your df:
# -------------------------
# df has: player_name, team, fantasy_pos, g, gs, fantasy_points_ppr, pos_rank, full_rank

# Example for 12-team, 1QB/2RB/2WR/1TE, using season totals:
df_v = compute_vorp_star(
    stats,
    teams=12,
    starters_per_team={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
    use_ppg=False,           # set True if you prefer per-game
    min_games_for_ppg=8,
    pool_factor=2.0,
    winsor_limits=(0.02, 0.98),
)

# Columns now available:
# ['points_used','rep_points','vorp_raw','pos_scale_robust','vorp_star',
#  'injury_flag','vorp_star_rank_overall','vorp_star_rank_pos', ...]
# Example view:
cols = [
    "player_name","team","fantasy_pos","fantasy_points_ppr","vorp_star","vorp_star_rank_overall", "vorp_star_rank_pos"
]
df_v_filt = df_v[cols].sort_values(["vorp_star"], ascending=False)


