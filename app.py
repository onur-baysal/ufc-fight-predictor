import streamlit as st
import pandas as pd
import numpy as np
import warnings
import re
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os

warnings.filterwarnings("ignore")

# Dosya yolu garantisi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGHTERS_CSV_PATH = os.path.join(BASE_DIR, "ufc_fighters_final.csv")

# ───────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="UFC Fight Night Predictor", page_icon="🥊", layout="wide",
                   initial_sidebar_state="expanded")

# ───────────────────────────────────────────────────────────────────────
# PREMIUM CSS THEME
# ───────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stApp  { font-family: 'Inter', -apple-system, sans-serif !important; }
    .stApp { background: radial-gradient(circle at top right, #1a1c2c 0%, #0a0b10 45%, #060709 100%) !important; }
    h1 { font-weight: 800 !important; letter-spacing: -0.5px; background: linear-gradient(90deg, #ffffff 0%, #ff5b5b 60%, #E10600 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding-bottom: 4px; }
    h2, h3 { color: #f5f5f7 !important; font-weight: 700 !important; letter-spacing: -0.3px; }
    p, span, label, div { color: #d6d8de; }
    hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.6em 0; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #14161f 0%, #0a0b10 100%) !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h1 { color: #ffffff !important; background: none !important; -webkit-text-fill-color: #ffffff !important; }
    .stButton>button { background: linear-gradient(135deg, #E10600 0%, #ff3b3b 100%) !important; color: #ffffff !important; font-weight: 700 !important; font-size: 0.95em; letter-spacing: 0.5px; border: none !important; border-radius: 12px !important; height: 3.1em; width: 100%; box-shadow: 0 6px 20px rgba(225, 6, 0, 0.35); transition: all 0.25s ease-in-out; }
    .stButton>button:hover { box-shadow: 0 8px 28px rgba(225, 6, 0, 0.55) !important; transform: translateY(-2px); }
    div[data-baseweb="select"] > div { background-color: #161922 !important; border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.08) !important; box-shadow: 0 4px 14px rgba(0,0,0,0.35); }
    div[data-baseweb="select"] span { color: #ffffff !important; }
    .glass-card { background: rgba(255,255,255,0.045) !important; backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 16px !important; padding: 20px 24px !important; margin-bottom: 16px !important; box-shadow: 0 10px 34px rgba(0,0,0,0.45); }
    .verdict-card { border-left: 4px solid #00E0B8 !important; background: linear-gradient(135deg, rgba(0,224,184,0.10), rgba(255,255,255,0.02)) !important; }
    .scenario-card { border-left: 4px solid #E10600 !important; background: linear-gradient(135deg, rgba(225,6,0,0.10), rgba(255,255,255,0.02)) !important; }
    .card-label { text-transform: uppercase !important; letter-spacing: 1.5px !important; font-size: 0.72em !important; font-weight: 700 !important; color: #9aa0ad !important; margin-bottom: 6px !important; }
    .card-main { font-size: 1.35em !important; font-weight: 800 !important; color: #ffffff !important; line-height: 1.35 !important; }
    .highlight-red { color: #ff5b5b !important; font-weight: 800; }
    .highlight-teal { color: #00E0B8 !important; font-weight: 800; }
    .card-sub { margin-top: 6px !important; font-size: 0.9em !important; color: #aeb3bd !important; font-weight: 500 !important; }
    .tot-container { background: rgba(20, 22, 31, 0.65) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 18px !important; padding: 20px 25px !important; box-shadow: 0 10px 34px rgba(0,0,0,0.5) !important; margin: 15px 0 !important; }
    .tot-header { display: flex !important; justify-content: space-between !important; align-items: center !important; padding-bottom: 25px !important; border-bottom: 1px solid rgba(255,255,255,0.12) !important; margin-bottom: 15px !important; }
    .fighter-name { font-size: 1.65em !important; font-weight: 900 !important; letter-spacing: 0.5px !important; width: 40% !important; }
    .name-red { color: #ff5b5b !important; text-align: left !important; }
    .name-blue { color: #4dd2ff !important; text-align: right !important; }
    .vs-container { width: 20% !important; display: flex !important; justify-content: center !important; align-items: center !important; }
    .vs-pill { font-size: 1.1em !important; font-weight: 900 !important; color: #0a0b10 !important; background: linear-gradient(135deg, #ffffff, #c8c8cf) !important; padding: 6px 20px !important; border-radius: 30px !important; letter-spacing: 2px; box-shadow: 0 0 15px rgba(255,255,255,0.2); text-align: center !important; }
    .tot-row { display: flex !important; justify-content: space-between !important; align-items: center !important; padding: 12px 0 !important; border-bottom: 1px solid rgba(255,255,255,0.05) !important; }
    .tot-row:last-child { border-bottom: none !important; }
    .tot-val { font-size: 1.1em !important; font-weight: 700 !important; width: 35% !important; }
    .red-side { text-align: left !important; color: #ffb3b3 !important; }
    .blue-side { text-align: right !important; color: #a9e8ff !important; }
    .tot-label { text-align: center !important; font-size: 0.8em !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 1.2px !important; color: #8b909c !important; width: 30% !important; }
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────
# 0. ORIGINAL LIVE WEB SCRAPER (STABIL PERFORMANS)
# ───────────────────────────────────────────────────────────────────────
SHERDOG_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _normalize_name(name):
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def scrape_fighter_stats(fighter_name, debug_box):
    try:
        clean_name = fighter_name.replace(" ", "+")
        search_url = f"https://www.sherdog.com/stats/fightfinder?SearchTxt={clean_name}"
        res = requests.get(search_url, headers=SHERDOG_HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        profile_url = None
        for a in soup.select('a[href^="/fighter/"]'):
            if fighter_name.split()[0].lower() in a.text.lower():
                profile_url = "https://www.sherdog.com" + a['href']
                break

        if not profile_url:
            debug_box.error("❌ Sherdog'da profil bulunamadı.")
            return None

        debug_box.write(f"🔗 Profil: {profile_url}")
        p_res = requests.get(profile_url, headers=SHERDOG_HEADERS, timeout=10)
        p_soup = BeautifulSoup(p_res.text, "html.parser")

        win = p_soup.select_one('.win span:nth-of-type(2)')
        lose = p_soup.select_one('.lose span:nth-of-type(2)')
        draw = p_soup.select_one('.draw span:nth-of-type(2)')

        w = int(win.text) if win else 0
        l = int(lose.text) if lose else 0
        d = int(draw.text) if draw else 0

        debug_box.success(f"🎯 Record Başarıyla Çekildi: {w}-{l}-{d}")
        return {"Wins": w, "Losses": l, "Draws": d}

    except Exception as e:
        debug_box.error(f"💥 Scraping Hatası: {str(e)}")
    return None


def update_fighter_csv(fighter_name, stats):
    try:
        fdf_raw = pd.read_csv(FIGHTERS_CSV_PATH)
        target = _normalize_name(fighter_name)
        name_series = fdf_raw["Fighter_Name"].astype(str).apply(_normalize_name)
        match_idx = fdf_raw.index[name_series == target]
        if len(match_idx) == 0: return False
        idx = match_idx[0]

        for col in ["Wins", "Losses", "Draws"]:
            if col in stats and pd.notna(stats[col]):
                fdf_raw.at[idx, col] = stats[col]

        fdf_raw.to_csv(FIGHTERS_CSV_PATH, index=False)
        return True
    except Exception as e:
        return False


# ───────────────────────────────────────────────────────────────────────
# 1. CORE DATA & MODEL FUNCTIONS
# ───────────────────────────────────────────────────────────────────────
def load_and_preprocess_data():
    fights = pd.read_csv(os.path.join(BASE_DIR, "ufc_gold_dataset_final.csv"))
    fighters = pd.read_csv(FIGHTERS_CSV_PATH)

    def parse_height(h):
        try:
            return int(str(h).split("'")[0]) * 30.48 + int(str(h).split("'")[1].replace('"', '').strip()) * 2.54
        except:
            return np.nan

    def parse_reach(r):
        try:
            return float(str(r).replace('"', '').strip())
        except:
            return np.nan

    def parse_pct(p):
        try:
            return float(str(p).replace('%', '').strip()) / 100.0
        except:
            return np.nan

    fighters["Height_cm"] = fighters["Height"].apply(parse_height)
    fighters["Reach_in"] = fighters["Reach"].apply(parse_reach)
    fighters["TD_Acc_f"] = fighters["TD_Acc"].apply(parse_pct)
    fighters["TD_Def_f"] = fighters["TD_Def"].apply(parse_pct)
    fighters["Str_Acc_f"] = fighters["Str_Acc"].apply(parse_pct)
    fighters["Str_Def_f"] = fighters["Str_Def"].apply(parse_pct)

    profile_cols = ["Fighter_Name", "Height_cm", "Reach_in", "SLpM", "SApM", "TD_Avg", "TD_Acc_f", "TD_Def_f",
                    "Sub_Avg", "Str_Acc_f", "Str_Def_f", "Wins", "Losses", "Draws"]
    fdf = fighters[profile_cols].copy()
    num_cols = profile_cols[1:]
    for c in num_cols: fdf[c] = pd.to_numeric(fdf[c], errors="coerce")
    fdf[num_cols] = fdf[num_cols].fillna(fdf[num_cols].median())

    df = fights.merge(fdf.rename(columns={c: f"P1_{c}" for c in fdf.columns if c != "Fighter_Name"}),
                      left_on="Fighter_1", right_on="Fighter_Name", how="left").drop(columns="Fighter_Name")
    df = df.merge(fdf.rename(columns={c: f"P2_{c}" for c in fdf.columns if c != "Fighter_Name"}), left_on="Fighter_2",
                  right_on="Fighter_Name", how="left").drop(columns="Fighter_Name")

    def map_method(m):
        m = str(m).strip()
        if "KO" in m or "TKO" in m or "Doctor" in m or "Continue" in m: return "KO_TKO"
        if "Sub" in m: return "Submission"
        if "Decision" in m: return "Decision"
        return None

    df["method_clean"] = df["Method"].apply(map_method)
    df = df[df["method_clean"].notna()].copy()
    df["f1_wins"] = (df["Winner"] == df["Fighter_1"]).astype(int)
    df["target_str"] = np.where(df["f1_wins"] == 1, "F1_" + df["method_clean"], "F2_" + df["method_clean"])
    return df, fighters, fdf


def grappling_score(td_acc, td_def, sub_avg, ctrl_sec, td_landed, td_att):
    return (td_acc * 0.25 + td_def * 0.20 + sub_avg * 0.20 + (np.where(td_att > 0, td_landed / td_att, 0)) * 0.20 + (
                np.log1p(ctrl_sec) / 10.0) * 0.15)


def striking_score(sig_landed, sig_att, kd, str_acc, str_def, slpm, sapm):
    return (str_acc * 0.25 + str_def * 0.20 + (np.where(sig_att > 0, sig_landed / sig_att, 0)) * 0.20 + kd * 0.15 + (
        np.where(sapm > 0, slpm / (slpm + sapm + 1e-9), 0.5)) * 0.20)


def train_model(df):
    for p, kd, sig_l, sig_a, ctrl, td_l, td_a, sub_a, td_acc, td_def, td_avg, str_acc, str_def, slpm, sapm in [
        ("F1", "F1_KD", "F1_Sig_Landed", "F1_Sig_Att", "F1_Ctrl_Sec", "F1_TD_Landed", "F1_TD_Att", "F1_Sub_Att",
         "P1_TD_Acc_f", "P1_TD_Def_f", "P1_TD_Avg", "P1_Str_Acc_f", "P1_Str_Def_f", "P1_SLpM", "P1_SApM"),
        ("F2", "F2_KD", "F2_Sig_Landed", "F2_Sig_Att", "F2_Ctrl_Sec", "F2_TD_Landed", "F2_TD_Att", "F2_Sub_Att",
         "P2_TD_Acc_f", "P2_TD_Def_f", "P2_TD_Avg", "P2_Str_Acc_f", "P2_Str_Def_f", "P2_SLpM", "P2_SApM"),
    ]:
        df[f"{p}_Grappling"] = grappling_score(df[td_acc], df[td_def], df[sub_a], df[ctrl], df[td_l], df[td_a])
        df[f"{p}_Striking"] = striking_score(df[sig_l], df[sig_a], df[kd], df[str_acc], df[str_def], df[slpm], df[sapm])

    df["Grappling_Diff"] = df["F1_Grappling"] - df["F2_Grappling"]
    df["Striking_Diff"] = df["F1_Striking"] - df["F2_Striking"]
    df["Ctrl_Diff"] = np.log1p(df["F1_Ctrl_Sec"]) - np.log1p(df["F2_Ctrl_Sec"])
    df["KD_Diff"] = df["F1_KD"] - df["F2_KD"]
    df["Reach_Diff"] = df["P1_Reach_in"] - df["P2_Reach_in"]
    df["Height_Diff"] = df["P1_Height_cm"] - df["P2_Height_cm"]
    df["WinRate_P1"] = df["P1_Wins"] / (df["P1_Wins"] + df["P1_Losses"] + 1e-6)
    df["WinRate_P2"] = df["P2_Wins"] / (df["P2_Wins"] + df["P2_Losses"] + 1e-6)
    df["WinRate_Diff"] = df["WinRate_P1"] - df["WinRate_P2"]
    df["SLpM_Diff"] = df["P1_SLpM"] - df["P2_SLpM"]
    df["Sub_Avg_Diff"] = df["P1_Sub_Avg"] - df["P2_Sub_Avg"]
    df["TD_Avg_Diff"] = df["P1_TD_Avg"] - df["P2_TD_Avg"]

    FEATURE_COLS = [
        "F1_Grappling", "F2_Grappling", "F1_Striking", "F2_Striking", "Grappling_Diff", "Striking_Diff", "Ctrl_Diff",
        "KD_Diff",
        "F1_Sig_Landed", "F2_Sig_Landed", "F1_TD_Landed", "F2_TD_Landed", "F1_Sub_Att", "F2_Sub_Att", "F1_Ctrl_Sec",
        "F2_Ctrl_Sec",
        "F1_Head", "F2_Head", "F1_Body", "F2_Body", "F1_Leg", "F2_Leg", "F1_Ground", "F2_Ground", "F1_Distance",
        "F2_Distance", "F1_Clinch", "F2_Clinch",
        "Reach_Diff", "Height_Diff", "WinRate_Diff", "WinRate_P1", "WinRate_P2", "SLpM_Diff", "Sub_Avg_Diff",
        "TD_Avg_Diff",
        "P1_SLpM", "P2_SLpM", "P1_SApM", "P2_SApM", "P1_TD_Acc_f", "P2_TD_Acc_f", "P1_Sub_Avg", "P2_Sub_Avg",
        "P1_Str_Acc_f", "P2_Str_Acc_f", "P1_Str_Def_f", "P2_Str_Def_f", "P1_TD_Def_f", "P2_TD_Def_f"
    ]
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["target_str"])
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="mlogloss", random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)
    return model, scaler, le, FEATURE_COLS, X.median()


# DATA YÜKLEME
df, fighters_raw, fdf = load_and_preprocess_data()
model, scaler, le, FEATURE_COLS, med_values = train_model(df)

st.sidebar.header("Matchup Selection")
fighter_list = sorted(fighters_raw["Fighter_Name"].unique())
fighter1_select = st.sidebar.selectbox("🔴 Corner 1", fighter_list, index=fighter_list.index(
    "Khamzat Chimaev") if "Khamzat Chimaev" in fighter_list else 0)
fighter2_select = st.sidebar.selectbox("🔵 Corner 2", fighter_list, index=fighter_list.index(
    "Sean Strickland") if "Sean Strickland" in fighter_list else 1)

st.sidebar.markdown("---")
update_clicked = st.sidebar.button("🔄 Verileri Güncelle (Live Scrape)")

db1 = st.sidebar.expander(f"🔍 Debug - {fighter1_select}")
db2 = st.sidebar.expander(f"🔍 Debug - {fighter2_select}")

if update_clicked:
    with st.spinner("Sherdog üzerinden taze profiller çekiliyor..."):
        stats1 = scrape_fighter_stats(fighter1_select, db1)
        if stats1:
            if update_fighter_csv(fighter1_select, stats1):
                db1.success("✅ CSV Güncellendi!")
            else:
                db1.error("⚠️ CSV satırı bulunamadı.")

        stats2 = scrape_fighter_stats(fighter2_select, db2)
        if stats2:
            if update_fighter_csv(fighter2_select, stats2):
                db2.success("✅ CSV Güncellendi!")
            else:
                db2.error("⚠️ CSV satırı bulunamadı.")
    st.sidebar.success("Veritabanı güncellendi, sayfa yenileniyor...")
    st.rerun()


def get_fighter_vector(name, fighters_df):
    row = fighters_df[fighters_df["Fighter_Name"] == name]
    if row.empty: return None
    r = row.iloc[0]
    return {
        "Height_cm": (int(str(r["Height"]).split("'")[0]) * 30.48 + int(
            str(r["Height"]).split("'")[1].replace('"', '')) * 2.54) if "'" in str(r["Height"]) else np.nan,
        "Reach_in": float(str(r["Reach"]).replace('"', '').strip()) if '"' in str(r["Reach"]) else np.nan,
        "SLpM": r["SLpM"], "SApM": r["SApM"], "TD_Avg": r["TD_Avg"], "Sub_Avg": r["Sub_Avg"],
        "Str_Acc_f": float(str(r["Str_Acc"]).replace('%', '')) / 100 if '%' in str(r["Str_Acc"]) else np.nan,
        "Str_Def_f": float(str(r["Str_Def"]).replace('%', '')) / 100 if '%' in str(r["Str_Def"]) else np.nan,
        "TD_Acc_f": float(str(r["TD_Acc"]).replace('%', '')) / 100 if '%' in str(r["TD_Acc"]) else np.nan,
        "TD_Def_f": float(str(r["TD_Def"]).replace('%', '')) / 100 if '%' in str(r["TD_Def"]) else np.nan,
        "Wins": r["Wins"], "Losses": r["Losses"], "Draws": r["Draws"]
    }


p1, p2 = get_fighter_vector(fighter1_select, fighters_raw), get_fighter_vector(fighter2_select, fighters_raw)

# ─────────────────── RENDER TALE OF THE TAPE (ORİJİNAL) ───────────────────
st.subheader("📋 Tale of the Tape")
if p1 and p2:
    def fmt(v, suffix=""):
        try:
            return f"{float(v):.1f}{suffix}"
        except:
            return f"{v}{suffix}"


    tot_rows = [
        ("Record", f"{int(p1['Wins'])}-{int(p1['Losses'])}-{int(p1['Draws'])}",
         f"{int(p2['Wins'])}-{int(p2['Losses'])}-{int(p2['Draws'])}"),
        ("Height (cm)", fmt(p1["Height_cm"]), fmt(p2["Height_cm"])),
        ("Reach (in)", fmt(p1["Reach_in"], "\""), fmt(p2["Reach_in"], "\"")),
        ("Sig. Strikes / Min (SLpM)", fmt(p1["SLpM"]), fmt(p2["SLpM"])),
        ("Strikes Absorbed / Min", fmt(p1["SApM"]), fmt(p2["SApM"])),
        ("Striking Accuracy", fmt(p1["Str_Acc_f"] * 100, "%"), fmt(p2["Str_Acc_f"] * 100, "%")),
        ("Striking Defense", fmt(p1["Str_Def_f"] * 100, "%"), fmt(p2["Str_Def_f"] * 100, "%")),
        ("Takedown Avg / 15min", fmt(p1["TD_Avg"]), fmt(p2["TD_Avg"])),
        ("Takedown Accuracy", fmt(p1["TD_Acc_f"] * 100, "%"), fmt(p2["TD_Acc_f"] * 100, "%")),
        ("Takedown Defense", fmt(p1["TD_Def_f"] * 100, "%"), fmt(p2["TD_Def_f"] * 100, "%")),
        ("Submission Avg / 15min", fmt(p1["Sub_Avg"]), fmt(p2["Sub_Avg"])),
    ]
    rows_html = "".join([
        f'<div class="tot-row"><div class="tot-val red-side">{v1}</div><div class="tot-label">{label}</div><div class="tot-val blue-side">{v2}</div></div>'
        for label, v1, v2 in tot_rows
    ])

    tot_html = f'''
    <div class="tot-container">
        <div class="tot-header">
            <div class="fighter-name name-red" style="width: 40%; text-align: left;">🔴 {fighter1_select}</div>
            <div class="vs-container" style="width: 20%; text-align: center;"><div class="vs-pill">VS</div></div>
            <div class="fighter-name name-blue" style="width: 40%; text-align: right;">{fighter2_select} 🔵</div>
        </div>
        {rows_html}
    </div>
    '''
    st.markdown(tot_html.strip(), unsafe_allow_html=True)

st.markdown("---")

if st.sidebar.button("💥 RUN SIMULATION"):
    def get_val(d, key, fallback_col):
        return med_values[fallback_col] if pd.isna(d.get(key, np.nan)) else d[key]


    wr1, wr2 = p1["Wins"] / (p1["Wins"] + p1["Losses"] + 1e-6), p2["Wins"] / (p2["Wins"] + p2["Losses"] + 1e-6)
    rnds = 3
    f1_s = get_val(p1, "SLpM", "P1_SLpM") * 5 * rnds * 0.5
    f2_s = get_val(p2, "SLpM", "P2_SLpM") * 5 * rnds * 0.5
    f1_t, f2_t = get_val(p1, "TD_Avg", "P1_TD_Avg") * rnds, get_val(p2, "TD_Avg", "P2_TD_Avg") * rnds
    f1_g = grappling_score(get_val(p1, "TD_Acc_f", "P1_TD_Acc_f"), get_val(p1, "TD_Def_f", "P1_TD_Def_f"),
                           get_val(p1, "Sub_Avg", "P1_Sub_Avg"), f1_t * 40, f1_t, max(f1_t + 1, 1))
    f2_g = grappling_score(get_val(p2, "TD_Acc_f", "P2_TD_Acc_f"), get_val(p2, "TD_Def_f", "P2_TD_Def_f"),
                           get_val(p2, "Sub_Avg", "P2_Sub_Avg"), f2_t * 40, f2_t, max(f2_t + 1, 1))
    f1_str = striking_score(f1_s, max(f1_s * 2, 1), 0.0, get_val(p1, "Str_Acc_f", "P1_Str_Acc_f"),
                            get_val(p1, "Str_Def_f", "P1_Str_Def_f"), get_val(p1, "SLpM", "P1_SLpM"),
                            get_val(p1, "SApM", "P1_SApM"))
    f2_str = striking_score(f2_s, max(f2_s * 2, 1), 0.0, get_val(p2, "Str_Acc_f", "P2_Str_Acc_f"),
                            get_val(p2, "Str_Def_f", "P2_Str_Def_f"), get_val(p2, "SLpM", "P2_SLpM"),
                            get_val(p2, "SApM", "P2_SApM"))

    feat = {
        "F1_Grappling": f1_g, "F2_Grappling": f2_g, "F1_Striking": f1_str, "F2_Striking": f2_str,
        "Grappling_Diff": f1_g - f2_g, "Striking_Diff": f1_str - f2_str,
        "Ctrl_Diff": np.log1p(f1_t * 40) - np.log1p(f2_t * 40), "KD_Diff": 0.0, "F1_Sig_Landed": f1_s,
        "F2_Sig_Landed": f2_s, "F1_TD_Landed": f1_t, "F2_TD_Landed": f2_t,
        "F1_Sub_Att": get_val(p1, "Sub_Avg", "P1_Sub_Avg") * rnds * 0.5,
        "F2_Sub_Att": get_val(p2, "Sub_Avg", "P2_Sub_Avg") * rnds * 0.5, "F1_Ctrl_Sec": f1_t * 40,
        "F2_Ctrl_Sec": f2_t * 40,
        "F1_Head": f1_s * 0.55, "F2_Head": f2_s * 0.55, "F1_Body": f1_s * 0.25, "F2_Body": f2_s * 0.25,
        "F1_Leg": f1_s * 0.20, "F2_Leg": f2_s * 0.20,
        "F1_Ground": f1_t * 40 * 0.3, "F2_Ground": f2_t * 40 * 0.3, "F1_Distance": f1_s * 0.6,
        "F2_Distance": f2_s * 0.6, "F1_Clinch": f1_s * 0.15, "F2_Clinch": f2_s * 0.15,
        "Reach_Diff": get_val(p1, "Reach_in", "P1_Reach_in") - get_val(p2, "Reach_in", "P2_Reach_in"),
        "Height_Diff": get_val(p1, "Height_cm", "P1_Height_cm") - get_val(p2, "Height_cm", "P2_Height_cm"),
        "WinRate_Diff": wr1 - wr2, "WinRate_P1": wr1, "WinRate_P2": wr2,
        "SLpM_Diff": get_val(p1, "SLpM", "P1_SLpM") - get_val(p2, "SLpM", "P2_SLpM"),
        "Sub_Avg_Diff": get_val(p1, "Sub_Avg", "P1_Sub_Avg") - get_val(p2, "Sub_Avg", "P2_Sub_Avg"),
        "TD_Avg_Diff": get_val(p1, "TD_Avg", "P1_TD_Avg") - get_val(p2, "TD_Avg", "P2_TD_Avg"),
        "P1_SLpM": get_val(p1, "SLpM", "P1_SLpM"), "P2_SLpM": get_val(p2, "SLpM", "P2_SLpM"),
        "P1_SApM": get_val(p1, "SApM", "P1_SApM"), "P2_SApM": get_val(p2, "SApM", "P2_SApM"),
        "P1_TD_Acc_f": get_val(p1, "TD_Acc_f", "P1_TD_Acc_f"), "P2_TD_Acc_f": get_val(p2, "TD_Acc_f", "P2_TD_Acc_f"),
        "P1_Sub_Avg": get_val(p1, "Sub_Avg", "P1_Sub_Avg"), "P2_Sub_Avg": get_val(p2, "Sub_Avg", "P2_Sub_Avg"),
        "P1_Str_Acc_f": get_val(p1, "Str_Acc_f", "P1_Str_Acc_f"),
        "P2_Str_Acc_f": get_val(p2, "Str_Acc_f", "P2_Str_Acc_f"),
        "P1_Str_Def_f": get_val(p1, "Str_Def_f", "P1_Str_Def_f"),
        "P2_Str_Def_f": get_val(p2, "Str_Def_f", "P2_Str_Def_f"),
        "P1_TD_Def_f": get_val(p1, "TD_Def_f", "P1_TD_Def_f"), "P2_TD_Def_f": get_val(p2, "TD_Def_f", "P2_TD_Def_f"),
    }

    # Kalan tahmin kartları ve annotasyon mantığı (DOKUNULMADI)
    FEATURE_COLS = [c for c in FEATURE_COLS if c != "Form_Diff"]  # Form diff temizliği
    row_df = pd.DataFrame([feat])[FEATURE_COLS].fillna(med_values)
    proba = model.predict_proba(scaler.transform(row_df))[0]
    results = [
        {"Fighter": (fighter1_select if c.split("_", 1)[0] == "F1" else fighter2_select), "Method": c.split("_", 1)[1],
         "Probability": p} for c, p in zip(le.classes_, proba)]
    res_df = pd.DataFrame(results).sort_values(by="Probability", ascending=False)

    st.subheader("🏆 Overall Win Probability")
    f1_total = res_df[res_df["Fighter"] == fighter1_select]["Probability"].sum()
    f2_total = res_df[res_df["Fighter"] == fighter2_select]["Probability"].sum()
    winner_champ = fighter1_select if f1_total > f2_total else fighter2_select

    fig_pie = go.Figure(data=[
        go.Pie(labels=[fighter1_select, fighter2_select], values=[f1_total, f2_total], hole=.55, sort=False,
               marker=dict(colors=['#E10600', '#00C2FF'], line=dict(color='#0a0b10', width=3)),
               textinfo='label+percent', textposition='outside',
               textfont=dict(color='#ffffff', size=18, family='Inter'), pull=[0.04, 0.04], rotation=90)])
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"),
                          showlegend=False, margin=dict(t=60, b=60, l=60, r=60))
    fig_pie.add_annotation(text="<b>WIN<br>PROB</b>", x=0.5, y=0.5, font=dict(size=15, color='#9aa0ad', family='Inter'),
                           showarrow=False)

    c_pie, c_text = st.columns([1, 1])
    with c_pie:
        st.plotly_chart(fig_pie, use_container_width=True)
    with c_text:
        top_scenario = res_df.iloc[0]
        st.markdown(
            f'<div class="glass-card verdict-card"><div class="card-label">Verdict</div><div class="card-main"><span class="highlight-teal">{winner_champ}</span> is favored to win</div><div class="card-sub">Total predicted win probability: <b>{max(f1_total, f2_total) * 100:.1f}%</b></div></div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="glass-card scenario-card"><div class="card-label">💡 Most Likely Target Scenario</div><div class="card-main"><span class="highlight-red">{top_scenario["Fighter"]}</span> via <i>{top_scenario["Method"]}</i></div><div class="card-sub">Scenario probability: <b>{top_scenario["Probability"] * 100:.1f}%</b></div></div>',
            unsafe_allow_html=True)

    # ───────────────────────────────────────────────────────────────────────
    # AI PREDICTION INSIGHTS
    # ───────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔮 AI Prediction Insights (Model Analiz Raporu)")
    insights = []
    if abs(feat["Striking_Diff"]) < 0.05:
        insights.append("🎯 <b>Ayakta Çatışma:</b> Striking metrikleri çok yakın — maç anlık bir refleksle çözülebilir.")
    elif feat["Striking_Diff"] > 0:
        insights.append(
            f"🥊 <b>Ayakta Üstünlük:</b> <b style='color:#ff5b5b;'>{fighter1_select}</b> daha yüksek isabet oranıyla ayaktaki kontrol avantajına sahip.")
    else:
        insights.append(
            f"🥊 <b>Ayakta Üstünlük:</b> <b style='color:#4dd2ff;'>{fighter2_select}</b> defansif striking metrikleri sayesinde ayaktaki değişimlerde öne çıkıyor.")

    if abs(feat["Grappling_Diff"]) < 0.08:
        insights.append("🤼 <b>Güreş Dengesi:</b> Grappling skoru dengede, net bir yerde domine beklentisi düşük.")
    elif feat["Grappling_Diff"] > 0:
        insights.append(
            f"🤼 <b>Yere Serme Tehdidi:</b> <b style='color:#ff5b5b;'>{fighter1_select}</b> dövüşü yere taşıyıp ({p1['TD_Avg']:.1f} TD/15dk) baskı kurabilir.")
    else:
        insights.append(
            f"🤼 <b>Yere Serme Tehdidi:</b> <b style='color:#4dd2ff;'>{fighter2_select}</b> daha efektif grappling profiliyle submission tehdidi oluşturuyor.")

    if feat["Reach_Diff"] > 1.5:
        insights.append(
            f"📐 <b>Mesafe Avantajı:</b> <b style='color:#ff5b5b;'>{fighter1_select}</b> {p1['Reach_in']:.0f}\" uzanma mesafesiyle dışarıdan vuruşlarda öne çıkıyor.")
    elif feat["Reach_Diff"] < -1.5:
        insights.append(
            f"📐 <b>Mesafe Avantajı:</b> <b style='color:#4dd2ff;'>{fighter2_select}</b> {abs(feat['Reach_Diff']):.0f}\" uzanma avantajıyla mesafeyi koruyabilir.")

    insights_html = "".join([
                                f'<div style="margin-bottom:16px;font-size:1.15em;line-height:1.7;color:#f5f5f7;padding:14px 18px;background:rgba(0,0,0,0.25);border-radius:10px;border-left:4px solid rgba(255,255,255,0.15);box-shadow:0 4px 10px rgba(0,0,0,0.2);">{ins}</div>'
                                for ins in insights])
    st.markdown(
        f'<div class="glass-card" style="border-top:4px solid #9aa0ad !important;background:linear-gradient(135deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01)) !important;padding:30px !important;"><div class="card-label" style="color:#ffffff !important;font-size:0.9em !important;margin-bottom:24px;letter-spacing:2px;">🤖 Taktiksel Metrik Analizi</div>{insights_html}</div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="glass-card verdict-card"><div class="card-main">Sidebar\'dan dövüşçüleri seçip <span class="highlight-red">\'RUN SIMULATION\'</span> butonuna basarak gecenin kazananını görebilirsin! 🥊</div></div>',
        unsafe_allow_html=True)
