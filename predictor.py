"""
UFC Fight Outcome & Method of Victory Predictor
Multi-class XGBoost Classifier
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# 1. LOAD & MERGE
# ─────────────────────────────────────────────

fights = pd.read_csv("ufc_gold_dataset_final.csv")
fighters = pd.read_csv("ufc_fighters_final.csv")

# ── Parse fighter profile helpers ──────────────
def parse_height(h):
    """5' 11\" → cm (float)"""
    try:
        parts = str(h).replace('"', '').split("'")
        return int(parts[0]) * 30.48 + int(parts[1].strip()) * 2.54
    except:
        return np.nan

def parse_reach(r):
    """66.0\" → float inches"""
    try:
        return float(str(r).replace('"', '').strip())
    except:
        return np.nan

def parse_pct(p):
    """'60%' → 0.60"""
    try:
        return float(str(p).replace('%', '').strip()) / 100.0
    except:
        return np.nan

fighters["Height_cm"] = fighters["Height"].apply(parse_height)
fighters["Reach_in"]  = fighters["Reach"].apply(parse_reach)
fighters["TD_Acc_f"]  = fighters["TD_Acc"].apply(parse_pct)
fighters["TD_Def_f"]  = fighters["TD_Def"].apply(parse_pct)
fighters["Str_Acc_f"] = fighters["Str_Acc"].apply(parse_pct)
fighters["Str_Def_f"] = fighters["Str_Def"].apply(parse_pct)

profile_cols = [
    "Fighter_Name", "Height_cm", "Reach_in", "SLpM", "SApM",
    "TD_Avg", "TD_Acc_f", "TD_Def_f", "Sub_Avg",
    "Str_Acc_f", "Str_Def_f", "Wins", "Losses", "Draws"
]
fdf = fighters[profile_cols].copy()

# Numeric fills
num_cols = profile_cols[1:]
for c in num_cols:
    fdf[c] = pd.to_numeric(fdf[c], errors="coerce")
fdf[num_cols] = fdf[num_cols].fillna(fdf[num_cols].median())

# ── Merge fighter profiles for F1 & F2 ─────────
def merge_fighter(fights_df, fdf, prefix, name_col):
    renamed = {c: f"{prefix}_{c}" for c in fdf.columns if c != "Fighter_Name"}
    fdf_r = fdf.rename(columns=renamed)
    return fights_df.merge(fdf_r, left_on=name_col, right_on="Fighter_Name", how="left").drop(columns="Fighter_Name")

df = merge_fighter(fights, fdf, "P1", "Fighter_1")
df = merge_fighter(df,    fdf, "P2", "Fighter_2")

# ─────────────────────────────────────────────
# 2. TARGET VARIABLE
# ─────────────────────────────────────────────

# Map Method → 3 clean categories
def map_method(m):
    m = str(m).strip()
    if "KO" in m or "TKO" in m or "Doctor" in m or "Continue" in m:
        return "KO_TKO"
    if "Sub" in m:
        return "Submission"
    if "Decision" in m:
        return "Decision"
    return None  # DQ / Overturned / Other → drop

df["method_clean"] = df["Method"].apply(map_method)
df = df[df["method_clean"].notna()].copy()

# Winner perspective: 1 = F1 wins, 0 = F2 wins
df["f1_wins"] = (df["Winner"] == df["Fighter_1"]).astype(int)

# Target: 6 classes  →  {F1_KO, F1_Sub, F1_Dec, F2_KO, F2_Sub, F2_Dec}
df["target_str"] = np.where(
    df["f1_wins"] == 1,
    "F1_" + df["method_clean"],
    "F2_" + df["method_clean"]
)

le = LabelEncoder()
df["target"] = le.fit_transform(df["target_str"])
print("Classes:", dict(enumerate(le.classes_)))

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def grappling_score(td_acc, td_def, sub_avg, ctrl_sec, td_landed, td_att):
    td_rate   = np.where(td_att > 0, td_landed / td_att, 0)
    ctrl_norm = np.log1p(ctrl_sec) / 10.0
    return (td_acc * 0.25 + td_def * 0.20 + sub_avg * 0.20 +
            td_rate * 0.20 + ctrl_norm * 0.15)

def striking_score(sig_landed, sig_att, kd, str_acc, str_def, slpm, sapm):
    sig_rate = np.where(sig_att > 0, sig_landed / sig_att, 0)
    eff      = np.where(sapm > 0, slpm / (slpm + sapm + 1e-9), 0.5)
    return (str_acc * 0.25 + str_def * 0.20 + sig_rate * 0.20 +
            kd * 0.15 + eff * 0.20)

for p, kd, sig_l, sig_a, ctrl, td_l, td_a, sub_a, td_acc, td_def, td_avg, str_acc, str_def, slpm, sapm in [
    ("F1", "F1_KD", "F1_Sig_Landed", "F1_Sig_Att", "F1_Ctrl_Sec",
     "F1_TD_Landed", "F1_TD_Att", "F1_Sub_Att",
     "P1_TD_Acc_f", "P1_TD_Def_f", "P1_TD_Avg", "P1_Str_Acc_f", "P1_Str_Def_f",
     "P1_SLpM", "P1_SApM"),
    ("F2", "F2_KD", "F2_Sig_Landed", "F2_Sig_Att", "F2_Ctrl_Sec",
     "F2_TD_Landed", "F2_TD_Att", "F2_Sub_Att",
     "P2_TD_Acc_f", "P2_TD_Def_f", "P2_TD_Avg", "P2_Str_Acc_f", "P2_Str_Def_f",
     "P2_SLpM", "P2_SApM"),
]:
    df[f"{p}_Grappling"] = grappling_score(
        df[td_acc], df[td_def], df[sub_a], df[ctrl], df[td_l], df[td_a]
    )
    df[f"{p}_Striking"] = striking_score(
        df[sig_l], df[sig_a], df[kd], df[str_acc], df[str_def], df[slpm], df[sapm]
    )

# Differential features
df["Grappling_Diff"]  = df["F1_Grappling"]  - df["F2_Grappling"]
df["Striking_Diff"]   = df["F1_Striking"]   - df["F2_Striking"]
df["Ctrl_Diff"]       = np.log1p(df["F1_Ctrl_Sec"]) - np.log1p(df["F2_Ctrl_Sec"])
df["KD_Diff"]         = df["F1_KD"]         - df["F2_KD"]
df["Reach_Diff"]      = df["P1_Reach_in"]   - df["P2_Reach_in"]
df["Height_Diff"]     = df["P1_Height_cm"]  - df["P2_Height_cm"]
df["WinRate_P1"]      = df["P1_Wins"] / (df["P1_Wins"] + df["P1_Losses"] + 1e-6)
df["WinRate_P2"]      = df["P2_Wins"] / (df["P2_Wins"] + df["P2_Losses"] + 1e-6)
df["WinRate_Diff"]    = df["WinRate_P1"] - df["WinRate_P2"]
df["SLpM_Diff"]       = df["P1_SLpM"]   - df["P2_SLpM"]
df["Sub_Avg_Diff"]    = df["P1_Sub_Avg"] - df["P2_Sub_Avg"]
df["TD_Avg_Diff"]     = df["P1_TD_Avg"]  - df["P2_TD_Avg"]

FEATURE_COLS = [
    # In-fight derived
    "F1_Grappling", "F2_Grappling", "F1_Striking", "F2_Striking",
    "Grappling_Diff", "Striking_Diff", "Ctrl_Diff", "KD_Diff",
    # Fight stats (raw)
    "F1_Sig_Landed", "F2_Sig_Landed", "F1_TD_Landed", "F2_TD_Landed",
    "F1_Sub_Att", "F2_Sub_Att", "F1_Ctrl_Sec", "F2_Ctrl_Sec",
    "F1_Head", "F2_Head", "F1_Body", "F2_Body", "F1_Leg", "F2_Leg",
    "F1_Ground", "F2_Ground", "F1_Distance", "F2_Distance",
    "F1_Clinch", "F2_Clinch",
    # Fighter profile differentials
    "Reach_Diff", "Height_Diff", "WinRate_Diff", "WinRate_P1", "WinRate_P2",
    "SLpM_Diff", "Sub_Avg_Diff", "TD_Avg_Diff",
    # Fighter profiles
    "P1_SLpM", "P2_SLpM", "P1_SApM", "P2_SApM",
    "P1_TD_Acc_f", "P2_TD_Acc_f", "P1_Sub_Avg", "P2_Sub_Avg",
    "P1_Str_Acc_f", "P2_Str_Acc_f", "P1_Str_Def_f", "P2_Str_Def_f",
    "P1_TD_Def_f", "P2_TD_Def_f",
]

X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
y = df["target"]

# ─────────────────────────────────────────────
# 4. TRAIN XGBoost
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X_train_s, y_train,
          eval_set=[(X_test_s, y_test)],
          verbose=False)

y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f}  ({acc*100:.1f}%)\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─────────────────────────────────────────────
# 5. PREDICT FUNCTION
# ─────────────────────────────────────────────

def build_fighter_feature_vector(name: str) -> dict:
    """Return a dict of profile features for a fighter (with fallback to median)."""
    row = fighters[fighters["Fighter_Name"].str.lower() == name.lower()]
    if row.empty:
        print(f"  ⚠️  '{name}' profil datasında bulunamadı, median değerler kullanılıyor.")
        return {c: np.nan for c in profile_cols[1:]}
    r = row.iloc[0]
    return {
        "Height_cm":  parse_height(r["Height"]),
        "Reach_in":   parse_reach(r["Reach"]),
        "SLpM":       r["SLpM"],
        "SApM":       r["SApM"],
        "TD_Avg":     r["TD_Avg"],
        "TD_Acc_f":   parse_pct(r["TD_Acc"]),
        "TD_Def_f":   parse_pct(r["TD_Def"]),
        "Sub_Avg":    r["Sub_Avg"],
        "Str_Acc_f":  parse_pct(r["Str_Acc"]),
        "Str_Def_f":  parse_pct(r["Str_Def"]),
        "Wins":       r["Wins"],
        "Losses":     r["Losses"],
        "Draws":      r["Draws"],
    }


def predict_fight(fighter1: str, fighter2: str):
    """
    Predict fight outcome scenarios between two fighters using only
    their career profile statistics (no in-fight data needed at prediction time).
    Prints all outcome scenarios sorted by probability.
    """
    print(f"\n{'═'*55}")
    print(f"  🥊  {fighter1.upper()}  vs  {fighter2.upper()}")
    print(f"{'═'*55}")

    p1 = build_fighter_feature_vector(fighter1)
    p2 = build_fighter_feature_vector(fighter2)

    # Fill NaN with dataset medians
    med = X.median()

    def get(d, key, fallback_key):
        v = d.get(key, np.nan)
        return med[fallback_key] if pd.isna(v) else v

    # Build a single-row feature vector (profile-only; in-fight stats set to 0)
    # For prediction we zero out in-fight raw stats and rely on engineered features
    p1_slpm     = get(p1, "SLpM", "P1_SLpM")
    p1_sapm     = get(p1, "SApM", "P1_SApM")
    p1_td_acc   = get(p1, "TD_Acc_f", "P1_TD_Acc_f")
    p1_td_def   = get(p1, "TD_Def_f", "P1_TD_Def_f")
    p1_td_avg   = get(p1, "TD_Avg", "P1_TD_Avg")
    p1_sub_avg  = get(p1, "Sub_Avg", "P1_Sub_Avg")
    p1_str_acc  = get(p1, "Str_Acc_f", "P1_Str_Acc_f")
    p1_str_def  = get(p1, "Str_Def_f", "P1_Str_Def_f")
    p1_wins     = get(p1, "Wins", "WinRate_P1")
    p1_losses   = get(p1, "Losses", "WinRate_P1")
    p1_reach    = get(p1, "Reach_in", "P1_Reach_in")
    p1_height   = get(p1, "Height_cm", "P1_Height_cm")

    p2_slpm     = get(p2, "SLpM", "P2_SLpM")
    p2_sapm     = get(p2, "SApM", "P2_SApM")
    p2_td_acc   = get(p2, "TD_Acc_f", "P2_TD_Acc_f")
    p2_td_def   = get(p2, "TD_Def_f", "P2_TD_Def_f")
    p2_td_avg   = get(p2, "TD_Avg", "P2_TD_Avg")
    p2_sub_avg  = get(p2, "Sub_Avg", "P2_Sub_Avg")
    p2_str_acc  = get(p2, "Str_Acc_f", "P2_Str_Acc_f")
    p2_str_def  = get(p2, "Str_Def_f", "P2_Str_Def_f")
    p2_wins     = get(p2, "Wins", "WinRate_P2")
    p2_losses   = get(p2, "Losses", "WinRate_P2")
    p2_reach    = get(p2, "Reach_in", "P2_Reach_in")
    p2_height   = get(p2, "Height_cm", "P2_Height_cm")

    wr1 = p1_wins / (p1_wins + p1_losses + 1e-6)
    wr2 = p2_wins / (p2_wins + p2_losses + 1e-6)

    # Estimated in-fight stats from career avgs (rough extrapolation for 3 rounds)
    rounds = 3
    f1_sig  = p1_slpm * 5 * rounds * 0.5  # ~half of sig strikes land
    f2_sig  = p2_slpm * 5 * rounds * 0.5
    f1_td   = p1_td_avg * rounds
    f2_td   = p2_td_avg * rounds
    f1_ctrl = f1_td * 40   # avg ctrl seconds per TD
    f2_ctrl = f2_td * 40
    f1_sub  = p1_sub_avg * rounds * 0.5
    f2_sub  = p2_sub_avg * rounds * 0.5
    f1_kd   = 0.0; f2_kd = 0.0   # conservative

    f1_grap = grappling_score(p1_td_acc, p1_td_def, p1_sub_avg, f1_ctrl, f1_td, max(f1_td+1,1))
    f2_grap = grappling_score(p2_td_acc, p2_td_def, p2_sub_avg, f2_ctrl, f2_td, max(f2_td+1,1))
    f1_stri = striking_score(f1_sig, max(f1_sig*2,1), f1_kd, p1_str_acc, p1_str_def, p1_slpm, p1_sapm)
    f2_stri = striking_score(f2_sig, max(f2_sig*2,1), f2_kd, p2_str_acc, p2_str_def, p2_slpm, p2_sapm)

    feat = {
        "F1_Grappling": f1_grap, "F2_Grappling": f2_grap,
        "F1_Striking": f1_stri,  "F2_Striking": f2_stri,
        "Grappling_Diff": f1_grap - f2_grap,
        "Striking_Diff":  f1_stri - f2_stri,
        "Ctrl_Diff":  np.log1p(f1_ctrl) - np.log1p(f2_ctrl),
        "KD_Diff":    f1_kd - f2_kd,
        "F1_Sig_Landed": f1_sig,  "F2_Sig_Landed": f2_sig,
        "F1_TD_Landed": f1_td,    "F2_TD_Landed": f2_td,
        "F1_Sub_Att": f1_sub,     "F2_Sub_Att": f2_sub,
        "F1_Ctrl_Sec": f1_ctrl,   "F2_Ctrl_Sec": f2_ctrl,
        "F1_Head": f1_sig*0.55,   "F2_Head": f2_sig*0.55,
        "F1_Body": f1_sig*0.25,   "F2_Body": f2_sig*0.25,
        "F1_Leg":  f1_sig*0.20,   "F2_Leg":  f2_sig*0.20,
        "F1_Ground": f1_ctrl*0.3, "F2_Ground": f2_ctrl*0.3,
        "F1_Distance": f1_sig*0.6,"F2_Distance": f2_sig*0.6,
        "F1_Clinch": f1_sig*0.15, "F2_Clinch": f2_sig*0.15,
        "Reach_Diff": p1_reach - p2_reach,
        "Height_Diff": p1_height - p2_height,
        "WinRate_Diff": wr1 - wr2, "WinRate_P1": wr1, "WinRate_P2": wr2,
        "SLpM_Diff": p1_slpm - p2_slpm,
        "Sub_Avg_Diff": p1_sub_avg - p2_sub_avg,
        "TD_Avg_Diff": p1_td_avg - p2_td_avg,
        "P1_SLpM": p1_slpm,   "P2_SLpM": p2_slpm,
        "P1_SApM": p1_sapm,   "P2_SApM": p2_sapm,
        "P1_TD_Acc_f": p1_td_acc, "P2_TD_Acc_f": p2_td_acc,
        "P1_Sub_Avg": p1_sub_avg, "P2_Sub_Avg": p2_sub_avg,
        "P1_Str_Acc_f": p1_str_acc, "P2_Str_Acc_f": p2_str_acc,
        "P1_Str_Def_f": p1_str_def, "P2_Str_Def_f": p2_str_def,
        "P1_TD_Def_f": p1_td_def,  "P2_TD_Def_f": p2_td_def,
    }

    row_df = pd.DataFrame([feat])[FEATURE_COLS].fillna(med)
    row_s  = scaler.transform(row_df)
    proba  = model.predict_proba(row_s)[0]

    EMOJIS = {
        "KO_TKO": "💥", "Submission": "🤼", "Decision": "📋"
    }

    results = []
    for cls, prob in zip(le.classes_, proba):
        winner_label, method = cls.split("_", 1)
        fighter_name = fighter1 if winner_label == "F1" else fighter2
        results.append((fighter_name, method, prob))

    results.sort(key=lambda x: -x[2])

    print(f"\n  {'Senaryo':<38} {'Olasılık':>8}")
    print(f"  {'─'*38} {'─'*8}")
    for fighter_name, method, prob in results:
        emoji = EMOJIS.get(method, "")
        label = f"{fighter_name} wins by {method}"
        bar   = "█" * int(prob * 30)
        print(f"  {emoji} {label:<36} {prob*100:>6.1f}%  {bar}")

    # Winner summary
    f1_total = sum(p for f, m, p in results if f == fighter1)
    f2_total = sum(p for f, m, p in results if f == fighter2)
    print(f"\n  📊 Toplam Kazanma Olasılığı:")
    print(f"     {fighter1:<25} → {f1_total*100:.1f}%")
    print(f"     {fighter2:<25} → {f2_total*100:.1f}%")
    print(f"{'═'*55}\n")

    return results


print("\n" + "=" * 55)
print("🥊 UFC TAHMİN MOTORU HAZIR! (Çıkmak için 'q' yazın)")
print("=" * 55)

while True:
    fighter1 = input("\n🔴 1. Dövüşçü (Örn: Khamzat Chimaev): ").strip()
    if fighter1.lower() == 'q':
        print("Sistemden çıkılıyor. Görüşürüz kral!")
        break

    fighter2 = input("🔵 2. Dövüşçü (Örn: Sean Strickland): ").strip()
    if fighter2.lower() == 'q':
        print("Sistemden çıkılıyor. Görüşürüz kral!")
        break

    print(f"\n⚙️ {fighter1.title()} vs {fighter2.title()} hesaplanıyor...\n")

    try:
        # İsimlerin baş harflerini otomatik büyütür (Örn: jon jones -> Jon Jones)
        predict_fight(fighter1.title(), fighter2.title())
    except Exception as e:
        print("\n⚠️ Bir hata oluştu! Dövüşçü isimlerini tam ve doğru yazdığına emin ol.")