from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore, chi2_contingency, mannwhitneyu, spearmanr

ANALYSIS_YEAR = "23"
INPUT_FILE = Path("../datasets/data.csv")
OUTPUT_DIR = Path(f"wyniki_analizy_{ANALYSIS_YEAR}")
MIN_EXAM_TAKERS = 10

def detect_prefix(df: pd.DataFrame, year_short: str) -> str:
    for p in (f"{year_short}.0", f"{year_short}"):
        if f"{p}_polski_mean" in df.columns:
            return p
    raise KeyError(f"Nie ma kolumny {year_short}_polski_mean / {year_short}.0_polski_mean")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def spearman_pair(x: pd.Series, y: pd.Series):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3:
        return np.nan, np.nan, len(tmp)
    r, p = spearmanr(tmp["x"], tmp["y"])
    return float(r), float(p), int(len(tmp))


def main():
    # wczytanie pliku i utworzenie dir
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono: {INPUT_FILE.resolve()}")
    df = pd.read_csv(INPUT_FILE, sep=";", low_memory=False)
    prefix = detect_prefix(df, ANALYSIS_YEAR)

    # wybór kolumn dla podanego roku
    col_wynik = f"{prefix}_polski_mean"
    col_prog = f"{prefix}_avg_threshold"
    col_zdaw = f"{prefix}_O_zdawalnosc"
    col_n = f"{prefix}_polski_liczba_zdajacych"

    # utworzenie tabeli roboczej work
    work = df.copy()
    work["wynik"] = to_num(work[col_wynik])
    work["prog"] = to_num(work[col_prog])
    work["zdawalnosc"] = to_num(work[col_zdaw]) if col_zdaw in work.columns else np.nan
    work["n_zdajacych"] = to_num(work[col_n]) if col_n in work.columns else np.nan
    work["res_wynik_zdaw"] = np.nan #zabezpieczenie na zdawalnosc - jesli nie ma zdawalnosci to nie wyrzuci bledu
    work["exp_zdawalnosc"] = np.nan
    # wybranie do dalszej analizy tylko szkol, ktore maja komplet danych - prog rekrutacyjny, sredni wynik z j polskiego, minimum 10 zdajacych
    work = work.dropna(subset=["wynik", "prog"]).copy()
    work = work[work["n_zdajacych"].fillna(0) >= MIN_EXAM_TAKERS].copy()
    if len(work) < 20:
        raise ValueError("Za mało szkół po filtrach (mniej niż 20). Sprawdź dane/rok/kolumny.")

    # linia trendu - przewidywany wynik z j polskiego
    m1 = LinearRegression().fit(work[["prog"]].values, work["wynik"].values)
    work["exp_wynik"] = m1.predict(work[["prog"]].values)
    work["res_prog_wynik"] = work["wynik"] - work["exp_wynik"]
    work["exp_zdawalnosc"] = np.nan
    # work["res_wynik_zdaw"] = np.nan

    if work["zdawalnosc"].notna().sum() >= 30:
        tmp = work.dropna(subset=["zdawalnosc", "wynik"]).copy()
        m2 = LinearRegression().fit(tmp[["wynik"]].values, tmp["zdawalnosc"].values)
        tmp["exp_zdawalnosc"] = m2.predict(tmp[["wynik"]].values)
        tmp["res_wynik_zdaw"] = tmp["zdawalnosc"] - tmp["exp_zdawalnosc"]
        work = work.drop(columns=["exp_zdawalnosc", "res_wynik_zdaw"]).merge(tmp[["RSPO", "exp_zdawalnosc", "res_wynik_zdaw"]], on="RSPO", how="left")
        if "res_wynik_zdaw" not in work.columns:
            work["res_wynik_zdaw"] = np.nan

    # odchylenie standardowe od srednieij
    work["z1"] = zscore(work["res_prog_wynik"].fillna(0))
    work["z2"] = zscore(work.get("res_wynik_zdaw", pd.Series(np.zeros(len(work)), index=work.index)).fillna(0))
    work["outlier_score"] = np.abs(work["z1"]) + np.abs(work["z2"])

    # podzial na liderow/maruderow - top/bottom 10%
    q_hi = work["res_prog_wynik"].quantile(0.90)
    q_lo = work["res_prog_wynik"].quantile(0.10)
    work["grupa"] = np.select(
        [work["res_prog_wynik"] >= q_hi, work["res_prog_wynik"] <= q_lo],
        ["Liderzy", "Maruderzy"],
        default="Srodek"
    )

    #heatmapa na podstawie spermana
    r1, p1, n1 = spearman_pair(work["prog"], work["wynik"])
    r2, p2, n2 = spearman_pair(work["wynik"], work["zdawalnosc"])
    corr = pd.DataFrame([
        {"para": "prog vs wynik", "spearman_r": r1, "p_value": p1, "N": n1},
        {"para": "wynik vs zdawalnosc", "spearman_r": r2, "p_value": p2, "N": n2},
    ])
    corr.to_csv(OUTPUT_DIR / "korelacje.csv", sep=";", index=False, encoding="utf-8-sig")
    corr_cols = [
        "prog", "wynik", "zdawalnosc", "n_zdajacych", "liczba_uczniow",
        "res_prog_wynik", "res_wynik_zdaw", "outlier_score"
    ]
    corr_cols = [c for c in corr_cols if c in work.columns]
    for c in corr_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    good_cols = [c for c in corr_cols if work[c].notna().mean() >= 0.30]  # min 30% nie-NaN
    corr_mat = work[good_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=True, center=0)
    plt.title(f"Mapa korelacji (20{ANALYSIS_YEAR})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_korelacje.png", dpi=200)
    plt.close()
    cols = [
        "RSPO", "nazwa_szkoly", "miejscowosc", "wojewodztwo", "publicznosc_status",
        "prog", "wynik", "zdawalnosc", "n_zdajacych",
        "res_prog_wynik", "res_wynik_zdaw", "outlier_score", "grupa",
        "kategoria_uczniow", "typ_gminy", "liczba_uczniow",
    ]
    cols = [c for c in cols if c in work.columns]

    # 50 szkół odstających
    top50 = work.sort_values("outlier_score", ascending=False).head(50).copy()
    top50[cols].to_csv(
        OUTPUT_DIR / "ranking_odstajace_top50.csv", sep=";", index=False, encoding="utf-8-sig"
    )

    # cechy wspólne szkół odstających (tylko dla top 50)
    liderzy = top50[top50["grupa"] == "Liderzy"]
    maruderzy = top50[top50["grupa"] == "Maruderzy"]

    rows = []

    for c in ["publicznosc_status", "typ_gminy", "kategoria_uczniow", "wojewodztwo"]:
        if c not in top50.columns:
            continue

        tab = pd.crosstab(top50["grupa"], top50[c])
        p = np.nan
        if tab.shape[0] >= 2 and tab.shape[1] >= 2:
            _, p, _, _ = chi2_contingency(tab)

        pct = pd.crosstab(top50["grupa"], top50[c], normalize="index") * 100
        if "Liderzy" in pct.index:
            topcats = pct.loc["Liderzy"].sort_values(ascending=False).head(3)
            for cat, val in topcats.items():
                val_m = pct.loc["Maruderzy", cat] if ("Maruderzy" in pct.index and cat in pct.columns) else np.nan
                rows.append({
                    "cecha": c,
                    "kategoria": str(cat),
                    "liderzy_%": float(val),
                    "maruderzy_%": float(val_m),
                    "p_value": float(p) if pd.notna(p) else np.nan
                })

    for c in ["liczba_uczniow", "n_zdajacych"]:
        if c not in work.columns:
            continue
        a = to_num(liderzy[c]).dropna()
        b = to_num(maruderzy[c]).dropna()
        p = np.nan
        if len(a) >= 10 and len(b) >= 10:
            _, p = mannwhitneyu(a, b, alternative="two-sided")

        rows.append({
            "cecha": c,
            "kategoria": "median",
            "liderzy_%": float(np.median(a)) if len(a) else np.nan,
            "maruderzy_%": float(np.median(b)) if len(b) else np.nan,
            "p_value": float(p) if pd.notna(p) else np.nan
        })

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "cechy_wspolne.csv", sep=";", index=False, encoding="utf-8-sig")

    # wykresy
    sns.set_theme(style="whitegrid")

    # wykres: prog vs wynik + linia trendu
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=work, x="prog", y="wynik", hue="grupa", alpha=0.6)
    tmp = work.sort_values("prog")
    plt.plot(tmp["prog"], tmp["exp_wynik"], "k--", lw=2)
    plt.title(f"Próg rekrutacyjny vs wynik z polskiego (20{ANALYSIS_YEAR})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wykres_prog_vs_wynik.png", dpi=200)
    plt.close()

    # wykres: wynik vs zdawalność
    if work["zdawalnosc"].notna().sum() >= 30:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=work, x="wynik", y="zdawalnosc", hue="grupa", alpha=0.6)
        if work.get("exp_zdawalnosc", pd.Series(index=work.index, dtype="float64")).notna().sum() >= 30:
            tmp2 = work.dropna(subset=["wynik", "exp_zdawalnosc"]).sort_values("wynik")
            plt.plot(tmp2["wynik"], tmp2["exp_zdawalnosc"], "k--", lw=2)
        plt.title(f"Wynik z polskiego vs zdawalność (20{ANALYSIS_YEAR})")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "wykres_wynik_vs_zdawalnosc.png", dpi=200)
        plt.close()

    # Nowy wykres: Liczba uczniów dla top 50 najbardziej odstających szkół
    if "liczba_uczniow" in top50.columns:
        plt.figure(figsize=(12, 15))
        # Sortowanie top50 po liczbie uczniów dla lepszej czytelności wykresu
        top50_sorted = top50.sort_values("liczba_uczniow", ascending=False)

        sns.scatterplot(
            data=top50_sorted,
            y="pelna_nazwa_szkoly",
            x="liczba_uczniow",
            hue="grupa",
            palette={"Liderzy": "green", "Maruderzy": "red", "Srodek": "gray"},
            s=100  # Wielkość punktów dla lepszej widoczności
        )

        plt.title(f"Liczba uczniów w top 50 najbardziej odstających szkołach (20{ANALYSIS_YEAR})")
        plt.xlabel("Liczba uczniów")
        plt.ylabel("Nazwa szkoły")
        plt.tight_layout()
        plt.subplots_adjust(left=0.6)
        plt.savefig(OUTPUT_DIR / "wykres_top50_liczba_uczniow.png", dpi=200)
        plt.close()

    work[cols].to_csv(OUTPUT_DIR / "wyniki_analizy_szkol.csv", sep=";", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
