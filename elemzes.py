"""
Formula 1 elemzés, komplexebb kérdésekkel
Kaggle adatcsomag: rohanrao/formula-1-world-championship-1950-2020

Könyvtárak: pandas, numpy, matplotlib
Három kérdés:
  1. Mi határozza meg a végső helyezést: a rajtpozíció, a csapat ereje
     vagy a versenyző tapasztalata? (többváltozós lineáris regresszió)
  2. Melyik pályán számít a legjobban a rajtpozíció? (korreláció pályánként)
  3. Mennyit számít a csapat, és kik teljesítenek a legjobban a csapattársukhoz
     képest? (szóráselemzés és csapattárs-összehasonlítás)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KAGGLE_NEV = "rohanrao/formula-1-world-championship-1950-2020"
ABRA_MAPPA = "abrak"
MODERN_KEZDET = 2010  # 2010-től egységes a pontozási rendszer


def adat_betoltes():
    """Letölti az adatcsomagot a Kaggle-ről (kagglehub), és beolvassa a CSV-ket."""
    try:
        import kagglehub

        mappa = kagglehub.dataset_download(KAGGLE_NEV)
    except Exception as hiba:
        print(f"Kaggle letöltés nem sikerült ({hiba}).")
        print("Tedd a CSV fájlokat az 'f1_adatok' mappába, vagy a script mellé.\n")
        mappa = "f1_adatok" if os.path.isdir("f1_adatok") else "."

    def olvas(fajl):
        # a hiányzó értékeket ebben az adatban \N jelöli
        return pd.read_csv(os.path.join(mappa, fajl), na_values=["\\N"])

    eredmenyek = olvas("results.csv")
    versenyzok = olvas("drivers.csv")
    futamok = olvas("races.csv")
    palyak = olvas("circuits.csv")

    print(f"Eredmények: {len(eredmenyek)} sor")
    print(f"Versenyzők: {len(versenyzok)} sor")
    print(f"Futamok: {len(futamok)} sor")
    print(f"Pályák: {len(palyak)} sor\n")
    return eredmenyek, versenyzok, futamok, palyak


def elokeszites(eredmenyek, futamok):
    """Összekapcsolja az eredményeket a futamokkal, és új jellemzőket számol."""
    adat = eredmenyek.merge(
        futamok[["raceId", "year", "date", "circuitId"]], on="raceId"
    )
    adat["date"] = pd.to_datetime(adat["date"])
    adat = adat.sort_values(["date", "raceId"])

    # tapasztalat: hány futamon indult már korábban a versenyző (teljes múlt)
    adat["tapasztalat"] = adat.groupby("driverId").cumcount()

    # csapat ereje: a csapat átlagos pontszáma az adott szezonban
    adat["csapat_ero"] = adat.groupby(["year", "constructorId"])["points"].transform(
        "mean"
    )
    return adat


def kerdes_1(adat):
    print("=" * 60)
    print("1. KÉRDÉS: Mi határozza meg a végső helyezést?")
    print("=" * 60)

    jellemzok = {
        "grid": "Rajtpozíció",
        "csapat_ero": "Csapat erőssége",
        "tapasztalat": "Tapasztalat",
    }

    modell = adat[(adat["year"] >= MODERN_KEZDET) & (adat["grid"] > 0)]
    modell = modell.dropna(subset=list(jellemzok) + ["positionOrder"])

    X = modell[list(jellemzok)].to_numpy(dtype=float)
    y = modell["positionOrder"].to_numpy(dtype=float)

    # standardizálás, hogy az együtthatók összehasonlíthatók legyenek
    X_z = (X - X.mean(axis=0)) / X.std(axis=0)
    y_z = (y - y.mean()) / y.std()

    # többváltozós lineáris regresszió legkisebb négyzetek módszerével
    A = np.column_stack([np.ones(len(X_z)), X_z])
    egyutthatok, *_ = np.linalg.lstsq(A, y_z, rcond=None)
    becsles = A @ egyutthatok
    r2 = 1 - ((y_z - becsles) ** 2).sum() / ((y_z - y_z.mean()) ** 2).sum()
    betak = egyutthatok[1:]

    # összehasonlítás: csak a rajtpozícióval mennyit magyaráz?
    r2_grid = np.corrcoef(X[:, 0], y)[0, 1] ** 2

    nevek = list(jellemzok.values())
    szinek = ["tab:red" if b > 0 else "tab:green" for b in betak]
    plt.figure(figsize=(9, 5))
    plt.barh(nevek, betak, color=szinek)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Mi befolyásolja a végső helyezést? (standardizált együtthatók)")
    plt.xlabel("β (pozitív: rontja, negatív: javítja a helyezést)")
    plt.tight_layout()
    plt.savefig(os.path.join(ABRA_MAPPA, "1_regresszio.png"), dpi=150)
    plt.show()

    print(f"A modell {len(modell)} rajtoló autó adatából készült "
          f"({MODERN_KEZDET} után).")
    for nev, b in zip(nevek, betak):
        irany = "rontja" if b > 0 else "javítja"
        print(f"  {nev}: β = {b:+.2f}, a növekedése {irany} a végső helyezést.")

    legfontosabb = nevek[int(np.argmax(np.abs(betak)))]
    print(f"A legerősebb tényező: {legfontosabb}.")
    print(f"A három tényező együtt a végső helyezés varianciájának "
          f"{r2 * 100:.0f}%-át magyarázza, a rajtpozíció egyedül "
          f"{r2_grid * 100:.0f}%-át.")
    print(f"A csapat erejének hozzáadása {(r2 - r2_grid) * 100:+.0f} "
          f"százalékpontot ad a magyarázóerőhöz.")
    print()


def kerdes_2(adat, palyak):
    print("=" * 60)
    print("2. KÉRDÉS: Melyik pályán számít a legjobban a rajtpozíció?")
    print("=" * 60)

    modern = adat[(adat["year"] >= 2000) & (adat["grid"] > 0)]
    meret = modern.groupby("circuitId").size()
    # csak azok a pályák, ahol legalább kb. 10 futam adata megvan
    modern = modern[modern["circuitId"].isin(meret[meret >= 200].index)]

    korrelaciok = {}
    for palya_id, csoport in modern.groupby("circuitId"):
        korrelaciok[palya_id] = np.corrcoef(
            csoport["grid"].to_numpy(dtype=float),
            csoport["positionOrder"].to_numpy(dtype=float),
        )[0, 1]

    korr = pd.Series(korrelaciok).sort_values(ascending=False)
    nevek = palyak.set_index("circuitId")["name"]

    n = min(8, len(korr) // 2)
    kiemelt = pd.concat([korr.head(n), korr.tail(n)])
    cimkek = [nevek.loc[i] for i in kiemelt.index]
    szinek = ["tab:red"] * n + ["tab:blue"] * n

    plt.figure(figsize=(11, 7))
    plt.barh(cimkek, kiemelt.values, color=szinek)
    plt.gca().invert_yaxis()
    plt.title("Rajtpozíció és végső helyezés korrelációja pályánként (2000 után)")
    plt.xlabel("Korrelációs együttható (r)")
    plt.tight_layout()
    plt.savefig(os.path.join(ABRA_MAPPA, "2_palyak.png"), dpi=150)
    plt.show()

    erosebb = nevek.loc[korr.index[0]]
    gyengebb = nevek.loc[korr.index[-1]]
    print(f"{len(korr)} pályát vizsgáltam (2000 után, legalább ~10 futam).")
    print(f"A rajtpozíció a(z) {erosebb} pályán számít a legtöbbet "
          f"(r = {korr.iloc[0]:.2f}).")
    print(f"A(z) {gyengebb} pályán számít a legkevesebbet "
          f"(r = {korr.iloc[-1]:.2f}), itt a rajt kevésbé határozza meg "
          f"a végeredményt.")
    print(f"A pályák átlagos korrelációja r = {korr.mean():.2f}.")
    print()


def kerdes_3(adat, versenyzok):
    print("=" * 60)
    print("3. KÉRDÉS: Mennyit számít a csapat, és ki a legjobb a csapattársához képest?")
    print("=" * 60)

    modern = adat[adat["year"] >= MODERN_KEZDET].copy()

    # a) mennyit magyaráz a csapat? (csoportok közötti szórás / teljes szórás)
    pont_atlag = modern["points"].mean()
    csapat_atlag = modern.groupby(["year", "constructorId"])["points"].transform("mean")
    ss_teljes = ((modern["points"] - pont_atlag) ** 2).sum()
    ss_csapat = ((csapat_atlag - pont_atlag) ** 2).sum()
    csapat_resz = ss_csapat / ss_teljes * 100

    # b) csapattársakhoz mért teljesítmény futamonként
    csoport = modern.groupby(["raceId", "constructorId"])["points"]
    modern["db"] = csoport.transform("count")
    modern["ossz"] = csoport.transform("sum")
    parok = modern[modern["db"] > 1].copy()
    csapattars_atlag = (parok["ossz"] - parok["points"]) / (parok["db"] - 1)
    parok["elonye"] = parok["points"] - csapattars_atlag

    versenyzo = parok.groupby("driverId").agg(
        elony=("elonye", "mean"), futamok=("elonye", "size")
    )
    versenyzo = versenyzo[versenyzo["futamok"] >= 50].reset_index()
    versenyzo = versenyzo.merge(
        versenyzok[["driverId", "forename", "surname"]], on="driverId"
    )
    versenyzo["nev"] = versenyzo["forename"] + " " + versenyzo["surname"]
    versenyzo = versenyzo.sort_values("elony", ascending=False)

    top = versenyzo.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top["nev"], top["elony"], color="tab:orange")
    plt.gca().invert_yaxis()
    plt.title("Átlagos pontelőny a csapattárshoz képest (futamonként, min. 50 futam)")
    plt.xlabel("Pontkülönbség futamonként")
    plt.tight_layout()
    plt.savefig(os.path.join(ABRA_MAPPA, "3_csapattars.png"), dpi=150)
    plt.show()

    legjobb = versenyzo.iloc[0]
    leggyengebb = versenyzo.iloc[-1]
    print(f"A futamonkénti pontszerzés varianciájának {csapat_resz:.0f}%-át "
          f"pusztán az magyarázza, melyik csapatban ül a versenyző az adott "
          f"évben.")
    print(f"A csapattársához képest a legjobban {legjobb['nev']} teljesített: "
          f"átlagosan {legjobb['elony']:+.2f} pont előnnyel futamonként "
          f"({int(legjobb['futamok'])} futam alapján).")
    print(f"A legnagyobb hátrányban {leggyengebb['nev']} volt "
          f"({leggyengebb['elony']:+.2f} pont futamonként).")
    print()


def main():
    os.makedirs(ABRA_MAPPA, exist_ok=True)
    eredmenyek, versenyzok, futamok, palyak = adat_betoltes()
    adat = elokeszites(eredmenyek, futamok)
    kerdes_1(adat)
    kerdes_2(adat, palyak)
    kerdes_3(adat, versenyzok)


if __name__ == "__main__":
    main()
