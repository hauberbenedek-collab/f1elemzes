"""
Formula 1 elemzés, komplexebb kérdésekkel
Kaggle adatcsomag: rohanrao/formula-1-world-championship-1950-2020

Könyvtárak: pandas, numpy, matplotlib
Öt kérdés:
  1. Mi határozza meg a végső helyezést: a rajtpozíció, a csapat ereje
     vagy a versenyző tapasztalata? (többváltozós lineáris regresszió)
  2. Melyik pályán számít a legjobban a rajtpozíció? (korreláció pályánként)
  3. Mennyit számít a csapat, és kik teljesítenek a legjobban a csapattársukhoz
     képest? (szóráselemzés és csapattárs-összehasonlítás)
  4. Hol nyer a legnagyobb eséllyel a pole pozícióból induló versenyző?
     (győzelmi arány pályánként, összevetve a 2. kérdés korrelációjával)
  5. Hogyan változott a csapatok dominanciája az évek során?
     (a legjobb csapat győzelmi aránya és a különböző győztesek száma)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KAGGLE_NEV = "rohanrao/formula-1-world-championship-1950-2020"
ABRA_MAPPA = "abrak"
MODERN_KEZDET = 2010  # 2010-től egységes a pontozási rendszer


def nevelo(ev):
    """Névelő az évtizedhez: az 1950-es, a 2010-es."""
    return "az" if ev < 2000 else "a"


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
    csapatok = olvas("constructors.csv")

    print(f"Eredmények: {len(eredmenyek)} sor")
    print(f"Versenyzők: {len(versenyzok)} sor")
    print(f"Futamok: {len(futamok)} sor")
    print(f"Pályák: {len(palyak)} sor\n")
    return eredmenyek, versenyzok, futamok, palyak, csapatok


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

    atlag = korr.mean()
    cimkek = [nevek.loc[i] for i in korr.index]
    # piros: az átlagnál erősebb, kék: az átlagnál gyengébb kapcsolat
    szinek = ["tab:red" if v >= atlag else "tab:blue" for v in korr.values]

    plt.figure(figsize=(11, max(7, len(korr) * 0.35)))
    plt.barh(cimkek, korr.values, color=szinek)
    plt.axvline(atlag, color="black", linestyle="--", linewidth=0.8,
                label=f"Átlag: {atlag:.2f}")
    plt.gca().invert_yaxis()
    plt.title("Rajtpozíció és végső helyezés korrelációja pályánként (2000 után)")
    plt.xlabel("Korrelációs együttható (r)")
    plt.legend()
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

    # a Monaco külön említése, ha benne van az elemzésben
    for palya_id, nev in nevek.items():
        if "Monaco" in nev and palya_id in korr.index:
            hely = list(korr.index).index(palya_id) + 1
            print(f"A(z) {nev} pályán r = {korr.loc[palya_id]:.2f}, ez a "
                  f"{len(korr)} pálya közül a(z) {hely}. legerősebb kapcsolat.")
    print()
    return korr


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


def kerdes_4(adat, palyak, korr):
    print("=" * 60)
    print("4. KÉRDÉS: Hol nyer a legnagyobb eséllyel a pole-ból induló?")
    print("=" * 60)

    pole = adat[(adat["year"] >= 2000) & (adat["grid"] == 1)].copy()
    pole["nyert"] = pole["positionOrder"] == 1

    csoport = pole.groupby("circuitId")["nyert"].agg(["mean", "size"])
    csoport = csoport[csoport["size"] >= 15]  # legalább 15 futam pályánként
    pole_arany = (csoport["mean"] * 100).sort_values(ascending=False)

    nevek = palyak.set_index("circuitId")["name"]
    atlag = pole["nyert"].mean() * 100
    cimkek = [nevek.loc[i] for i in pole_arany.index]
    szinek = ["tab:red" if v >= atlag else "tab:blue" for v in pole_arany.values]

    plt.figure(figsize=(11, max(7, len(pole_arany) * 0.35)))
    plt.barh(cimkek, pole_arany.values, color=szinek)
    plt.axvline(atlag, color="black", linestyle="--", linewidth=0.8,
                label=f"Átlag: {atlag:.0f}%")
    plt.gca().invert_yaxis()
    plt.title("A pole pozícióból induló versenyzők győzelmi aránya pályánként (2000 után)")
    plt.xlabel("Győzelmi arány (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ABRA_MAPPA, "4_pole.png"), dpi=150)
    plt.show()

    print(f"A pole-ból induló versenyzők átlagosan {atlag:.0f}%-a nyert "
          f"2000 után.")
    print(f"A legnagyobb esély a(z) {nevek.loc[pole_arany.index[0]]} "
          f"pályán van ({pole_arany.iloc[0]:.0f}%), a legkisebb a(z) "
          f"{nevek.loc[pole_arany.index[-1]]} pályán ({pole_arany.iloc[-1]:.0f}%).")

    for palya_id, nev in nevek.items():
        if "Monaco" in nev and palya_id in pole_arany.index:
            hely = list(pole_arany.index).index(palya_id) + 1
            print(f"A(z) {nev} pályán a pole-ból induló {pole_arany.loc[palya_id]:.0f}%"
                  f"-ban nyer, ez a {len(pole_arany)} pálya közül a(z) {hely}. helyen áll.")

    # összevetés a 2. kérdés korrelációjával
    kozos = pole_arany.index.intersection(korr.index)
    r = np.corrcoef(
        pole_arany.loc[kozos].to_numpy(dtype=float),
        korr.loc[kozos].to_numpy(dtype=float),
    )[0, 1]
    kapcsolat = "erősen összefügg" if r > 0.7 else "csak részben függ össze"
    print(f"A pole-győzelmi arány és a 2. kérdés korrelációja pályánként "
          f"r = {r:.2f}, tehát a két mérőszám {kapcsolat}.")
    print()


def kerdes_5(adat, csapatok):
    print("=" * 60)
    print("5. KÉRDÉS: Hogyan változott a csapatok dominanciája?")
    print("=" * 60)

    gyoztesek = adat[adat["positionOrder"] == 1]
    futamok_szama = gyoztesek.groupby("year").size()

    eves_csapat = gyoztesek.groupby(["year", "constructorId"]).size()
    legjobb_db = eves_csapat.groupby(level="year").max()
    legjobb_idx = eves_csapat.groupby(level="year").idxmax()

    dominancia = legjobb_db / futamok_szama * 100
    kulonbozo = gyoztesek.groupby("year")["driverId"].nunique()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax1.plot(dominancia.index, dominancia.values, marker="o", markersize=3,
             label="Éves érték")
    ax1.plot(dominancia.index, dominancia.rolling(5, center=True).mean(),
             color="red", label="5 éves átlag")
    ax1.set_ylabel("A legjobb csapat győzelmi aránya (%)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(kulonbozo.index, kulonbozo.values, marker="o", markersize=3,
             color="tab:green")
    ax2.set_ylabel("Különböző futamgyőztesek száma")
    ax2.set_xlabel("Év")
    ax2.grid(alpha=0.3)
    fig.suptitle("Csapatdominancia és a győztesek sokfélesége évenként")
    plt.tight_layout()
    plt.savefig(os.path.join(ABRA_MAPPA, "5_dominancia.png"), dpi=150)
    plt.show()

    # évtizedes átlagok numpy maszkolással
    evek = dominancia.index.to_numpy()
    ertekek = dominancia.to_numpy()
    for evtized in sorted(set((evek // 10) * 10)):
        maszk = (evek >= evtized) & (evek < evtized + 10)
        print(f"{nevelo(int(evtized)).capitalize()} {int(evtized)}-es években "
              f"a legjobb csapat átlagosan a futamok "
              f"{np.mean(ertekek[maszk]):.0f}%-át nyerte.")

    csapat_nev = csapatok.set_index("constructorId")["name"]
    for ev in dominancia.nlargest(3).index:
        csapat = csapat_nev.loc[legjobb_idx.loc[ev][1]]
        print(f"{ev}: a(z) {csapat} a futamok {dominancia.loc[ev]:.0f}%-át "
              f"nyerte, ez a legdominánsabb szezonok egyike.")

    r = np.corrcoef(dominancia.to_numpy(), kulonbozo.loc[dominancia.index].to_numpy())[0, 1]
    print(f"A csapatdominancia és a különböző győztesek száma közti "
          f"korreláció r = {r:.2f}, vagyis ahol egy csapat jobban uralkodik, "
          f"ott {'kevesebb' if r < 0 else 'több'} különböző versenyző nyer.")
    print()


def main():
    os.makedirs(ABRA_MAPPA, exist_ok=True)
    eredmenyek, versenyzok, futamok, palyak, csapatok = adat_betoltes()
    adat = elokeszites(eredmenyek, futamok)
    kerdes_1(adat)
    korr = kerdes_2(adat, palyak)
    kerdes_3(adat, versenyzok)
    kerdes_4(adat, palyak, korr)
    kerdes_5(adat, csapatok)


if __name__ == "__main__":
    main()
