import os

import numpy as np
import pandas as pd


# File di partenza e file da creare.
# Devono trovarsi nella stessa cartella di questo script.
CARTELLA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

INPUT_XLSX = os.path.join(
    CARTELLA_SCRIPT,
    "DATI alfa1antitripsina 104 CAMPIONI (1).xlsx"
)
SHEET_NAME = "DATI A1AT 104 CAMPIONI"
OUTPUT_CSV = os.path.join(CARTELLA_SCRIPT, "dataset_finale.csv")


def pulisci_numero(valore):
    """Converte i valori del foglio Excel in numeri utilizzabili da Python."""
    if pd.isna(valore):
        return np.nan

    testo = str(valore).strip().replace(",", ".")

    # Nel foglio Excel la PCR può comparire come "<0,5".
    # Per avere un valore numerico unico uso metà del limite indicato.
    if testo.startswith("<"):
        testo = testo.replace("<", "").strip()
        numero = pd.to_numeric(testo, errors="coerce")
        if pd.isna(numero):
            return np.nan
        return numero / 2

    return pd.to_numeric(testo, errors="coerce")


print("Leggo il file Excel...")
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME, header=0)

# Dopo i 104 pazienti nel file Excel ci sono righe vuote o di legenda.
# Tengo solo le righe che hanno il numero progressivo nella prima colonna.
prima_colonna = df.columns[0]
df = df[df[prima_colonna].notna()].copy()

# Conversioni di formato usate negli script della tesi.
df[prima_colonna] = df[prima_colonna].astype(int)
df["SESSO"] = df["SESSO"].map({"F": 0, "M": 1}).astype(int)
df["ETA'"] = df["ETA'"].astype(int)

marker_cols = [
    "ALFA1 %\n (2,9 – 4,9 %)",
    "ALFA1#\n(0,20-0,35 g/dl)",
    "A1AT\n(90-200 mg/dl)",
    "PCR\n(0,0-5,0 mg/l)",
]

for colonna in marker_cols:
    df[colonna] = df[colonna].apply(pulisci_numero)

# Nel dataset finale A1AT è salvato come intero.
df["A1AT\n(90-200 mg/dl)"] = df["A1AT\n(90-200 mg/dl)"].astype(int)

# Salvataggio del CSV finale.
df.to_csv(OUTPUT_CSV, index=False)

print(f"Dataset salvato: {OUTPUT_CSV}")
print(f"Righe salvate: {df.shape[0]}")
print(f"Colonne salvate: {df.shape[1]}")
