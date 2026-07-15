# -*- coding: utf-8 -*-


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==========================================================
# 0. PARAMETRI GENERALI
# ==========================================================

RANDOM_SEED = 42

# Soglie A1AT (mg/dL)
SOGLIA_DEFICIT_SEVERO = 50
SOGLIA_RANGE_DATASET_MIN = 90
SOGLIA_APPROFONDIMENTO = 110
SOGLIA_RANGE_DATASET_MAX = 200

NOME_FIGURA = "diagnostica_1d_soglia_a1at"


# ==========================================================
# 1. PERCORSI FILE
# ==========================================================

if "__file__" in globals():
    base_dir = Path(__file__).resolve().parent
    nome_script = Path(__file__).stem
else:
    base_dir = Path.cwd()
    nome_script = "output_figura_diagnostica_1d"

csv_path = base_dir / "dataset_finale.csv"

cartella_output = base_dir / nome_script
cartella_output.mkdir(parents=True, exist_ok=True)

png_path = cartella_output / f"{NOME_FIGURA}.png"


# ==========================================================
# 2. COLONNE DEL DATASET
# ==========================================================

COL_ID = "NUMERO\nPROGRESSIVO"
COL_ALLELICA = "CARATTERIZZAZIONE\nALLELICA"

COL_ALFA1_PERC = "ALFA1 %\n (2,9 – 4,9 %)"
COL_ALFA1_ASS = "ALFA1#\n(0,20-0,35 g/dl)"
COL_A1AT = "A1AT\n(90-200 mg/dl)"
COL_PCR = "PCR\n(0,0-5,0 mg/l)"

FEATURES = [
    COL_ALFA1_PERC,
    COL_ALFA1_ASS,
    COL_A1AT,
    COL_PCR,
]


# ==========================================================
# 3. CARICAMENTO E PREPARAZIONE DATI
# ==========================================================

df = pd.read_csv(csv_path)
df_num = df.copy()

# Conversione numerica robusta delle feature
for col in FEATURES:
    df_num[col] = pd.to_numeric(
        df_num[col].astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce"
    )

df_num["ALLELICA_clean"] = df_num[COL_ALLELICA].astype(str).str.strip()

# Target binario: MM = NORMALE, tutto il resto = MUTAZIONE
df_num["CLASSE"] = np.where(
    df_num["ALLELICA_clean"] == "MM",
    "NORMALE",
    "MUTAZIONE"
)

# Rimuove solo eventuali righe senza A1AT
df_num = df_num.dropna(subset=[COL_A1AT]).reset_index(drop=True)


# ==========================================================
# 4. GRUPPI DA DISEGNARE
# ==========================================================

pazienti_normali = df_num[df_num["CLASSE"] == "NORMALE"].copy()

mutati_invisibili = df_num[
    (df_num["CLASSE"] == "MUTAZIONE") &
    (df_num[COL_A1AT] >= SOGLIA_APPROFONDIMENTO)
].copy()

mutati_visibili = df_num[
    (df_num["CLASSE"] == "MUTAZIONE") &
    (df_num[COL_A1AT] < SOGLIA_APPROFONDIMENTO)
].copy()

n_tot = len(df_num)
n_normali = len(pazienti_normali)
n_mutati = len(mutati_visibili) + len(mutati_invisibili)
n_invisibili = len(mutati_invisibili)


# ==========================================================
# 5. CONTROLLO A VIDEO
# ==========================================================

print("=" * 80)
print("CONTROLLO DATI - FIGURA DIAGNOSTICA 1D A1AT")
print("=" * 80)
print(f"Dataset letto da: {csv_path}")
print(f"Pazienti totali analizzati: {n_tot}")
print(f"Pazienti MM/NORMALE: {n_normali}")
print(f"Pazienti MUTAZIONE: {n_mutati}")
print(f"Mutati invisibili con A1AT >= {SOGLIA_APPROFONDIMENTO} mg/dL: {n_invisibili}")

if n_invisibili > 0:
    print("\nDettaglio mutati invisibili:")
    print(
        mutati_invisibili[
            [COL_ID, COL_ALLELICA, COL_A1AT, COL_ALFA1_PERC, COL_ALFA1_ASS, COL_PCR]
        ]
        .sort_values(by=COL_A1AT)
        .to_string(index=False)
    )
else:
    print("\nNessun mutato invisibile secondo la soglia scelta.")


# ==========================================================
# 6. IMPOSTAZIONI GRAFICHE
# ==========================================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"

rng = np.random.default_rng(RANDOM_SEED)

x_min = 30
x_max = max(220, int(np.ceil((df_num[COL_A1AT].max() + 10) / 10) * 10))

fig, ax = plt.subplots(figsize=(14.5, 7.0), dpi=150)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")


# ==========================================================
# 7. FASCE DI SFONDO
# ==========================================================

ax.axvspan(
    x_min,
    SOGLIA_DEFICIT_SEVERO,
    color="#f4b6b2",
    alpha=0.55,
    zorder=0
)

ax.axvspan(
    SOGLIA_DEFICIT_SEVERO,
    SOGLIA_APPROFONDIMENTO,
    color="#f6df8f",
    alpha=0.60,
    zorder=0
)

ax.axvspan(
    SOGLIA_APPROFONDIMENTO,
    x_max,
    color="#bfe6bd",
    alpha=0.62,
    zorder=0
)

# Evidenzia il range A1AT del dataset: 90-200 mg/dL
ax.axvspan(
    SOGLIA_RANGE_DATASET_MIN,
    SOGLIA_RANGE_DATASET_MAX,
    color="none",
    ec="#4f4f4f",
    lw=1.3,
    linestyle=":",
    alpha=0.80,
    zorder=1
)


# ==========================================================
# 8. LINEE DI RIFERIMENTO
# ==========================================================

ax.axhline(
    0,
    color="#222222",
    linewidth=1.6,
    zorder=2
)

for soglia, label, color in [
    (SOGLIA_DEFICIT_SEVERO, "50", "#6b0000"),
    (SOGLIA_RANGE_DATASET_MIN, "90", "#555555"),
    (SOGLIA_APPROFONDIMENTO, "110", "#111111"),
    (SOGLIA_RANGE_DATASET_MAX, "200", "#555555"),
]:
    ax.axvline(
        soglia,
        color=color,
        linestyle="--" if soglia in [SOGLIA_DEFICIT_SEVERO, SOGLIA_APPROFONDIMENTO] else ":",
        linewidth=1.2,
        alpha=0.75,
        zorder=2
    )

    ax.text(
        soglia,
        0.88,
        f"{label} mg/dL",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.80
        ),
        zorder=5
    )


# ==========================================================
# 9. PUNTI REALI DEL DATASET
# ==========================================================

y_normali = rng.uniform(0.08, 0.32, size=len(pazienti_normali))
y_mutati_visibili = -rng.uniform(0.08, 0.32, size=len(mutati_visibili))

# Per i mutati invisibili uso posizioni ordinate, così le annotazioni sono più stabili
mutati_invisibili = mutati_invisibili.sort_values(by=COL_A1AT).reset_index(drop=True)
if len(mutati_invisibili) > 0:
    y_mutati_invisibili = -np.linspace(0.12, 0.22, len(mutati_invisibili))
else:
    y_mutati_invisibili = np.array([])

# Pazienti MM
ax.scatter(
    pazienti_normali[COL_A1AT],
    y_normali,
    s=62,
    marker="o",
    color="#1f5be3",
    edgecolor="white",
    linewidth=0.7,
    alpha=0.78,
    zorder=3
)

# Pazienti mutati sotto soglia
ax.scatter(
    mutati_visibili[COL_A1AT],
    y_mutati_visibili,
    s=80,
    marker="D",
    color="#d92d27",
    edgecolor="white",
    linewidth=0.7,
    alpha=0.82,
    zorder=4
)

# Mutati invisibili
ax.scatter(
    mutati_invisibili[COL_A1AT],
    y_mutati_invisibili,
    s=220,
    marker="D",
    color="#d92d27",
    edgecolor="black",
    linewidth=1.7,
    alpha=0.98,
    zorder=6
)


# ==========================================================
# 10. ANNOTAZIONI MUTATI INVISIBILI
# ==========================================================

if n_invisibili > 0:
    annotation_positions = [
        (-14, -0.72),
        (18, -0.72),
        (-20, -0.58),
        (22, -0.58),
    ]

    for i, (_, row) in enumerate(mutati_invisibili.iterrows()):
        x = row[COL_A1AT]
        y = y_mutati_invisibili[i]
        dx, dy = annotation_positions[i % len(annotation_positions)]

        id_paz = row[COL_ID] if COL_ID in row.index else ""
        genotipo = row["ALLELICA_clean"]

        ax.annotate(
            f"ID {id_paz} - {genotipo}\nA1AT = {x:.0f} mg/dL",
            xy=(x, y),
            xytext=(x + dx, y + dy),
            ha="center",
            va="top",
            fontsize=8.8,
            color="#8f0000",
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="->",
                color="#8f0000",
                lw=1.35,
                shrinkA=4,
                shrinkB=4
            ),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#8f0000",
                linewidth=1.0,
                alpha=0.94
            ),
            zorder=7
        )


# ==========================================================
# 11. TESTI DELLE ZONE
# ==========================================================

ax.text(
    (x_min + SOGLIA_DEFICIT_SEVERO) / 2,
    0.73,
    "Deficit\nsevero",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#8f0000",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="none",
        alpha=0.70
    ),
    zorder=5
)

ax.text(
    (SOGLIA_DEFICIT_SEVERO + SOGLIA_APPROFONDIMENTO) / 2,
    0.73,
    "Area di sospetto\n/ deficit intermedio",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#7a5a00",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="none",
        alpha=0.70
    ),
    zorder=5
)

ax.text(
    (SOGLIA_APPROFONDIMENTO + x_max) / 2,
    0.73,
    f"Area non sospetta secondo soglia 1D\nA1AT ≥ {SOGLIA_APPROFONDIMENTO} mg/dL",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#1c6e22",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="none",
        alpha=0.70
    ),
    zorder=5
)

ax.text(
    (SOGLIA_RANGE_DATASET_MIN + SOGLIA_RANGE_DATASET_MAX) / 2,
    0.53,
    "Range A1AT riportato nel dataset: 90–200 mg/dL",
    ha="center",
    va="center",
    fontsize=9,
    color="#444444",
    bbox=dict(
        boxstyle="round,pad=0.28",
        facecolor="white",
        edgecolor="#666666",
        alpha=0.85
    ),
    zorder=5
)


# ==========================================================
# 12. TITOLO, LEGENDA
# ==========================================================

ax.set_title(
    "Diagnostica 1D: soglia su A1AT",
    fontsize=18,
    pad=40,
    color="#222222"
)

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Pazienti MM",
        markerfacecolor="#1f5be3",
        markeredgecolor="white",
        markersize=9
    ),
    Line2D(
        [0], [0],
        marker="D",
        color="w",
        label="Pazienti mutati sotto soglia",
        markerfacecolor="#d92d27",
        markeredgecolor="white",
        markersize=9
    ),
    Line2D(
        [0], [0],
        marker="D",
        color="w",
        label='Mutati "invisibili"',
        markerfacecolor="#d92d27",
        markeredgecolor="black",
        markersize=12
    ),
]

ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.14),
    ncol=3,
    frameon=False,
    fontsize=10
)


# ==========================================================
# 13. FORMATTAZIONE ASSI
# ==========================================================

ax.set_xlim(x_min, x_max)
ax.set_ylim(-1.28, 1.02)

ax.set_yticks([])

xticks = [50, 90, 110, 150, 200]
xticks = [x for x in xticks if x_min <= x <= x_max]
ax.set_xticks(xticks)

ax.set_xlabel(
    "A1AT (mg/dL)",
    fontsize=12,
    labelpad=12,
    color="#222222"
)

ax.tick_params(axis="x", labelsize=10, colors="#333333")

for spine in ["left", "right", "top"]:
    ax.spines[spine].set_visible(False)

ax.spines["bottom"].set_color("#333333")
ax.spines["bottom"].set_linewidth(1.1)

ax.grid(
    axis="x",
    linestyle=":",
    linewidth=0.8,
    alpha=0.35
)

# ==========================================================
# 14. SALVATAGGIO
# ==========================================================

plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.90])

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nFigura salvata in:")
print(png_path)

# -*- coding: utf-8 -*-
"""
Diagnostica 2D: A1AT + ALFA1%
--------------------------------
Genera una figura basata sui dati reali del dataset_finale.csv.

La figura mostra:
- pazienti MM;
- pazienti mutati con A1AT sotto soglia;
- mutati "invisibili" in 1D, cioè con A1AT >= 110 mg/dL;
- soglia verticale su A1AT;
- soglia orizzontale su ALFA1%.

Input:
    dataset_finale.csv

Output:
    cartella con lo stesso nome dello script:
    - diagnostica_2d_a1at_alfa1.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==========================================================
# 0. PARAMETRI GENERALI
# ==========================================================

RANDOM_SEED = 42

SOGLIA_A1AT_1D = 110          # mg/dL
SOGLIA_A1AT_RANGE_MIN = 90    # mg/dL
SOGLIA_A1AT_RANGE_MAX = 200   # mg/dL
SOGLIA_ALFA1_PERC_MIN = 2.9   # %

NOME_FIGURA = "diagnostica_2d_a1at_alfa1"


# ==========================================================
# 1. PERCORSI FILE
# ==========================================================

if "__file__" in globals():
    base_dir = Path(__file__).resolve().parent
    nome_script = Path(__file__).stem
else:
    base_dir = Path.cwd()
    nome_script = "output_figura_diagnostica_2d"

csv_path = base_dir / "dataset_finale.csv"

cartella_output = base_dir / nome_script
cartella_output.mkdir(parents=True, exist_ok=True)

png_path = cartella_output / f"{NOME_FIGURA}.png"


# ==========================================================
# 2. COLONNE DEL DATASET
# ==========================================================

COL_ID = "NUMERO\nPROGRESSIVO"
COL_ALLELICA = "CARATTERIZZAZIONE\nALLELICA"

COL_ALFA1_PERC = "ALFA1 %\n (2,9 – 4,9 %)"
COL_ALFA1_ASS = "ALFA1#\n(0,20-0,35 g/dl)"
COL_A1AT = "A1AT\n(90-200 mg/dl)"
COL_PCR = "PCR\n(0,0-5,0 mg/l)"

FEATURES = [
    COL_ALFA1_PERC,
    COL_ALFA1_ASS,
    COL_A1AT,
    COL_PCR,
]


# ==========================================================
# 3. CARICAMENTO E PREPARAZIONE DATI
# ==========================================================

df = pd.read_csv(csv_path)
df_num = df.copy()

for col in FEATURES:
    df_num[col] = pd.to_numeric(
        df_num[col].astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce"
    )

df_num["ALLELICA_clean"] = df_num[COL_ALLELICA].astype(str).str.strip()

df_num["CLASSE"] = np.where(
    df_num["ALLELICA_clean"] == "MM",
    "NORMALE",
    "MUTAZIONE"
)

df_num = df_num.dropna(subset=[COL_A1AT, COL_ALFA1_PERC]).reset_index(drop=True)


# ==========================================================
# 4. GRUPPI DA DISEGNARE
# ==========================================================

pazienti_normali = df_num[df_num["CLASSE"] == "NORMALE"].copy()

mutati_sotto_soglia = df_num[
    (df_num["CLASSE"] == "MUTAZIONE") &
    (df_num[COL_A1AT] < SOGLIA_A1AT_1D)
].copy()

mutati_invisibili = df_num[
    (df_num["CLASSE"] == "MUTAZIONE") &
    (df_num[COL_A1AT] >= SOGLIA_A1AT_1D)
].copy()

mutati_invisibili = mutati_invisibili.sort_values(by=COL_A1AT).reset_index(drop=True)

n_tot = len(df_num)
n_normali = len(pazienti_normali)
n_mutati = len(mutati_sotto_soglia) + len(mutati_invisibili)
n_invisibili = len(mutati_invisibili)


# ==========================================================
# 5. CONTROLLO A VIDEO
# ==========================================================

print("=" * 90)
print("CONTROLLO DATI - FIGURA DIAGNOSTICA 2D A1AT + ALFA1%")
print("=" * 90)
print(f"Dataset letto da: {csv_path}")
print(f"Pazienti totali analizzati: {n_tot}")
print(f"Pazienti MM/NORMALE: {n_normali}")
print(f"Pazienti MUTAZIONE: {n_mutati}")
print(f"Mutati invisibili in 1D con A1AT >= {SOGLIA_A1AT_1D} mg/dL: {n_invisibili}")

if n_invisibili > 0:
    print("\nDettaglio mutati invisibili in 1D:")
    print(
        mutati_invisibili[
            [COL_ID, COL_ALLELICA, COL_A1AT, COL_ALFA1_PERC, COL_ALFA1_ASS, COL_PCR]
        ]
        .sort_values(by=COL_A1AT)
        .to_string(index=False)
    )


# ==========================================================
# 6. IMPOSTAZIONI GRAFICHE
# ==========================================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"

rng = np.random.default_rng(RANDOM_SEED)

x_min = 30
x_max = max(220, int(np.ceil((df_num[COL_A1AT].max() + 10) / 10) * 10))

y_min = 1.75
y_max = max(6.4, float(df_num[COL_ALFA1_PERC].max()) + 0.5)

fig, ax = plt.subplots(figsize=(14.8, 7.4), dpi=150)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")


# ==========================================================
# 7. SFONDO: ZONE DIAGNOSTICHE
# ==========================================================

# Area sospetta secondo A1AT
ax.axvspan(
    x_min,
    SOGLIA_A1AT_1D,
    color="#f4b6b2",
    alpha=0.48,
    zorder=0
)

# Area non sospetta secondo A1AT
ax.axvspan(
    SOGLIA_A1AT_1D,
    x_max,
    color="#bfe6bd",
    alpha=0.55,
    zorder=0
)

# Banda ALFA1% sotto range
ax.axhspan(
    y_min,
    SOGLIA_ALFA1_PERC_MIN,
    color="#f6df8f",
    alpha=0.36,
    zorder=1
)


# ==========================================================
# 8. LINEE DI RIFERIMENTO
# ==========================================================

ax.axvline(
    SOGLIA_A1AT_1D,
    color="#111111",
    linestyle="--",
    linewidth=1.4,
    alpha=0.85,
    zorder=3
)

ax.axhline(
    SOGLIA_ALFA1_PERC_MIN,
    color="#9a6a00",
    linestyle="--",
    linewidth=1.35,
    alpha=0.90,
    zorder=3
)

# Range A1AT del dataset: 90-200 mg/dL
for soglia in [SOGLIA_A1AT_RANGE_MIN, SOGLIA_A1AT_RANGE_MAX]:
    ax.axvline(
        soglia,
        color="#555555",
        linestyle=":",
        linewidth=1.2,
        alpha=0.70,
        zorder=3
    )


# ==========================================================
# 9. ETICHETTE DELLE SOGLIE
# ==========================================================

ax.text(
    SOGLIA_A1AT_1D,
    y_max - 0.10,
    f"A1AT = {SOGLIA_A1AT_1D} mg/dL",
    ha="center",
    va="top",
    fontsize=9.5,
    fontweight="bold",
    color="#111111",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="none",
        alpha=0.90
    ),
    zorder=7
)

ax.text(
    x_min + 2,
    SOGLIA_ALFA1_PERC_MIN + 0.05,
    f"ALFA1% = {SOGLIA_ALFA1_PERC_MIN:.1f}%",
    ha="left",
    va="bottom",
    fontsize=9.5,
    fontweight="bold",
    color="#7a5a00",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="none",
        alpha=0.88
    ),
    zorder=7
)

ax.text(
    SOGLIA_A1AT_RANGE_MIN,
    y_max - 0.10,
    "90",
    ha="center",
    va="top",
    fontsize=8.8,
    color="#555555",
    fontweight="bold",
    zorder=7
)

ax.text(
    SOGLIA_A1AT_RANGE_MAX,
    y_max - 0.10,
    "200",
    ha="center",
    va="top",
    fontsize=8.8,
    color="#555555",
    fontweight="bold",
    zorder=7
)


# ==========================================================
# 10. TESTI DELLE AREE, IN ALTO E NON SUI PUNTI
# ==========================================================

ax.text(
    67,
    y_max - 0.52,
    f"Area sospetta\nA1AT < {SOGLIA_A1AT_1D} mg/dL",
    ha="center",
    va="top",
    fontsize=9.5,
    fontweight="bold",
    color="#8f0000",
    bbox=dict(
        boxstyle="round,pad=0.28",
        facecolor="white",
        edgecolor="none",
        alpha=0.72
    ),
    zorder=6
)

ax.text(
    165,
    y_max - 0.52,
    f"Area non sospetta secondo soglia 1D\nA1AT ≥ {SOGLIA_A1AT_1D} mg/dL",
    ha="center",
    va="top",
    fontsize=9.5,
    fontweight="bold",
    color="#1c6e22",
    bbox=dict(
        boxstyle="round,pad=0.28",
        facecolor="white",
        edgecolor="none",
        alpha=0.72
    ),
    zorder=6
)

ax.text(
    54,
    y_min + 0.12,
    "ALFA1% sotto range",
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
    color="#7a5a00",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="none",
        alpha=0.70
    ),
    zorder=6
)


# ==========================================================
# 11. PUNTI REALI DEL DATASET
# ==========================================================

def jitter_x(values, scale=0.45):
    return values.to_numpy(dtype=float) + rng.normal(0, scale, size=len(values))

def jitter_y(values, scale=0.030):
    return values.to_numpy(dtype=float) + rng.normal(0, scale, size=len(values))


# Pazienti MM
ax.scatter(
    jitter_x(pazienti_normali[COL_A1AT]),
    jitter_y(pazienti_normali[COL_ALFA1_PERC]),
    s=62,
    marker="o",
    color="#1f5be3",
    edgecolor="white",
    linewidth=0.7,
    alpha=0.78,
    zorder=4
)

# Mutati sotto soglia A1AT
ax.scatter(
    jitter_x(mutati_sotto_soglia[COL_A1AT]),
    jitter_y(mutati_sotto_soglia[COL_ALFA1_PERC]),
    s=82,
    marker="D",
    color="#d92d27",
    edgecolor="white",
    linewidth=0.7,
    alpha=0.82,
    zorder=5
)

# Mutati invisibili in 1D
ax.scatter(
    mutati_invisibili[COL_A1AT],
    mutati_invisibili[COL_ALFA1_PERC],
    s=230,
    marker="D",
    color="#f39c12",
    edgecolor="black",
    linewidth=1.7,
    alpha=0.98,
    zorder=8
)


# ==========================================================
# 12. ANNOTAZIONI DEI MUTATI INVISIBILI
# ==========================================================

if n_invisibili > 0:
    # Posizioni scelte per non coprire la nuvola dei punti
    annotation_positions = [
        (-33, 1.15),   # primo invisibile, spostato in alto a sinistra
        (34, 1.35),    # secondo invisibile, spostato in alto a destra
        (-30, -0.80),
        (30, -0.80),
    ]

    for i, (_, row) in enumerate(mutati_invisibili.iterrows()):
        x = row[COL_A1AT]
        y = row[COL_ALFA1_PERC]
        dx, dy = annotation_positions[i % len(annotation_positions)]

        id_paz = row[COL_ID] if COL_ID in row.index else ""
        genotipo = row["ALLELICA_clean"]

        ax.annotate(
            f"ID {id_paz} - {genotipo}\nA1AT = {x:.0f} mg/dL | ALFA1% = {y:.2f}%",
            xy=(x, y),
            xytext=(x + dx, y + dy),
            ha="center",
            va="center",
            fontsize=8.6,
            color="#8a4f00",
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="->",
                color="#8a4f00",
                lw=1.25,
                shrinkA=4,
                shrinkB=4
            ),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#8a4f00",
                linewidth=1.0,
                alpha=0.94
            ),
            zorder=9
        )


# ==========================================================
# 13. TITOLO E LEGENDA
# ==========================================================

ax.set_title(
    "Diagnostica 2D: A1AT + ALFA1%",
    fontsize=18,
    pad=42,
    color="#222222"
)

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Pazienti MM",
        markerfacecolor="#1f5be3",
        markeredgecolor="white",
        markersize=9
    ),
    Line2D(
        [0], [0],
        marker="D",
        color="w",
        label="Pazienti mutati sotto soglia A1AT",
        markerfacecolor="#d92d27",
        markeredgecolor="white",
        markersize=9
    ),
    Line2D(
        [0], [0],
        marker="D",
        color="w",
        label='Mutati "invisibili" in 1D',
        markerfacecolor="#f39c12",
        markeredgecolor="black",
        markersize=12
    ),
]

ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.14),
    ncol=3,
    frameon=False,
    fontsize=10
)


# ==========================================================
# 14. FORMATTAZIONE ASSI
# ==========================================================

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel(
    "A1AT (mg/dL)",
    fontsize=12,
    labelpad=12,
    color="#222222"
)

ax.set_ylabel(
    "ALFA1% (%)",
    fontsize=12,
    labelpad=12,
    color="#222222"
)

xticks = [50, 90, 110, 150, 200]
xticks = [x for x in xticks if x_min <= x <= x_max]
ax.set_xticks(xticks)

ax.tick_params(axis="both", labelsize=10, colors="#333333")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.spines["bottom"].set_color("#333333")
ax.spines["left"].set_color("#333333")
ax.spines["bottom"].set_linewidth(1.1)
ax.spines["left"].set_linewidth(1.1)

ax.grid(
    True,
    linestyle=":",
    linewidth=0.8,
    alpha=0.28
)


# ==========================================================
# 15. SALVATAGGIO
# ==========================================================

plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.90])

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nFigura salvata in:")
print(png_path)