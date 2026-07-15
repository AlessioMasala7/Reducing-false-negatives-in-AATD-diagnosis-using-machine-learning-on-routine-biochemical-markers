# 13/06/26

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=1.05
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fontsize"] = 11

# Colori scelti
palette_classi = {
    "MUTAZIONE": "#4C72B0",
    "NORMALE": "#DD8452"
}

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================

df = pd.read_csv("dataset_finale.csv")

feature = [
    "ALFA1 %\n (2,9 – 4,9 %)",
    "ALFA1#\n(0,20-0,35 g/dl)",
    "A1AT\n(90-200 mg/dl)",
    "PCR\n(0,0-5,0 mg/l)",
]

feat_short = {
    "ALFA1 %\n (2,9 – 4,9 %)": "ALFA1%",
    "ALFA1#\n(0,20-0,35 g/dl)": "ALFA1#",
    "A1AT\n(90-200 mg/dl)": "A1AT",
    "PCR\n(0,0-5,0 mg/l)": "PCR",
}

target_col = "CARATTERIZZAZIONE\nALLELICA"

# Cartella di output
cartella_output = "test11.6 ss grafici per cap 3"
os.makedirs(cartella_output, exist_ok=True)

# ==========================================
# 2. PREPARAZIONE DATI
# ==========================================

df_num = df.copy()

# Conversione dei marker in formato numerico
# Gestisce virgole decimali e valori tipo "<0.5"
for col in feature:
    df_num[col] = (
        df_num[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("<", "", regex=False)
        .str.strip()
    )
    df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

# Rinomina le colonne dei marker per avere grafici più leggibili
df_plot = df_num.rename(columns=feat_short)

markers = ["ALFA1%", "ALFA1#", "A1AT", "PCR"]

range_marker = {
    "ALFA1%": (2.9, 4.9),
    "ALFA1#": (0.20, 0.35),
    "A1AT": (90, 200),
    "PCR": (0, 5)
}

titoli_marker = {
    "ALFA1%": "ALFA1% (%)",
    "ALFA1#": "ALFA1# (g/dL)",
    "A1AT": "A1AT (mg/dL)",
    "PCR": "PCR (mg/L)"
}

# Target binario
df_plot["CLASSE"] = np.where(
    df_num[target_col].astype(str).str.strip() == "MM",
    "NORMALE",
    "MUTAZIONE"
)

ordine_classi = ["MUTAZIONE", "NORMALE"]

print("Distribuzione classi:")
print(df_plot["CLASSE"].value_counts())
print()

summary = df_plot.groupby("CLASSE")[markers].agg(["mean", "median", "std", "min", "max"])
print("Statistiche descrittive dei marker per classe:")
print(summary.round(3))
print()

summary.round(3).to_excel(
    os.path.join(cartella_output, "statistiche_descrittive_marker_per_classe.xlsx")
)

df_long = df_plot.melt(
    id_vars="CLASSE",
    value_vars=markers,
    var_name="Marcatore",
    value_name="Valore"
)

# ==========================================
# 3. BOXPLOT MIGLIORATO DEI MARKER PER CLASSE
# ==========================================

plt.figure(figsize=(14, 7))

ax = sns.boxplot(
    data=df_long,
    x="Marcatore",
    y="Valore",
    hue="CLASSE",
    hue_order=ordine_classi,
    palette=palette_classi,
    width=0.65,
    linewidth=1.4,
    fliersize=0,
    showmeans=True,
    meanprops={
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 6
    },
    medianprops={
        "color": "black",
        "linewidth": 2
    },
    boxprops={
        "edgecolor": "black",
        "linewidth": 1.2
    },
    whiskerprops={
        "color": "black",
        "linewidth": 1.2
    },
    capprops={
        "color": "black",
        "linewidth": 1.2
    }
)

sns.stripplot(
    data=df_long,
    x="Marcatore",
    y="Valore",
    hue="CLASSE",
    hue_order=ordine_classi,
    palette=palette_classi,
    dodge=True,
    alpha=0.55,
    size=4,
    edgecolor="black",
    linewidth=0.3
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[:2],
    labels[:2],
    title="Classe",
    loc="upper right",
    frameon=True
)

ax.set_title("Distribuzione dei marker biochimici per classe")
ax.set_xlabel("Marcatore")
ax.set_ylabel("Valore misurato")

plt.tight_layout()

plt.savefig(
    os.path.join(cartella_output, "figura_3_3_boxplot_marker_per_classe_migliorato.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 3.b BOXPLOT SINGOLI PER OGNI MARKER
# ==========================================


fig, axes = plt.subplots(
    1,
    4,
    figsize=(22, 6),
    sharey=False
)

for ax, marker in zip(axes, markers):

    low, high = range_marker[marker]

    # Evidenzia il range di normalità
    ax.axhspan(
        low,
        high,
        color="green",
        alpha=0.12
    )

    # Boxplot
    sns.boxplot(
        data=df_plot,
        x="CLASSE",
        y=marker,
        hue="CLASSE",
        order=ordine_classi,
        hue_order=ordine_classi,
        palette=palette_classi,
        dodge=False,
        width=0.55,
        linewidth=1.4,
        fliersize=0,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5
        },
        medianprops={
            "color": "black",
            "linewidth": 2
        },
        boxprops={
            "edgecolor": "black",
            "linewidth": 1.2
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.2
        },
        capprops={
            "color": "black",
            "linewidth": 1.2
        },
        legend=False,
        ax=ax
    )

    # Punti reali sovrapposti
    sns.stripplot(
        data=df_plot,
        x="CLASSE",
        y=marker,
        hue="CLASSE",
        order=ordine_classi,
        hue_order=ordine_classi,
        palette=palette_classi,
        dodge=False,
        jitter=0.18,
        alpha=0.45,
        size=4,
        edgecolor="black",
        linewidth=0.3,
        legend=False,
        ax=ax
    )

    # Linee del range di normalità
    ax.axhline(low, color="green", linestyle="--", linewidth=1)
    ax.axhline(high, color="green", linestyle="--", linewidth=1)

    ax.set_title(titoli_marker[marker], fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Valore misurato")
    ax.tick_params(axis="x", rotation=20)

plt.suptitle(
    "Distribuzione dei marker biochimici per classe - boxplot separati",
    fontsize=18,
    fontweight="bold",
    y=1.03
)

plt.tight_layout()

plt.savefig(
    os.path.join(cartella_output, "figura_3_3_boxplot_singoli_marker.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 4. SCATTER PLOT A COPPIE MIGLIORATO
# ==========================================


try:
    pair = sns.pairplot(
        df_plot,
        vars=markers,
        hue="CLASSE",
        hue_order=ordine_classi,
        palette=palette_classi,
        corner=True,
        diag_kind="kde",
        height=3.0,
        plot_kws={
            "alpha": 0.75,
            "s": 55,
            "edgecolor": "black",
            "linewidth": 0.35
        },
        diag_kws={
            "fill": True,
            "alpha": 0.25,
            "linewidth": 2
        }
    )

except Exception as e:
    print("Attenzione: KDE non riuscita nel pairplot. Uso istogrammi sulla diagonale.")
    print("Errore:", e)

    pair = sns.pairplot(
        df_plot,
        vars=markers,
        hue="CLASSE",
        hue_order=ordine_classi,
        palette=palette_classi,
        corner=True,
        diag_kind="hist",
        height=3.0,
        plot_kws={
            "alpha": 0.75,
            "s": 55,
            "edgecolor": "black",
            "linewidth": 0.35
        },
        diag_kws={
            "alpha": 0.45
        }
    )

pair.fig.suptitle(
    "Relazioni a coppie tra i marker biochimici",
    y=1.02,
    fontsize=18,
    fontweight="bold"
)

if pair._legend is not None:
    pair._legend.set_title("Classe")

pair.savefig(
    os.path.join(cartella_output, "figura_3_3_scatter_plot_coppie_migliorato.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 4B. BOXPLOT SEPARATI PER OGNI FEATURE (2x2)
# ==========================================

range_normali = {
    "ALFA1%": (2.9, 4.9),
    "ALFA1#": (0.20, 0.35),
    "A1AT": (90, 200),
    "PCR": (0.0, 5.0),
}

titoli_marker = {
    "ALFA1%": "ALFA1% (%)",
    "ALFA1#": "ALFA1# (g/dL)",
    "A1AT": "A1AT (mg/dL)",
    "PCR": "PCR (mg/L)",
}

# griglia 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, marker in enumerate(markers):
    ax = axes[i]

    low, high = range_normali[marker]
    ax.axhspan(low, high, color="green", alpha=0.12, zorder=0)
    ax.axhline(low, color="green", linestyle="--", linewidth=1)
    ax.axhline(high, color="green", linestyle="--", linewidth=1)

    sns.boxplot(
        data=df_plot,
        x="CLASSE",
        y=marker,
        ax=ax,
        width=0.55,
        linewidth=1.4,
        fliersize=0,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6
        },
        medianprops={
            "color": "black",
            "linewidth": 2
        },
        boxprops={
            "edgecolor": "black",
            "linewidth": 1.2
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.2
        },
        capprops={
            "color": "black",
            "linewidth": 1.2
        }
    )

    sns.stripplot(
        data=df_plot,
        x="CLASSE",
        y=marker,
        ax=ax,
        alpha=0.55,
        size=4,
        edgecolor="black",
        linewidth=0.3
    )

    # Solo per PCR usiamo la scala logaritmica 0-5 leggibile, >5 compresso
    if marker == "PCR":
        ax.set_yscale("symlog", linthresh=5, linscale=1.0, base=10)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 10, 20, 50, 100])
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    ax.set_title(titoli_marker[marker], fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Valore misurato")
    ax.tick_params(axis="x", rotation=0)

fig.suptitle(
    "Distribuzione dei marker biochimici per classe - boxplot separati",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    os.path.join(cartella_output, "figura_3_3_boxplot_separati_2x2.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 5. HEATMAP DI CORRELAZIONE
# ==========================================

corr = df_plot[markers].corr(method="pearson")

plt.figure(figsize=(7, 6))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    square=True,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Correlazione di Pearson"}
)
plt.tight_layout()

plt.savefig(
    os.path.join(cartella_output, "figura_3_3_heatmap_correlazione.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 6. VIOLIN PLOT STANDARDIZZATO
# ==========================================

df_std = df_plot.copy()

for col in markers:
    df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()

df_long_std = df_std.melt(
    id_vars="CLASSE",
    value_vars=markers,
    var_name="Marcatore",
    value_name="Valore standardizzato"
)

plt.figure(figsize=(14, 7))

try:
    ax = sns.violinplot(
        data=df_long_std,
        x="Marcatore",
        y="Valore standardizzato",
        hue="CLASSE",
        hue_order=ordine_classi,
        palette=palette_classi,
        inner="quartile",
        cut=0,
        bw_adjust=0.8,
        linewidth=1.5,
        density_norm="width"
    )

except TypeError:
    ax = sns.violinplot(
        data=df_long_std,
        x="Marcatore",
        y="Valore standardizzato",
        hue="CLASSE",
        hue_order=ordine_classi,
        palette=palette_classi,
        inner="quartile",
        cut=0,
        bw_adjust=0.8,
        linewidth=1.5,
        scale="width"
    )

sns.stripplot(
    data=df_long_std,
    x="Marcatore",
    y="Valore standardizzato",
    hue="CLASSE",
    hue_order=ordine_classi,
    palette=palette_classi,
    dodge=True,
    alpha=0.35,
    size=3,
    edgecolor="black",
    linewidth=0.25
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[:2],
    labels[:2],
    title="Classe",
    loc="upper right",
    frameon=True
)

ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)

ax.set_title("Distribuzione standardizzata dei marker biochimici per classe")
ax.set_xlabel("Marcatore")
ax.set_ylabel("Valore standardizzato")

plt.tight_layout()

plt.savefig(
    os.path.join(cartella_output, "figura_3_3_violinplot_marker_standardizzati.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 6.b VIOLIN PLOT SINGOLI PER OGNI MARKER 2x2
# ==========================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
    sharey=False
)

axes = axes.flatten()

for ax, marker in zip(axes, markers):

    # Range di normalità del marker
    low, high = range_marker[marker]

    # Evidenzia il range di normalità
    ax.axhspan(
        low,
        high,
        color="green",
        alpha=0.12,
        zorder=0
    )

    ax.axhline(low, color="green", linestyle="--", linewidth=1)
    ax.axhline(high, color="green", linestyle="--", linewidth=1)

    # Diagramma a violino
    try:
        sns.violinplot(
            data=df_plot,
            x="CLASSE",
            y=marker,
            hue="CLASSE",
            order=ordine_classi,
            hue_order=ordine_classi,
            palette=palette_classi,
            inner="quartile",
            cut=0,
            bw_adjust=0.8,
            linewidth=1.5,
            density_norm="width",
            dodge=False,
            legend=False,
            ax=ax
        )

    except TypeError:
        sns.violinplot(
            data=df_plot,
            x="CLASSE",
            y=marker,
            hue="CLASSE",
            order=ordine_classi,
            hue_order=ordine_classi,
            palette=palette_classi,
            inner="quartile",
            cut=0,
            bw_adjust=0.8,
            linewidth=1.5,
            scale="width",
            dodge=False,
            ax=ax
        )

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Punti reali dei pazienti
    sns.stripplot(
        data=df_plot,
        x="CLASSE",
        y=marker,
        hue="CLASSE",
        order=ordine_classi,
        hue_order=ordine_classi,
        palette=palette_classi,
        dodge=False,
        alpha=0.45,
        size=4,
        edgecolor="black",
        linewidth=0.3,
        jitter=0.18,
        legend=False,
        ax=ax
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Stesse considerazioni di prima per PCR
    if marker == "PCR":
        ax.set_yscale("symlog", linthresh=5, linscale=1.0, base=10)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 10, 20, 50, 100])
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    ax.set_title(titoli_marker[marker], fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Valore misurato")
    ax.tick_params(axis="x", rotation=0)

fig.suptitle(
    "Distribuzione dei marker biochimici per classe con range di normalità",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    os.path.join(cartella_output, "figura_3_3_violinplot_singoli_marker_2x2.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()