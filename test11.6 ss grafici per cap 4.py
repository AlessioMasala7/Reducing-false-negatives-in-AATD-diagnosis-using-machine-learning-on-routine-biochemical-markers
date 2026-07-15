# ============================================================
# Figura 4.3 - esempio didattico lineare vs RBF
# Dataset sintetico: make_moons di scikit-learn
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from matplotlib.colors import ListedColormap


nome_script = os.path.splitext(os.path.basename(__file__))[0]

cartella_output = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    nome_script
)

os.makedirs(cartella_output, exist_ok=True)

OUTPUT_PNG = os.path.join(
    cartella_output,
    "figura_4_3_make_moons_lineare_vs_rbf.png"
)


# =========================
# 1. Dataset sintetico
# =========================

X, y = make_moons(
    n_samples=120,
    noise=0.18,
    random_state=42
)


# =========================
# 2. Modelli
# =========================

modelli = [
    (
        "Modello lineare",
        make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=1.0,
                max_iter=200000,
                random_state=42,
                dual="auto"
            )
        )
    ),
    (
        "SVC con kernel RBF",
        make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma=1.2,
                random_state=42
            )
        )
    )
]


# =========================
# 3. Griglia per frontiere decisionali
# =========================

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]


# =========================
# 4. Figura
# =========================

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)

cmap_background = ListedColormap(["#dbe5f1", "#fde9d9"])

colore_classe_0 = "#1f77b4"
colore_classe_1 = "#e41a1c"

for ax, (titolo, modello) in zip(axes, modelli):

    modello.fit(X, y)

    # Regioni decisionali
    Z = modello.predict(grid).reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        Z,
        cmap=cmap_background,
        alpha=0.75
    )

    # Frontiera decisionale
    decision = modello.decision_function(grid).reshape(xx.shape)

    ax.contour(
        xx,
        yy,
        decision,
        levels=[0],
        colors="black",
        linewidths=2
    )

    # Punti
    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        c=colore_classe_0,
        edgecolors="white",
        linewidths=0.8,
        s=55,
        label="Classe 0"
    )

    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c=colore_classe_1,
        edgecolors="white",
        linewidths=0.8,
        s=55,
        label="Classe 1"
    )

    ax.set_title(titolo, fontsize=14)
    ax.set_xlabel("Feature 1")
    ax.grid(alpha=0.20)

axes[0].set_ylabel("Feature 2")

fig.suptitle(
    "Confronto tra frontiera decisionale lineare e non lineare",
    fontsize=16,
    y=1.02
)

plt.tight_layout()

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")


plt.show()

print(f"Figura PNG salvata in: {OUTPUT_PNG}")


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# Figura - Effetto del parametro gamma in SVC con kernel RBF
# Dataset sintetico bidimensionale
# ============================================================

OUTPUT_PNG = os.path.join(
    cartella_output,
    "figura_gamma_svc_rbf.png"
)



# =========================
# 1. Dataset sintetico
# =========================

X, y = make_blobs(
    n_samples=70,
    centers=[[-1.5, -1.2], [1.2, 1.1]],
    cluster_std=[0.75, 0.75],
    random_state=7
)

# Trasformo le etichette in -1 e +1, solo per coerenza con la legenda
y = np.where(y == 0, -1, 1)


# =========================
# 2. Valori di gamma
# =========================

gammas = [1, 10, 100]

titoli = [
    r"$\gamma$ basso ($\gamma = 1$)",
    r"$\gamma$ medio ($\gamma = 10$)",
    r"$\gamma$ alto ($\gamma = 100$)"
]


# =========================
# 3. Griglia per frontiere
# =========================

x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 600),
    np.linspace(y_min, y_max, 600)
)

grid = np.c_[xx.ravel(), yy.ravel()]


# =========================
# 4. Figura
# =========================

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)

for ax, gamma, titolo in zip(axes, gammas, titoli):

    modello = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=100,
            gamma=gamma
        )
    )

    modello.fit(X, y)

    # Valori della funzione decisionale
    decision = modello.decision_function(grid).reshape(xx.shape)

    # Margini e frontiera:
    # decision = 0  -> frontiera decisionale
    # decision = ±1 -> margini
    ax.contour(
        xx,
        yy,
        decision,
        levels=[-1, 0, 1],
        colors=["gray", "black", "gray"],
        linestyles=["--", "-", "--"],
        linewidths=[1.0, 1.7, 1.0]
    )

    # Punti classe +1
    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="#006d6f",
        s=35,
        edgecolors="white",
        linewidths=0.5,
        label="Classe +1"
    )

    # Punti classe -1
    ax.scatter(
        X[y == -1, 0],
        X[y == -1, 1],
        c="#9b0000",
        s=35,
        edgecolors="white",
        linewidths=0.5,
        label="Classe -1"
    )

    ax.set_title(titolo, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

# Legenda unica
handles, labels = axes[0].get_legend_handles_labels()

# Aggiungo linee fittizie per legenda frontiera/margini
line_frontiera, = axes[0].plot([], [], color="black", linewidth=1.7, label="Frontiera di decisione")
line_margine, = axes[0].plot([], [], color="gray", linestyle="--", linewidth=1.0, label="Margini (±1)")

handles = handles + [line_frontiera, line_margine]
labels = labels + ["Frontiera di decisione", "Margini (±1)"]

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04)
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")


plt.show()

print(f"Figura PNG salvata in: {OUTPUT_PNG}")
