# AGGIRONATO AL 15/05/26

# Questo script serve a scegliere gli iperparametri principali dei classificatori usati nello script principale
# Per ogni combinazione di parametri viene eseguita una 5x2CV manuale,
# cioè 5 ripetizioni di split 50/50, ciascuna valutata in entrambe le direzioni,
# train A / test B e train B / test A. La metrica usata è la balanced accuracy, più adatta dell'accuracy standard perché il dataset è sbilanciato tra NORMALE e MUTAZIONE

# Librerie per gestione dati, calcoli numerici e grafici
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Classificatori da confrontare
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Pipeline, normalizzazione, split e metrica di valutazione
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
# Libreria per gestire percorsi e cartelle di output
import os

# Ricava il nome del file .py
nome_script = os.path.splitext(os.path.basename(__file__))[0]
# Crea il percorso della cartella di output accanto allo script,
# usando come nome cartella lo stesso nome del file Python
cartella_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), nome_script)
# Crea la cartella se non esiste già, exist_ok=True evita errore se la cartella è già presente
os.makedirs(cartella_output, exist_ok=True)

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================

# Carica il dataset finale contenente i pazienti,
# i marcatori biochimici e la caratterizzazione allelica
df = pd.read_csv("dataset_finale.csv")

# Lista delle quattro feature biochimiche usate dai modelli
# Sono gli stessi marcatori utilizzati nello script principale
feature = [
    "ALFA1 %\n (2,9 – 4,9 %)",
    "ALFA1#\n(0,20-0,35 g/dl)",
    "A1AT\n(90-200 mg/dl)",
    "PCR\n(0,0-5,0 mg/l)",
]

# Alcuni valori possono contenere la virgola decimale, quindi trasformiamo in stringa e poi convertiamo sostituendo la virgola con il punto
df_num = df.copy()
for col in feature:
    df_num[col] = pd.to_numeric(
        df_num[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
# Matrice delle feature usata dai classificatori
X = df_num[feature].values
# il target caratterizzazione allelica
y = df["CARATTERIZZAZIONE\nALLELICA"].astype(str)
# il target binario usato per la classificazione
# MM viene considerato NORMALE, tutti gli altri genotipi MUTAZIONE
yAcc = pd.Series(np.where(y == "MM", "NORMALE", "MUTAZIONE"))
# Array degli indici dei pazienti
idx = np.arange(len(df))

N_REP = 5


# ==========================================
# 2. FUNZIONE 5x2cv
# ==========================================

# Questa funzione applica lo schema 5x2CV a un insieme di pipeline
# È una versione più compatta rispetto allo script principale, qui non servono matrici di confusione, test statistici o analisi per paziente,
# ma solo la balanced accuracy media per ogni combinazione di iperparametri, quindi si è voluto semplificare lo stesso metodo



def grid_search_5x2cv(pipelines_dict, X, y, n_rep=5):
    # pipelines_dict è un dizionario del tipo:
    # nome_modello_o_parametro -> Pipeline sklearn
    # ad esempio, "LR_C1" -> Pipeline(MinMaxScaler + LogisticRegression(C=1))
    # la funzione restituisce un dizionario nome_modello_o_parametro -> balanced accuracy media sui 10 round
    risultati = {}
    # impostiamo il seed globale per rendere riproducibile gli split generati da train_test_split
    np.random.seed(42)

    for nome, pipe in pipelines_dict.items():   # itera su ogni pipeline
        np.random.seed(42)        # Reset del seed prima di ogni pipeline
        scores = []        # Lista delle balanced accuracy ottenute nei 10 round
        for i in range(n_rep): # Divide il dataset in due metà A e B mantenendo quando possibile, la proporzione tra NORMALE e MUTAZIONE
            try:
                idxA, idxB, yA, yB = train_test_split(
                    idx, y, test_size=0.5,
                    random_state=None, stratify=y
                )
            except ValueError:       # come nel principale in caso di stratificazione fallita fa un 50 e 50
                idxA, idxB, yA, yB = train_test_split(
                    idx, y, test_size=0.5,
                    random_state=None, stratify=None
                )
            # round A, addestra la pipeline sul fold A e valuta sul fold B
            pipe.fit(X[idxA], yA)
            scores.append(balanced_accuracy_score(yB, pipe.predict(X[idxB])))
            # round B, addestra la pipeline sul fold B e valuta sul fold A
            pipe.fit(X[idxB], yB)
            scores.append(balanced_accuracy_score(yA, pipe.predict(X[idxA])))
        # alla fine dei 10 round salva la media delle balanced accuracy
        risultati[nome] = np.mean(scores)
    # restituisce una tabella compatta dei risultati con una media finale per ogni pipeline testata
    return risultati


# ==========================================
# 3. RANDOM FOREST — tuning n_estimators
# ==========================================

# In questa sezione si valuta l'effetto del numero di alberi della Random Forest sulle prestazioni del modello
# n_estimators indica quanti alberi decisionali compongono la foresta in generale, più alberi rendono la stima più stabile, ma aumentano (inutilmente) anche il tempo di calcolo
# seguendo un approccio prudente su un dataset piccolo (n=104), viene ottimizzato solo questo iperparametro principale, mentre gli altri parametri del Random Forest vengono mantenuti fissi

# L'unico iperparametro rilevante secondo Raschka è n_estimators, più alberi = più stabile
# Gli altri parametri (max_depth, min_samples ecc.) hanno effetti trascurabili su dataset piccoli come il nostro

# Lista dei valori di n_estimators da testare
# Ogni valore corrisponde a una Random Forest con un diverso numero di alberi
n_estimators_values = [10, 50, 100, 200, 500]

# Crea un dizionario di pipeline, una per ogni valore di n_estimators, la chiave del dizionario è il valore del parametro convertito in stringa,
# ad esempio 10, 50 o 100
pipes_rf = {
    str(n): Pipeline([
        # MinMaxScaler normalizza le feature tra 0 e 1
        # Anche se Random Forest non richiede strettamente scaling, lo manteniamo nella pipeline per coerenza con gli altri modelli
        ("scaler", MinMaxScaler()),
        # Classificatore Random Forest.
        ("clf", RandomForestClassifier(
            # Numero di alberi della foresta, variabile del tuning.
            n_estimators=n,
            # Pesa le classi in modo inversamente proporzionale alla loro frequenza e ricalcola i pesi sui campioni bootstrap di ciascun albero,
            # è utile perché il dataset è sbilanciato
            class_weight="balanced_subsample",
            # Rende riproducibile la componente casuale della Random Forest
            random_state=42
        ))
    ])
    for n in n_estimators_values
}

print("=" * 50)
print("RANDOM FOREST — tuning n_estimators")
print("=" * 50)
# dopo aver applicato la 5x2 CV a tutte le pipeline di RF otteniamo un dizionario
# n_estimators -> balanced accuracy media sui 10 round
res_rf = grid_search_5x2cv(pipes_rf, X, yAcc)
# Stampa la balanced accuracy media ottenuta da ogni valore di n_estimators
for n, score in res_rf.items():
    print(f"  n_estimators={n:<6}  bal_acc={score:.6f}")
# Seleziona il valore di n_estimators con balanced accuracy media più alta e lo stampiamo
best_rf = max(res_rf, key=res_rf.get)
print(f"\n→ Miglior n_estimators: {best_rf}  (bal_acc={res_rf[best_rf]:.6f})")
print(f"  Default n_estimators=200 già usato: bal_acc={res_rf.get('200', 'N/A')}")

# ==========================================
# 4. SVC RBF — tuning C e gamma (heatmap)
# ==========================================

# In questa sezione si ottimizzano i due principali iperparametri di SVC con kernel RBF
# C regola il compromesso tra margine ampio ed errori di classificazione:
# - valori più alti penalizzano di più gli errori sul training set
# - mentre valori più bassi permettono un margine più morbido
# gamma controlla l'ampiezza dell'influenza dei singoli campioni nel kernel RBF:
# - valori alti rendono il modello più locale e potenzialmente più complesso
# - valori bassi rendono il confine decisionale più regolare
# Poiché C e gamma interagiscono tra loro, vengono valutate tutte le combinazioni della griglia tramite 5x2CV

C_values = [0.01, 0.1, 1, 10, 100, 1000]
# Oltre ai valori numerici, si includono anche "scale" e "auto", cioè le due strategie automatiche offerte da scikit-learn
gamma_values = [0.001, 0.01, 0.1, 1, 10, "scale", "auto"]
# Matrice che conterrà la balanced accuracy media
risultati_svc = np.zeros((len(C_values), len(gamma_values)))

# Cicla su tutte le combinazioni della griglia ci e gi sono gli indici usati per salvare il risultato nella matrice
for ci, C in enumerate(C_values):
    for gi, gamma in enumerate(gamma_values):
        # la funzione grid_search_5x2cv resetta già il seed, non c'è bisogno di resettare il seme
        # Crea una pipeline SVC RBF per la specifica combinazione di C e gamma.
        pipes = {
            "svc": Pipeline([
                # Normalizza le feature tra 0 e 1
                # Per SVC è importante perché il modello è sensibile alla scala delle variabili
                ("scaler", MinMaxScaler()),
                # Classificatore SVC con kernel RBF
                ("clf", SVC(
                    kernel="rbf",
                    # Valore di C testato in questa iterazione
                    C=C,
                    # Valore di gamma testato in questa iterazione
                    gamma=gamma,
                    # Bilancia il peso delle classi per compensare lo sbilanciamento
                    class_weight="balanced",
                    # Rende riproducibili eventuali componenti interne legate alla casualità
                    random_state=42
                ))
            ])
        }
        # Valuta la pipeline con lo schema 5x2CV
        res = grid_search_5x2cv(pipes, X, yAcc)
        # Salva la balanced accuracy media nella cella corrispondente alla combinazione C-gamma corrente
        risultati_svc[ci, gi] = res["svc"]

# Trova la posizione della combinazione con balanced accuracy più alta e convertiamo in stringa per il grafico
best_svc = np.unravel_index(np.argmax(risultati_svc), risultati_svc.shape)
gamma_labels = [str(g) for g in gamma_values]

print("\n" + "=" * 50)
print("SVC RBF — tuning C × gamma (balanced accuracy)")
print("=" * 50)

print(f"\n{'C \\ gamma':<10}", end="")
for gl in gamma_labels:
    print(f"{gl:>10}", end="")
print()
print("-" * (10 + 10 * len(gamma_values)))

for ci, C in enumerate(C_values):
    print(f"{C:<10}", end="")
    for gi in range(len(gamma_values)):
        val = risultati_svc[ci, gi]
        marker = "*" if (ci, gi) == best_svc else " "
        print(f"{val:>9.6f}{marker}", end="")
    print()

print(f"\n→ Miglior combinazione: C={C_values[best_svc[0]]}, gamma={gamma_values[best_svc[1]]}")
print(f"  Balanced Acc: {risultati_svc[best_svc]:.6f}")
print(f"  Default (C=1, gamma=scale): {risultati_svc[C_values.index(1), gamma_values.index('scale')]:.6f}")
print(f"  Delta (ottimizzato - default): "
      f"{risultati_svc[best_svc] - risultati_svc[C_values.index(1), gamma_values.index('scale')]:+.6f}")

# Nota metodologica:
# questo tuning è esplorativo, perché viene svolto sullo stesso dataset poi usato per la valutazione finale. Per questo i risultati vanno interpretati
# come scelta ragionata degli iperparametri, non come stima indipendente della performance generalizzabile

# ==========================================
# 5. LOGISTIC REGRESSION e LINEAR SVC — tuning C
# ==========================================

# In questa sezione viene ottimizzato il parametro C per due classificatori lineari:
# - Logistic Regression
# - LinearSVC
# In entrambi i casi C controlla l'intensità della regolarizzazione:
# - valori piccoli di C impongono una regolarizzazione più forte,
# - valori grandi di C lasciano il modello più libero di adattarsi ai dati.
# Poiché il dataset è piccolo e sbilanciato, il confronto viene fatto usando sempre la balanced accuracy media ottenuta tramite 5x2CV

# Valori di C da testare per entrambi i modelli lineari
C_values_lin = [0.01, 0.1, 1, 10, 100, 1000]
# Dizionario che conterrà tutte le pipeline da valutare
pipes_lin = {}
# Per ogni valore di C si costruiscono due pipeline:
# una per Logistic Regression e una per LinearSVC.
for C in C_values_lin:
    # Pipeline per Logistic Regression con il valore corrente di C
    pipes_lin[f"LR_C{C}"] = Pipeline([
        # Normalizza le feature tra 0 e 1
        # Per i modelli lineari è importante perché i coefficienti dipendono dalla scala delle variabili.
        ("scaler", MinMaxScaler()),
        # class_weight="balanced" per compensare lo sbilanciamento tra NORMALE e MUTAZIONE.
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    # Pipeline per LinearSVC con il valore corrente di C.
    pipes_lin[f"LSVC_C{C}"] = Pipeline([
        # Normalizza le feature tra 0 e 1.
        ("scaler", MinMaxScaler()),
        ("clf", LinearSVC(
            C=C,
            class_weight="balanced",
            random_state=42,
            # Aumenta il numero massimo di iterazioni per ridurre il rischio di mancata convergenza
            max_iter=10000
        ))
    ])
# Applica la 5x2CV a tutte le pipeline lineari, il risultato contiene sia le Logistic Regression sia le LinearSVC
res_lin = grid_search_5x2cv(pipes_lin, X, yAcc)

# Separa i risultati della Logistic Regression da quelli della LinearSVC
# Le chiavi vengono ripulite lasciando solo il valore di C
res_lr = {
    k.replace("LR_C", ""): v
    for k, v in res_lin.items()
    if k.startswith("LR")
}

res_lsvc = {
    k.replace("LSVC_C", ""): v
    for k, v in res_lin.items()
    if k.startswith("LSVC")
}
# stampa migliore risultato per LR
print("\n" + "=" * 50)
print("LOGISTIC REGRESSION — tuning C")
print("=" * 50)
# Mostra la balanced accuracy media per ogni valore di C
for C, score in res_lr.items():
    print(f"  C={C:<8}  bal_acc={score:.6f}")
# Seleziona il valore di C con balanced accuracy media più alta.
best_lr = max(res_lr, key=res_lr.get)
print(f"\n→ Miglior C: {best_lr}  (bal_acc={res_lr[best_lr]:.6f})")
# stessa cosa per LSVC
print("\n" + "=" * 50)
print("LINEAR SVC — tuning C")
print("=" * 50)
for C, score in res_lsvc.items():
    print(f"  C={C:<8}  bal_acc={score:.6f}")
best_lsvc = max(res_lsvc, key=res_lsvc.get)
print(f"\n→ Miglior C: {best_lsvc}  (bal_acc={res_lsvc[best_lsvc]:.6f})")

# ==========================================
# 6. RIEPILOGO FINALE
# ==========================================

print("\n" + "=" * 50)
print("RIEPILOGO")
print("=" * 50)
print(f"  Random Forest:       n_estimators={best_rf}")
print(f"  SVC RBF:             C={C_values[best_svc[0]]}, gamma={gamma_values[best_svc[1]]}")
print(f"  Logistic Regression: C={best_lr}")
print(f"  Linear SVC:          C={best_lsvc}")

# ==========================================
# 7. GRAFICI
# ==========================================

# In questa sezione vengono prodotti tre grafici riassuntivi:
# - tuning di n_estimators per Random Forest
# - tuning di C per Logistic Regression e LinearSVC
# - tuning combinato C-gamma per SVC RBF
# I grafici servono a visualizzare rapidamente quale configurazione ottiene la balanced accuracy più alta e quanto il risultato cambia
# al variare degli iperparametri

# Ogni pannello mostra il tuning di una famiglia di modelli

# creazione grafico per random forest


fig, ax = plt.subplots(figsize=(6, 5))

barre_rf = ax.bar(
    [str(n) for n in n_estimators_values],
    [res_rf[str(n)] for n in n_estimators_values],
    color="#4C72B0",
    edgecolor="black"
)

ax.axhline(
    0.5,
    color="red",
    linestyle="--",
    label="Dummy baseline"
)

# Numeri sopra le barre
for barra in barre_rf:
    altezza = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        altezza + 0.006,
        f"{altezza:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        rotation=90
    )

ax.set_title(
    "Random Forest\nn_estimators",
    fontweight="bold"
)

ax.set_xlabel("n_estimators")
ax.set_ylabel("Balanced Accuracy")
ax.set_ylim(0.4, 1.05)
ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(cartella_output, "tuning_random_forest_n_estimators.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# creazione grafico per logistic regression e linearsvc

fig, ax = plt.subplots(figsize=(6, 5))

x = np.arange(len(C_values_lin))
w = 0.35

barre_lr = ax.bar(
    x - w / 2,
    [res_lr[str(c)] for c in C_values_lin],
    w,
    label="Logistic Regression",
    color="#4C72B0",
    edgecolor="black"
)

barre_lsvc = ax.bar(
    x + w / 2,
    [res_lsvc[str(c)] for c in C_values_lin],
    w,
    label="Linear SVC",
    color="#DD8452",
    edgecolor="black"
)

ax.axhline(
    0.5,
    color="red",
    linestyle="--",
    label="Dummy baseline"
)

# Numeri sopra le barre Logistic Regression
for barra in barre_lr:
    altezza = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        altezza + 0.006,
        f"{altezza:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        rotation=90
    )

# Numeri sopra le barre Linear SVC
for barra in barre_lsvc:
    altezza = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        altezza + 0.006,
        f"{altezza:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        rotation=90
    )

ax.set_xticks(x)
ax.set_xticklabels([f"C={c}" for c in C_values_lin])

ax.set_title(
    "LR vs LinearSVC\ntuning C",
    fontweight="bold"
)

ax.set_xlabel("C")
ax.set_ylim(0.4, 1.05)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0
)

plt.tight_layout()

plt.savefig(
    os.path.join(cartella_output, "tuning_lr_linearsvc_C.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# creazione grafico per svc rbf

fig, ax = plt.subplots(figsize=(6, 5))

gamma_plot_idx = [i for i, g in enumerate(gamma_values)]
gamma_plot_labels = [str(gamma_values[i]) for i in gamma_plot_idx]

ris_plot = risultati_svc[:, gamma_plot_idx]

im = ax.imshow(
    ris_plot,
    cmap="RdYlGn",
    vmin=0.5,
    vmax=1.0,
    aspect="auto"
)

plt.colorbar(
    im,
    ax=ax,
    label="Balanced Accuracy"
)

ax.set_xticks(range(len(gamma_plot_labels)))
ax.set_xticklabels([f"γ={g}" for g in gamma_plot_labels])

ax.set_yticks(range(len(C_values)))
ax.set_yticklabels([f"C={c}" for c in C_values])

ax.set_title(
    "SVC RBF\nC × gamma",
    fontweight="bold"
)

# Scrive il valore numerico della balanced accuracy dentro ogni cella
for ci in range(len(C_values)):
    for gi in range(len(gamma_plot_labels)):
        val = ris_plot[ci, gi]
        ax.text(
            gi,
            ci,
            f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if val < 0.7 else "black",
            fontweight="bold"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(cartella_output, "tuning_svc_rbf_C_gamma.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 8. SALVATAGGIO STATISTICHE IN UN UNICO FOGLIO EXCEL
# ==========================================

# In questa sezione i risultati numerici del tuning vengono raccolti in un unico file Excel
# Il file contiene:
# - un riepilogo dei migliori iperparametri per ogni modello
# - il dettaglio del tuning Random Forest
# - la matrice C-gamma dello SVC RBF
# - il dettaglio del tuning C per Logistic Regression e LinearSVC
# Salvare questi risultati permette di documentare in modo trasparente da quali valori derivano gli iperparametri scelti nello script principale

# Crea una tabella sintetica con il miglior iperparametro trovato per ciascun classificatore e la relativa balanced accuracy
df_best = pd.DataFrame([
    {"Sezione": "RIEPILOGO MIGLIORI", "Modello": "Random Forest", "Parametro": f"n_estimators={best_rf}",
     "Balanced_Accuracy": res_rf[best_rf]},
    {"Sezione": "RIEPILOGO MIGLIORI", "Modello": "SVC RBF",
     "Parametro": f"C={C_values[best_svc[0]]}, gamma={gamma_values[best_svc[1]]}",
     "Balanced_Accuracy": risultati_svc[best_svc]},
    {"Sezione": "RIEPILOGO MIGLIORI", "Modello": "Logistic Regression", "Parametro": f"C={best_lr}",
     "Balanced_Accuracy": res_lr[best_lr]},
    {"Sezione": "RIEPILOGO MIGLIORI", "Modello": "Linear SVC", "Parametro": f"C={best_lsvc}",
     "Balanced_Accuracy": res_lsvc[best_lsvc]}
])
# Trasforma il dizionario dei risultati Random Forest in una tabella ogni riga contiene un valore di n_estimators e la balanced accuracy media
df_rf_det = pd.DataFrame(list(res_rf.items()), columns=['n_estimators', 'Balanced_Accuracy'])
df_rf_det.insert(0, "Sezione",  " RF")
# Converte la matrice dei risultati C-gamma in un DataFrame
df_svc_det = pd.DataFrame(risultati_svc, index=[f"C={c}" for c in C_values], columns=[f"g={g}" for g in gamma_labels])
# Crea una tabella con i risultati ottenuti da Logistic Regression e LinearSVC per ogni valore di C testato
df_lin_det = pd.DataFrame({
    'C': [f"C={c}" for c in C_values_lin],
    'Logistic_Regression': [res_lr[str(c)] for c in C_values_lin],
    'Linear_SVC': [res_lsvc[str(c)] for c in C_values_lin]
})
df_lin_det.insert(0, "Sezione", " LINEARI")

# Il file viene salvato nella cartella output dello script
percorso_excel = os.path.join(cartella_output, f"{nome_script}_risultati.xlsx")

with pd.ExcelWriter(percorso_excel, engine='openpyxl') as writer:
    # Scrive il riepilogo dei migliori risultati all'inizio del foglio
    df_best.to_excel(writer, sheet_name='Risultati_Tuning', index=False, startrow=0)

    df_rf_det.to_excel(writer, sheet_name='Risultati_Tuning', index=False, startrow=len(df_best) + 2)

    pd.DataFrame(["SVC RBF (C vs Gamma)"]).to_excel(writer, sheet_name='Risultati_Tuning', index=False, header=False, startrow=len(df_best) + len(df_rf_det) + 4)
    df_svc_det.to_excel(writer, sheet_name='Risultati_Tuning', index=True, startrow=len(df_best) + len(df_rf_det) + 5)

    df_lin_det.to_excel(writer, sheet_name='Risultati_Tuning', index=False, startrow=len(df_best) + len(df_rf_det) + len(df_svc_det) + 8)

print(f" Tutte le statistiche sono state salvate in Excel: {percorso_excel}")
