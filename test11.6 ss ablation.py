# AGGIORNATO AL 15/05/26

# Analisi di ablation, valuta l'impatto della rimozione di ogni feature su 4 modelli
# Lo script misura quanto cambiano le prestazioni quando si elimina un marker alla volta

# Lo studio di ablation consiste nel rimuovere una feature alla volta e misurare
# quanto peggiorano le prestazioni dei classificatori, più cala l'accuratezza,
# più quella feature era importante.

# Librerie per manipolazione dati e calcoli numerici
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, math
# Test statistici
from scipy import stats
from scipy.stats import f as f_dist
# Strumenti scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef,
)
# Ricava il nome del file .py
nome_script = os.path.splitext(os.path.basename(__file__))[0]
# Crea il percorso della cartella di output accanto allo script,
# usando come nome cartella lo stesso nome del file Python
cartella_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), nome_script)
# Crea la cartella se non esiste già, exist_ok=True evita errore se la cartella è già presente
os.makedirs(cartella_output, exist_ok=True)

# ==========================================
# 1. FUNZIONI DI COMODO
# ==========================================

# Calcola tutte le metriche di classificazione in un unico dizionario
# chiamata per ogni round (A e B) di ogni ripetizione 5x2CV
def eval_metrics(y_true, y_pred):
    return {
        "acc":         accuracy_score(y_true, y_pred),         # Accuracy standard: è la percentuale totale di predizioni corrette,può essere fuorviante con classi sbilanciate
        "bal_acc":     balanced_accuracy_score(y_true, y_pred), # Balanced accuracy: è la media del recall delle due classi, è più corretta quando NORMALE e MUTAZIONE non hanno la stessa numerosità
        "f1_macro":    f1_score(y_true, y_pred, average="macro"), # F1 macro: media semplice dell'F1 delle classi, dà lo stesso peso a NORMALE e MUTAZIONE
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),         # F1 weighted: media pesata dell'F1 in base al numero di campioni per classe
        "mcc":         matthews_corrcoef(y_true, y_pred), #        # Matthews Correlation Coefficient: metrica robusta per classificazione binaria sbilanciata
    }

# Implementa i due test statistici descritti in Dietterich (1998):
# Test t di Dietterich (5x2CV t-test):
#      Confronta il classificatore vs baseline usando le differenze dei 5 round A.
#      Usa la varianza stimata su tutte le 5 coppie (A,B) come denominatore, è uguale a quello presente nello script principale
# Test F di Alpaydin (5x2CV F-test):
#      # Variante generalmente più potente del t-test, usa tutte le 10 differenze (5 round A + 5 round B)
#      Distribuito come F(10, 5)

def dietterich_alpaydin(diff_A, diff_B):
    # Varianza per ciascuna delle 5 ripetizioni media di diff_A e diff_B al quadrato
    s2 = np.array([(dA - .5*(dA+dB))**2 + (dB - .5*(dA+dB))**2
                   for dA, dB in zip(diff_A, diff_B)])
    # Denominatore del t-test: radice della varianza media pesata (formula Dietterich)
    denom = math.sqrt((1/5) * s2.sum())
    # Statistica t usa solo la prima differenza al numeratore (convenzione Dietterich)
    t     = diff_A[0] / denom if denom > 0 else float("inf")
    p_t   = 2 * (1 - stats.t.cdf(abs(t), df=5))
    num   = sum(dA**2 + dB**2 for dA, dB in zip(diff_A, diff_B))
    # Statistica F di Alpaydin usa la somma di tutti i quadrati delle differenze
    f_s   = num / (2*s2.sum()) if s2.sum() > 0 else float("inf")
    # 10 fold totali, 5 ripetizioni
    p_f   = 1 - f_dist.cdf(f_s, dfn=10, dfd=5)
    return t, p_t, f_s, p_f

# ==========================================
# 2. PREPARAZIONE DATI
# ==========================================

# Carica il dataset con i campioni e accorciamo i nomi delle features
df = pd.read_csv("dataset_finale.csv")
feature = ["ALFA1 %\n (2,9 – 4,9 %)", "ALFA1#\n(0,20-0,35 g/dl)", "A1AT\n(90-200 mg/dl)", "PCR\n(0,0-5,0 mg/l)"]
feat_short = ["ALFA1%", "ALFA1#", "A1AT", "PCR"]
# zip accoppia i due elenchi elemento per elemento, dict li trasforma in un dizionario:
f2s = dict(zip(feature, feat_short))

# breve ciclo for per convertire le virgole in punti all'interno dei dataset perchè pandas potrebbe confondersi
df_num = df.copy()
for col in feature:
    df_num[col] = pd.to_numeric(df_num[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")

# Legge la colonna del genotipo dal CSV e la converte in stringa
y_raw = df["CARATTERIZZAZIONE\nALLELICA"].astype(str)
# Crea il target binario:
# - MM diventa NORMALE
# - tutti gli altri genotipi diventano MUTAZIONE.
yAcc  = pd.Series(np.where(y_raw == "MM", "NORMALE", "MUTAZIONE"))

# Si costruisce una lista di esperimenti, ognuno con un nome e un sottoinsieme di feature, dove poniamo tutte le liste e dizionari chiariti precedentemente
# Il + aggiunge un esperimento per ogni feature rimossa, la riga [ff for ff in feature if ff != f] significa proprio "prendi tutte le feature tranne quella corrente f" a seguito per ogni f in lista feature
ablations = [("ALL", feature)] + [(f"NO_{f2s[f]}", [ff for ff in feature if ff != f]) for f in feature]
# ablations è una lista di 5 esperimenti. Per ognuno, il modello viene riaddestrato da zero usando solo quelle feature
# Confrontando le performance, vediamo quanto pesa ogni marcatore, se togliendo A1AT le performance crollano, vuol dire che A1AT è molto importante

# ==========================================
# 3. DEFINIZIONE DEI MODELLI
# ==========================================
# Ogni Pipeline combina MinMaxScaler (normalizza i dati) + classificatore
# I parametri sono allineati a quelli ottimizzati nel file test11.6, a loro volta provenienti dal tuning
pipe_rf = Pipeline([("scaler", MinMaxScaler()), ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", random_state=42))])
pipe_svc = Pipeline([("scaler", MinMaxScaler()), ("clf", SVC(kernel="rbf", C=1000, gamma="auto", class_weight="balanced", random_state=42))])
pipe_lr = Pipeline([("scaler", MinMaxScaler()), ("clf", LogisticRegression(class_weight="balanced", C=1000, max_iter=1000, random_state=42))])
pipe_lsvc = Pipeline([("scaler", MinMaxScaler()), ("clf", LinearSVC(C=1000, class_weight="balanced", random_state=42, max_iter=10000))])
dummy = DummyClassifier(strategy="most_frequent")

# ==========================================
# 4. CICLO ABLATION
# ==========================================

all_results  = []   # accumula tutti i risultati riga per riga (per il CSV), alla fine conterrà le metriche di ogni modello, per ogni ablation, per ogni fold
summary_rows = []   # accumula una riga riassuntiva per ogni coppia modello/ablation, ogni riga conterrà medie delle metriche e risultati dei test statistici

for tag, feat_list in ablations:
    # tag è l'etichetta dell'esperimento (es. "ALL", "NO_PCR")
    # feat_list è la lista delle feature da usare in questo esperimento
    print(f"\n{'=' * 50}\nABLATION: {tag}  ->  {[f2s[f] for f in feat_list]}\n{'=' * 50}")
    # Estrae la matrice X con sole le feature dell'esperimento in corso
    X_abl = df_num[feat_list].values

    # Lista dei 4 modelli da confrontare
    model_configs = [
        ("RF", pipe_rf),
        ("SVC", pipe_svc),
        ("LR", pipe_lr),
        ("LSVC", pipe_lsvc)
    ]

    # per ogni name nella lista model_configs qua sopra
    for name, pipe in model_configs:
        # RESET DEL SEED garantisce che ogni modello veda gli stessi identici split per avere un confronto equo,
        # qua è necessario perchè iteriamo su tutti i classificatori invece nel principale non c'è bisogno
        np.random.seed(42)

        rows_model = []  # risultati di questo modello in questo esperimento
        diff_A, diff_B = [], []  # differenze (modello - dummy) per il test statistico

        for i in range(5):
            # 5x2CV: 5 ripetizioni, ognuna con uno split 50/50 casuale
            # stratify=yAcc mantiene le proporzioni di classe nei due fold
            try:
                idxA, idxB, yA, yB = train_test_split(
                    np.arange(len(df)), yAcc, test_size=0.5,
                    random_state=None, stratify=yAcc)
            # Se la stratificazione fallisce, ad esempio in dataset molto piccoli,
            # usa uno split non stratificato come fallback
            # Crea comunque due fold 50/50, ma senza preservare le proporzioni di classe
            except ValueError:
                idxA, idxB, yA, yB = train_test_split(
                    np.arange(len(df)), yAcc, test_size=0.5, random_state=None)
            # matrici X per i due fold
            XA, XB = X_abl[idxA], X_abl[idxB]

            # Round A, addestramento su fold A, predizione su fold B
            p_B = pipe.fit(XA, yA).predict(XB)  # predizione del modello
            p_dum_B = dummy.fit(XA, yA).predict(XB)  # predizione del dummy (baseline)
            m_A = eval_metrics(yB, p_B)  # metriche del modello, richiamando la funzione sopra
            m_dum_A = eval_metrics(yB, p_dum_B)  # metriche del dummy
            diff_A.append(m_A["bal_acc"] - m_dum_A["bal_acc"])  # differenza per test stat
            rows_model.append({"rep": i + 1, "round": "A", "model": name, **m_A}) # salva i risultati di ogni round in una lista di dizionari

            # Round B, addestramento su fold B, predizione su fold A
            p_A = pipe.fit(XB, yB).predict(XA)
            p_dum_A = dummy.fit(XB, yB).predict(XA)
            m_B = eval_metrics(yA, p_A)
            m_dum_B = eval_metrics(yA, p_dum_A)
            diff_B.append(m_B["bal_acc"] - m_dum_B["bal_acc"])
            rows_model.append({"rep": i + 1, "round": "B", "model": name, **m_B})

        # Calcolo statistiche Dietterich e Alpaydin richiamando la funzione definita prima
        t, pt, f, pf = dietterich_alpaydin(diff_A, diff_B)

        # Salva tutti i risultati (10 righe, 5 rep × 2 round) con il tag ablation
        res_df = pd.DataFrame(rows_model) # Converte la lista di dizionari in una tabella
        res_df["ablation"] = tag # Aggiunge una colonna con il tag dell'esperimento corrente (es. "ALL", "NO_PCR", ecc.) a tutte le 10 righe
        all_results.append(res_df) # Aggiunge questo DataFrame alla lista all_results, conterrà 200 righe totali

        # riga di riepilogo per questo modello/ablation
        summary_rows.append({
            "ablation": tag,
            "model": name,
            "bal_acc_mean": np.mean([r["bal_acc"] for r in rows_model]),
            "f1_macro_mean": np.mean([r["f1_macro"] for r in rows_model]),
            "mcc_mean": np.mean([r["mcc"] for r in rows_model]),
            "t_dietterich": t,
            "p_dietterich": pt,
            "f_alpaydin": f,
            "p_alpaydin": pf
        })

# ==========================================
# 5. SALVATAGGIO E STAMPA
# ==========================================

# Unisce tutti i DataFrame dei singoli esperimenti in uno solo
df_all = pd.concat(all_results, ignore_index=True)
df_summary = pd.DataFrame(summary_rows)

# Salva il riepilogo con valori arrotondati per leggibilità
df_summary_fmt = df_summary.copy()
for col in ["bal_acc_mean", "f1_macro_mean", "mcc_mean"]: df_summary_fmt[col] = df_summary_fmt[col].round(6)
for col in ["t_dietterich", "f_alpaydin"]: df_summary_fmt[col] = df_summary_fmt[col].round(3)

# stampa dei risultati
print("\n=== TABELLA RIASSUNTIVA FINALE ===")
print(df_summary[["ablation", "model", "bal_acc_mean", "p_dietterich", "p_alpaydin"]].to_string(index=False))


# Definisce una funzione dedicata al salvataggio dei risultati in Excel ricevendo:
# - df_all_rows risultati dettagliati di tutti i round
# - df_sum tabella riassuntiva finale
def salva_risultati_excel_ablation(df_all_rows, df_sum):
    # Prende il nome del file script corrente, crea una cartella di output e salva su 2 fogli excel
    percorso_file = os.path.join(cartella_output, f"{nome_script}_risultati.xlsx")

    with pd.ExcelWriter(percorso_file, engine='openpyxl') as writer:
        df_sum.to_excel(writer, sheet_name='Summary', index=False) # Salva il riepilogo nel foglio Summary
        df_all_rows.to_excel(writer, sheet_name='All_Replications', index=False) # Salva tutti i risultati dettagliati nel foglio All_Replications


# ==========================================
# 6. GENERAZIONE GRAFICO ABLATION
# ==========================================

# Crea la figura con dimensioni adatte alla stampa/presentazione
fig, ax = plt.subplots(figsize=(12, 6))
# Estrae i tag univoci degli esperimenti e i nomi dei modelli dalla tabella riassuntiva
ablations_tags = df_summary["ablation"].unique()
models = df_summary["model"].unique()
# Posizioni sull'asse X (una per ogni condizione di ablation)
x = np.arange(len(ablations_tags))
width = 0.2
# un colore diverso per modello
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
# Per ogni modello, raccoglie i valori medi di balanced accuracy per ogni condizione e disegna il gruppo di barre
for i, model in enumerate(models):
    scores = []
    for tag in ablations_tags:
        val = df_summary[(df_summary["model"] == model) & (df_summary["ablation"] == tag)]["bal_acc_mean"].values
        scores.append(val[0] if len(val) > 0 else 0)

    offset = (i - len(models) / 2 + 0.5) * width
    ax.bar(x + offset, scores, width, label=model, color=colors[i], edgecolor='black')

ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Baseline Dummy (0.5)')
# Etichette e titolo
ax.set_ylabel('Balanced Accuracy Mean', fontweight='bold', fontsize=12)
ax.set_title('Ablation Study: Impatto della rimozione dei singoli marcatori biochimici', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(ablations_tags, fontweight='bold', fontsize=11)
ax.set_ylim(0.4, 1.05)  # Lascia spazio per l'etichetta dei numeri sopra la barra

ax.legend(title="Classificatore", bbox_to_anchor=(1.01, 1), loc='upper left')

for container in ax.containers:
    ax.bar_label(container, fmt='%.6f', padding=3, fontsize=9, rotation=90)

plt.tight_layout()

# Salva il grafico nella cartella di output creata all'inizio dello script
percorso_grafico = os.path.join(cartella_output, "grafico_ablation.png")
plt.savefig(percorso_grafico, dpi=300, bbox_inches="tight")
plt.show()

salva_risultati_excel_ablation(df_all, df_summary)

