# AGGIRONATO AL 15/05/26

#Questo script implementa:
#- 5x2cv per confronto Random Forest/SVC vs Dummy Classifier
#- Test statistici di Dietterich e Alpaydin
#- Valutazione su dataset sbilanciato (n=104)

#Dataset: dataset_finale.csv
#Feature: ALFA1%, ALFA1#, A1AT, PCR (marcatori biochimici)
#Target originale: CARATTERIZZAZIONE ALLELICA (MM, Mmalton, S, Z, ...)
#Target usato qui: BINARIO (NORMALE=MM, MUTAZIONE=resto)


# Librerie base per manipolazione dati (pandas, numpy) e visualizzazione (matplotlib)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# scikit-learn: pipeline, preprocessing, classificatori e metriche
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, make_scorer
from mlxtend.evaluate import paired_ttest_5x2cv, combined_ftest_5x2cv
from collections import defaultdict
# Metriche di valutazione e visualizzazione della matrice di confusione
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from scipy import stats              # distribuzioni statistiche (t-test)
from scipy.stats import f as f_dist  # distribuzione F (test Alpaydin)
import math                          # sqrt per il denominatore del t-test
import os                            # gestione percorsi e cartelle di output

# Ricava il nome del file .py
nome_script = os.path.splitext(os.path.basename(__file__))[0]
# Crea il percorso della cartella di output accanto allo script,
# usando come nome cartella lo stesso nome del file Python
cartella_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), nome_script)
# Crea la cartella se non esiste già, exist_ok=True evita errore se la cartella è già presente
os.makedirs(cartella_output, exist_ok=True)

#==========================================
#1. CARICAMENTO DATI
#==========================================

#legge il file csv e lo mette in df (abbreviazione di dataframe)
df = pd.read_csv("dataset_finale.csv")

#in questa fase prototipale escludiamo sesso ed età per semplificare il modello e concentrarci solo sui dati biochimici,
#anche perchè sono scarsamente rilevanti nelle predizioni (stilando i capitoli dovrei dimostrarlo o mi baso sul consiglio ricevuto?)
feature = [
    "ALFA1 %\n (2,9 – 4,9 %)",
    "ALFA1#\n(0,20-0,35 g/dl)",
    "A1AT\n(90-200 mg/dl)",
    "PCR\n(0,0-5,0 mg/l)",
]

#seleziono la colonna target che contiene le etichette e la imposto come vettore di etichette
#La conversione a stringa evita problemi con valori misti o formati non uniformi
y = df["CARATTERIZZAZIONE\nALLELICA"].astype(str)
#stampo quante prove abbiamo per ogni classe
print("Distribuzione classi (multiclasse):")
print(y.value_counts(), "\n")

# NORMALE = MM, MUTAZIONE = tutto il resto
yBin = np.where(y == "MM", "NORMALE", "MUTAZIONE")
#Trasforma il vettore binario in una Series pandas
yAcc = pd.Series(yBin)
# stampa numero di MM e mutati e ribadiamo a video che il modello userà solo quelle 2 classi
print("Distribuzione classi (binaria):")
print(yAcc.value_counts(), "\n")
print("Classi usate dal modello:", sorted(yAcc.unique()))

# ==========================================
# 1.b ANALISI SOGLIE PER PAZIENTE (SOTTO / IN_RANGE / SOPRA)
# ==========================================

# Si salvano in due variabili i nomi esatti delle colonne del CSV
id_col = "NUMERO\nPROGRESSIVO" # Nome della colonna che identifica ogni paziente/campione
target_col = "CARATTERIZZAZIONE\nALLELICA" # Nome della colonna contenente il genotipo originale

# Crea una copia del dataset per convertire le feature in formato numerico
# Si itera su ogni feature biochimica e si eseguono tre operazioni in cascata:
df_num = df.copy()
for col in feature:
    # converte tutto in stringa, gestendo eventuali celle miste
    df_num[col] = pd.to_numeric(
        # converte in float, se una cella non è convertibile (es. testo libero, celle vuote), restituisce NaN invece di lanciare un'eccezione
        df_num[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
# formiamo una matrice numerica X (104 PAZIENTI x 4 MARCATORI BIOCHIMICI)
X = df_num[feature].values
# Soglie dei marcatori (come da nomi colonna), non è stata scelta una lista perchè usare un dizionario indicizzato per nome
# colonna rende il codice più leggibile e meno soggetto a errori di indice se l'ordine delle feature cambiasse
thr = {
    feature[0]: (2.9, 4.9),    # ALFA1 %
    feature[1]: (0.20, 0.35),  # ALFA1#
    feature[2]: (90.0, 200.0), # A1AT
    feature[3]: (0.0, 5.0),    # PCR
}

# funzione che prende un valore numerico e i limiti del range, e restituisce una delle quattro etichette possibili
def status(val, lo, hi):
    if pd.isna(val):
        return "MANCANTE"
    if val < lo:
        return "SOTTO"
    if val > hi:
        return "SOPRA"
    return "IN_RANGE"

# si crea una tabella di riepilogo con tre colonne base
tab = pd.DataFrame({
    "ID_PAZIENTE": df_num[id_col].astype(str),
    "ALLELICA": df_num[target_col].astype(str),
    # np.where funziona come un "if vettoriale" per ogni riga se il genotipo è "MM" = "NORMALE", altrimenti = "MUTAZIONE"
    "BINARIO": np.where(df_num[target_col].astype(str) == "MM", "NORMALE", "MUTAZIONE")
})

# si crea una tabella del tipo qua sotto, che poi salviamo in csv, a cosa serve? per avere un riepilogo di quali valori sono in range o no, come ragionerebbe un medico
# iteriamo su ciascun marker biochimico
for col in feature:
    lo, hi = thr[col]     # Recupera i limiti di normalità associati di prima
    tab[f"{col}_value"] = df_num[col]     # Aggiunge alla tabella il valore numerico
    tab[f"{col}_status"] = df_num[col].apply(lambda v: status(v, lo, hi)) # Aggiunge lo stato clinico del marker SOTTO, IN_RANGE, SOPRA oppure MANCANTE
# Crea la lista delle colonne che contengono lo stato dei quattro marker
status_cols = [f"{c}_status" for c in feature]
# Crea una tabella per ogni casistica, tutto true va nella prima, in ANY_SOTTO almeno 1 marker sottosoglia e in ANY_SOPRA con almeno 1 marker soprasoglia
tab["CON_VALORI_NELLA_NORMA"] = tab[status_cols].eq("IN_RANGE").all(axis=1)
tab["ANY_SOTTO"] = tab[status_cols].eq("SOTTO").any(axis=1)
tab["ANY_SOPRA"] = tab[status_cols].eq("SOPRA").any(axis=1)

print("\nPAZIENTI CHE HANNO I VALORI TUTTI NELLA NORMA (TRUE) E ALMENO UN VALORE NON NELLA NORMA (FALSE):")
# Conta quanti pazienti NORMALE/MUTAZIONE hanno tutti i marker nel range e quanti invece hanno almeno un marker fuori range
print(tab.groupby(["BINARIO", "CON_VALORI_NELLA_NORMA"]).size())

print("\nPAZIENTI CON TUTTI I MARCATORI NELLA NORMA:")
# Mostra ID, genotipo e classe binaria dei pazienti con tutti i marker nei range che siano NORMALI o MUTATI
print(tab.loc[tab["CON_VALORI_NELLA_NORMA"], ["ID_PAZIENTE", "ALLELICA", "BINARIO"]].to_string(index=False))

tab.to_csv(os.path.join(cartella_output, "tabella_soglie_4marker_binario.csv"), index=False)

#==========================================
# 2. SCELTA MODELLLO DA UTENTE + DEFINIZIONE MODELLI E METRICHE
#==========================================

# In questa sezione vengono definiti:
# - il Dummy Classifier, cioè la baseline statistica minima;
# - due baseline cliniche basate su soglia A1AT;
# - il classificatore ML scelto dall'utente;
# - la funzione comune per calcolare le metriche.

# Le pipeline combinano MinMaxScaler e classificatore.
# MinMaxScaler viene fittato solo sul training set di ogni fold e poi applicato al test set.
# Questo evita data leakage, perché le informazioni del test set non influenzano la normalizzazione.

# class_weight="balanced" o "balanced_subsample" compensa lo sbilanciamento tra classi.
# Nel dataset, NORMALE e MUTAZIONE non hanno la stessa numerosità:
# senza pesi, un classificatore potrebbe favorire la classe più frequente.

# I parametri dei modelli derivano dal tuning iperparametrico:
# - Random Forest: n_estimators=100
# - SVC RBF: C=1000, gamma="auto"
# - Logistic Regression: C=1000
# - LinearSVC: C=1000

# La baseline a soglia usa A1AT perché è il marker biochimico più direttamente
# associato al deficit di alfa-1-antitripsina. Gli altri marker sono informativi,
# ma A1AT rappresenta la misura clinica principale della concentrazione sierica

class TraditionalThreshold(BaseEstimator, ClassifierMixin):
    # Baseline tradizionale, A1AT < 90 mg/dl -> MUTAZIONE (soglia clinica nota)
    def __init__(self, threshold=90.0, feature_idx=2):  # threshold è la soglia clinica di A1AT, feature_idx=2 indica che A1AT è la terza feature nella matrice X
        self.threshold = threshold         # Salva la soglia come attributo dell'oggetto
        self.feature_idx = feature_idx         # Salva l'indice della feature A1AT
    def fit(self, X, y):      # ricordiamoci che questa baseline non impara parametri dai dati, perché usa una soglia fissa
        return self
    def predict(self, X): # Riceve una matrice X e restituisce una predizione per ogni paziente
        return np.where(X[:, self.feature_idx] < self.threshold,
                        "MUTAZIONE", "NORMALE")         # Se il valore A1AT è sotto soglia, predice MUTAZIONE, altrimenti predice NORMALE

# La soglia esplorativa viene cercata su A1AT perché vogliamo confrontare
# il modello ML con una regola clinica semplice basata sul marker principale,
# ottimizzando però il cut-off sul training set invece di fissarlo a priori.

class ExploratoryThreshold(BaseEstimator, ClassifierMixin):
    # Definisce una baseline esplorativa basata su A1AT, trova la soglia ottimale su A1AT nel training set, a differenza della tradizionale
    def __init__(self, feature_idx=2):
        self.feature_idx = feature_idx
        self.best_thr = None
    def fit(self, X, y):         # Cerca la soglia A1AT che massimizza la balanced accuracy sul training set
        vals = np.sort(np.unique(X[:, self.feature_idx]))         # Estrae tutti i valori unici di A1AT nel training set e li ordina
        best_score, best_t = -1, vals[0] # Inizializza il miglior punteggio e la migliore soglia, best_score parte da -1 per essere sicuramente superato dal primo valore
        for t in vals:             # Prova ogni valore A1AT come possibile soglia
            pred = np.where(X[:, self.feature_idx] < t, "MUTAZIONE", "NORMALE")             # Genera le predizioni usando t come soglia
            score = balanced_accuracy_score(y, pred)             # Valuta la soglia tramite balanced accuracy sul training set
            if score > best_score:                 # Se questa soglia è migliore delle precedenti, la salva e aggiorna il miglior punteggio e soglia
                best_score, best_t = score, t
        self.best_thr = best_t
        return self
    def predict(self, X):         # Predice usando la soglia ottimizzata nel training set
        return np.where(X[:, self.feature_idx] < self.best_thr,
                        "MUTAZIONE", "NORMALE")    # Se A1AT è sotto la soglia appresa, predice MUTAZIONE, altrimenti predice NORMALE.
# menù iterattivo
print("\n" + "="*50)
print("SELEZIONE CLASSIFICATORE")
print("="*50)
print("1 = Random Forest")
print("2 = Support Vector Classifier (SVC)")
print("3 = Logistic Regression")
print("4 = Linear SVC")
print("="*50)

# semplice ciclo while per chiedere all'utente quale classificatore vuole
while True:
    scelta = input("Inserisci 1, 2, 3 o 4: ").strip()
    if scelta == "1":
        model_type = "RF"
        nome_modello = "Random Forest"
        break
    elif scelta == "2":
        model_type = "SVC"
        nome_modello = "SVC"
        break
    elif scelta == "3":
        model_type = "LR"
        nome_modello = "LogisticRegression"
        break
    elif scelta == "4":
        model_type = "LSVC"
        nome_modello = "LinearSVC"
        break
    else:
        print("Inserisci un numero tra 1, 2, 3 o 4.")

print(f"\n Hai selezionato: {nome_modello}")

# baseline casuale, predice sempre la classe più frequente (NORMALE=68/104)
# otterrà accuracy +-65% ma balanced_accuracy = 0.5 per definizione
casuale = DummyClassifier(strategy="most_frequent")

# Modello serio in base alla scelta
if model_type == "RF":
    serio = Pipeline(steps=[
        ("scaler", MinMaxScaler()),
        ("clf", RandomForestClassifier(
            # empiricamente rascka dimostra che 200 alberi è il numero tipicamente abbondante sennò si rallenterebbe il codice inutilmente senza guadagnare in performance
            #ATTENZIONE: dal tuning si è visto che per un infinitesimo la balanced accuracy con 100 alberi aumenta quindi ci atteniamo al tuning effettuato
            n_estimators=100,
            random_state=42,
            class_weight="balanced_subsample"
        ))
    ])
elif model_type == "SVC":
    serio = Pipeline(steps=[
        ("scaler", MinMaxScaler()),
        ("clf", SVC(
            kernel="rbf", # Trasforma i dati in uno spazio ad alta dimensione per trovare confini non lineari
            C=1000, # empiricamente in script apposito
            gamma="auto", # valore migliore individuato nel tuning SVC RBF
            class_weight="balanced",
            random_state=42
        ))
    ])
elif model_type == "LR":
    serio = Pipeline(steps=[
        ("scaler", MinMaxScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1000,
            random_state=42
        ))
    ])
elif model_type == "LSVC":
    serio = Pipeline(steps=[
        ("scaler", MinMaxScaler()),
        ("clf", LinearSVC(
            C=1000,
            class_weight="balanced",
            random_state=42,
            max_iter=10000
        ))
    ])

# eval_metrics è una funzione di comodo che prende le etichette vere (y_true) e quelle predette (y_pred) e
# restituisce un dizionario con 5 metriche. Viene chiamata 10 volte durante il 5x2cv (una per ogni round)
# sia sul modello serio che sul Dummy.
def eval_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred), # Utile ma ingannevole: il Dummy ottiene +-65% senza fare nulla
        "bal_acc": balanced_accuracy_score(y_true, y_pred), # media del recall per classe non premia chi ignora la classe rara
        "f1_macro": f1_score(y_true, y_pred, average="macro"), # Media F1 per classe senza peso
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"), # Media F1 pesata per la numerosità
        "mcc": matthews_corrcoef(y_true, y_pred),
        # perchè ho voluto usare il Matthews Correlation Coefficient (MCC)? A differenza dell'accuracy standard, il MCC considera tutti gli elementi della matrice di confusione,
        # rendendolo particolarmente adatto per dataset sbilanciati come il nostro (P330). tiene conto di TP, TN, FP, FN simultaneamente
    }

def dietterich_alpaydin(diff_A, diff_B):
    # Calcola manualmente il paired t-test 5x2CV di Dietterich
    # e il combined F-test 5x2CV di Alpaydin a partire dalle differenze
    # di balanced accuracy tra due classificatori.
    s2 = np.array([
        (dA - 0.5 * (dA + dB)) ** 2 + (dB - 0.5 * (dA + dB)) ** 2
        for dA, dB in zip(diff_A, diff_B)
    ])

    denom = math.sqrt((1.0 / 5.0) * sum(s2))
    t_stat = diff_A[0] / denom if denom > 0 else float("inf")
    p_t = 2 * (1 - stats.t.cdf(abs(t_stat), df=5))

    numerator = sum(dA ** 2 + dB ** 2 for dA, dB in zip(diff_A, diff_B))
    denominator = 2 * sum(s2)
    f_stat = numerator / denominator if denominator > 0 else float("inf")
    p_f = 1 - f_dist.cdf(f_stat, dfn=10, dfd=5)

    return t_stat, p_t, f_stat, p_f

def salva_tabella(df_out, nome_file):
    # Salva una tabella nella cartella output dello script.
    # Centralizzare il salvataggio evita ripetizioni di os.path.join.
    df_out.to_csv(os.path.join(cartella_output, nome_file), index=False)

#==========================================
#3. SCHEMA 5x2cv
#==========================================

# In questa sezione viene implementata manualmente la 5x2 cross-validation.
# Il dataset viene diviso 5 volte in due metà stratificate.
# Per ogni divisione vengono eseguiti due round:
# - train su A, test su B;
# - train su B, test su A.


# Si inizializzano liste vuote che verranno riempite durante il ciclo. Ogni lista ha uno scopo preciso:
rows = [] # dizionario metriche per ogni run
diff_acc_A = []   # differenze modello ML - Dummy, round A
diff_acc_B = []   # differenze modello ML - Dummy, round B

diff_trad_A = []  # differenze modello ML - soglia tradizionale, round A
diff_trad_B = []  # differenze modello ML - soglia tradizionale, round B

diff_expl_A = []  # differenze modello ML - soglia esplorativa, round A
diff_expl_B = []  # differenze modello ML - soglia esplorativa, round B

all_acc_serio = [] # lista delle accuracy del modello scelto nei 10 round
all_y_true = [] # etichette vere di ogni fold
all_y_pred_serio = []      # predizioni del modello ML scelto
all_y_pred_casuale = []    # predizioni del Dummy Classifier
all_y_pred_trad = []       # predizioni della baseline tradizionale A1AT < 90
all_y_pred_expl = []       # predizioni della baseline esplorativa
all_idx_test = []          # indici dei pazienti nel test set

# matrice X (104x4): righe=pazienti, colonne=4 feature biochimiche
X = df_num[feature].values

# Il seed globale viene impostato qui, una sola volta prima del ciclo
# idx è un array con i numeri di riga (0..103): train_test_split divide gli indici, non i dati direttamente
np.random.seed(42)
# creazione di un array con gli indici dei pazienti, parte da 0 fino a 103
# un array è semplicemente una lista ordinata di valori dello stesso tipo, disposti in sequenza
# idx serve come etichetta numerica di ogni paziente, per poter risalire a lui dopo che i dati sono stati mescolati e divisi precedentemente
idx = np.arange(len(df))
# Contatore delle eventuali ripetizioni in cui la stratificazione fallisce, non deve succedere con questo dataset
n_stratify_fail = 0

# il seed globale viene impostato una sola volta prima del ciclo, ogni chiamata successiva a train_test_split usa random_state=None,
# quindi consuma lo stato casuale globale di NumPy, in questo modo gli split sono diversi tra le 5 ripetizioni,
# ma l'intera sequenza rimane riproducibile a ogni esecuzione dello script. Anche mlxtend riceve un seed fisso (random_seed=42), ma genera internamente
# i propri split 5x2CV: per questo i valori t e F possono essere vicini, ma non necessariamente identici a quelli calcolati manualmente.
for i in range(5):
    # esegue le 5 ripetizioni previste dal protocollo 5x2CV, ogni ripetizione produrrà due round A e B
    # Il blocco try/except gestisce un caso limite, con dataset molto piccoli e classi rare, la stratificazione può fallire se una classe ha troppo pochi elementi per essere divisa equamente
    # In quel caso si degrada silenziosamente a split non stratificato, stampando un avviso
    try:
        # Divide gli indici dei pazienti in due fold:
        # - idxA: indici del fold A
        # - idxB: indici del fold B
        # Divide anche le etichette corrispondenti:
        # - yA: classi del fold A
        # - yB: classi del fold B
        # test_size=0.5 crea due metà di uguale dimensione
        # stratify=yAcc mantiene proporzioni simili tra NORMALE e MUTAZIONE
        idxA, idxB, yA, yB = train_test_split(
            idx, yAcc, test_size=0.5, random_state=None, stratify=yAcc
        )
    except ValueError as e:
        print(f"  [ATTENZIONE] rep {i + 1}: stratify fallita ({e}), uso stratify=None")
        # conta le ripetizioni non stratificate
        n_stratify_fail += 1
        idxA, idxB, yA, yB = train_test_split(
            idx, yAcc, test_size=0.5, random_state=None, stratify=None
        )
    # Estrae dalla matrice X le righe corrispondenti ai due fold
    # XA contiene le feature dei pazienti del fold A e XB contiene le feature dei pazienti del fold B
    XA, XB = X[idxA], X[idxB]

    # round A (train A / test B)
    # Entrambi i modelli (4 in totale) vengono addestrati su A e testati su B,.fit() sulla Pipeline esegue in sequenza MinMaxScaler (impara min/max da XA) e poi il classificatore
    # il test set XB non "vede" mai i dati di training durante la normalizzazione
    # la chiamata a fit sul modello tradizionale non stima parametri, serve solo a mantenere la stessa interfaccia fit/predict dei modelli scikit-learn
    # il fit sul modello esplorativo invece serve perchè cerca la siglia di A1AT che massimizzza la balanced accuracy
    serio_A = serio.fit(XA, yA)
    casuale_A = casuale.fit(XA, yA)
    trad_A = TraditionalThreshold().fit(XA, yA)
    expl_A = ExploratoryThreshold().fit(XA, yA)

    pred_serio_B = serio_A.predict(XB)
    pred_casuale_B = casuale_A.predict(XB)
    pred_trad_B = trad_A.predict(XB)
    pred_expl_B = expl_A.predict(XB)

    # si chiama la funzione precedentemente creata con le metriche per tutti i modelli confrontando le predizioni con le etichette vere yB
    m_serio_A = eval_metrics(yB, pred_serio_B)
    m_casuale_A = eval_metrics(yB, pred_casuale_B)
    m_trad_A = eval_metrics(yB, pred_trad_B)
    m_expl_A = eval_metrics(yB, pred_expl_B)

    # Si salvano i risultati del round A nelle liste globali
    all_acc_serio.append(m_serio_A["acc"]) # Salva l'accuracy del modello scelto per il round A
    all_y_true.append(yB)    # Salva le etichette vere del test set B
    all_y_pred_serio.append(pred_serio_B)   # Salva le predizioni del modello scelto sul test set B
    all_y_pred_casuale.append(pred_casuale_B)  # Salva le predizioni del Dummy sul test set B
    all_y_pred_trad.append(pred_trad_B)
    all_y_pred_expl.append(pred_expl_B)
    all_idx_test.append(idxB)     # Salva gli indici dei pazienti presenti nel test set B, serviranno nel blocco 6
    # differenza di balanced accuracy (modello serio - Dummy) per il round A, usata dai test statistici
    diff_acc_A.append(m_serio_A["bal_acc"] - m_casuale_A["bal_acc"])
    # differenze anche per le baseline a soglia e esplorativa
    diff_trad_A.append(m_serio_A["bal_acc"] - m_trad_A["bal_acc"])
    diff_expl_A.append(m_serio_A["bal_acc"] - m_expl_A["bal_acc"])
    # Aggiunge al riepilogo le metriche del modello scelto nel round A
    rows.append({"rep": i + 1, "round": "A (train A / test B)", "model": nome_modello, **m_serio_A})
    # Aggiunge le metriche del Dummy nel round A
    rows.append({"rep": i + 1, "round": "A (train A / test B)", "model": "Dummy", **m_casuale_A})
    # Aggiunge le metriche della baseline clinica tradizionale nel round A
    rows.append({"rep": i + 1, "round": "A (train A / test B)", "model": "Trad (A1AT<90)", **m_trad_A})
    # Aggiunge le metriche della baseline esplorativa nel round A
    rows.append({"rep": i + 1, "round": "A (train A / test B)", "model": "Expl threshold", **m_expl_A})


    # round B (train B / test A)
    # Questa volta si addestra su B e si testa su A. È la logica speculare del round A :)
    # Da notare che serio viene ri-addestrato da zero, il modello del round A viene sovrascritto, ma i risultati erano già stati salvati nelle liste nessuna perdita di dati.
    serio_B = serio.fit(XB, yB)
    casuale_B = casuale.fit(XB, yB)
    trad_B = TraditionalThreshold().fit(XB, yB)
    expl_B = ExploratoryThreshold().fit(XB, yB)

    pred_serio_A = serio_B.predict(XA)
    pred_casuale_A = casuale_B.predict(XA)
    pred_trad_A = trad_B.predict(XA)
    pred_expl_A = expl_B.predict(XA)

    m_serio_B = eval_metrics(yA, pred_serio_A)
    m_casuale_B = eval_metrics(yA, pred_casuale_A)
    m_trad_B = eval_metrics(yA, pred_trad_A)
    m_expl_B = eval_metrics(yA, pred_expl_A)

    all_acc_serio.append(m_serio_B["acc"])
    all_y_true.append(yA)
    all_y_pred_serio.append(pred_serio_A)
    all_y_pred_casuale.append(pred_casuale_A)
    all_y_pred_trad.append(pred_trad_A)
    all_y_pred_expl.append(pred_expl_A)
    all_idx_test.append(idxA)

    diff_acc_B.append(m_serio_B["bal_acc"] - m_casuale_B["bal_acc"])
    diff_trad_B.append(m_serio_B["bal_acc"] - m_trad_B["bal_acc"])
    diff_expl_B.append(m_serio_B["bal_acc"] - m_expl_B["bal_acc"])

    rows.append({"rep": i + 1, "round": "B (train B / test A)", "model": nome_modello, **m_serio_B})
    rows.append({"rep": i + 1, "round": "B (train B / test A)", "model": "Dummy", **m_casuale_B})
    rows.append({"rep": i + 1, "round": "B (train B / test A)", "model": "Trad (A1AT<90)", **m_trad_B})
    rows.append({"rep": i + 1, "round": "B (train B / test A)", "model": "Expl threshold", **m_expl_B})

# Controlla se almeno una ripetizione non è stata stratificata.
if n_stratify_fail > 0:
    print(f"\n  ATTENZIONE: {n_stratify_fail}/5 ripetizioni NON stratificate!")
    print("   I risultati potrebbero essere meno affidabili.")
else:
    print("\n Tutte le 5 ripetizioni stratificate correttamente.")

# rows a questo punto contiene 40 dizionari (5 ripetizioni x 2 round x 4 modelli = 40) pd Pandas li converte automaticamente in un DataFrame con una riga per dizionario
res = pd.DataFrame(rows)
# Stampa un'intestazione per il riepilogo delle metriche sui 10 round
print("5x2cv (10 run) - metriche: media + std ")
# Raggruppa le righe per modello, per accuracy, balanced accuracy e MCC calcola media e deviazione standard
agg = res.groupby("model")[["acc", "bal_acc", "mcc"]].agg(["mean", "std"])
print(agg, "\n")

print("F1-SCORE")
# Raggruppa per modello e calcola media/deviazione standard di F1 macro e F1 weighted.
f1_only = res.groupby("model")[["f1_macro", "f1_weighted"]].agg(["mean", "std"])
print(f1_only)

salva_tabella(res, f"metriche_5x2cv_dettaglio_{nome_modello}.csv")
agg_export = agg.copy()
agg_export.columns = [f"{metrica}_{stat}" for metrica, stat in agg_export.columns]
salva_tabella(agg_export.reset_index(), f"metriche_5x2cv_summary_{nome_modello}.csv")
f1_export = f1_only.copy()
f1_export.columns = [f"{metrica}_{stat}" for metrica, stat in f1_export.columns]
salva_tabella(f1_export.reset_index(), f"f1_5x2cv_summary_{nome_modello}.csv")

#==========================================
#4. TEST STATISTICI DI DIETTERICH e ALPAYDIN (MANUALE)
#==========================================

# Il modello scelto è davvero migliore del Dummy Classifier, oppure la differenza osservata può essere dovuta al caso?

# In questa sezione vengono calcolati manualmente due test statistici 5x2CV:
# - il paired t-test di Dietterich
# - il combined F-test di Alpaydin
# Entrambi confrontano il modello scelto con il Dummy Classifier usando come metrica la balanced accuracy

# Confronto manuale tra modello ML vs Dummy
t, p, f_stat, p_f = dietterich_alpaydin(diff_acc_A, diff_acc_B)

# Confronto manuale tra modello ML vs soglia tradizionale A1AT<90
t_trad, p_trad, f_stat_trad, p_f_trad = dietterich_alpaydin(diff_trad_A, diff_trad_B)

# Confronto manuale tra modello ML vs soglia esplorativa A1AT
t_expl, p_expl, f_stat_expl, p_f_expl = dietterich_alpaydin(diff_expl_A, diff_expl_B)

print(f"\nTEST STATISTICI MANUALI: {nome_modello} vs Dummy")
print(f"{'Test':<30} {'Statistic':>12} {'p-value':>15}")
print(f"{'Dietterich t-test':<30} {t:>12.4f} {p:>15.6f}")
print(f"{'Alpaydin F-test':<30} {f_stat:>12.4f} {p_f:>15.6f}")

# La soglia p < 0.05 è la convenzione standard se la probabilità che la differenza osservata sia dovuta al caso è inferiore al 5%,
# si rigetta l'ipotesi nulla (i due modelli sono equivalenti), questa non è una prova che il modello sia migliore è solo evidenza statistica che la differenza è improbabilmente casuale
print(f"Al livello di significatività 0.05, {nome_modello} mostra una differenza statisticamente significativa rispetto al Dummy? {'SÌ' if p_f < 0.05 else 'NO'}")
# Un p-value inferiore a 0.05 indica che la differenza osservata tra modello e Dummy
# è difficilmente attribuibile al caso secondo il test statistico scelto
# Questo non dimostra che il modello sia perfetto, ma fornisce evidenza
# che performi meglio della baseline Dummy sul protocollo 5x2CV

print(f"\nTEST STATISTICI MANUALI: {nome_modello} vs soglia tradizionale A1AT<90")
print(f"{'Test':<30} {'Statistic':>12} {'p-value':>15}")
print(f"{'Dietterich t-test':<30} {t_trad:>12.4f} {p_trad:>15.6f}")
print(f"{'Alpaydin F-test':<30} {f_stat_trad:>12.4f} {p_f_trad:>15.6f}")
print(
    f"Al livello di significatività 0.05, {nome_modello} mostra una differenza "
    f"statisticamente significativa rispetto alla soglia tradizionale? "
    f"{'SÌ' if p_f_trad < 0.05 else 'NO'}"
)

print(f"\nTEST STATISTICI MANUALI: {nome_modello} vs soglia esplorativa A1AT")
print(f"{'Test':<30} {'Statistic':>12} {'p-value':>15}")
print(f"{'Dietterich t-test':<30} {t_expl:>12.4f} {p_expl:>15.6f}")
print(f"{'Alpaydin F-test':<30} {f_stat_expl:>12.4f} {p_f_expl:>15.6f}")
print(
    f"Al livello di significatività 0.05, {nome_modello} mostra una differenza "
    f"statisticamente significativa rispetto alla soglia esplorativa? "
    f"{'SÌ' if p_f_expl < 0.05 else 'NO'}"
)

#==========================================
# 4.b VERIFICA CON MLXTEND
#==========================================

# Questa sezione confronta il calcolo manuale dei test 5x2CV con le implementazioni disponibili nella libreria mlxtend
# Lo scopo non è ottenere valori identici al decimale, ma verificare che la conclusione statistica sia coerente

# I valori numerici possono differire leggermente dalla verifica con mlxtend perché:
# - mlxtend genera internamente gli split del 5x2CV a partire da random_seed=42;
# - lo script manuale usa np.random.seed(42) e poi train_test_split(..., random_state=None)
#   per costruire le 5 coppie di fold A/B;
# - la metrica è stata resa coerente usando scorer_bal, cioè balanced_accuracy,
#   più adatta a un dataset sbilanciato (68 NORMALE vs 36 MUTAZIONE).

# Crea uno scorer compatibile con mlxtend basato sulla balanced accuracy
# Senza questo parametro, mlxtend userebbe la metrica di default cioè l'accuracy standard
scorer_bal = make_scorer(balanced_accuracy_score)

t_mlx, p_mlx = paired_ttest_5x2cv(
    estimator1=serio, # estimator1 è il modello scelto
    estimator2=casuale, # estimator2 è il Dummy Classifier
    X=X, # X e y sono dati e target binario
    y=yAcc,
    scoring=scorer_bal, # scoring=scorer_bal forza l'uso della balanced accuracy
    random_seed=42 # random_seed=42 rende riproducibile la procedura interna di mlxtend
)

f_mlx, p_f_mlx = combined_ftest_5x2cv(
    estimator1=serio,
    estimator2=casuale,
    X=X,
    y=yAcc,
    scoring=scorer_bal,   #
    random_seed=42
)

'''
# in seguito si è deciso di voler confrontare anche tra modello serio e
# baseline tradizionale perchè in confronto non deve essere solo con il classificatore casuale
trad = TraditionalThreshold()

t_mlx_trad, p_mlx_trad = paired_ttest_5x2cv(
    estimator1=serio,
    estimator2=trad,
    X=X,
    y=yAcc,
    scoring=scorer_bal,
    random_seed=42
)

f_mlx_trad, p_f_mlx_trad = combined_ftest_5x2cv(
    estimator1=serio,
    estimator2=trad,
    X=X,
    y=yAcc,
    scoring=scorer_bal,
    random_seed=42
)
'''

print(f"\nCONFRONTO MANUALE vs MLXTEND (entrambi su balanced_accuracy)")
print(f"{'':30} {'t mio':>12} {'t mlxtend':>12}")
print(f"{'Dietterich t':30} {t:>12.4f} {t_mlx:>12.4f}")
print(f"{'p-value':30} {p:>12.6f} {p_mlx:>12.6f}")
print(f"{'':30} {'F mio':>12} {'F mlxtend':>12}")
print(f"{'Alpaydin F':30} {f_stat:>12.4f} {f_mlx:>12.4f}")
print(f"{'p-value':30} {p_f:>12.6f} {p_f_mlx:>12.6f}")
print("Nota: mlxtend costruisce internamente i propri split 5x2CV, il confronto serve come verifica indipendente della stessa conclusione statistica.")

'''
print(f"\nCONFRONTO MLXTEND: {nome_modello} vs soglia tradizionale A1AT<90")
print(f"{'Test':<30} {'Statistic':>12} {'p-value':>15}")
print(f"{'Dietterich t-test':<30} {t_mlx_trad:>12.4f} {p_mlx_trad:>15.6f}")
print(f"{'Alpaydin F-test':<30} {f_mlx_trad:>12.4f} {p_f_mlx_trad:>15.6f}")
print(f"Al livello di significatività 0.05, {nome_modello} è diverso dalla soglia tradizionale? {'SÌ' if p_f_mlx_trad < 0.05 else 'NO'}")
'''

test_stats = pd.DataFrame([
    {"metodo": "manuale", "confronto": f"{nome_modello} vs Dummy", "test": "Dietterich t-test", "statistic": t, "p_value": p},
    {"metodo": "manuale", "confronto": f"{nome_modello} vs Dummy", "test": "Alpaydin F-test", "statistic": f_stat, "p_value": p_f},

    {"metodo": "manuale", "confronto": f"{nome_modello} vs soglia tradizionale A1AT<90", "test": "Dietterich t-test", "statistic": t_trad, "p_value": p_trad},
    {"metodo": "manuale", "confronto": f"{nome_modello} vs soglia tradizionale A1AT<90", "test": "Alpaydin F-test", "statistic": f_stat_trad, "p_value": p_f_trad},

    {"metodo": "manuale", "confronto": f"{nome_modello} vs soglia esplorativa A1AT", "test": "Dietterich t-test", "statistic": t_expl, "p_value": p_expl},
    {"metodo": "manuale", "confronto": f"{nome_modello} vs soglia esplorativa A1AT", "test": "Alpaydin F-test", "statistic": f_stat_expl, "p_value": p_f_expl},

    {"metodo": "mlxtend", "confronto": f"{nome_modello} vs Dummy", "test": "Dietterich t-test", "statistic": t_mlx, "p_value": p_mlx},
    {"metodo": "mlxtend", "confronto": f"{nome_modello} vs Dummy", "test": "Alpaydin F-test", "statistic": f_mlx, "p_value": p_f_mlx},

    # {"metodo": "mlxtend", "confronto": f"{nome_modello} vs Trad (A1AT<90)", "test": "Dietterich t-test", "statistic": t_mlx_trad, "p_value": p_mlx_trad},
    # {"metodo": "mlxtend", "confronto": f"{nome_modello} vs Trad (A1AT<90)", "test": "Alpaydin F-test", "statistic": f_mlx_trad, "p_value": p_f_mlx_trad},
])
salva_tabella(test_stats, f"test_statistici_{nome_modello}.csv")

#==========================================
#5. MATRICE DI CONFUSIONE AGGREGATA
#==========================================

# In questa sezione vengono aggregate le predizioni dei 10 round del 5x2CV
# L'obiettivo è valutare il comportamento complessivo del modello scelto e confrontarlo con il Dummy Classifier

# Aggregare i risultati di tutti i fold riduce la dipendenza dal singolo split aspetto importante con un dataset piccolo (n=104)
# In questo modo non si seleziona il "fold migliore", ma si valuta il modello sull'insieme completo delle predizioni prodotte durante la cross-validation

# La matrice di confusione aggregata contiene i conteggi assoluti su tutti gli eventi di test dei 10 round
# Nel 5x2CV ogni paziente compare una volta per ripetizione nel test set quindi si ottengono 104 x 5 = 520 eventi di predizione

# Per il grafico la matrice viene poi normalizzata per riga, vuol dire che per ogni riga viene effettuata una divisione per il totale dei veri appartenenti a quella classe
# facciamo un esempio, matrice aggregata (520 eventi, 5x2CV)
#
#                  Pred. NORMALE   Pred. MUTAZIONE
#  Vero NORMALE        310              15          -> totale riga: 325
#  Vero MUTAZIONE       28             167          -> totale riga: 195
#
# Dopo la normalizzazione per riga (ogni cella / totale della sua riga):
#
#                  Pred. NORMALE   Pred. MUTAZIONE
#  Vero NORMALE     310/325=0.95    15/325=0.05     -> recall NORMALE  95%
#  Vero MUTAZIONE    28/195=0.14   167/195=0.86     -> recall MUTAZIONE 86%
#
# Senza normalizzazione la riga NORMALE avrebbe valori assoluti molto più grandi
# (più pazienti MM nel dataset), rendendo difficile il confronto visivo tra classi
# Con la normalizzazione entrambe le righe sono in scala 0-1 e mostrano direttamente il recall per classe, indipendentemente dallo sbilanciamento

# Concatena le etichette vere dei test set di tutti i 10 round
y_true_all = np.concatenate(all_y_true)
# Concatena tutte le predizioni del modello scelto nei 10 round
y_pred_serio_all = np.concatenate(all_y_pred_serio)
# Concatena tutte le predizioni del Dummy Classifier nei 10 round
y_pred_casuale_all = np.concatenate(all_y_pred_casuale)
# concatena anche i risultati dei classificatori umani
y_pred_trad_all = np.concatenate(all_y_pred_trad)
y_pred_expl_all = np.concatenate(all_y_pred_expl)
# concatena gli indici dei pazienti presenti nei test set dei 10 round
idx_test_all = np.concatenate(all_idx_test)

# ora vogliamo contare quanti pazienti su 104 vengono effettivamente predetti come mutazione o MM calcolando anche numero esatto di FN E TN

per_paziente = []

for paz_idx in np.unique(idx_test_all):
    mask = idx_test_all == paz_idx

    vero = y_true_all[mask][0]
    predizioni = y_pred_serio_all[mask]

    n_mut = np.sum(predizioni == "MUTAZIONE")
    n_norm = np.sum(predizioni == "NORMALE")

    # voto di maggioranza su 5 predizioni
    pred_finale = "MUTAZIONE" if n_mut >= 3 else "NORMALE"

    per_paziente.append({
        "idx": paz_idx,
        "ID_PAZIENTE": df.iloc[paz_idx][id_col],
        "classe_vera": vero,
        "predizioni_MUTAZIONE": n_mut,
        "predizioni_NORMALE": n_norm,
        "predizione_finale_maggioranza": pred_finale,
        "corretto": vero == pred_finale
    })

df_per_paziente = pd.DataFrame(per_paziente)

print("\n" + "="*70)
print("CONTEGGIO PER PAZIENTE UNICO - VOTO DI MAGGIORANZA")
print("="*70)

print("\nPredizione finale del modello sui 104 pazienti:")
print(df_per_paziente["predizione_finale_maggioranza"].value_counts())

print("\nMatrice di confusione per paziente unico:")
cm_pazienti = confusion_matrix(
    df_per_paziente["classe_vera"],
    df_per_paziente["predizione_finale_maggioranza"],
    labels=["MUTAZIONE", "NORMALE"]
)

print(cm_pazienti)

TP, FN, FP, TN = cm_pazienti[0, 0], cm_pazienti[0, 1], cm_pazienti[1, 0], cm_pazienti[1, 1]

print(f"\nMutati identificati: {TP}/36")
print(f"Mutati non identificati: {FN}/36")
print(f"Falsi positivi tra i normali: {FP}/68")
print(f"Normali corretti: {TN}/68")

salva_tabella(df_per_paziente, f"predizioni_per_paziente_maggioranza_{nome_modello}.csv")

def cm_row(nome, y_pred):
    cm = confusion_matrix(y_true_all, y_pred, labels=["MUTAZIONE", "NORMALE"])
    TP, FN, FP, TN = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return {
        "metodo": nome,
        "TP_mutazione": TP,
        "FN_mutazione": FN,
        "FP_normale": FP,
        "TN_normale": TN,
        "recall_mutazione": TP / (TP + FN) if (TP + FN) > 0 else 0,
        "recall_normale": TN / (TN + FP) if (TN + FP) > 0 else 0,
    }

cm_summary = pd.DataFrame([
    cm_row(nome_modello, y_pred_serio_all),
    cm_row("Dummy", y_pred_casuale_all),
    cm_row("Trad (A1AT<90)", y_pred_trad_all),
    cm_row("Expl threshold", y_pred_expl_all),
])

print("\nRIEPILOGO MATRICI DI CONFUSIONE AGGREGATE:")
print(cm_summary.to_string(index=False))
salva_tabella(cm_summary, f"confusion_matrix_summary_tutti_metodi_{nome_modello}.csv")

# normalità per marker (range)
col_alfa1_perc = "ALFA1 %\n (2,9 – 4,9 %)"
col_alfa1_abs  = "ALFA1#\n(0,20-0,35 g/dl)"
col_a1at       = "A1AT\n(90-200 mg/dl)"
col_pcr        = "PCR\n(0,0-5,0 mg/l)"

# Crea una maschera booleana (cioè una lista di True/False, una per ogni paziente) lunga quanto il dataset originale
# True indica che il paziente ha tutti e quattro i marker nel range clinico
# False indica che almeno un marker è fuori range
normal_all_patients = (
    df_num[col_alfa1_perc].between(2.9, 4.9)   # ALFA1% nel range?
  & df_num[col_alfa1_abs].between(0.20, 0.35)  # ALFA1# nel range?
  & df_num[col_a1at].between(90, 200)           # A1AT nel range?
  & df_num[col_pcr].between(0.0, 5.0)           # PCR nel range?
)

# Allinea la maschera di normalità (104 valori) ai 520 eventi del test set
# per ogni predizione nei 10 round, recupera se quel paziente aveva tutti i marker in range
normal_in_test = normal_all_patients.iloc[idx_test_all].to_numpy()

# Filtra i 520 eventi dove il paziente è realmente MUTAZIONE E ha tutti i marker nei range. Sono i pazienti clinicamente più rilevanti
# quelli che sfuggirebbero a un'analisi tradizionale basata solo sui valori ematici perché sembrano "normali"
mut_marker_in_range = (y_true_all == "MUTAZIONE") & normal_in_test

# Conta i pazienti unici, non gli eventi, lo stesso paziente può comparire in più fold, quindi serve np.unique
n_mut_in_range_pat = len(np.unique(idx_test_all[mut_marker_in_range]))

# Aggiunge un terzo filtro, pazienti NORMALI classificati come MUTAZIONE con marker nei range, sono i falsi positivi clinicamente innocui
fp_marker_in_range = (y_true_all == "NORMALE") & (y_pred_serio_all == "MUTAZIONE") & normal_in_test
# Conta quanti pazienti unici rientrano in questa categoria
n_fp_marker_in_range_pat = len(np.unique(idx_test_all[fp_marker_in_range]))

''' print("\nMutati con tutti i marcatori nella norma (clinicamente 'nascosti'):")
print(f"- Pazienti MUTAZIONE con tutti i marcatori in range (conteggiati nel dataset): {n_mut_in_range_pat}")
print(f"- Pazienti NORMALE con tutti i marcatori in range predetti MUTAZIONE almeno una volta dal modello: {n_fp_marker_in_range_pat}")
'''
# Mutazioni correttamente previste con marker in range
correct = (y_true_all == y_pred_serio_all)
is_mutation = (y_true_all == "MUTAZIONE")
# Seleziona gli eventi in cui il paziente è realmente MUTAZIONE, il modello lo predice correttamente come MUTAZIONE e
# tutti i marker sono nel range. Questi sono i casi in cui il modello aggiunge potenziale valore clinico
mask_interest = correct & is_mutation & normal_in_test
# Seleziona tutti gli eventi in cui un paziente NORMALE viene classificato come MUTAZIONE
false_positive = (y_true_all == "NORMALE") & (y_pred_serio_all == "MUTAZIONE")
# Conta il numero totale di eventi falsi positivi nei 10 round
n_fp_events = int(false_positive.sum())
# Conta il numero di pazienti unici coinvolti in almeno 1 falso positivo
n_fp_patients = int(len(np.unique(idx_test_all[false_positive])))
# Stampa del conteggio predizioni
print("\nFalsi positivi (ovvero NORMALI classificati come MUTAZIONE):")
print(f"- Eventi (predizioni CV): {n_fp_events}")
print(f"- Pazienti unici: {n_fp_patients}")

# Conta gli eventi in cui una mutazione con marker normali è stata correttamente identificata
n_events = int(mask_interest.sum())
# Estrae gli indici unici dei pazienti identificati correttamente
idx_unique = np.unique(idx_test_all[mask_interest])
# Conta il numero di pazienti unici identificati correttamente
n_patients_unique = int(len(idx_unique))

''' print("\nMutazioni correttamente previste con 4 marcatori nella norma:")
print(f"- Eventi (predizioni CV): {n_events}")
print(f"- Pazienti unici: {n_patients_unique}")
'''
# Estrae dal dataset originale i pazienti mutati con marker normali che il modello ha identificato correttamente almeno una volta
df_flagged = df.iloc[idx_unique].copy()
print("\nCampioni (pazienti unici) individuati:")
# Stampa numero, genotipo e marker biochimici dei pazienti individuati
print(
    df_flagged[[id_col, "CARATTERIZZAZIONE\nALLELICA"] + feature]
    .sort_values(by=id_col)
    .to_string(index=False)
)
salva_tabella(
    df_flagged[[id_col, "CARATTERIZZAZIONE\nALLELICA"] + feature].sort_values(by=id_col),
    f"mutazioni_marker_in_range_identificate_{nome_modello}.csv"
)
print(f"Campioni nel dataset: {len(yAcc)}")
print(f"Totale predizioni: {len(y_true_all)}")
print(f"Predizioni medie per campione: {len(y_true_all) / len(yAcc):.1f}")

# Estraiamo i nomi delle classi ordinate
class_names = np.unique(yAcc)

# Calcola la matrice di confusione aggregata del modello serio scelto usando tutte le predizioni dei 10 round
conf_matrix_model = confusion_matrix(
    y_true=y_true_all,
    y_pred=y_pred_serio_all,
    labels=class_names
)

# Calcola la matrice di confusione aggregata del Dummy Classifier
conf_matrix_dummy = confusion_matrix(
    y_true=y_true_all,
    y_pred=y_pred_casuale_all,
    labels=class_names
)

# Calcola la matrice di confusione aggregata della baseline tradizionale A1AT < 90
conf_matrix_trad = confusion_matrix(
    y_true=y_true_all,
    y_pred=y_pred_trad_all,
    labels=class_names
)

# Calcola la matrice di confusione aggregata della baseline esplorativa
conf_matrix_expl = confusion_matrix(
    y_true=y_true_all,
    y_pred=y_pred_expl_all,
    labels=class_names
)

print("MATRICE DI CONFUSIONE - Baseline tradizionale A1AT<90:")
print(conf_matrix_trad)
print()

print("MATRICE DI CONFUSIONE - Baseline esplorativa A1AT:")
print(conf_matrix_expl)
print()

print(f"MATRICE DI CONFUSIONE - {nome_modello} (valori assoluti riferiti al numero totale di predizioni):")
print(conf_matrix_model)
print()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# Normalizza la matrice del modello per riga, ogni riga somma a 1 e mostra la distribuzione delle predizioni rispetto alla classe vera
conf_model_norm = conf_matrix_model.astype("float") / conf_matrix_model.sum(axis=1)[:, np.newaxis]
conf_dummy_norm = conf_matrix_dummy.astype("float") / conf_matrix_dummy.sum(axis=1)[:, np.newaxis]
# Senza normalizzazione la matrice mostrerebbe valori assoluti su 520 predizioni

# Visualizzazione titoli, colori e aggiustamento margini
disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_model_norm, display_labels=class_names)
disp1.plot(cmap="Blues", ax=axes[0], values_format=".2f")
for testo in disp1.text_.ravel():
    testo.set_fontsize(18)
axes[0].set_title(f"{nome_modello}\n(Matrice normalizzata per riga)", fontsize=12, fontweight="bold")

disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_dummy_norm, display_labels=class_names)
disp2.plot(cmap="Reds", ax=axes[1], values_format=".2f")
for testo in disp2.text_.ravel():
    testo.set_fontsize(18)
axes[1].set_title("Dummy Classifier (baseline)\n(Matrice normalizzata per riga)", fontsize=12, fontweight="bold")
for ax in axes:
    ax.set_xlabel("Classe Predetta", fontsize=10)
    ax.set_ylabel("Classe Vera", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.suptitle("Confronto Matrici di Confusione - Aggregazione 5x2cv (10 fold)",
             fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(cartella_output, f"confusion_matrix_{nome_modello}_5x2cv.png"), dpi=300, bbox_inches="tight")
plt.show()

#==========================================
# 5.b MATRICI DI CONFUSIONE DEI SINGOLI ROUND 5x2CV
#==========================================

# In questa sezione viene costruita una griglia con le matrici di confusione normalizzate di ciascun round del 5x2CV
# Serve a verificare la stabilità del modello sui singoli split

# Ogni matrice rappresenta un singolo test set da circa 52 pazienti
# La normalizzazione per riga permette di confrontare i round anche se le due classi hanno numerosità diversa
# Il valore diagonale di ciascuna riga corrisponde al recall della classe

# griglia 2 righe x 5 colonne: riga 0 = round A, riga 1 = round B
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
# Trasforma la griglia 2x5 in un array lineare di 10 assi
axes = axes.flatten()

# Lista che conterrà le etichette descrittive dei 10 round
round_labels = []
for i in range(5):
    # aggiungiamo l'etichetta del round A della ripetizione corrente
    round_labels.append(f"Rep {i+1} - Round A\n(train A / test B)")
    # aggiungiamo l'etichetta del round B della ripetizione corrente
    round_labels.append(f"Rep {i+1} - Round B\n(train B / test A)")

for k in range(10):
    # matrice di confusione del singolo round k (52 predizioni)
    cm_k = confusion_matrix(all_y_true[k], all_y_pred_serio[k], labels=class_names)
    # normalizza la matrice per riga dividendo la cella per il totale della classe vera corrispondente
    cm_k_norm = cm_k.astype("float") / cm_k.sum(axis=1)[:, np.newaxis]
    # Crea l'oggetto grafico per visualizzare la matrice normalizzata.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_k_norm, display_labels=class_names)
    # colorbar=False: evita 10 barre colore che affollano la figura
    disp.plot(cmap="Blues", ax=axes[k], values_format=".2f", colorbar=False)

    # Aumenta la dimensione dei numeri visualizzati dentro le celle
    for testo in disp.text_.ravel():
        testo.set_fontsize(20)

    # Calcola la balanced accuracy del modello nel singolo round k
    bal = balanced_accuracy_score(all_y_true[k], all_y_pred_serio[k])
    axes[k].set_title(f"{round_labels[k]}\nbal_acc={bal:.3f}", fontsize=15, fontweight="bold")
    axes[k].set_xlabel("Predetta", fontsize=12)
    axes[k].set_ylabel("Vera", fontsize=12)
    axes[k].tick_params(labelsize=12)

plt.suptitle(
    f"Matrici di Confusione dei 10 round 5x2CV\n"
    f"(N=520 eventi di predizione su 104 pazienti unici, "
    f"ogni paziente appare in media 5 volte nel test set)",
    fontsize=18, fontweight="bold", y=1.0
)
plt.tight_layout(rect=[0, 0, 1, 0.90], h_pad=3.5, w_pad=1.5)
fig.subplots_adjust(hspace=0.65)
plt.savefig(os.path.join(cartella_output, f"confusion_matrix_10fold_{nome_modello}.png"), dpi=300, bbox_inches="tight")
plt.show()

#==========================================
# 5.c MATRICI DI CONFUSIONE BASELINE A SOGLIA
#==========================================

# Normalizza le matrici per riga, come fatto per modello ML e Dummy
conf_trad_norm = conf_matrix_trad.astype("float") / conf_matrix_trad.sum(axis=1)[:, np.newaxis]
conf_expl_norm = conf_matrix_expl.astype("float") / conf_matrix_expl.sum(axis=1)[:, np.newaxis]

# Crea una figura con due matrici affiancate soglia tradizionale e soglia esplorativa
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Matrice della baseline tradizionale A1AT < 90
disp3 = ConfusionMatrixDisplay(
    confusion_matrix=conf_trad_norm,
    display_labels=class_names
)
disp3.plot(cmap="Greens", ax=axes[0], values_format=".2f")
for testo in disp3.text_.ravel():
    testo.set_fontsize(18)
axes[0].set_title(
    "Baseline tradizionale A1AT < 90\n(Matrice normalizzata per riga)",
    fontsize=12,
    fontweight="bold"
)

# Matrice della baseline esplorativa
disp4 = ConfusionMatrixDisplay(
    confusion_matrix=conf_expl_norm,
    display_labels=class_names
)
disp4.plot(cmap="Purples", ax=axes[1], values_format=".2f")
for testo in disp4.text_.ravel():
    testo.set_fontsize(18)
axes[1].set_title(
    "Baseline esplorativa A1AT\n(Matrice normalizzata per riga)",
    fontsize=12,
    fontweight="bold"
)

# Imposta etichette e rotazione dei nomi classe
for ax in axes:
    ax.set_xlabel("Classe Predetta", fontsize=10)
    ax.set_ylabel("Classe Vera", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.suptitle(
    "Confronto matrici di confusione con baseline a soglia\n"
    "Aggregazione 5x2CV (10 fold)",
    fontsize=13,
    fontweight="bold",
    y=0.98
)

plt.tight_layout()
plt.savefig(
    os.path.join(cartella_output, "confusion_matrix_baseline_soglie_5x2cv.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#==========================================
# 5.d MUTATI "NASCOSTI" e FN CON_VALORI_NELLA_NORMA
#==========================================

# Tra i pazienti MUTAZIONE con tutti i marker nella norma, quanti vengono comunque persi dal modello?

# In questa sezione si analizzano i pazienti realmente MUTAZIONE, con tutti i marker biochimici nel range,
# ma classificati dal modello come NORMALE.

# Seleziona gli eventi in cui:
# - la classe vera è MUTAZIONE
# - il modello predice NORMALE
# - tutti i marker biochimici sono nel range di normalità
fn_marker_in_range = (y_true_all == "MUTAZIONE") & (y_pred_serio_all == "NORMALE") & normal_in_test
# Conta il numero di eventi falsi negativi con marker in range, un paziente può contribuire più volte se viene sbagliato in più round
n_fn_marker_in_range_events = int(fn_marker_in_range.sum())
# Estrae gli indici unici dei pazienti coinvolti in questi falsi negativi
idx_fn_marker_in_range = np.unique(idx_test_all[fn_marker_in_range])
# Conta il numero di pazienti unici falsi negativi con marker in range
n_fn_marker_in_range_patients = int(len(idx_fn_marker_in_range))

# Se esistono pazienti falsi negativi con marker in range allora stampa il loro dettaglio
if n_fn_marker_in_range_patients > 0:
    df_fn = df.iloc[idx_fn_marker_in_range].copy()
    print("\nDettaglio pazienti sbagliati almeno 1 volta dal modello ma coi marcatori in range:")
    print(
        df_fn[[id_col, "CARATTERIZZAZIONE\nALLELICA"] + feature]
        .sort_values(by=id_col)
        .to_string(index=False)
    )
    salva_tabella(
        df_fn[[id_col, "CARATTERIZZAZIONE\nALLELICA"] + feature].sort_values(by=id_col),
        f"falsi_negativi_marker_in_range_{nome_modello}.csv"
    )

# conteggio per paziente unico (non per evento):
# un paziente è TP se identificato correttamente ALMENO UNA VOLTA nei 10 round
# un paziente è FN puro solo se NON è mai stato identificato in nessun round
idx_mut_marker_in_range = idx_test_all[mut_marker_in_range]
tp_set  = set(idx_test_all[mask_interest])   # identificati correttamente  almeno una volta
fn_puri = set(idx_mut_marker_in_range) - tp_set    # mai identificati correttamente  in nessun round

print("\n" + "="*70)
print("ANALISI MUTATI CON TUTTI I MARCATORI IN RANGE")
print("="*70)
print(f"- Pazienti realmente MUTAZIONE con tutti i marker in range: {n_mut_in_range_pat}")
print(f"- Identificati almeno una volta dal modello: {len(tp_set)}")
print(f"- Mai identificati dal modello: {len(fn_puri)}")
print(f"- Eventi TP tra i mutati con tutti i marker in range: {n_events}")
print(f"- Eventi FN tra i mutati con tutti i marker in range: {n_fn_marker_in_range_events}")
print(f"- Pazienti NORMALE con tutti i marker in range predetti MUTAZIONE almeno una volta: {n_fp_marker_in_range_pat}")

# Controlla che il totale dei mutati con marker in range sia uguale alla somma tra identificati almeno una volta e mai identificati
# Se non torna, interrompe lo script con un messaggio di errore
assert n_mut_in_range_pat == len(tp_set) + len(fn_puri), \
    "ATTENZIONE: errore logico nel calcolo TP/FN dei mutati con marker in range"

# mostra i pazienti FN puri se esistono e ne stampa i dettagli, mai identificati
if fn_puri:
    df_fn_puri = df.iloc[sorted(fn_puri)].copy()
    print("\nPazienti mai identificati (FN puri):")
    print(
        df_fn_puri[[id_col, "CARATTERIZZAZIONE\nALLELICA"] + feature]
        .sort_values(by=id_col)
        .to_string(index=False)
    )
    salva_tabella(
        df_fn_puri[[id_col, "CARATTERIZZAZIONE\nALLELICA"] + feature].sort_values(by=id_col),
        f"fn_puri_marker_in_range_{nome_modello}.csv"
    )


#==========================================
# 6. TRACCIAMENTO PER-PAZIENTE
#==========================================

# Quali pazienti vengono sbagliati? Quante volte?
# In quali round? Sono sempre sbagliati o solo in alcuni split?

# In questa sezione si ricostruisce lo storico delle predizioni per ogni paziente
# L'obiettivo è capire quali pazienti vengono classificati male, quante volte vengono sbagliati e in quali round del 5x2CV

# Crea un dizionario in cui ogni chiave è l'indice di un paziente
# Ogni valore è una lista di eventi di test associati a quel paziente e
# defaultdict(list) permette di aggiungere eventi senza inizializzare manualmente la lista
storico = defaultdict(list)

# Ricostruiamo lo storico dai dati già raccolti nel 5x2cv
# all_idx_test, all_y_true, all_y_pred_serio sono liste di 10 array
# (5 rep x 2 round), in ordine: rep1_roundA, rep1_roundB, rep2_roundA, ...

for fold_idx, (idx_fold, yt, yp) in enumerate(zip(all_idx_test, all_y_true, all_y_pred_serio)):
    # Itera sui 10 round del 5x2CV
    # Per ogni round recupera:
    # - idx_fold indici dei pazienti nel test set
    # - yt etichette vere
    # - yp predizioni del modello scelto
    rep    = (fold_idx // 2) + 1          # Ricava il numero della ripetizione, e siccome ogni ripetizione
                                          # ha 2 round si ha 0,0,1,1,2,2,... che poi diventa 1,1,2,2,...
    round_ = "A" if fold_idx % 2 == 0 else "B" # Ricava se il round è A o B facendo pari o dispari

    # Questo blocco tiene traccia di come è stato classificato ogni paziente in ogni round della cross-validation
    for paziente_idx, vero, predetto in zip(idx_fold, yt, yp):        # Itera sui pazienti del test set del round corrente
        storico[paziente_idx].append({
            "rep": rep,
            "round": round_,
            "vero": vero,
            "predetto": predetto,
            "corretto": vero == predetto
        })

# stampa intestazione errori per paziente
print("\n" + "="*70)
print(f"ANALISI ERRORI PER PAZIENTE — {nome_modello}")
print("="*70)
print(f"{'ID':<20} {'Classe vera':<12} {'Fold test':>10} "
      f"{'Corretti':>10} {'Errori':>8} {'Fold con errore'}")
print("-"*70)

# Lista che conterrà solo i pazienti sbagliati almeno una volta
pazienti_con_errori = []

for paz_idx in sorted(storico.keys()):  # Itera su tutti i pazienti presenti nello storico, ordinandoli per indice originale nel dataset
    eventi = storico[paz_idx]   # Recupera tutti gli eventi di test associati al paziente corrente
    n_fold   = len(eventi)    # Conta quante volte il paziente è comparso nei test set
    n_ok     = sum(1 for e in eventi if e["corretto"])    # Conta quante volte il paziente è stato classificato correttamente
    n_errori = sum(1 for e in eventi if not e["corretto"])    # Conta quante volte il paziente è stato classificato male

    if n_errori == 0:
        continue  # paziente sempre corretto, non ci interessa

    # Crea una lista con le ripetizioni/round in cui il paziente è stato sbagliato
    # ad esempio: rep1A, rep3B, rep5A
    fold_errori = [
        f"rep{e['rep']}{e['round']}"
        for e in eventi if not e["corretto"]
    ]
    # Recupera la classe vera del paziente
    classe_vera = eventi[0]["vero"]
    # Recupera l'ID paziente dal DataFrame originale usando l'indice
    id_paziente = df.iloc[paz_idx][id_col]

    # Salva un riepilogo del paziente sbagliato almeno una volta
    # Questa lista verrà usata per contare errori per classe e identificare pazienti difficili
    pazienti_con_errori.append({
        "idx": paz_idx,
        "id": id_paziente,
        "classe": classe_vera,
        "n_fold": n_fold,
        "n_ok": n_ok,
        "n_errori": n_errori,
        "fold_errori": fold_errori
    })
    # Stampa una riga della tabella per il paziente corrente
    # mostriamo quante volte è stato testato, quante volte corretto, quante volte sbagliato e in quali round
    print(f"{str(id_paziente):<20} {classe_vera:<12} {n_fold:>10} "
          f"{n_ok:>10} {n_errori:>8}   {', '.join(fold_errori)}")

print("-"*70)
print(f"Totale pazienti con almeno un errore: {len(pazienti_con_errori)}")

# Filtra i pazienti con errori appartenenti alla classe MUTAZIONE
err_mutazione = [p for p in pazienti_con_errori if p["classe"] == "MUTAZIONE"]
# Filtra i pazienti con errori appartenenti alla classe NORMALE
err_normale   = [p for p in pazienti_con_errori if p["classe"] == "NORMALE"]

print(f"  - MUTAZIONE classificati male almeno una volta: {len(err_mutazione)}")
print(f"  - NORMALE   classificati male almeno una volta: {len(err_normale)}")

# Pazienti "difficili" sbagliati in più della metà dei fold
print("\n" + "="*70)
print("PAZIENTI INSTABILI O SISTEMATICAMENTE MISCLASSIFICATI (errori > metà dei fold in cui appaiono):")
print("="*70)

# Seleziona i pazienti sbagliati in più della metà delle apparizioni nei test set
# Questi sono i casi più instabili o problematici per il modello
difficili = [p for p in pazienti_con_errori
             if p["n_errori"] > p["n_fold"] / 2]

if difficili:    # Se esistono pazienti difficili, stampa il dettaglio.
    for p in difficili:
        print(f"  {p['id']:<20} classe={p['classe']:<12} "
              f"errori={p['n_errori']}/{p['n_fold']}  "
              f"fold: {', '.join(p['fold_errori'])}")
else:
    print("Nessun paziente sbagliato in più della metà dei fold.")

df_pazienti_con_errori = pd.DataFrame(pazienti_con_errori)
if not df_pazienti_con_errori.empty:
    df_pazienti_con_errori["fold_errori"] = df_pazienti_con_errori["fold_errori"].apply(", ".join)
salva_tabella(df_pazienti_con_errori, f"pazienti_con_errori_{nome_modello}.csv")

df_pazienti_difficili = pd.DataFrame(difficili)
if not df_pazienti_difficili.empty:
    df_pazienti_difficili["fold_errori"] = df_pazienti_difficili["fold_errori"].apply(", ".join)
salva_tabella(df_pazienti_difficili, f"pazienti_difficili_{nome_modello}.csv")

