# Riduzione dei falsi negativi nella diagnosi di AATD mediante machine learning applicato a marcatori biochimici di routine

Questo repository contiene il codice sorgente, il flusso di preparazione dei dati, le analisi statistiche e le figure sviluppate nell’ambito di un progetto di tesi dedicato al **supporto mediante machine learning alla diagnosi della carenza di alfa-1-antitripsina (AATD)**.

Lo studio valuta se quattro marcatori biochimici di routine possano contribuire a distinguere i pazienti con genotipo normale **MM** dai pazienti portatori di almeno una mutazione del gene **SERPINA1**, con particolare attenzione alla riduzione dei falsi negativi diagnostici.

> **Importante:** questo progetto costituisce un prototipo di ricerca e non rappresenta uno strumento diagnostico clinico. Il test genetico rimane il metodo di riferimento per confermare la presenza di varianti del gene SERPINA1.

---

## Obiettivo del progetto

L’attuale percorso diagnostico di primo livello si basa principalmente sulla concentrazione sierica di alfa-1-antitripsina. L’utilizzo di una soglia fissa può tuttavia non identificare alcuni pazienti il cui profilo biochimico appare normale nonostante la presenza di una variante genetica.

Questo progetto valuta se l’analisi combinata di quattro marcatori biochimici di routine possa migliorare l’individuazione dei pazienti portatori di mutazioni:

- **ALFA1%** — percentuale della frazione alfa-1 nell’elettroforesi delle proteine sieriche
- **ALFA1#** — concentrazione assoluta della frazione alfa-1
- **A1AT** — concentrazione sierica di alfa-1-antitripsina
- **PCR / CRP** — proteina C-reattiva, utilizzata per contestualizzare lo stato infiammatorio

Il target di classificazione binaria è definito nel seguente modo:

- `NORMALE`: genotipo MM
- `MUTAZIONE`: qualsiasi altro genotipo SERPINA1 osservato nel dataset

---

## Dataset

Il dataset contiene **104 pazienti**:

| Classe | Numero di pazienti |
|---|---:|
| MM / normale | 68 |
| Mutazione SERPINA1 | 36 |
| **Totale** | **104** |

La classe MUTAZIONE comprende varianti quali S, Z, Mmalton, S/Mmalton, Mmalton/Mmalton e SZ.

### Protezione dei dati

Devono essere pubblicati esclusivamente dati completamente anonimizzati o correttamente de-identificati. Prima di mantenere il file Excel originale in un repository pubblico, è necessario verificare che non contenga nomi, date di nascita, numeri di cartella clinica, recapiti o altre informazioni che possano consentire l’identificazione dei pazienti.

---

## Modelli

Sono stati valutati i seguenti classificatori supervisionati:

- Random Forest
- Support Vector Classifier con kernel RBF
- Logistic Regression
- Linear Support Vector Classifier

I modelli sono stati confrontati con:

- Dummy Classifier
- Soglia clinica tradizionale: `A1AT < 90 mg/dL`
- Soglia esplorativa su A1AT ottimizzata esclusivamente sul training fold

Il `MinMaxScaler` viene applicato all’interno di ciascuna pipeline di machine learning, in modo da evitare che le informazioni provenienti dai test fold influenzino la fase di normalizzazione.

---

## Validazione e analisi statistica

La valutazione principale utilizza una procedura di **5×2 cross-validation**:

1. Il dataset viene suddiviso in due metà stratificate, indicate come A e B.
2. Il modello viene addestrato su A e valutato su B.
3. Il modello viene addestrato su B e valutato su A.
4. La procedura viene ripetuta cinque volte, producendo complessivamente dieci round di test.

Le principali metriche di valutazione sono:

- Accuracy
- Balanced accuracy
- F1 macro
- F1 weighted
- Matthews Correlation Coefficient
- Matrici di confusione
- Recall della classe MUTAZIONE
- Analisi dei falsi negativi

I confronti statistici includono:

- Paired t-test 5×2CV di Dietterich
- Combined F-test 5×2CV di Alpaydin
- Verifica indipendente mediante la libreria `mlxtend`

Il repository comprende inoltre il tuning degli iperparametri e un ablation study, nel quale viene rimosso un marcatore biochimico alla volta per valutarne il contributo alle prestazioni dei classificatori.

---

## Risultati principali

Prestazioni medie ottenute mediante il protocollo 5×2CV:

| Metodo | Accuracy | Balanced accuracy | MCC |
|---|---:|---:|---:|
| Dummy | 0,654 ± 0,000 | 0,500 ± 0,000 | 0,000 ± 0,000 |
| Soglia tradizionale A1AT | 0,875 ± 0,030 | 0,833 ± 0,043 | 0,723 ± 0,069 |
| Soglia esplorativa A1AT | 0,902 ± 0,025 | 0,905 ± 0,024 | 0,800 ± 0,043 |
| Random Forest | 0,912 ± 0,034 | 0,904 ± 0,047 | 0,811 ± 0,075 |
| SVC RBF | **0,921 ± 0,029** | **0,932 ± 0,027** | **0,840 ± 0,057** |
| Logistic Regression | 0,910 ± 0,037 | 0,915 ± 0,035 | 0,814 ± 0,073 |
| LinearSVC | 0,912 ± 0,034 | 0,919 ± 0,034 | 0,819 ± 0,070 |

SVC con kernel RBF ha ottenuto la balanced accuracy media più elevata. L’ablation study ha mostrato che la rimozione di **A1AT** determina il maggiore calo delle prestazioni, confermando il contributo dominante di questo marcatore all’interno del pannello di quattro variabili.

I risultati sono riferiti esclusivamente al dataset e al protocollo di validazione utilizzati nel presente studio. Prima di qualsiasi possibile applicazione clinica sono necessarie una validazione esterna, prospettica e possibilmente multicentrica.

---

## Contenuto del repository

```text
.
├── test11.6 ss.py
├── test11.6 ss tuning.py
├── test11.6 ss ablation.py
├── test11.6 ss creazione file csv.py
├── test11.6 ss grafici per cap 1.py
├── test11.6 ss grafici per cap 3.py
├── test11.6 ss grafici per cap 4.py
├── results.zip
├── LICENSE
└── README.md
```

### Descrizione degli script

| File | Funzione |
|---|---|
| `test11.6 ss.py` | Esperimento principale mediante 5×2CV, baseline, metriche, test statistici, matrici di confusione e analisi a livello di singolo paziente |
| `test11.6 ss tuning.py` | Tuning degli iperparametri di Random Forest, SVC RBF, Logistic Regression e LinearSVC |
| `test11.6 ss ablation.py` | Ablation study ottenuto rimuovendo un marcatore alla volta |
| `test11.6 ss creazione file csv.py` | Preparazione del dataset e generazione del file CSV |
| `test11.6 ss grafici per cap 1.py` | Generazione delle figure utilizzate nel capitolo introduttivo |
| `test11.6 ss grafici per cap 3.py` | Generazione delle figure relative all’analisi esplorativa dei dati |
| `test11.6 ss grafici per cap 4.py` | Generazione delle figure relative alla metodologia |
| `results.zip` | Tabelle, figure e risultati prodotti dagli esperimenti |

---

# Reducing False Negatives in AATD Diagnosis Using Machine Learning on Routine Biochemical Markers

This repository contains the code, data-processing workflow, statistical analyses, and figures developed for a thesis project on **machine-learning support for the diagnosis of alpha-1 antitrypsin deficiency (AATD)**.

The study investigates whether four routine biochemical markers can help distinguish patients with a normal **MM** genotype from patients carrying at least one **SERPINA1** mutation, with particular attention to reducing diagnostic false negatives.

> **Important:** this project is a research prototype and is not a clinical diagnostic tool. Genetic testing remains the reference method for confirming SERPINA1 variants.

---

## Project objective

The current first-level diagnostic pathway often relies heavily on serum alpha-1 antitrypsin concentration. A fixed threshold may fail to identify patients whose biochemical profile appears normal despite the presence of a pathogenic variant.

This project evaluates whether the combined analysis of four routine markers can improve mutation screening:

- **ALFA1%** — percentage of the alpha-1 fraction in serum protein electrophoresis
- **ALFA1#** — absolute concentration of the alpha-1 fraction
- **A1AT** — serum alpha-1 antitrypsin concentration
- **CRP / PCR** — C-reactive protein, used to contextualize inflammatory status

The binary classification target is:

- `NORMALE`: MM genotype
- `MUTAZIONE`: any other observed SERPINA1 genotype

---

## Dataset

The dataset contains **104 patients**:

| Class | Patients |
|---|---:|
| MM / normal | 68 |
| SERPINA1 mutation | 36 |
| **Total** | **104** |

The mutation class includes variants such as S, Z, Mmalton, S/Mmalton, Mmalton/Mmalton, and SZ.

### Data privacy

Only fully anonymized or properly de-identified data should be published. Before keeping the original Excel file in a public repository, verify that it contains no names, dates of birth, medical record numbers, contact details, or other identifying information.

---

## Models

The following supervised classifiers are evaluated:

- Random Forest
- Support Vector Classifier with RBF kernel
- Logistic Regression
- Linear Support Vector Classifier

The models are compared with:

- Dummy Classifier
- Traditional clinical threshold: `A1AT < 90 mg/dL`
- Exploratory A1AT threshold optimized on the training fold

`MinMaxScaler` is applied inside each machine-learning pipeline to avoid leakage from the test folds.

---

## Validation and statistical analysis

The main evaluation uses **5×2 cross-validation**:

1. The dataset is divided into two stratified halves, A and B.
2. The model is trained on A and tested on B.
3. The model is trained on B and tested on A.
4. The procedure is repeated five times, producing ten test rounds.

Main evaluation metrics:

- Accuracy
- Balanced accuracy
- F1 macro
- F1 weighted
- Matthews correlation coefficient
- Confusion matrices
- Mutation recall and false-negative analysis

Statistical comparisons include:

- Dietterich paired 5×2CV t-test
- Alpaydin combined 5×2CV F-test
- Independent verification with `mlxtend`

The repository also includes hyperparameter tuning and an ablation study in which one biochemical marker is removed at a time.

---

## Main results

Mean performance over the 5×2CV protocol:

| Method | Accuracy | Balanced accuracy | MCC |
|---|---:|---:|---:|
| Dummy | 0.654 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 |
| Traditional A1AT threshold | 0.875 ± 0.030 | 0.833 ± 0.043 | 0.723 ± 0.069 |
| Exploratory A1AT threshold | 0.902 ± 0.025 | 0.905 ± 0.024 | 0.800 ± 0.043 |
| Random Forest | 0.912 ± 0.034 | 0.904 ± 0.047 | 0.811 ± 0.075 |
| SVC RBF | **0.921 ± 0.029** | **0.932 ± 0.027** | **0.840 ± 0.057** |
| Logistic Regression | 0.910 ± 0.037 | 0.915 ± 0.035 | 0.814 ± 0.073 |
| LinearSVC | 0.912 ± 0.034 | 0.919 ± 0.034 | 0.819 ± 0.070 |

SVC RBF achieved the highest mean balanced accuracy. The ablation study showed the largest performance decrease when **A1AT** was removed, confirming its dominant contribution within the four-marker panel.

These results refer only to this dataset and validation design. External and prospective validation is required before any clinical application.

---

## Repository contents

```text
.
├── test11.6 ss.py
├── test11.6 ss tuning.py
├── test11.6 ss ablation.py
├── test11.6 ss creazione file csv.py
├── test11.6 ss grafici per cap 1.py
├── test11.6 ss grafici per cap 3.py
├── test11.6 ss grafici per cap 4.py
├── results.zip
├── LICENSE
└── README.md
```

### Script overview

| File | Purpose |
|---|---|
| `test11.6 ss.py` | Main 5×2CV experiment, baselines, metrics, statistical tests, confusion matrices, and patient-level analysis |
| `test11.6 ss tuning.py` | Hyperparameter tuning for Random Forest, SVC RBF, Logistic Regression, and LinearSVC |
| `test11.6 ss ablation.py` | Ablation study obtained by removing one marker at a time |
| `test11.6 ss creazione file csv.py` | Dataset preparation and CSV generation |
| `test11.6 ss grafici per cap 1.py` | Figures used in the introductory chapter |
| `test11.6 ss grafici per cap 3.py` | Exploratory-data-analysis figures |
| `test11.6 ss grafici per cap 4.py` | Methodology-related figures |
| `results.zip` | Generated tables, figures, and experiment outputs |
