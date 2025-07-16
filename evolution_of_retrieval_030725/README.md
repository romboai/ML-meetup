# Evoluzione dei Sistemi di Retrieval: Da TF-IDF a BGE-M3

Questo progetto dimostra l'evoluzione dei sistemi di information retrieval attraverso l'implementazione e il confronto di diverse tecniche, da metodi classici come TF-IDF e BM25 fino ai moderni embedding contestuali come BGE-M3.

## 📋 Panoramica del Progetto

Il progetto analizza e confronta diversi approcci per il retrieval di informazioni su un dataset di domande in lingua sarda basato su Wikipedia. L'obiettivo è mostrare come le tecniche di retrieval sono evolute nel tempo e come i modelli moderni di embedding contestuali hanno migliorato significativamente le prestazioni.

### Metodologie Implementate

1. **Bag-of-Words con TF-IDF / BM25** - Metodi classici basati su frequenza
2. **Word Embeddings statici** - FastText per rappresentazioni vettoriali di parole
3. **BERT ed embeddings contestuali** - Sentence-BERT per embeddings contestuali
4. **Dense Retrieval (DPR)** - Retrieval denso con modelli pre-addestrati
5. **BGE-M3 (2024)** - Modello multilingue avanzato con embedding Matryoshka
6. **RAG – Retrieval Augmented Generation** - Sistema completo di generazione di risposte

## 🗂️ Struttura del Progetto

```
evolution_of_retrieval_030725/
├── evolution_of_retrieval.ipynb    # Notebook principale con tutti gli esperimenti
├── data_io.py                      # Funzioni per caricamento dati
├── metrics.py                      # Metriche di valutazione (Recall@k, MRR, Precision@k)
├── download_wiki.py                # Script per scaricare pagine Wikipedia
├── extract_para.py                 # Estrazione paragrafi da HTML Wikipedia
├── nq_sc_extract.py                # Elaborazione dataset Natural Questions
├── translate_questions.py          # Traduzione domande
├── requirements.txt                # Dipendenze Python
├── img/                           # Immagini per il notebook
└── README.md                      # Questo file
```

## 📊 Dataset

### Input
- **paragraphs.jsonl**: Corpus di paragrafi estratti da Wikipedia in sardo
  - Formato: JSONL con campi `paragraph_id`, `lang`, `page_title`, `text`
  - Dimensione: ~14,049 paragrafi
  
- **nq_sc_sc.csv**: Dataset di domande Natural Questions tradotte in sardo
  - Formato: CSV con domande e URL di riferimento
  - Dimensione: ~2,677 domande

### Output Attesi
- **Metriche di valutazione** per ogni metodo:
  - **Recall@k**: Frazione di query con almeno una risposta corretta nei top-k risultati
  - **MRR (Mean Reciprocal Rank)**: Media dei ranghi reciproci delle risposte corrette
  - **Precision@k**: Frazione di documenti recuperati che sono corretti
  - **Tempo di query**: Tempo medio per elaborare una query

## 🚀 Installazione e Setup

### Prerequisiti
- Python 3.8+
- CUDA-compatible GPU (raccomandato per modelli BGE-M3)
- ~8GB RAM
- ~10GB spazio disco

### Installazione

1. **Clona il repository e naviga nella directory:**
```bash
cd evolution_of_retrieval_030725
```

2. **Installa le dipendenze:**
```bash
pip install -r requirements.txt
```

3. **Scarica i dati NLTK necessari:**
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

## 📈 Risultati delle Prestazioni

| Metodo     | Recall@10 | MRR     | Δ Recall vs BM25 | ms/query |
|------------|-----------|---------|------------------|----------|
| TF-IDF     | 0.547     | 0.406   | – 0.023          | 3.9      |
| BM25       | 0.570     | 0.439   | Baseline         | 10.0     |
| FastText   | 0.263     | 0.210   | – 0.307          | 4.2      |
| SBERT      | 0.511     | 0.403   | – 0.059          | 19.2     |
| **BGE-M3** | **0.854** | **0.713** | **+0.284**      | 84.6     |

### Osservazioni Chiave
- **BGE-M3** mostra un miglioramento significativo (+28.4% in Recall@10)
- I metodi classici (TF-IDF, BM25) sono più veloci ma meno accurati
- Gli embedding contestuali moderni offrono il miglior trade-off precisione/velocità

## 🔧 Utilizzo

### 1. Esecuzione del Notebook Principale
```bash
jupyter notebook evolution_of_retrieval.ipynb
```

Il notebook contiene sezioni per ogni metodologia con:
- Implementazione del metodo
- Valutazione delle prestazioni
- Confronto con altri approcci

### 2. Preparazione dei Dati (se necessario)

**Scaricamento pagine Wikipedia:**
```bash
python download_wiki.py nq_sc.csv --output corpus --workers 16
```

**Estrazione paragrafi:**
```bash
python extract_para.py --root corpus --langs sc -o paragraphs.jsonl --jobs 8
```

### 3. Valutazione di un Metodo Personalizzato

```python
from data_io import load_paragraphs, load_questions
from metrics import eval_retriever

# Carica dati
df_para = load_paragraphs()
df_q_sc = load_questions()

# Definisci funzione di retrieval
def my_retriever(query: str, k: int = 10) -> list[str]:
    # Implementa la tua logica di retrieval
    pass

# Valuta
scores = eval_retriever(my_retriever, df_q_sc, k=10)
print(f"Recall@10: {scores['recall@k']:.3f}")
print(f"MRR: {scores['mrr']:.3f}")
```

## 🎯 Funzionalità Principali

### 1. Preprocessing del Testo
- Tokenizzazione con NLTK
- Rimozione stopwords italiane
- Normalizzazione (lowercase, rimozione caratteri non alfabetici)

### 2. Metodi di Retrieval Implementati

#### TF-IDF
- Vettorizzazione con scikit-learn
- Similarità coseno
- Normalizzazione L2

#### BM25
- Implementazione con rank-bm25
- Parametri ottimizzati per il dataset

#### FastText
- Embeddings pre-addestrati
- Similarità coseno su vettori medi

#### Sentence-BERT
- Modello `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings contestuali per frasi

#### BGE-M3
- Modello multilingue avanzato
- Embedding Matryoshka per compressione
- Indicizzazione FAISS per ricerca veloce

### 3. Sistema RAG Completo
- Retrieval con BGE-M3
- Reranking con FlagReranker
- Generazione risposte con GPT-4

## 📊 Metriche di Valutazione

### Recall@k
Frazione di query dove almeno una risposta corretta appare nei top-k risultati.

### Mean Reciprocal Rank (MRR)
Media dei ranghi reciproci delle prime risposte corrette. Formula:
```
MRR = (1/rank_1 + 1/rank_2 + ... + 1/rank_n) / n
```

### Precision@k
Frazione di documenti recuperati che sono effettivamente corretti.

## 🔍 Analisi dei Risultati

### Vantaggi dei Metodi Moderni
1. **Capacità semantica**: Comprensione del significato oltre le parole esatte
2. **Robustezza multilingue**: BGE-M3 gestisce bene lingue minoritarie come il sardo
3. **Scalabilità**: Indicizzazione FAISS per grandi corpus

### Trade-off Prestazioni
- **Velocità**: TF-IDF/BM25 > FastText > SBERT > BGE-M3
- **Accuratezza**: BGE-M3 > SBERT > BM25 > TF-IDF > FastText

## 🛠️ Personalizzazione

### Aggiungere un Nuovo Metodo
1. Implementa la funzione di retrieval
2. Aggiungi la valutazione nel notebook
3. Confronta con i metodi esistenti

### Modificare i Parametri
- **BM25**: Parametri `k1` e `b`
- **BGE-M3**: Dimensioni embedding, batch size
- **FAISS**: Numero di probe per la ricerca

## 📝 Note Tecniche

### Requisiti Hardware
- **CPU**: Minimo 4 core per preprocessing
- **RAM**: 8GB+ per dataset completo
- **GPU**: NVIDIA GPU con 8GB+ VRAM per BGE-M3

### Ottimizzazioni
- Indicizzazione FAISS per ricerca veloce
- Batch processing per embedding
- Caching dei risultati intermedi

## 🤝 Contributi

Per contribuire al progetto:
1. Fork del repository
2. Crea un branch per la feature
3. Implementa e testa le modifiche
4. Submit una pull request

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT.

## 📞 Contatti

Per domande o supporto, apri una issue su GitHub.

---

**Nota**: Questo progetto è stato sviluppato per scopi educativi e di ricerca nell'ambito dell'information retrieval e del natural language processing. 