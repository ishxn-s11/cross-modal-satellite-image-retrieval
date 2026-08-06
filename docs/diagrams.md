# System diagrams

Mermaid diagrams (render natively on GitHub). They reflect the actual
implementation, not a target.

## System architecture

```mermaid
graph TB
    subgraph UI["User interfaces"]
        SPA["FastAPI SPA (web/index.html)"]
        ST["Streamlit UI (web/streamlit_app.py)"]
    end

    subgraph API["API layer"]
        FA["FastAPI (api/app.py) — /health /predict /retrieve /batch-retrieve /metrics /history /model-info"]
    end

    subgraph CORE["Core pipeline (src/)"]
        PREP["prepare_dataset — DatasetInterface + modality-aware preprocessing"]
        ENC["ModalityAdaptiveEncoder — adapters + ResNet/ViT + projection"]
        TRN["trainer — InfoNCE + SupCon + CE + geo/hard-negative losses"]
        ENG["RetrievalEngine — two-stage FAISS + re-ranker"]
        EVAL["evaluation — P/R/F1/mAP/NDCG, latency, scalability, baselines"]
        XAI["xai — Grad-CAM / ViT attention; embedding viz"]
    end

    subgraph PERSIST["Persistence"]
        E["EmbeddingStore (.npz)"]
        F["IndexStore (FAISS .index)"]
        S["MetadataStore (SQLite: images/galleries/embeddings/evaluation_results/model_versions)"]
    end

    subgraph DATA["Data"]
        SYN["Synthetic (offline)"]
        EU["EuroSAT"]
        S12["SEN12MS / So2Sat / BigEarthNet-MM (optional, user-downloaded)"]
    end

    SPA --> FA
    ST --> CORE
    FA --> CORE
    DATA --> PREP
    PREP --> ENC --> TRN
    TRN --> ENG --> EVAL
    ENG --> E
    ENG --> F
    EVAL --> S
    ENC --> XAI
```

## Retrieval pipeline

```mermaid
flowchart LR
    Q["Query image"] --> DET["modality detection / selection"]
    DET --> PRE["modality-specific preprocessing"]
    PRE --> EMB["encoder → shared embedding"]
    EMB --> S1["FAISS top-candidate_k (stage 1)"]
    S1 --> R["re-ranker (geo / MLP, optional)"]
    R --> S2["top-k results"]
    S2 --> M["similarity + metadata + geo distance"]
    M --> X["explainability (Grad-CAM / attention)"]
    M --> UI["UI: cards + map + analytics"]
```

## Data-flow

```mermaid
flowchart LR
    DS["DatasetInterface"] --> PATCH["patches {modality: (N,C,H,W)}"]
    DS --> MD["metadata: ImageMetadata[]"]
    PATCH --> STATS["normalization stats"]
    STATS --> TF["per-modality transforms (+ augmentation on train)"]
    TF --> DL["DataLoader"]
    DL --> LOSS["losses → gradients → optimizer"]
    LOSS --> CKPT["best_model.pt"]
    CKPT --> EMBS["embeddings (cached .npz)"]
    EMBS --> IDX["FAISS galleries (cached .index)"]
    IDX --> RET["retrieval results"]
    MD --> DB[(SQLite metadata)]
    MD --> RET
    RET --> EVALM["metrics + latency + benchmarks"]
```

## Database (ER)

```mermaid
erDiagram
    images ||--o{ retrieval_logs : queried_by
    images {
        int id PK
        text class_name
        text split
        text dataset
        real latitude
        real longitude
        text acquisition_date
    }
    galleries {
        int id PK
        text name UK
        text modality
        int num_vectors
        text index_path
        text config_hash
    }
    embeddings {
        int id PK
        text modality
        text config_hash
        int dim
        int n_vectors
    }
    evaluation_results {
        int id PK
        text config_hash
        text pair
        text kind
        int k
        text metric
        real value
    }
    model_versions {
        int id PK
        text name UK
        text config_hash
        text path
        text metrics
    }
    datasets {
        int id PK
        text name UK
        text sensor
        int n_images
    }
```

## Deployment

```mermaid
flowchart LR
    DEV["Host / Docker"] --> API["uvicorn api.app:app :8000"]
    API --> VOL["volumes: models/, data/, outputs/, embeddings/, faiss/, database/"]
    STAPP["streamlit run web/streamlit_app.py"] --> SRC["shared src/ pipeline"]
    API --> SRC
    subgraph Config
        Y["configs/*.yaml"] 
        ENV["RETRIEVAL_* env vars"]
    end
    Y --> API
    ENV --> API
```

## UI (Streamlit) wireframe

```mermaid
flowchart TB
    H["Home: dataset + system stats"]
    R["Retrieval: query id / upload, query modality, gallery modality, top-K, re-rank"]
    RS["Results: query + ranked cards (similarity, sensor, land cover, date, distance, time)"]
    M["Map: query + retrieved locations"]
    A["Analytics: F1@5/10, P/R, per-pair table, latency"]
    E["Embeddings: PCA / t-SNE / UMAP by class or modality"]
    X["Explainability: Grad-CAM / ViT attention overlay"]
    G["Gallery: browse images + metadata"]
    HST["History: recent retrievals (SQLite)"]
    H --> R --> RS
    RS --> M
    RS --> X
    H --> A
    H --> E
    H --> G
    H --> HST
```
