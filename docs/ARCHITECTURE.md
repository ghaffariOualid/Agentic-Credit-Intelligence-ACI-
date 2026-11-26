# Architecture Technique — Agentic Credit Intelligence (ACI)

> **Périmètre métier** : plateforme distribuée d'assistance à l'analyse financière des entreprises, **sans décision automatique de crédit**. Chaque valeur financière et chaque constat de risque est relié à une preuve documentaire vérifiable, tous les calculs sont déterministes et la validation finale appartient à l'analyste (HITL).

---

## 1. Architecture Globale Hybride (Local + Google Colab 1× NVIDIA T4)

ACI adopte une architecture modulaire permettant d'exécuter l'applicatif et l'orchestration localement tout en déportant l'inférence LLM sur un **GPU NVIDIA Tesla T4 (16 Go VRAM)** via **vLLM** et un tunnel HTTPS Cloudflare, avec bascule automatique sur un fallback local structuré hors ligne.

```mermaid
flowchart TB
    subgraph UI["Portail Analyste (Next.js 15 · Port 3001)"]
        V1["Vue générale & KPI"]
        V2["Dossiers de crédit + Upload PDF/Excel + HITL"]
        V3["Exécutions agents (Graphe & traces)"]
        V4["Routeur intelligent (Explicabilité)"]
        V5["Studio RL & Infrastructure GPU"]
    end

    subgraph API["Backend FastAPI (Monolithe Modulaire · Port 8000)"]
        GW_API["API Métier (/api/v1/cases, /runs, /report/pdf)"]
        RT["Runtime Multi-Agents (LangGraph + MemorySaver)"]
        DE["Decision Engine (Supervisé / Bandit / PPO)"]
        INF["Inference Gateway (OpenAI /v1/chat/completions)"]
    end

    subgraph AGENTS["Agents & Fonctions Déterministes"]
        DOC["Document Parser (PDF / XLSX / TXT)"]
        FIN["Financial Calculator (Ratios F03)"]
        VER["Verification Guardrails (F05)"]
        PDF["PDF Exporter (ReportLab F06)"]
    end

    subgraph GPU["Serving LLM (Google Colab · 1x NVIDIA T4 16 Go)"]
        TUN["Cloudflare Quick Tunnel (HTTPS)"]
        VLLM["vLLM (--dtype half --quantization awq)"]
        M1["Small: Qwen2.5-3B-AWQ"]
        M2["Medium: Qwen2.5-7B-AWQ"]
        M3["Large: Qwen2.5-14B-AWQ"]
    end

    UI -->|REST & SSE| GW_API
    GW_API --> RT
    GW_API --> DE
    RT --> AGENTS
    RT --> DE
    RT --> INF
    INF -->|HTTPS + Bearer Key| TUN
    TUN --> VLLM
    VLLM --- M1
    VLLM --- M2
    VLLM --- M3
    INF -.->|Mode auto hors ligne| MOCK["Fallback local structuré"]
```

---

## 2. Workflow Multi-Agents LangGraph & Contrôle Humain (HITL)

Le graphe d'exécution garantit que **le routeur IA ne contourne jamais les calculs financiers ni les contrôles bloquants**. Lorsqu'une incohérence documentaire (ex. écart entre le bilan et `Dettes.xlsx`) ou une confiance d'extraction insuffisante est détectée, `Verification Agent` suspend l'exécution via `langgraph.types.interrupt()`.

```mermaid
stateDiagram-v2
    [*] --> DocumentAgent: Import PDF / Excel / TXT (F01-F02)
    DocumentAgent --> FinancialAgent: Contrat FinancialStatement validé
    FinancialAgent --> RiskComplianceAgent: Ratios & rapprochement déterministes (F03)
    RiskComplianceAgent --> VerificationAgent: Constats sourcés + synthèse LLM (F04)

    VerificationAgent --> HITL_Pause: Anomalie ou preuve faible (F05)
    VerificationAgent --> ReportAgent: Tous les contrôles valides

    state HITL_Pause {
        [*] --> AttenteAnalyste: Checkpoint MemorySaver persisté
        AttenteAnalyste --> CorrectionHumaine: POST /api/v1/runs/{id}/resume (F07)
        CorrectionHumaine --> RecalculDeterministe: Application des field_overrides
    }

    HITL_Pause --> ReportAgent: Reprise validée + audit journalisé
    ReportAgent --> [*]: CreditReport v1 + Export PDF (F06)
```

---

## 3. Séquence d'Exécution de Bout en Bout

Ce diagramme détaille les échanges temps réel entre l'analyste, le portail Next.js, le runtime LangGraph, le Decision Engine et le serveur vLLM sur Colab T4.

```mermaid
sequenceDiagram
    actor Analyste
    participant Portal as Portail Next.js
    participant API as FastAPI + LangGraph
    participant Router as Decision Engine
    participant Gateway as Inference Gateway
    participant T4 as vLLM (Colab T4)

    Analyste->>Portal: Importer Bilan.pdf & Dettes.xlsx
    Portal->>API: POST /api/v1/cases/upload
    API-->>Portal: Dossier créé (fichiers isolés)

    Analyste->>Portal: Cliquer sur Lancer l'analyse
    Portal->>API: POST /api/v1/cases/{id}/analyze
    API->>API: Document Agent + Financial Agent (déterministe)
    API->>Router: predict(RouterState: complexité, VRAM, budget)
    Router-->>API: RouterDecision (tier=medium, contraintes appliquées)
    API->>Gateway: POST /v1/chat/completions (model=medium)
    Gateway->>T4: Requête OpenAI-compatible (Qwen2.5-7B-AWQ)
    T4-->>Gateway: Synthèse factuelle structurée + usage tokens
    Gateway-->>API: Réponse + aci_meta (latence, backend)

    API->>API: Verification Agent (F05)
    alt Incohérence détectée (HITL requis)
        API-->>Portal: status=interrupted + motif de suspension
        Analyste->>Portal: Saisir correction / validation + commentaire
        Portal->>API: POST /api/v1/runs/{id}/resume (HITLDecision)
        API->>API: Recalcul F03 + Journalisation F07 + Report Agent
    end
    API-->>Portal: status=completed + CreditReport v1
    Analyste->>Portal: Télécharger le rapport PDF
    Portal->>API: GET /api/v1/cases/{id}/report/pdf
    API-->>Analyste: ACI_Rapport_{case_id}_v1.pdf
```

---

## 4. Moteur de Décision Entraînable (PyTorch Supervisé, Bandit LinUCB & RL PPO)

Le moteur de décision apprend à optimiser le compromis **qualité / coût GPU / latence** sur le profil matériel **NVIDIA T4 (16 Go)** sans jamais décider de l'octroi de crédit.

- **Vecteur d'état normalisé (`STATE_DIM = 12`)** : type de tâche (one-hot × 4), taille du contexte, complexité estimée, charge GPU, VRAM T4, profondeur de file, budget restant, index d'étape, confiance précédente.
- **Masquage des actions interdites** :
  - `small` masqué si `complexité >= 0,80` sur tâche critique (`financial_extraction`, `risk_analysis`).
  - `large` masqué si `VRAM >= 90 %` ou `budget < 0,05 $`.
  - Contrôles documentaires toujours obligatoires.
- **Fonction de récompense** : `R = 0,60·Q − 0,15·C − 0,15·L − 0,10·E`.

```mermaid
flowchart LR
    S["État s (12 features normalisées)"] --> M["Masque de sécurité (compute_action_mask)"]
    M --> POL

    subgraph POL["Politiques Entraînables"]
        SUP["Supervisé (PyTorch MLP)"]
        BND["Bandit Contextuel (LinUCB)"]
        PPO["RL Séquentiel (PyTorch PPO + GAE)"]
    end

    POL --> ENV["Simulateur Gymnasium T4 (4 étapes / dossier)"]
    ENV -->|"Transition (s, a, r, s', done)"| LOG["Contrat RLTransition"]
    LOG --> BENCH["Benchmark Multi-Seeds (Static vs Bandit vs MLP vs PPO)"]
    POL --> ART["Artefacts versionnés (policy-*.pt)"]
    ART -->|Activation contrôlée| RT_LIVE["Runtime LangGraph en direct"]
```

---

## 5. Modèle de Données & Contrats Pydantic

Tous les agents manipulent des contrats strictement validés par Pydantic afin d'empêcher les erreurs d'unités (`UNIT` / `THOUSAND` / `MILLION`), les mélanges de devises ou les constats sans preuve.

```mermaid
erDiagram
    CREDIT_CASE ||--o{ DOCUMENT_REF : contient
    CREDIT_CASE ||--|| FINANCIAL_STATEMENT : possede
    FINANCIAL_STATEMENT ||--o{ SOURCED_AMOUNT : compose_de
    SOURCED_AMOUNT ||--|| EVIDENCE_REF : cite
    CREDIT_CASE ||--o{ RUN_RECORD : execute
    RUN_RECORD ||--|| FINANCIAL_METRICS : calcule
    RUN_RECORD ||--o{ RISK_FINDING : identifie
    RISK_FINDING }o--|{ EVIDENCE_REF : prouve_par
    RUN_RECORD ||--|| VERIFICATION_RESULT : controle
    RUN_RECORD ||--o{ HITL_DECISION : valide_par
    RUN_RECORD ||--o| CREDIT_REPORT : produit
    RUN_RECORD ||--o{ RL_TRANSITION : journalise
```

---

## 6. Organisation du Dépôt

| Répertoire | Responsabilité |
| :--- | :--- |
| `apps/analyst-portal/` | Portail Next.js 15 (6 vues, bannière tapis marocain, upload PDF/Excel, export PDF) |
| `services/contracts/` | Contrats Pydantic partagés (`domain.py`, `intelligence.py`, `inference.py`) |
| `services/api/` | API FastAPI métier (`/cases`, `/upload`, `/analyze`, `/runs`, `/events` SSE, `/resume`, `/report/pdf`) |
| `services/runtime/` | Orchestration LangGraph (`StateGraph`, `MemorySaver`, `interrupt()`, traces) |
| `services/router/` | Service Decision Engine (`/router/predict`, `/training/jobs`) |
| `services/inference/` | Inference Gateway OpenAI-compatible (`/v1/chat/completions`, quotas, retry, fallback) |
| `agents/document/` | Parseur multi-formats (`.pdf`, `.xlsx`, `.txt`, `.csv`) et extraction sourcée |
| `agents/financial/` | Moteur de calculs financiers déterministes (ratios, DSCR, rapprochement dettes) |
| `agents/report/` | Générateur PDF bancaire versionné (ReportLab) |
| `intelligence/decision_model/` | Encodage d'état, masquage d'actions, MLP supervisé PyTorch, Bandit LinUCB |
| `intelligence/reinforcement_learning/` | Simulateur Gymnasium calibré T4 et entraîneur PPO PyTorch |
| `datasets/synthetic/` | Générateur reproductible de dossiers annotés & vérité terrain |
| `benchmarks/` | Protocole expérimental multi-seeds (`Static` vs `Bandit` vs `Supervised` vs `PPO`) |
| `infrastructure/colab/` | Notebook Google Colab prêt à l'emploi pour vLLM sur 1× NVIDIA T4 (16 Go) |
| `tests/` | Suite complète `pytest` (26 tests unitaires et d'intégration) |
