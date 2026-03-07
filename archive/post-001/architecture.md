# Flex API Architecture

This document describes the architecture of the Flex API, which analyzes Steam gaming profiles to generate personality insights and clustering signatures.

## Overview

The Flex API is a Flask-based REST API that:
- Retrieves user gaming achievements from a PostgreSQL database
- Clusters users into personality signatures using ML pipelines
- Provides OpenAPI/Swagger documentation via flask-smorest

**Production URL**: https://flex-beta-engine.onrender.com

## Directory Structure

```
api/
├── server.py                 # Application entry point
├── api/                      # Flask application package
│   ├── schemas.py            # Marshmallow request/response schemas
│   ├── __init__.py           # App factory (create_app)
│   ├── middleware/
│   │   └── auth.py           # Authentication middleware
│   └── routes/
│       ├── __init__.py
│       ├── achievements.py   # GET /user/achievements/{steam_id}
│       ├── signatures.py     # GET /user/signatures/{steam_id}
│       └── docs.py           # GET / (landing page)
├── services/
│   ├── __init__.py
│   └── achievements_service.py  # Database queries, API logging
├── ml/                       # Machine learning pipeline
│   ├── __init__.py
│   ├── clustering_pipeline.py   # ClusteringPipeline class
│   ├── model_storage.py         # ModelStorage for versioning
│   └── interpreters.py          # Cluster/model interpretation
├── utils/
│   ├── __init__.py
│   └── database.py           # Database connection utilities
└── models/                   # Trained ML models (joblib files)
```

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Flask Application                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ api/routes/  │  │ api/routes/  │  │ api/routes/  │              │
│  │achievements.py│ │signatures.py │  │  docs.py     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘              │
│         │                 │                                         │
│  ┌──────▼─────────────────▼──────┐                                 │
│  │  services/achievements_service │                                 │
│  │  - getAchievementBySteamId    │                                 │
│  │  - getSignaturesBySteamId     │                                 │
│  │  - api_call_logger decorator  │                                 │
│  └──────────────┬────────────────┘                                 │
│                 │                                                   │
│  ┌──────────────▼────────────────┐                                 │
│  │     utils/database.py         │                                 │
│  │  - create_db_connection       │                                 │
│  │  - execute_query              │                                 │
│  │  - close_db_connection        │                                 │
│  └──────────────┬────────────────┘                                 │
│                 │                                                   │
│                 ▼                                                   │
│          ┌──────────────┐                                          │
│          │  PostgreSQL  │                                          │
│          │   Database   │                                          │
│          └──────────────┘                                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     ML Pipeline (Offline Jobs)                      │
│  ┌────────────────────┐  ┌────────────────────┐                    │
│  │ ClusteringPipeline │                                              │
│  │ - train_clusters   │                                              │
│  │ - assign_to_cluster│                                              │
│  └─────────┬──────────┘                                              │
│            │                                                        │
│  ┌─────────▼──────────┐  ┌────────────────────┐                    │
│  │   ModelStorage     │  │   Interpreters     │                    │
│  │ - save/load models │  │ - describe_cluster │                    │
│  │ - version control  │  │ - analyze_quality  │                    │
│  └────────────────────┘  └────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### API Request Flow

```
Client Request
      │
      ▼
┌─────────────────┐
│ Flask Blueprint │  Request validation via marshmallow schema
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ achievements_   │  Database query via raw psycopg2
│ service.py      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │  Fetch achievements/signatures
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Response Schema │  Serialize response via marshmallow
└────────┬────────┘
         │
         ▼
   JSON Response
```

### ML Clustering Flow (Offline)

```
┌─────────────────┐
│ Fetch all users │  Database query
│ achievements    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Selection│  Filter rare achievements to build
│ (vocabulary)     │  a manageable binary feature space
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Binary Matrix   │  Sparse achievement vectors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ L2 Normalize    │  Row-wise normalization (sparse)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TruncatedSVD   │  Dimensionality reduction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    HDBSCAN      │  Density-based cluster discovery
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Save medoids    │  Store in ClusterCentroid table
│ to database     │
└─────────────────┘

Notes:
- HDBSCAN discovers cluster count from data density (no `n_clusters` input).
- Noise points are labeled `-1` and represent users with unique playstyles.

Why HDBSCAN for achievement data:
- Player achievement histories form uneven densities (hardcore completionists vs casual players).
- HDBSCAN adapts to variable cluster sizes and leaves true outliers as noise instead of forcing them into a cluster.
- Cluster representatives are medoids (real users) rather than centroids (synthetic means).
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Landing page with API info |
| GET | `/user/achievements/{steam_id}` | Get user achievements by steam_id |
| GET | `/user/signatures/{steam_id}` | Get user clustering signatures |
| GET | `/docs` | OpenAPI Swagger documentation |
| GET | `/apidoc` | ReDoc documentation |
| GET | `/openapi.json` | Raw OpenAPI specification |

## API Request/Response Validation

Uses `flask-smorest` with marshmallow schemas defined in `api/schemas.py`:
- `@blp.arguments()` decorator for request validation
- `@blp.response()` decorator for response serialization
- Automatic OpenAPI/Swagger documentation generation

## Database Schema

### Key Tables

- **UserAchievement**: Steam user achievements (unique_key, game_id, unlocktime)
- **UserSignature**: Cluster assignments for users
- **ClusterCentroid**: Cluster representative vectors (medoids for HDBSCAN; stored in SVD space)
- **ApiLogs**: API call logging (input, output, timestamp)

### Connection Management

Database connections use raw psycopg2 (not SQLAlchemy ORM):
```python
from utils.database import create_db_connection, execute_query, close_db_connection

conn = create_db_connection()
results = execute_query(conn, sql_query)
close_db_connection(conn)
```

Environment variables required:
- `DB_USERNAME`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`

## ML Pipeline Components

### ClusteringPipeline

Core class for training and inference:
- `select_features(all_users_achievements)`: Filter rare achievements and build vocabulary
- `extract_binary_features(achievements)`: Convert achievements to sparse binary vectors
- `train_clusters(all_users_achievements)`: Build sparse matrix, normalize, TruncatedSVD, HDBSCAN (min_cluster_size)
- `assign_to_clusters(achievements, medoids)`: Assign single user using vocabulary + SVD

HDBSCAN specifics:
- `min_cluster_size` controls the minimum members per cluster (higher = fewer, larger clusters).
- Cluster labels include noise points with label `-1`.
- Medoids (actual data points) replace centroids for representative vectors.

### ModelStorage

Version-controlled model persistence:
- Saves models with metadata (version, timestamp, metrics)
- Auto-versioning (v1, v2, v3...)
- List/load/delete model versions

### Interpreters

Model explainability:
- `ClusterInterpreter`: Describe clusters in human terms
- `ModelInterpreter`: TruncatedSVD component analysis, feature importance

### Cluster-Defining Achievements

Cluster achievements are not stored as a dedicated table. They are derived at runtime
from the active cluster medoids plus the achievement vocabulary:

- **Source of truth**: `ClusterCentroid` rows contain medoid vectors (stored in SVD space).
- **Achievement vocabulary**: Loaded by `ClusteringPipeline` from MLModel table or
  training-run files (`models/*/vocabulary.pkl` or `models/*/vocabulary.json`).
- **Derivation**: `ClusterInterpreter.get_cluster_features(cluster_id, cluster_data, top_n=...)`
  inverse-transforms medoid vectors back into achievement space (via SVD) and ranks
  the top achievements by absolute feature importance.
- **Metadata lookup**: The top achievement `unique_key` values can be joined to
  `GameAchievement` (name, description, appid) and `Game` (game_name) for display.

Practical retrieval flow for a `cluster_id`:
1. Load active medoids from `ClusterCentroid` (see `services/achievements_service._load_centroids`).
2. Load models + vocabulary via `ClusteringPipeline` (SVD + achievement_vocabulary).
3. Use `ClusterInterpreter.get_cluster_features(..., top_n=N)` to get top unique_keys.
4. Query details:
   - `SELECT ga.unique_key, ga.name, ga.description, g.name AS game_name
      FROM "GameAchievement" ga JOIN "Game" g ON g.appid = ga.appid
      WHERE ga.unique_key IN (...)`

Limitations:
- Requires SVD + vocabulary to be loaded; otherwise cluster achievements cannot be derived.
- Medoids are SVD-space vectors; the inverse-transform is required to map to achievement space.
- Results are typically **top N** achievements per cluster (e.g., N=5 or N=10), not exhaustive.

## Clustering Philosophy & Tuning

The clustering model is designed to discover **micro-clusters** — small groups of 2-6 users
who share specific gaming behavior patterns (e.g. tactical combat enthusiasts, completionists,
cooperative players). Each micro-cluster represents a potential behavioral **signature**.

### Current Configuration (as of 2026-02-10)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `min_cluster_size` | 2 | Allow the smallest possible groups to form clusters |
| `cluster_selection_method` | `leaf` | Keep granular micro-clusters instead of merging them upward |
| `min_samples` | 1 | Minimum density requirement (already at floor) |
| `n_components` (SVD) | 50 (capped) | Dimensionality of compressed feature space |

**Key insight**: The default HDBSCAN configuration (`eom`, `min_cluster_size=50`) produced only
2 clusters with 84.9% noise. Switching to `leaf` with `min_cluster_size=2` discovered 152
micro-clusters with 40.6% coverage — exactly the small signature groups the model is designed to find.

### Cluster Selection Methods

- **`eom`** (Excess of Mass): Prefers fewer, larger clusters. Merges small groups up the
  dendrogram hierarchy. Appropriate when looking for broad population segments.
- **`leaf`**: Keeps the most granular clusters at the bottom of the dendrogram. Produces many
  small clusters. Appropriate when looking for niche behavioral signatures.

### Noise is Expected

With micro-clustering, a significant noise percentage (40-60%) is healthy and expected. Noise
users have genuinely unique profiles that don't share patterns with enough other users to form
a cluster. As the dataset grows, some noise users will find matches and form new micro-clusters.

## Scalability Guidance

### Current State (992 users, 2026-02-10)

The pipeline processes 855k achievement records across 992 users in ~3 minutes. The leaf/size=2
configuration discovers 152 micro-clusters. This is a small dataset; scaling characteristics
are projected below.

### Pipeline Stage Scaling

| Stage | Complexity | 1k users | 10k users | 50k users | 100k+ users |
|-------|-----------|----------|-----------|-----------|-------------|
| Feature selection | O(n * a) | trivial | seconds | seconds | seconds |
| Sparse binary matrix | O(nnz) | trivial | trivial | trivial | trivial |
| L2 normalization | O(nnz) | trivial | trivial | trivial | trivial |
| TruncatedSVD | O(n * k * d) | instant | seconds | minutes | minutes |
| **HDBSCAN** | **O(n^2 log n)** | instant | minutes | **10+ GB RAM** | **not viable** |
| Medoid computation | O(c * m^2) | trivial | trivial | trivial | trivial |
| Query-time similarity | O(c * k) | trivial | trivial | trivial | trivial |

Where n=users, a=achievements/user, nnz=non-zero entries, k=components, d=vocab size,
c=clusters, m=cluster members.

### Scaling Recommendations

**Up to ~10k users** (near-term): Current architecture works with minor tuning.
- Raise `n_components` cap from 50 to 100-200 to preserve finer structure in a more diverse dataset.
- Monitor vocabulary growth — with `min_users=2`, vocabulary scales with user count.
- Database query for fetching all achievements will need indexing on the JOIN columns.

**At 50k+ users** (future): HDBSCAN becomes the bottleneck due to O(n^2) pairwise distances.
- **Option A**: Train on a representative sample (e.g. 10k users), then assign remaining users
  to discovered clusters via `assign_to_clusters()` (cosine similarity to medoids). The pipeline
  already supports this pattern.
- **Option B**: Switch to approximate HDBSCAN (`algorithm='boruvka_kdtree'`) which trades
  exactness for sub-quadratic scaling.
- **Option C**: Batch/incremental clustering — cluster in chunks and merge overlapping clusters.

**SVD dimensionality**: The current cap of 50 components is sufficient for ~1k users but may
compress away fine-grained differences at larger scales. Consider scaling `n_components`
proportionally (e.g. `min(200, n_users // 5, n_features - 1)`).

**Vocabulary growth**: More users means more achievements pass the `min_users=2` threshold.
The sparse matrix handles large vocabularies efficiently, but SVD must compress a larger space.
Monitor explained variance ratio — if it drops below ~60%, increase `n_components`.

## Deployment

Deployed to Render:
- **Staging**: https://beta-engine-staging.onrender.com
- **Production**: https://flex-beta-engine.onrender.com
