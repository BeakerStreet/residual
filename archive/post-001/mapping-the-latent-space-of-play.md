Mapping the Latent Space of Play: How Flex Finds Your Gaming Signature
A technical deep-dive into the clustering pipeline powering Flex's personality engine — and an announcement of our latest model.

What Is Flex?
Before we get into the machine learning, a quick framing: Flex is a platform that analyzes your Steam gaming history to surface personality insights. Not the usual "you played 400 hours of Dark Souls, therefore you're masochistic" surface-level stuff — but something more structurally interesting. The hypothesis is that the specific pattern of which achievements you've earned, across which games, at what moments, encodes something real about who you are as a player and, extrapolating further, who you are as a person.

The inputs are humble: a Steam ID, a list of games, and the achievements earned in each. The output is a signature — a position in a learned latent space of gaming behavior, and a set of nearest-neighbor clusters that describe which other players you most resemble.

This article is a full technical walkthrough of how that works, and an announcement of the latest version of the model powering it.

The Data: What We're Actually Working With
Everything starts with the Steam Web API. Two endpoints do the heavy lifting:

IPlayerService/GetOwnedGames — returns every game a user owns, with playtime in minutes.
ISteamUserStats/GetPlayerAchievements — per-game, returns every achievement and whether the user has earned it, along with the Unix timestamp of unlock.
We call these after a user authenticates via Steam OpenID 2.0, then persist the results incrementally to PostgreSQL. The ingestion runs in a background thread so it doesn't block the login response. Progress is tracked per-game in real time (a recent improvement — more on that later).

What lands in the database is a normalized structure:

Game — Steam appid, name, platform.
GameAchievement — a global catalog of achievements across all games we've seen, keyed by a unique_key (a compound of appid + apiname).
UserAchievement — the per-user unlock records, linking a steam_id to a unique_key, with unlock timestamp and playtime at the moment of unlock.
At time of writing, we have ~992 users with roughly 855,000 achievement records in the database. That's the training corpus.

The ML Pipeline: Overview
The clustering pipeline lives in jobs/clustering_job.py and runs offline (not in the API's hot path). A training run proceeds in five stages:

Feature Engineering — Build a sparse binary matrix from achievement records.
Dimensionality Reduction — Compress to a dense low-dimensional embedding via TruncatedSVD.
Clustering — Run HDBSCAN over the embedding space to discover groups.
Confidence Scoring — Assign a per-user confidence value for their cluster membership.
Persistence — Write cluster centroids, user assignments, and top cluster-defining achievements back to the database.
Let's walk through each stage in detail.

Stage 1: Feature Engineering
The raw data is user × achievement co-occurrence. For each user, we have a set of unique_keys representing the achievements they've earned. The feature engineering step turns this into a matrix.

First, a vocabulary is built: we filter down to achievements that appear in the records of at least 2 distinct users. Achievements earned by only a single user are extremely sparse and contribute noise without signal — there's no basis for comparison. The resulting vocabulary maps each surviving unique_key to a column index.

Then, for each user, we construct a binary vector: 1 if they earned achievement at index i, 0 otherwise. The result is a sparse CSR matrix of shape (n_users, n_achievements).

Because we're using binary presence/absence rather than counts, this is effectively a set-based representation — we don't distinguish between a user who unlocked an achievement immediately at game launch versus one who ground for 80 hours. That's a deliberate choice: the which, not the when, is what we're clustering on at this stage. (Temporal and playtime signals are present in the raw data and represent future feature engineering opportunities.)

This matrix is extremely sparse — most users have earned achievements in a small fraction of all games in the corpus, so the vast majority of entries are zero. Scipy's sparse matrix format handles this efficiently without materializing the zeros.

Stage 2: Dimensionality Reduction via TruncatedSVD
A sparse binary matrix with tens of thousands of columns is not a great input to a density-based clusterer. The curse of dimensionality is real: in high-dimensional spaces, distance metrics become meaningless because all points tend toward equal mutual distance.

We apply Truncated SVD (also known as Latent Semantic Analysis in the NLP literature) to compress the achievement space into 50 dense components. Before decomposition, each user's vector is L2-normalized, so the SVD operates on direction rather than magnitude.

Why Truncated SVD specifically? Because it's designed for sparse matrices — unlike full PCA which requires a dense eigen-decomposition, Truncated SVD uses a randomized algorithm (ARPACK under the hood in scikit-learn) that never needs to materialize the full covariance matrix. It's both memory-efficient and fast on the kind of data we have.

At 50 components over ~992 users, we capture roughly 60% of explained variance. The SVD transform becomes a key artifact: it's serialized to disk (and to the MLModel database table) after training, and used at query time to project new users into the same latent space.

The output of this stage is a dense matrix of shape (n_users, 50) — each row is a user's embedding. This is the space in which we cluster.

Stage 3: HDBSCAN Clustering
This is the core algorithmic choice, and it's worth explaining in detail.

Why Not K-Means?
K-Means is the default choice for many clustering problems, and we did use it in earlier iterations. But it has properties that make it a poor fit here:

Requires specifying k upfront. We have no strong prior on how many distinct player archetypes exist. Gaming behavior is heterogeneous and the cluster count is itself something we want the data to tell us.
Produces synthetic centroids. K-Means centroids are the mean of all member vectors — they don't correspond to any real user. This makes interpretation harder.
Forces all points into clusters. K-Means has no concept of noise or outliers. Every user gets assigned to a cluster, even users whose gaming history is genuinely singular.
Assumes spherical clusters. The Euclidean distance minimization in K-Means implicitly assumes roughly circular (in 2D) or hyperspherical (in high-D) cluster shapes. Gaming behavior communities don't necessarily have that structure.
Why HDBSCAN?
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) addresses each of these:

Discovers cluster count from data density. No k parameter required — clusters emerge where the embedding space is locally dense.
Produces medoids, not means. Each cluster's representative is the actual user who sits most centrally within it — a real profile, not a synthetic average.
Leaves genuine outliers as noise (label -1). Users with genuinely unique gaming histories aren't forced into the nearest cluster — they're flagged as individuals.
Handles arbitrary cluster shapes. The density-based formulation finds clusters of any geometry, connected through chains of dense neighborhoods.
The algorithm builds a hierarchy of clusters by varying a density threshold — think of it as watching clusters form and split as you lower a water level over the embedding landscape. HDBSCAN then selects the "most persistent" clusters across that hierarchy.

Cluster Selection: leaf vs eom
This is a tuning decision with large practical consequences, worth explaining.

HDBSCAN's cluster hierarchy is a tree. At the top are large, coarse groups. As you descend, they split into smaller, more specific sub-clusters. The cluster selection method determines which level of the hierarchy you extract as your final clusters.

eom (Excess of Mass) — the default — selects clusters that maximize a stability score summed across the hierarchy. It tends to favor large, persistent clusters and collapses smaller sub-clusters back into their parents. With our early configurations (eom, min_cluster_size=50), the result was stark: 2 clusters, with 84.9% of users labeled as noise. Most users had genuinely unique gaming fingerprints that didn't fit into the two large archetypes that eom extracted.

leaf — selects the leaf nodes of the cluster tree, i.e., the smallest, most specific clusters before they'd be considered noise. This is exactly the right choice for our use case. We want to find the micro-communities — the small groups of 3, 7, or 20 players who share a surprisingly specific achievement fingerprint. With leaf and min_cluster_size=2, the latest training run produced 152 clusters, with 40.6% user coverage.

The practical difference: eom finds the two big gaming archetypes. leaf finds the people who all, inexplicably, have the same obscure achievement in Dwarf Fortress, and played the exact same chapter of a visual novel.

Current Training Configuration
PYTHONPATH=. python jobs/clustering_job.py \
  --output-dir results \
  --log-level INFO \
  --min-cluster-size 2 \
  --cluster-selection-method leaf
min_cluster_size=2: The minimum size of a valid cluster. Setting this to 2 means a pair of users with unusually similar achievement fingerprints constitutes a discoverable group.
min_samples=1: The minimum number of core points required for density estimation. With 1, even isolated dense pairs are kept.
cluster_selection_method=leaf: As above.
Stage 4: Confidence Scoring
Once users are assigned to clusters, we compute a confidence score — a float in [0, 1] representing how centrally a user sits within their assigned cluster.

For HDBSCAN, this uses the model's native membership probabilities. HDBSCAN internally computes the probability that each point belongs to its assigned cluster (as opposed to being near the cluster boundary or near noise). These are directly exposed via hdbscan.probabilities_ after fitting.

For legacy K-Means runs, we use cosine similarity between the user's SVD-transformed vector and the cluster centroid.

Confidence scores have two uses:

Flagging outliers — users with maximum similarity below a threshold (default: 25th percentile of membership probabilities, configurable via SIGNATURE_OUTLIER_THRESHOLD) are marked is_outlier=True in the signature response. This surfaces users whose gaming behavior doesn't map cleanly onto any discovered community.
UI signal — the confidence score is exposed in the API response so the frontend can communicate how strongly a user belongs to their cluster, not just which cluster they're in.
Stage 5: Persistence
After training, the following artifacts are written to the database and disk:

ClusterCentroid — Stores the medoid vector (the actual SVD-space embedding of the medoid user) for each cluster, along with the run_id and an is_active flag. Old centroids are deactivated, not deleted — giving us a versioned history of model runs.

UserSignature — Maps each user to their assigned cluster, with confidence score, run_id, and is_active. Noise points (cluster -1) are excluded — we don't store a signature for users with no cluster assignment.

ClusterAchievement — Pre-computed top achievements for each cluster. We inverse-transform the medoid's SVD vector back to achievement space and rank achievements by importance (weighting by both the component loadings and the achievement's prevalence in the cluster). The top-ranked achievements become the human-readable description of what a cluster represents.

MLModel — The serialized SVD transformer, HDBSCAN model, and vocabulary are stored as BYTEA in the database (and as .pkl files on disk). The API loads these at startup to handle real-time signature queries.

Run tracking uses a structured run_id format:

{training_type}_{ISO8601_timestamp}_v{version}
# e.g.: full_2026-02-10T19:24:48Z_v5
This is stored in both ClusterCentroid and UserSignature, enabling rollback and provenance tracking across model versions.

The Real-Time Signature API
When a user calls /user/signatures/{steam_id}, the system doesn't look up a stored assignment. It recomputes similarity in real time:

Query the user's UserAchievement records from the database.
Reconstruct their binary achievement vector against the current vocabulary.
L2-normalize and apply the stored SVD transform.
Compute cosine similarity between the user's embedding and every active ClusterCentroid vector.
Retrieve the top-ranked ClusterAchievement records for each cluster.
Return the full ranked list of similarities across all clusters.
This means the signature response reflects the user's current achievement state — as they earn new achievements, their similarity scores shift without requiring a re-training run. It also means every user gets similarity scores against every cluster, not just the one they were assigned to during training. The response looks like:

{
  "signatures": [
    {
      "cluster_id": 47,
      "similarity": 0.847,
      "cluster_name": null,
      "member_count": 12,
      "achievements": [
        {
          "unique_key": "427520_ACH.HARD_MODE",
          "name": "Masochist",
          "description": "Complete the game on hard difficulty",
          "game_name": "Factorio"
        }
      ]
    },
    // ... all other clusters, ranked by similarity ...
  ],
  "is_outlier": false,
  "model_run_id": "full_2026-02-10T19:24:48Z_v5"
}
The member_count field (added in the most recent model release) shows how many active users belong to each cluster — a small but meaningful piece of context for interpreting similarity scores. Finding yourself at 0.85 similarity to a cluster with 4 members is a very different signal than finding yourself at 0.85 similarity to a cluster with 200.

Announcing: Model v5 (full_2026-02-10T19:24:48Z_v5)
The latest training run represents a meaningful shift in model philosophy.

Previous configuration (EOM, min_cluster_size=50):

Clusters discovered: 2
Users covered: ~15%
Noise rate: 84.9%
Current configuration (LEAF, min_cluster_size=2):

Clusters discovered: 152
Users covered: 40.6%
Noise rate: 59.4%
The 152 clusters represent the first real population of micro-communities the system has discovered. These aren't broad archetypes ("competitive players" vs. "casual players"). They're specific: small groups of users who share a surprisingly particular achievement fingerprint across games that don't obviously belong together.

The move from eom to leaf is the key change. Rather than extracting the two most statistically stable aggregations of the entire user space, we're now reading the fine-grained leaf structure of the density hierarchy — the actual specific communities that exist within the data, before they'd be collapsed upward into their parent clusters by a stability criterion that was designed for a different kind of clustering problem.

What 40.6% coverage means: Six in ten users are still flagged as noise — their gaming history is specific enough that no cluster of 2+ users with highly similar fingerprints exists for them yet. As the user base grows, that number will fall. More users means more opportunities for rare achievement patterns to co-occur, and more micro-clusters will emerge. The noise users are not failures of the model — they're correctly identified individuals.

Why the noise rate is still high: The achievement space is enormous and user profiles are sparse. Even with 992 users and 855k records, most of the achievement vocabulary is covered by only a handful of users. Density-based clustering requires that users be nearby in the embedding space — and with 50 SVD dimensions representing the variance in tens of thousands of achievements, many users are genuinely isolated in that space.

Implementation Details Worth Noting
No ORM. The entire database layer uses raw psycopg2 queries. This is intentional — it gives fine-grained control over query structure and makes the SQL readable and debuggable without an abstraction layer hiding what's happening.

Sparse matrices throughout feature engineering. The achievement matrix is never materialized as a dense array until after SVD compression. Scipy CSR format keeps memory usage tractable even as the achievement vocabulary grows.

Module-level caching for centroids. At API startup, centroid vectors and cluster achievements are loaded into module-level dicts. Subsequent requests hit in-memory structures rather than the database. The cache is refreshed on deploy (i.e., when the process restarts after a new training run is deployed). This is a simple caching strategy that works well at current scale.

Background ingestion. After Steam auth callback, achievement ingestion runs in a background thread. The login response returns immediately with a redirect; ingestion proceeds asynchronously. Progress is tracked in the User table via processedGames (updated per game, not at end of run) and exposed via /user/ingestion-status/{steam_id}.

Medoids, not centroids. HDBSCAN's cluster representatives are actual user profiles — the user who sits most centrally in the cluster's density region. When the system stores ClusterCentroid vectors, it's storing the real embedding of a real user. The ClusterAchievement records derived from these are the real achievements of a real person who represents their cluster.

Scaling Considerations
The current architecture handles ~1k users comfortably. Looking forward:

Stage	Complexity	~1k users	~10k users	~50k users
Feature engineering	O(n × a)	trivial	seconds	seconds
TruncatedSVD	O(n × k × d)	instant	seconds	minutes
HDBSCAN	O(n² log n)	instant	minutes	prohibitive
Real-time similarity	O(c × k)	trivial	trivial	trivial
HDBSCAN's quadratic complexity is the bottleneck. At 50k+ users, the full pairwise distance computation becomes both time and memory prohibitive. The solutions we're considering:

Sampling: Train HDBSCAN on a 10k sample of the most active users, then assign remaining users to their nearest discovered cluster via cosine similarity to medoids.
Approximate nearest neighbors: Swap HDBSCAN's exact distance computations for ANN-based approximations (FAISS, hnswlib) — giving up exactness for tractable runtime.
Hierarchical runs: Cluster on clusters — find macro-communities at scale, then run granular clustering within each macro-community.
The real-time similarity computation is already O(c × k) — linear in the number of clusters and the SVD dimensionality — so that path scales independently and isn't a concern.

What's Next
A few directions we're actively thinking about:

Temporal features. The unlocktime field is in every achievement record and is currently unused in clustering. Achievement unlock order, time between unlocks, and unlock patterns relative to game release dates are all potentially rich signals about player behavior — the speedrunner, the completionist who returns years later, the player who abandons a game at 70%.

Rarity weighting. Not all achievements are equal. An achievement earned by 80% of players who own a game is a weak clustering signal; one earned by 0.3% is strong. We have the data to weight achievements by their global rarity, which would put more discriminative power on the rare achievements that most strongly differentiate player types.

Named clusters. The 152 clusters currently have no names — they're identified by integer IDs. Adding human-readable names (either manually curated or generated by feeding the top achievements into a language model) would make the signatures meaningfully interpretable to users.

Achievement graph structure. Achievements within a game often have dependency relationships (e.g., you can't earn "Chapter 5 Complete" without "Chapter 4 Complete"). Incorporating this structure into the feature representation — rather than treating achievements as independent binary variables — could sharpen the clustering signal.

Closing Thoughts
The core claim of Flex is that gaming behavior is legible — that the specific set of games you've played, the achievements you've chosen to pursue, and the order and timing in which you've pursued them constitute a meaningful personal signal. The ML pipeline we've built is a first pass at making that legibility computational.

What we've found so far: with the right clustering configuration, 40% of users are discoverable as members of specific micro-communities, and the micro-communities that emerge are specific enough to be genuinely surprising. Players who've never met, playing games with no obvious relationship, end up with nearly identical achievement fingerprints. That's the signal we're trying to surface.

Model v5 is the best version of this pipeline we've shipped. We expect it to get considerably more interesting as the user base grows and the achievement space becomes more densely covered.

The Flex API is deployed at https://flex-beta-engine.onrender.com with Swagger documentation at /docs. The clustering pipeline source is in jobs/clustering_job.py and the signature service is in services/achievements_service.py.