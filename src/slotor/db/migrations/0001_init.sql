-- Slotor schema v1 (SQLite >= 3.37, STRICT tables).
-- Source de vérité du schéma. Vue d'ensemble : docs/SCHEMA.md. Décision : DECISIONS.md D18.
-- Conventions : mm / degrés en base (le cœur convertit en radians, D3) ; booléens 0/1 ; dates ISO-8601 UTC ;
-- JSON dans des colonnes TEXT validées par json_valid(). Le validateur et les optimiseurs ne lisent JAMAIS la base
-- dans leurs boucles : le catalogue est chargé en objets immuables au démarrage.
-- À appliquer avec PRAGMA foreign_keys = ON (par connexion).

------------------------------------------------------------------------------------------------------------------
-- 1. CATALOGUE (statique, versionné, avec provenance — D5)
------------------------------------------------------------------------------------------------------------------

CREATE TABLE source (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL CHECK (kind IN ('track_architect','official_doc','measurement','user','other')),
    url          TEXT,
    retrieved_on TEXT,
    note         TEXT
) STRICT;

CREATE TABLE catalog_version (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,            -- ex. 'd132-2026.1'
    product_line   TEXT NOT NULL DEFAULT 'D132',
    content_sha256 TEXT NOT NULL,                   -- hash du fichier seed d'origine
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    note           TEXT
) STRICT;

CREATE TABLE piece_type (
    id                 INTEGER PRIMARY KEY,
    catalog_version_id INTEGER NOT NULL REFERENCES catalog_version(id) ON DELETE CASCADE,
    slug               TEXT NOT NULL,               -- identifiant stable, ex. 'curve_r1_60'
    category           TEXT NOT NULL CHECK (category IN (
                           'straight','curve','banked_curve','start','lane_change','switch','control_unit',
                           'ramp_up','ramp_down','bridge_deck','pillar','accessory')),
    name               TEXT NOT NULL,
    ta_code            TEXT,                        -- code Track Architect si connu
    manufacturer_part_number TEXT,                  -- référence fabricant (nomenclature, phase 8)
    length_mm          REAL CHECK (length_mm  IS NULL OR length_mm  > 0),
    radius_mm          REAL CHECK (radius_mm  IS NULL OR radius_mm  > 0),
    -- Ligne à laquelle radius_mm est mesuré (D20). NULL = inconnu : la pièce ne peut pas être vérifiée.
    radius_ref         TEXT CHECK (radius_ref IS NULL OR radius_ref IN
                           ('centerline','lane1','lane2','inner_edge','outer_edge')),
    turn_angle_deg     REAL,                        -- angle de virage non signé ; le signe vient du miroir
    mirrorable         INTEGER NOT NULL DEFAULT 0 CHECK (mirrorable IN (0,1)),
    width_mm           REAL CHECK (width_mm IS NULL OR width_mm > 0),
    -- Ponts (D4 révisée) : 2D à plat, DEUX niveaux discrets, aucune pente modélisée.
    -- Pièce normale 0->0 ; pièce surélevée 1->1 ; rampe montante 0->1 ; rampe descendante 1->0.
    -- Une rampe (level_in <> level_out) occupe les deux niveaux : elle entre en collision avec tout recouvrement.
    level_in           INTEGER NOT NULL DEFAULT 0 CHECK (level_in  IN (0,1)),
    level_out          INTEGER NOT NULL DEFAULT 0 CHECK (level_out IN (0,1)),
    verified           INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0,1)),   -- 0 = interdit à l'optimiseur
    source_id          INTEGER REFERENCES source(id),
    note               TEXT,
    UNIQUE (catalog_version_id, slug),
    CHECK (category <> 'ramp_up'   OR (level_in = 0 AND level_out = 1)),
    CHECK (category <> 'ramp_down' OR (level_in = 1 AND level_out = 0))
) STRICT;

CREATE TABLE piece_connector (
    id            INTEGER PRIMARY KEY,
    piece_type_id INTEGER NOT NULL REFERENCES piece_type(id) ON DELETE CASCADE,
    role          TEXT NOT NULL CHECK (role IN ('in','out','aux')),
    idx           INTEGER NOT NULL DEFAULT 0,       -- aiguillages : plusieurs sorties
    x_mm          REAL NOT NULL,
    y_mm          REAL NOT NULL,
    theta_deg     REAL NOT NULL,
    UNIQUE (piece_type_id, role, idx)
) STRICT;

CREATE TABLE piece_outline (                        -- contours dans le repère local de la pièce
    piece_type_id INTEGER NOT NULL REFERENCES piece_type(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN ('body','support','border')),   -- 'support' = pilier, 'border' = bordure/barrière (optionnels, absents en V1)
    idx           INTEGER NOT NULL DEFAULT 0,
    polygon_json  TEXT NOT NULL CHECK (json_valid(polygon_json)),     -- [[x,y],...] anneau extérieur, mm
    PRIMARY KEY (piece_type_id, kind, idx)
) STRICT;

CREATE TABLE piece_lane_path (                      -- trajectoire de chaque voie (longueurs par voie)
    piece_type_id INTEGER NOT NULL REFERENCES piece_type(id) ON DELETE CASCADE,
    lane          INTEGER NOT NULL CHECK (lane IN (1,2)),
    variant       INTEGER NOT NULL DEFAULT 0,       -- aiguillages : chemin droit / dévié
    path_json     TEXT NOT NULL CHECK (json_valid(path_json)),
    length_mm     REAL NOT NULL CHECK (length_mm > 0),
    PRIMARY KEY (piece_type_id, lane, variant)
) STRICT;

CREATE TABLE piece_constraint (                     -- orientation de montage, contraintes électriques, etc.
    id            INTEGER PRIMARY KEY,
    piece_type_id INTEGER NOT NULL REFERENCES piece_type(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    params_json   TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(params_json))
) STRICT;

CREATE TABLE piece_set (                            -- sets officiels (ex. référence 30144)
    id                 INTEGER PRIMARY KEY,
    catalog_version_id INTEGER NOT NULL REFERENCES catalog_version(id) ON DELETE CASCADE,
    reference          TEXT NOT NULL,
    name               TEXT,
    UNIQUE (catalog_version_id, reference)
) STRICT;

CREATE TABLE piece_set_item (
    set_id        INTEGER NOT NULL REFERENCES piece_set(id) ON DELETE CASCADE,
    piece_type_id INTEGER NOT NULL REFERENCES piece_type(id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (set_id, piece_type_id)
) STRICT;

CREATE TABLE piece_price (                          -- phase 8 (« quels rails acheter ? »)
    piece_type_id INTEGER NOT NULL REFERENCES piece_type(id) ON DELETE CASCADE,
    as_of         TEXT NOT NULL,
    amount_cents  INTEGER NOT NULL CHECK (amount_cents >= 0),
    currency      TEXT NOT NULL DEFAULT 'EUR',
    source_id     INTEGER REFERENCES source(id),
    PRIMARY KEY (piece_type_id, as_of)
) STRICT;

------------------------------------------------------------------------------------------------------------------
-- 2. DONNÉES UTILISATEUR (projets) — base locale, ignorée par Git (D17)
------------------------------------------------------------------------------------------------------------------

CREATE TABLE inventory (
    id                 INTEGER PRIMARY KEY,
    catalog_version_id INTEGER NOT NULL REFERENCES catalog_version(id),
    name               TEXT NOT NULL UNIQUE,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    note               TEXT
) STRICT;

CREATE TABLE inventory_item (
    inventory_id  INTEGER NOT NULL REFERENCES inventory(id) ON DELETE CASCADE,
    piece_type_id INTEGER NOT NULL REFERENCES piece_type(id),   -- invariant applicatif : même catalogue que l'inventaire
    quantity      INTEGER NOT NULL CHECK (quantity >= 0),
    PRIMARY KEY (inventory_id, piece_type_id)
) STRICT;

CREATE TABLE surface (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    width_mm       REAL NOT NULL CHECK (width_mm > 0),
    depth_mm       REAL NOT NULL CHECK (depth_mm > 0),
    margin_mm      REAL NOT NULL DEFAULT 0 CHECK (margin_mm >= 0),
    obstacles_json TEXT CHECK (obstacles_json IS NULL OR json_valid(obstacles_json))   -- futur : zones interdites
) STRICT;

CREATE TABLE tolerance_config (                     -- D3/D8/D9 : eps_area, eps_pos, eps_angle…
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    config_json    TEXT NOT NULL CHECK (json_valid(config_json)),
    content_sha256 TEXT NOT NULL
) STRICT;

CREATE TABLE project (
    id                  INTEGER PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    catalog_version_id  INTEGER NOT NULL REFERENCES catalog_version(id),
    inventory_id        INTEGER NOT NULL REFERENCES inventory(id),
    surface_id          INTEGER NOT NULL REFERENCES surface(id),
    tolerance_config_id INTEGER NOT NULL REFERENCES tolerance_config(id),
    allow_bridges       INTEGER NOT NULL DEFAULT 0 CHECK (allow_bridges IN (0,1)),   -- D4 : 0 = à plat
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    note                TEXT
) STRICT;

------------------------------------------------------------------------------------------------------------------
-- 3. OPTIMISATIONS ET SOLUTIONS
------------------------------------------------------------------------------------------------------------------

CREATE TABLE optimization_run (
    id            INTEGER PRIMARY KEY,
    project_id    INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    method        TEXT NOT NULL CHECK (method IN
                      ('seed','random','greedy','beam','lns','rl_ppo','rl_beam','rl_lns')),
    seed          INTEGER,
    budget_s      REAL,
    config_json   TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(config_json)),
    rl_model_id   INTEGER REFERENCES rl_model(id),
    catalog_sha256 TEXT,
    git_sha       TEXT,
    git_dirty     INTEGER CHECK (git_dirty IN (0,1)),
    status        TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','done','failed','cancelled')),
    started_at    TEXT,
    finished_at   TEXT,
    mlflow_run_id TEXT,
    error         TEXT
) STRICT;

CREATE TABLE solution (
    id                    INTEGER PRIMARY KEY,
    run_id                INTEGER NOT NULL REFERENCES optimization_run(id) ON DELETE CASCADE,
    rank                  INTEGER NOT NULL CHECK (rank >= 1),
    length_centerline_mm  REAL NOT NULL CHECK (length_centerline_mm > 0),   -- objectif (D7)
    length_lane1_mm       REAL NOT NULL,
    length_lane2_mm       REAL NOT NULL,
    piece_count           INTEGER NOT NULL CHECK (piece_count > 0),
    bridge_count          INTEGER NOT NULL DEFAULT 0 CHECK (bridge_count >= 0),
    width_mm              REAL NOT NULL,                                    -- boîte englobante
    depth_mm              REAL NOT NULL,
    validated             INTEGER NOT NULL CHECK (validated IN (0,1)),      -- validateur COMPLET passé (D6)
    validator_version     TEXT,
    circuit_hash          TEXT NOT NULL,                                    -- déduplication
    svg                   TEXT,
    created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE (run_id, rank)
) STRICT;

CREATE TABLE solution_piece (
    solution_id   INTEGER NOT NULL REFERENCES solution(id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL CHECK (seq >= 0),
    piece_type_id INTEGER NOT NULL REFERENCES piece_type(id),
    mirrored      INTEGER NOT NULL DEFAULT 0 CHECK (mirrored IN (0,1)),
    x_mm          REAL NOT NULL,                                           -- pose d'entrée (cache, recalculable)
    y_mm          REAL NOT NULL,
    theta_deg     REAL NOT NULL,
    level         INTEGER NOT NULL DEFAULT 0 CHECK (level IN (0,1)),
    PRIMARY KEY (solution_id, seq)
) STRICT, WITHOUT ROWID;

CREATE TABLE run_progress (                         -- courbe « anytime » longueur(t) pour le benchmark
    run_id         INTEGER NOT NULL REFERENCES optimization_run(id) ON DELETE CASCADE,
    t_s            REAL NOT NULL,
    best_length_mm REAL NOT NULL,
    PRIMARY KEY (run_id, t_s)
) STRICT, WITHOUT ROWID;

------------------------------------------------------------------------------------------------------------------
-- 4. R&D RL ET BENCHMARK (phase 7)
------------------------------------------------------------------------------------------------------------------

CREATE TABLE rl_model (
    id               INTEGER PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    algo             TEXT NOT NULL,                  -- ex. 'maskable_ppo'
    formulation      TEXT NOT NULL CHECK (formulation IN ('a_construct','b_beam_prior','c_lns_destroy')),
    env_version      TEXT NOT NULL,
    catalog_sha256   TEXT NOT NULL,
    obs_spec_json    TEXT CHECK (obs_spec_json IS NULL OR json_valid(obs_spec_json)),
    hyperparams_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(hyperparams_json)),
    total_timesteps  INTEGER,
    seed             INTEGER,
    artifact_path    TEXT,                           -- fichier du modèle (hors Git)
    mlflow_run_id    TEXT,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
) STRICT;

CREATE TABLE benchmark_suite (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    note TEXT,
    UNIQUE (name, version)
) STRICT;

CREATE TABLE benchmark_instance (
    id                    INTEGER PRIMARY KEY,
    suite_id              INTEGER NOT NULL REFERENCES benchmark_suite(id) ON DELETE CASCADE,
    name                  TEXT NOT NULL,
    split                 TEXT NOT NULL CHECK (split IN ('train','test')),
    project_id            INTEGER NOT NULL REFERENCES project(id),
    best_known_length_mm  REAL,
    UNIQUE (suite_id, name)
) STRICT;

CREATE TABLE benchmark_result (
    id                       INTEGER PRIMARY KEY,
    instance_id              INTEGER NOT NULL REFERENCES benchmark_instance(id) ON DELETE CASCADE,
    run_id                   INTEGER NOT NULL UNIQUE REFERENCES optimization_run(id) ON DELETE CASCADE,
    length_mm                REAL,                   -- NULL = aucune solution valide dans le budget
    time_to_first_valid_s    REAL
) STRICT;

------------------------------------------------------------------------------------------------------------------
-- 5. ACHATS (phase 8)
------------------------------------------------------------------------------------------------------------------

CREATE TABLE purchase_plan (
    id           INTEGER PRIMARY KEY,
    project_id   INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    budget_cents INTEGER NOT NULL CHECK (budget_cents >= 0),
    currency     TEXT NOT NULL DEFAULT 'EUR',
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    note         TEXT
) STRICT;

CREATE TABLE purchase_plan_item (
    plan_id       INTEGER NOT NULL REFERENCES purchase_plan(id) ON DELETE CASCADE,
    piece_type_id INTEGER NOT NULL REFERENCES piece_type(id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (plan_id, piece_type_id)
) STRICT;

------------------------------------------------------------------------------------------------------------------
-- Index
------------------------------------------------------------------------------------------------------------------
CREATE INDEX idx_piece_type_category ON piece_type(catalog_version_id, category);
CREATE INDEX idx_run_project         ON optimization_run(project_id, method);
CREATE INDEX idx_solution_run        ON solution(run_id);
CREATE INDEX idx_solution_hash       ON solution(circuit_hash);
CREATE INDEX idx_bench_inst_suite    ON benchmark_instance(suite_id, split);

PRAGMA user_version = 1;
