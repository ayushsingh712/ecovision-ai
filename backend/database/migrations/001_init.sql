CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    public_id TEXT NOT NULL UNIQUE,
    original_filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    top3_json TEXT NOT NULL,
    model_stage TEXT NOT NULL,
    model_version TEXT NOT NULL,
    original_file TEXT NOT NULL,
    heatmap_file TEXT NOT NULL,
    overlay_file TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_class
    ON predictions(predicted_class);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL UNIQUE
        REFERENCES predictions(id) ON DELETE CASCADE,
    is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
    corrected_class TEXT,
    comment TEXT,
    verified INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_correct
    ON feedback(is_correct);
