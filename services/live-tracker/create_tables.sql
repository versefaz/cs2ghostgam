\c cs2_predictions;

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY,
    match_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    team1 VARCHAR(100) NOT NULL,
    team2 VARCHAR(100) NOT NULL,
    predicted_winner VARCHAR(100) NOT NULL,
    win_probability DECIMAL(5,4) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    odds_team1 DECIMAL(6,3),
    odds_team2 DECIMAL(6,3),
    expected_value DECIMAL(8,4),
    features_used JSONB,
    actual_winner VARCHAR(100),
    is_correct BOOLEAN,
    profit_loss DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy DECIMAL(5,4),
    total_profit_loss DECIMAL(12,2),
    roi DECIMAL(6,4),
    avg_confidence DECIMAL(5,4),
    high_conf_accuracy DECIMAL(5,4),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(date);
