# Barcelona Bicing Bike-Sharing Forecast System

A production-ready machine learning system for forecasting bike availability at Barcelona's Bicing bike-sharing stations. The system leverages linear regression models with gap-aware preprocessing to predict `num_bikes_available` across multiple time horizons.

## Overview

This project provides:
- **Multi-horizon forecasting** for bike availability (1, 3, 6, and 12 steps ahead)
- **Gap-aware preprocessing** to handle temporal gaps in station data
- **REST API** for real-time predictions
- **Redis-backed caching** for station snapshots and historical data
- **Stateless sliding window architecture** for memory-efficient processing

### Key Features

- **Linear Regression Models**: Comparison of LinearRegression, Ridge, and Lasso regressors
- **Best Model Performance**: Lasso (alpha=0.001) achieves MAE of 0.174 bikes vs. baseline of 12.584
- **Real-time Data Integration**: Automatic polling from Barcelona's Open Data API
- **Scalable Architecture**: Async FastAPI server with background tasks
- **Robust Gap Handling**: Automatic backfilling for short gaps, reset for long gaps

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.9 or higher
- Redis server (optional, falls back to in-memory cache)
- CUDA-compatible GPU (optional, for LSTM models)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ElizabethCF01/bicing-bcn-forecast-model-api.git
   cd updated
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root (use `.env.example` as template):

```bash
# Redis connection (optional)
REDIS_URL="redis://localhost:6379/0"

# Bicing API authentication
BICING_API_TOKEN="your_api_token_here"

# Model paths (optional, defaults shown)
BICING_META_DIR="bicing_lstm_artifacts_stateless"
BICING_LINEAR_MODEL_PATH="bicing_linear_artifacts/best_linear_model.joblib"
```

### Data Source

The system fetches live data from Barcelona's Open Data platform:
```
https://opendata-ajuntament.barcelona.cat/data/dataset/6aa3416d-ce1a-494d-861b-7bd07f069600/resource/1b215493-9e63-4a12-8980-2d7e0fa19f85/download
```

Data Set : 
```
https://opendata-ajuntament.barcelona.cat/data/en/dataset/estat-estacions-bicing
```

## Project Structure

```
updated/
├── Bicing_Linear_Regression_Gaps.ipynb  # Training notebook
├── api_lr.py                             # FastAPI production server
├── requirements.txt                      # Python dependencies
├── .env.example                          # Environment template
├── bicing_linear_artifacts/              # Trained model artifacts
│   └── best_linear_model.joblib
└── bicing_lstm_artifacts_stateless/      # Metadata from LSTM workflow
    └── meta.json
```

## Training the Model

### Step 1: Prepare Training Data

Place your historical Bicing CSV data in the `data/` directory:
```
data/2025_09_Setembre_BicingNou_ESTACIONS.csv
```

The CSV should contain these columns:
- `station_id`, `num_bikes_available`, `num_bikes_available_types.mechanical`
- `num_bikes_available_types.ebike`, `num_docks_available`, `last_reported`
- `is_charging_station`, `status`, `is_installed`, `is_renting`, `is_returning`
- `last_updated`, `ttl`

### Step 2: Run Training Notebook

Open and execute [Bicing_Linear_Regression_Gaps.ipynb](Bicing_Linear_Regression_Gaps.ipynb):

```bash
jupyter notebook Bicing_Linear_Regression_Gaps.ipynb
```

The notebook will:
1. Load metadata from the LSTM workflow (`meta.json`)
2. Stream and preprocess the CSV data
3. Assemble sliding windows with gap-aware features
4. Train and compare multiple linear models
5. Save the best model to `bicing_linear_artifacts/best_linear_model.joblib`

### Configuration Parameters (in notebook)

```python
HORIZONS = [1, 3, 6, 12]          # Forecast horizons (in steps)
SEQ_LEN = 90                       # Lookback window size
TRAIN_MAX_SAMPLES = 20_000         # Training samples to collect
VAL_MAX_SAMPLES = 5_000            # Validation samples
TRAIN_SUBSAMPLE_PROB = 0.20        # Subsampling probability
BASE_STEP_SEC = 240                # Time step duration (4 minutes)
SHORT_GAP_STEPS = 3                # Gap threshold for backfilling
LONG_GAP_STEPS = 15                # Gap threshold for buffer reset
```

## Running the API

### Start the Server

```bash
python api_lr.py
```

Or with Uvicorn for production:
```bash
uvicorn api_lr:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will:
- Load the trained model from `bicing_linear_artifacts/`
- Connect to Redis (or use in-memory fallback)
- Start background polling of Bicing API (every 5 minutes)
- Serve predictions at `http://localhost:8000`

### Health Check

Visit the interactive API documentation:
```
http://localhost:8000/docs
```

## API Endpoints

### 1. Forecast Bike Availability

**GET** `/forecast`

Predict future bike availability for a specific station.

**Query Parameters**:
- `station_id` (int, required): Target station identifier
- `horizon_minutes` (int, default=60): Forecast horizon in minutes (max: 240)

**Example Request**:
```bash
curl "http://localhost:8000/forecast?station_id=1&horizon_minutes=60"
```

**Example Response**:
```json
{
  "station_id": 1,
  "horizon_minutes": 60,
  "forecast": [
    {
      "eta": "2025-10-20T10:04:00Z",
      "bikes_available": 12,
      "docks_available": 8
    },
    {
      "eta": "2025-10-20T10:12:00Z",
      "bikes_available": 10,
      "docks_available": 10
    },
    {
      "eta": "2025-10-20T10:24:00Z",
      "bikes_available": 8,
      "docks_available": 12
    },
    {
      "eta": "2025-10-20T10:48:00Z",
      "bikes_available": 5,
      "docks_available": 15
    }
  ],
  "issued_at": "2025-10-20T10:00:00Z"
}
```

### 2. Latest Station Snapshot

**GET** `/stations/latest`

Retrieve the most recent availability snapshot for all stations.

**Example Request**:
```bash
curl "http://localhost:8000/stations/latest"
```

**Example Response**:
```json
[
  {
    "station_id": 1,
    "bikes_available": 12,
    "docks_available": 8,
    "collected_at": "2025-10-20T10:00:00Z",
    "mechanical_bikes_available": 7,
    "ebikes_available": 5,
    "status": "IN_SERVICE",
    "is_installed": true,
    "is_renting": true,
    "is_returning": true
  }
]
```

### 3. Station History

**GET** `/stations/history?limit=10`

Fetch cached historical snapshots (ordered oldest to newest).

**Query Parameters**:
- `limit` (int, default=10, max=500): Number of snapshots to return

**Example Request**:
```bash
curl "http://localhost:8000/stations/history?limit=5"
```

## Model Architecture

### Feature Engineering

The model uses a comprehensive feature set including:

**Continuous Features**:
- `num_bikes_available`, `num_bikes_available_types.mechanical`, `num_bikes_available_types.ebike`
- `num_docks_available`, `is_installed`, `is_renting`, `is_returning`, `ttl`
- `sin_hour`, `cos_hour`, `sin_dow`, `cos_dow` (cyclical time encoding)
- `delta_steps`, `log1p_delta_steps`, `is_gap` (gap-aware features)

**Categorical Features** (one-hot encoded):
- `station_id` (546 unique stations)
- `status` (e.g., "IN_SERVICE", "OUT_OF_SERVICE")

**Total Feature Dimension**: 1,424 (after one-hot encoding)

### Gap-Aware Preprocessing

The system handles temporal gaps intelligently:

1. **Short Gaps** (≤3 steps): Backfill with last observation
2. **Long Gaps** (>15 steps): Reset station buffer
3. **Delta Features**: Track time since last observation

### Model Comparison Results

| Model | Overall MAE | h1 MAE | h3 MAE | h6 MAE | h12 MAE |
|-------|-------------|--------|--------|--------|---------|
| **Lasso (α=0.001)** | **0.174** | **0.060** | **0.139** | **0.207** | **0.290** |
| Ridge (α=100.0) | 0.179 | 0.068 | 0.144 | 0.211 | 0.294 |
| Ridge (α=10.0) | 0.183 | 0.068 | 0.146 | 0.216 | 0.303 |
| LinearRegression | 0.185 | 0.068 | 0.147 | 0.217 | 0.305 |
| Ridge (α=1.0) | 0.185 | 0.069 | 0.148 | 0.219 | 0.306 |
| Baseline (carry-forward) | 12.584 | 12.581 | 12.583 | 12.584 | 12.587 |

## Performance Metrics

### Model Performance

- **Best Model**: Lasso (alpha=0.001)
- **Average MAE**: 0.174 bikes across all horizons
- **Improvement over Baseline**: 98.6% reduction in MAE
- **Training Samples**: 20,000 windows
- **Validation Samples**: 5,000 windows

### Forecast Horizons

| Horizon | Steps | Minutes | MAE (bikes) |
|---------|-------|---------|-------------|
| h1 | 1 | 4 | 0.060 |
| h3 | 3 | 12 | 0.139 |
| h6 | 6 | 24 | 0.207 |
| h12 | 12 | 48 | 0.290 |

### API Performance

- **Prediction Latency**: <50ms per station
- **Background Polling**: Every 5 minutes
- **Station Coverage**: 546 stations across Barcelona

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black .

# Type checking
mypy api_lr.py

# Linting
flake8 api_lr.py
```

### Adding New Features

To add new features to the model:

1. Update `FEAT_CONT` list in the training notebook
2. Modify `_prepare()` method in `BicingLinearForecastService`
3. Retrain the model
4. Update `scaler_mean` and `scaler_std` in metadata

## Troubleshooting

### Common Issues

**1. Model artifacts not found**

```
FileNotFoundError: Meta file not found: bicing_lstm_artifacts_stateless/meta.json
```

**Solution**: Run the LSTM training notebook first to generate `meta.json`, or create it manually with required metadata.

**2. Redis connection failed**

```
Failed to connect to Redis. Using in-memory FakeRedis instead.
```

**Solution**: This is expected behavior. The API will use an in-memory cache if Redis is unavailable. To use Redis:
```bash
# Install Redis
brew install redis  # macOS
sudo apt-get install redis-server  # Ubuntu

# Start Redis
redis-server
```

**3. Insufficient history for inference**

```
HTTPException 503: Insufficient history for the requested station
```

**Solution**: The model requires 90 historical observations. Wait for more data to accumulate or use `/stations/history` endpoint to verify data availability.

**4. Station not in training metadata**

```
HTTPException 404: Station {id} not present in trained model metadata
```

**Solution**: Retrain the model with updated station data, or verify the station ID exists in the current Bicing network.

### Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via environment variable:
```bash
export LOG_LEVEL=DEBUG
python api_lr.py
```

## Dependencies

Key dependencies:
- **FastAPI**: Async web framework
- **Scikit-learn**: Linear regression models
- **NumPy/Pandas**: Data processing
- **Redis**: Caching layer (optional)
- **Joblib**: Model serialization
- **PyTorch Lightning**: For LSTM models (optional)

See [requirements.txt](requirements.txt) for complete list.

## License

[Add your license here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with tests

## Authors

[Add your name/team here]

## Acknowledgments

- Barcelona City Council for providing open data
- Bicing bike-sharing system
- PyTorch and scikit-learn communities

---

**Last Updated**: October 2025
**Version**: 1.0.0
