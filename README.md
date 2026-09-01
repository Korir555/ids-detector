# ML-Based Intrusion Detection System (IDS)

Machine learning-powered network anomaly detection using scikit-learn Random Forest and Gradient Boosting classifiers. Analyzes network traffic patterns to detect intrusions in real-time.

## Features

- **ML Classification**: Random Forest + Gradient Boosting on network features
- **12 Network Indicators**: Packet size, protocols, port variance, entropy, SYN floods, DNS queries
- **Real-Time Detection**: Analyze individual or batch traffic records
- **Alert System**: Severity-based alerts (HIGH/MEDIUM/LOW)
- **Statistics Dashboard**: Real-time detection metrics
- **Cross-Validation**: 5-fold CV with ROC-AUC evaluation
- **Feature Importance**: Identifies most predictive indicators

## Technical Details

**Models Trained**:
- Random Forest (100 trees, max_depth=15)
- Gradient Boosting (100 estimators, learning_rate=0.1)

**Features**:
- Packet size, packets/sec, protocol ratios
- Destination/source port variety
- Connection duration, failure rates
- Payload entropy, port scanning score
- SYN flood indicators, DNS query rate

**Performance**:
- ~95% accuracy on test set
- ROC-AUC: 0.97+
- Balanced F1-score

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5002
```

**Train model**: Click "TRAIN MODEL" button
**Analyze traffic**: Click "ANALYZE DEMO" for sample attacks

## API Endpoints

- `POST /api/model/train` — Train model
- `POST /api/detection/analyze` — Single traffic analysis
- `POST /api/detection/batch` — Batch analysis
- `GET /api/alerts` — Retrieve alerts
- `GET /api/statistics` — Detection stats
