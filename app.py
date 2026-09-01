from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from model_trainer import IDSModelTrainer
import joblib
import os
from datetime import datetime, timedelta
from collections import deque

app = Flask(__name__, template_folder='templates')
CORS(app)

# Initialize IDS model
ids_trainer = IDSModelTrainer()
model_loaded = False

# Alert history (in-memory, would be database in production)
alerts_history = deque(maxlen=1000)
traffic_stats = {
    'total_packets': 0,
    'anomalies_detected': 0,
    'detection_rate': 0.0,
    'last_updated': None
}

@app.route('/')
def index():
    """Serve IDS dashboard UI"""
    return render_template('index.html')

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """Train IDS model on synthetic data"""
    try:
        from model_trainer import IDSModelTrainer as Trainer
        
        # Generate training data
        trainer = Trainer()
        df = trainer.generate_synthetic_training_data(samples=5000)
        
        X = df.drop('is_attack', axis=1)
        y = df['is_attack']
        trainer.feature_names = X.columns.tolist()
        
        # Train
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = trainer.train_model(X_train, y_train, X_test, y_test, 'random_forest')
        
        # Save globally
        global ids_trainer, model_loaded
        ids_trainer = trainer
        model_loaded = True
        
        return jsonify({
            'status': 'success',
            'message': f'Model trained on {len(X_train)} samples',
            'accuracy': results['test_accuracy'],
            'roc_auc': results['roc_auc'],
            'features': trainer.feature_names
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/analyze', methods=['POST'])
def analyze_traffic():
    """Analyze network traffic for intrusions"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not trained. Train model first.'}), 400
        
        data = request.json
        
        # Extract features from submitted traffic
        features_dict = data.get('features', {})
        
        # Validate features
        required_features = ids_trainer.feature_names
        if not all(f in features_dict for f in required_features):
            return jsonify({'error': 'Missing required features'}), 400
        
        # Create feature vector
        X = np.array([[features_dict[f] for f in required_features]])
        
        # Predict
        predictions, probabilities = ids_trainer.predict(X)
        
        is_attack = bool(predictions[0])
        attack_probability = float(probabilities[0][1])  # Probability of attack
        
        # Create alert
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'is_attack': is_attack,
            'attack_probability': attack_probability,
            'confidence': max(probabilities[0]),
            'features': features_dict,
            'severity': 'HIGH' if (is_attack and attack_probability > 0.8) else
                       'MEDIUM' if (is_attack and attack_probability > 0.5) else
                       'LOW'
        }
        
        alerts_history.append(alert)
        
        # Update stats
        traffic_stats['total_packets'] += 1
        if is_attack:
            traffic_stats['anomalies_detected'] += 1
        traffic_stats['detection_rate'] = (
            traffic_stats['anomalies_detected'] / traffic_stats['total_packets']
            if traffic_stats['total_packets'] > 0 else 0
        )
        traffic_stats['last_updated'] = datetime.utcnow().isoformat()
        
        return jsonify(alert), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/batch', methods=['POST'])
def batch_analyze():
    """Analyze batch of traffic records"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not trained'}), 400
        
        data = request.json
        traffic_list = data.get('traffic', [])
        
        results = []
        for traffic in traffic_list:
            features = traffic.get('features', {})
            required_features = ids_trainer.feature_names
            
            if all(f in features for f in required_features):
                X = np.array([[features[f] for f in required_features]])
                predictions, probabilities = ids_trainer.predict(X)
                
                results.append({
                    'is_attack': bool(predictions[0]),
                    'attack_probability': float(probabilities[0][1]),
                    'confidence': float(max(probabilities[0]))
                })
            else:
                results.append({'error': 'Missing features'})
        
        return jsonify({
            'total': len(results),
            'attacks_detected': sum(1 for r in results if r.get('is_attack')),
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Retrieve recent alerts"""
    limit = request.args.get('limit', 100, type=int)
    severity = request.args.get('severity', None)
    
    alerts_list = list(alerts_history)[-limit:]
    
    if severity:
        alerts_list = [a for a in alerts_list if a.get('severity') == severity]
    
    return jsonify({
        'total': len(alerts_list),
        'alerts': alerts_list
    }), 200

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get IDS statistics"""
    return jsonify(traffic_stats), 200

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not model_loaded:
        return jsonify({'status': 'not_trained'}), 200
    
    return jsonify({
        'status': 'trained',
        'features': ids_trainer.feature_names,
        'feature_count': len(ids_trainer.feature_names),
        'model_type': 'Random Forest (Scikit-learn)'
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'ids-detector'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5002)
