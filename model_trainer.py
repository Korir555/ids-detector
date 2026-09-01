"""
Machine Learning Intrusion Detection System (IDS)
Trains and evaluates models on network traffic patterns
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib
import json
from datetime import datetime

class IDSModelTrainer:
    """Train ML models for intrusion detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_history = []
    
    @staticmethod
    def generate_synthetic_training_data(samples=5000):
        """
        Generate synthetic network traffic data for training
        Features based on common IDS indicators
        """
        np.random.seed(42)
        
        # Normal traffic patterns
        normal_samples = samples // 2
        normal_data = {
            'packet_size': np.random.normal(512, 100, normal_samples),  # avg 512 bytes
            'packets_per_sec': np.random.normal(50, 10, normal_samples),
            'protocol_ratio_tcp': np.random.uniform(0.4, 0.8, normal_samples),
            'protocol_ratio_udp': np.random.uniform(0.1, 0.4, normal_samples),
            'dst_port_variance': np.random.normal(5, 2, normal_samples),  # few dest ports
            'src_port_variety': np.random.normal(10, 5, normal_samples),  # varied src ports
            'connection_duration': np.random.exponential(5, normal_samples),  # typical session length
            'failed_connection_rate': np.random.beta(2, 20, normal_samples),  # low failure rate
            'payload_entropy': np.random.uniform(4.5, 7.0, normal_samples),  # moderate entropy
            'port_scanning_score': np.random.uniform(0, 0.2, normal_samples),  # low port scan
            'syn_flood_indicator': np.random.uniform(0, 0.1, normal_samples),  # low SYN flood
            'dns_query_rate': np.random.normal(5, 2, normal_samples),
            'is_attack': np.zeros(normal_samples)
        }
        
        # Attack traffic patterns
        attack_samples = samples // 2
        attack_type = np.random.choice(['port_scan', 'ddos', 'syn_flood', 'brute_force'], attack_samples)
        
        attack_data = {
            'packet_size': np.random.normal(256, 150, attack_samples),  # smaller packets
            'packets_per_sec': np.random.normal(500, 200, attack_samples),  # high rate
            'protocol_ratio_tcp': np.random.uniform(0.7, 1.0, attack_samples),
            'protocol_ratio_udp': np.random.uniform(0.0, 0.2, attack_samples),
            'dst_port_variance': np.random.normal(200, 100, attack_samples),  # many dest ports
            'src_port_variety': np.random.normal(50, 20, attack_samples),  # varied src ports
            'connection_duration': np.random.exponential(0.5, attack_samples),  # short sessions
            'failed_connection_rate': np.random.beta(5, 5, attack_samples),  # high failure rate
            'payload_entropy': np.random.uniform(0.5, 4.0, attack_samples),  # low entropy
            'port_scanning_score': np.random.uniform(0.5, 1.0, attack_samples),  # high port scan
            'syn_flood_indicator': np.random.uniform(0.3, 1.0, attack_samples),  # high SYN flood
            'dns_query_rate': np.random.normal(50, 20, attack_samples),  # high DNS queries
            'is_attack': np.ones(attack_samples)
        }
        
        # Combine
        df_normal = pd.DataFrame(normal_data)
        df_attack = pd.DataFrame(attack_data)
        df = pd.concat([df_normal, df_attack], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)  # shuffle
        
        return df
    
    def train_model(self, X_train, y_train, X_test=None, y_test=None, model_type='random_forest'):
        """Train ML model for intrusion detection"""
        
        # Fit scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',  # handle class imbalance
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        results = {
            'model_type': model_type,
            'trained_at': datetime.utcnow().isoformat(),
            'train_samples': len(X_train),
            'feature_count': X_train.shape[1]
        }
        
        # Training accuracy
        train_acc = self.model.score(X_train_scaled, y_train)
        results['train_accuracy'] = float(train_acc)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        results['cv_mean'] = float(cv_scores.mean())
        results['cv_std'] = float(cv_scores.std())
        
        # Test set evaluation
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            test_acc = self.model.score(X_test_scaled, y_test)
            results['test_accuracy'] = float(test_acc)
            
            # Predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            results['classification_report'] = class_report
            
            # ROC-AUC
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            results['roc_auc'] = float(roc_auc)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
            
            # Precision-Recall
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            results['precision_at_95_recall'] = float(precision[np.argmax(recall >= 0.95)])
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        results['feature_importance'] = {
            self.feature_names[i]: float(feature_importance[i])
            for i in range(len(self.feature_names))
        }
        
        self.training_history.append(results)
        return results
    
    def predict(self, X):
        """Predict if traffic is attack or normal"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def save_model(self, path):
        """Save trained model and scaler"""
        joblib.dump(self.model, f'{path}/model.pkl')
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
    
    def load_model(self, path):
        """Load trained model and scaler"""
        self.model = joblib.load(f'{path}/model.pkl')
        self.scaler = joblib.load(f'{path}/scaler.pkl')


def run_training_pipeline():
    """Complete training pipeline"""
    
    print("=" * 60)
    print("ML-BASED INTRUSION DETECTION SYSTEM TRAINING")
    print("=" * 60)
    
    # Generate data
    print("\n[1/4] Generating synthetic training data...")
    df = IDSModelTrainer.generate_synthetic_training_data(samples=10000)
    print(f"    Generated {len(df)} samples")
    print(f"    Normal: {sum(df['is_attack'] == 0)}, Attack: {sum(df['is_attack'] == 1)}")
    
    # Prepare features
    print("\n[2/4] Preparing features...")
    X = df.drop('is_attack', axis=1)
    y = df['is_attack']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train set: {len(X_train)} samples")
    print(f"    Test set: {len(X_test)} samples")
    
    # Train models
    print("\n[3/4] Training models...")
    trainer = IDSModelTrainer()
    trainer.feature_names = X.columns.tolist()
    
    # Random Forest
    print("    Training Random Forest...")
    rf_results = trainer.train_model(X_train, y_train, X_test, y_test, 'random_forest')
    print(f"    ✓ Train Acc: {rf_results['train_accuracy']:.4f}")
    print(f"    ✓ Test Acc: {rf_results['test_accuracy']:.4f}")
    print(f"    ✓ ROC-AUC: {rf_results['roc_auc']:.4f}")
    
    # Gradient Boosting
    print("    Training Gradient Boosting...")
    gb_results = trainer.train_model(X_train, y_train, X_test, y_test, 'gradient_boosting')
    print(f"    ✓ Train Acc: {gb_results['train_accuracy']:.4f}")
    print(f"    ✓ Test Acc: {gb_results['test_accuracy']:.4f}")
    print(f"    ✓ ROC-AUC: {gb_results['roc_auc']:.4f}")
    
    # Top features
    print("\n[4/4] Feature importance (Random Forest):")
    feature_importance = sorted(
        rf_results['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for feature, importance in feature_importance[:5]:
        print(f"    {feature}: {importance:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return trainer, rf_results, gb_results


if __name__ == '__main__':
    trainer, rf_results, gb_results = run_training_pipeline()
