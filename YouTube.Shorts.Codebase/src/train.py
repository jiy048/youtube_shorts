"""
Training pipeline for YouTube Shorts recommendation system
Uses latest ML practices and optimizations
"""
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
from feature_engineering import FeatureEngineer
from data_validation import validate_processed_data
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, mlflow_tracking_uri="http://localhost:5000"):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.best_model = None
        self.best_score = 0
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("youtube_shorts_recommendation")
        
    def prepare_features(self, df, target_col='label'):
        """Prepare features for training"""
        logger.info("Preparing features...")
        
        # Exclude non-feature columns
        exclude_cols = [
            'user_id', 'video_id', 'timestamp', 'label', 'label_engagement', 
            'label_completion', 'day_period'  # Categorical already encoded
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle any remaining categorical columns
        for col in feature_cols:
            if df[col].dtype == 'object':
                logger.warning(f"Found categorical column {col}, removing from features")
                feature_cols.remove(col)
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        logger.info(f"Prepared {len(feature_cols)} features for training")
        return X, y
    
    def train_baseline_models(self, X_train, y_train, X_val, y_val):
        """Train baseline models"""
        logger.info("Training baseline models...")
        
        models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000,
                solver='lbfgs'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            with mlflow.start_run(run_name=f"baseline_{name}"):
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_pred_proba)
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("auc", auc)
                mlflow.log_param("model_type", name)
                
                # Log model
                signature = infer_signature(X_train, y_pred_proba)
                mlflow.sklearn.log_model(model, name, signature=signature)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return results
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model with advanced features"""
        logger.info("Training LightGBM model...")
        
        # Prepare data for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True  # Handle class imbalance
        }
        
        with mlflow.start_run(run_name="lightgbm_baseline"):
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Predictions
            y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Metrics
            accuracy = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # Log metrics and parameters
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc", auc)
            mlflow.log_params(params)
            
            # Log model
            mlflow.lightgbm.log_model(model, "lightgbm_model")
            
            logger.info(f"LightGBM - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            return {
                'model': model,
                'accuracy': accuracy,
                'auc': auc
            }
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Hyperparameter optimization using Optuna"""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'verbose': -1,
                'random_state': 42,
                'is_unbalance': True
            }
            
            # Prepare data
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=500,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=0)  # Silent
                ]
            )
            
            # Predict and calculate AUC
            y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            return auc
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True
        })
        
        logger.info(f"Best AUC: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        with mlflow.start_run(run_name="lightgbm_optimized"):
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                best_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Final metrics
            y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # Log everything
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc", auc)
            mlflow.log_params(best_params)
            mlflow.lightgbm.log_model(model, "lightgbm_optimized")
            
            logger.info(f"Optimized LightGBM - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            return {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'params': best_params
            }
    
    def evaluate_model(self, model, X_test, y_test, model_name="model"):
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:  # LightGBM
            y_pred_proba = model.predict(X_test)
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"{model_name} Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  Precision: {report['1']['precision']:.4f}")
        logger.info(f"  Recall: {report['1']['recall']:.4f}")
        logger.info(f"  F1-Score: {report['1']['f1-score']:.4f}")
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score']
        }
    
    def save_model(self, model, model_name, metrics=None):
        """Save model and metadata"""
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / f'{model_name}.pkl'
        joblib.dump(model, model_path)
        
        # Save feature columns
        with open(models_dir / 'feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'feature_count': len(self.feature_columns),
            'metrics': metrics or {}
        }
        
        with open(models_dir / f'{model_name}_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved: {model_path}")
        
        return model_path
    
    def train_pipeline(self, df, target_col='label', test_size=0.2, optimize=True):
        """Complete training pipeline"""
        logger.info("Starting training pipeline...")
        
        # Prepare features
        X, y = self.prepare_features(df, target_col)
        
        # Train-test split (chronological)
        # Sort by timestamp first for realistic split
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            split_idx = int(len(df_sorted) * (1 - test_size))
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            # Further split training into train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
        logger.info(f"Target distribution - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
        
        # Train baseline models
        baseline_results = self.train_baseline_models(X_train, y_train, X_val, y_val)
        
        # Train LightGBM
        lgb_result = self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Hyperparameter optimization
        if optimize:
            lgb_optimized = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            lgb_optimized = lgb_result
        
        # Find best model
        all_results = {**baseline_results, 'lightgbm': lgb_result, 'lightgbm_optimized': lgb_optimized}
        best_model_name = max(all_results.keys(), key=lambda k: all_results[k]['auc'])
        self.best_model = all_results[best_model_name]['model']
        self.best_score = all_results[best_model_name]['auc']
        
        logger.info(f"Best model: {best_model_name} with AUC: {self.best_score:.4f}")
        
        # Final evaluation on test set
        test_metrics = self.evaluate_model(self.best_model, X_test, y_test, best_model_name)
        
        # Save best model
        model_path = self.save_model(self.best_model, f'best_model_{best_model_name}', test_metrics)
        
        return {
            'best_model': self.best_model,
            'best_model_name': best_model_name,
            'test_metrics': test_metrics,
            'model_path': model_path,
            'all_results': all_results
        }

def main():
    """Main training execution"""
    # Check if processed data exists
    processed_data_path = Path('data/processed_interactions.csv')
    
    if not processed_data_path.exists():
        logger.info("Processed data not found. Running feature engineering...")
        
        # Load raw data
        users_df = pd.read_csv('data/users.csv')
        videos_df = pd.read_csv('data/videos.csv')
        interactions_df = pd.read_csv('data/interactions_cleaned.csv')
        
        # Run feature engineering
        fe = FeatureEngineer()
        processed_df = fe.process_features(interactions_df, users_df, videos_df)
        processed_df.to_csv(processed_data_path, index=False)
        fe.save_artifacts()
    else:
        logger.info("Loading processed data...")
        processed_df = pd.read_csv(processed_data_path)
    
    # Validate processed data
    validate_processed_data(processed_df)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train models
    results = trainer.train_pipeline(processed_df, target_col='label', optimize=True)
    
    logger.info("Training complete!")
    logger.info(f"Best model: {results['best_model_name']}")
    logger.info(f"Test AUC: {results['test_metrics']['auc']:.4f}")
    logger.info(f"Model saved to: {results['model_path']}")

if __name__ == "__main__":
    main()