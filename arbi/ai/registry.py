"""
Model Registry for Versioned Model Management

This module provides a SQLite-based model registry for storing, versioning,
and retrieving ML models with full metadata, reproducibility information,
and audit trails.

Key Features:
- Model versioning with timestamps
- Dataset hash tracking for reproducibility
- Hyperparameter and performance logging
- Model artifact path management
- Query interface for latest/best models
"""

import sqlite3
import json
import hashlib
import pickle
import joblib
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Complete model metadata structure"""
    model_id: str
    symbol: str
    model_type: str
    version: str
    created_at: datetime
    
    # Training metadata
    dataset_hash: str
    training_samples: int
    feature_count: int
    random_seed: int
    
    # Model configuration
    hyperparameters: Dict[str, Any]
    preprocessing_config: Dict[str, Any]
    
    # Performance metrics
    validation_score: float
    test_score: Optional[float]
    cross_val_scores: List[float]
    performance_metrics: Dict[str, float]
    
    # File paths
    model_path: str
    scaler_path: str
    metadata_path: str
    
    # Additional info
    training_duration_seconds: float
    feature_importance: Optional[Dict[str, float]]
    notes: str = ""

class ModelRegistry:
    """SQLite-based model registry for production model management"""
    
    def __init__(self, db_path: str = "model_registry.db", models_dir: str = "models"):
        self.db_path = Path(db_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self._initialize_database()
        logger.info(f"Model registry initialized: {self.db_path}")
    
    def _initialize_database(self):
        """Initialize the model registry database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    
                    -- Training metadata
                    dataset_hash TEXT NOT NULL,
                    training_samples INTEGER NOT NULL,
                    feature_count INTEGER NOT NULL,
                    random_seed INTEGER NOT NULL,
                    
                    -- Model configuration (JSON)
                    hyperparameters TEXT NOT NULL,
                    preprocessing_config TEXT NOT NULL,
                    
                    -- Performance metrics
                    validation_score REAL NOT NULL,
                    test_score REAL,
                    cross_val_scores TEXT NOT NULL,  -- JSON array
                    performance_metrics TEXT NOT NULL,  -- JSON dict
                    
                    -- File paths
                    model_path TEXT NOT NULL,
                    scaler_path TEXT NOT NULL,
                    metadata_path TEXT NOT NULL,
                    
                    -- Additional info
                    training_duration_seconds REAL NOT NULL,
                    feature_importance TEXT,  -- JSON dict, optional
                    notes TEXT DEFAULT ''
                )
            """)
            
            # Create indexes for efficient queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON models(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON models(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_validation_score ON models(validation_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_score ON models(symbol, validation_score DESC)")
            
            logger.info("Model registry database schema initialized")
    
    def register_model(
        self,
        model_obj: Any,
        scaler_obj: Any,
        symbol: str,
        model_type: str,
        hyperparameters: Dict[str, Any],
        preprocessing_config: Dict[str, Any],
        validation_score: float,
        performance_metrics: Dict[str, float],
        dataset_hash: str,
        training_samples: int,
        feature_count: int,
        random_seed: int,
        cross_val_scores: List[float],
        training_duration_seconds: float,
        test_score: Optional[float] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        notes: str = ""
    ) -> str:
        """
        Register a new model in the registry with full metadata
        
        Returns:
            model_id: Unique identifier for the registered model
        """
        # Generate unique model ID
        timestamp = datetime.now(timezone.utc)
        model_id = f"{symbol}_{model_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{random_seed}"
        version = timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Create model directory
        model_dir = self.models_dir / symbol / timestamp.strftime('%Y%m%d_%H%M%S')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        model_path = model_dir / "model.pkl"
        scaler_path = model_dir / "scaler.pkl"
        metadata_path = model_dir / "meta.json"
        
        # Save model and scaler
        joblib.dump(model_obj, model_path)
        joblib.dump(scaler_obj, scaler_path)
        
        # Create metadata object
        metadata = ModelMetadata(
            model_id=model_id,
            symbol=symbol,
            model_type=model_type,
            version=version,
            created_at=timestamp,
            dataset_hash=dataset_hash,
            training_samples=training_samples,
            feature_count=feature_count,
            random_seed=random_seed,
            hyperparameters=hyperparameters,
            preprocessing_config=preprocessing_config,
            validation_score=validation_score,
            test_score=test_score,
            cross_val_scores=cross_val_scores,
            performance_metrics=performance_metrics,
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            metadata_path=str(metadata_path),
            training_duration_seconds=training_duration_seconds,
            feature_importance=feature_importance,
            notes=notes
        )
        
        # Save metadata JSON
        metadata_dict = asdict(metadata)
        metadata_dict['created_at'] = metadata_dict['created_at'].isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        # Insert into database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO models VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                model_id,
                symbol,
                model_type,
                version,
                timestamp.isoformat(),
                dataset_hash,
                training_samples,
                feature_count,
                random_seed,
                json.dumps(hyperparameters),
                json.dumps(preprocessing_config),
                validation_score,
                test_score,
                json.dumps(cross_val_scores),
                json.dumps(performance_metrics),
                str(model_path),
                str(scaler_path),
                str(metadata_path),
                training_duration_seconds,
                json.dumps(feature_importance) if feature_importance else None,
                notes
            ))
        
        logger.info(f"✅ Model registered: {model_id} (validation_score: {validation_score:.4f})")
        return model_id
    
    def get_latest_model(self, symbol: str, model_type: Optional[str] = None) -> Optional[ModelMetadata]:
        """Get the latest model for a symbol, optionally filtered by type"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT * FROM models 
                WHERE symbol = ?
            """
            params = [symbol]
            
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type)
                
            query += " ORDER BY created_at DESC LIMIT 1"
            
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                return self._row_to_metadata(row)
        
        return None
    
    def get_best_model(self, symbol: str, model_type: Optional[str] = None, metric: str = "validation_score") -> Optional[ModelMetadata]:
        """Get the best performing model for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if metric == "validation_score":
                order_col = "validation_score DESC"
            elif metric == "test_score":
                order_col = "test_score DESC"
            else:
                # For custom metrics, we'd need to parse the JSON
                order_col = "validation_score DESC"
                logger.warning(f"Custom metric {metric} not supported, using validation_score")
            
            query = f"""
                SELECT * FROM models 
                WHERE symbol = ?
            """
            params = [symbol]
            
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type)
                
            query += f" ORDER BY {order_col} LIMIT 1"
            
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                return self._row_to_metadata(row)
        
        return None
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelMetadata]:
        """Get a specific model by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_metadata(row)
        
        return None
    
    def list_models(
        self, 
        symbol: Optional[str] = None, 
        model_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ModelMetadata]:
        """List models with optional filtering"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM models WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type)
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_metadata(row) for row in rows]
    
    def load_model(self, model_id: str) -> Tuple[Any, Any, ModelMetadata]:
        """Load model, scaler, and metadata by ID"""
        metadata = self.get_model_by_id(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Load model and scaler
        model_obj = joblib.load(metadata.model_path)
        scaler_obj = joblib.load(metadata.scaler_path)
        
        logger.info(f"Loaded model {model_id} from {metadata.model_path}")
        return model_obj, scaler_obj, metadata
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry and filesystem"""
        metadata = self.get_model_by_id(model_id)
        if not metadata:
            return False
        
        # Delete from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
        
        # Delete files
        try:
            Path(metadata.model_path).unlink(missing_ok=True)
            Path(metadata.scaler_path).unlink(missing_ok=True)
            Path(metadata.metadata_path).unlink(missing_ok=True)
            
            # Try to remove directory if empty
            model_dir = Path(metadata.model_path).parent
            try:
                model_dir.rmdir()
            except OSError:
                pass  # Directory not empty
                
        except Exception as e:
            logger.warning(f"Error deleting model files for {model_id}: {e}")
        
        logger.info(f"Deleted model {model_id}")
        return True
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total models
            total_count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            
            # Models by symbol
            symbol_counts = dict(conn.execute("""
                SELECT symbol, COUNT(*) FROM models GROUP BY symbol
            """).fetchall())
            
            # Models by type
            type_counts = dict(conn.execute("""
                SELECT model_type, COUNT(*) FROM models GROUP BY model_type
            """).fetchall())
            
            # Performance statistics
            perf_stats = conn.execute("""
                SELECT 
                    AVG(validation_score) as avg_val_score,
                    MAX(validation_score) as max_val_score,
                    MIN(validation_score) as min_val_score
                FROM models
            """).fetchone()
            
        return {
            "total_models": total_count,
            "models_by_symbol": symbol_counts,
            "models_by_type": type_counts,
            "performance_stats": {
                "avg_validation_score": perf_stats[0],
                "max_validation_score": perf_stats[1], 
                "min_validation_score": perf_stats[2]
            }
        }
    
    def _row_to_metadata(self, row: sqlite3.Row) -> ModelMetadata:
        """Convert database row to ModelMetadata object"""
        return ModelMetadata(
            model_id=row["model_id"],
            symbol=row["symbol"],
            model_type=row["model_type"],
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            dataset_hash=row["dataset_hash"],
            training_samples=row["training_samples"],
            feature_count=row["feature_count"],
            random_seed=row["random_seed"],
            hyperparameters=json.loads(row["hyperparameters"]),
            preprocessing_config=json.loads(row["preprocessing_config"]),
            validation_score=row["validation_score"],
            test_score=row["test_score"],
            cross_val_scores=json.loads(row["cross_val_scores"]),
            performance_metrics=json.loads(row["performance_metrics"]),
            model_path=row["model_path"],
            scaler_path=row["scaler_path"],
            metadata_path=row["metadata_path"],
            training_duration_seconds=row["training_duration_seconds"],
            feature_importance=json.loads(row["feature_importance"]) if row["feature_importance"] else None,
            notes=row["notes"]
        )

def compute_dataset_hash(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> str:
    """Compute deterministic hash of training dataset for reproducibility"""
    # Create a string representation of the dataset
    data_str = f"X_shape:{X.shape}|X_sum:{X.sum():.6f}|y_sum:{y.sum():.6f}|features:{','.join(sorted(feature_names))}"
    
    # Add some sample data points for uniqueness
    if len(X) > 10:
        sample_indices = [0, len(X)//4, len(X)//2, 3*len(X)//4, len(X)-1]
        sample_data = X[sample_indices].flatten()
        data_str += f"|sample_data:{sample_data.sum():.6f}"
    
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]

# Global registry instance
_registry: Optional[ModelRegistry] = None

def get_model_registry(db_path: str = "model_registry.db", models_dir: str = "models") -> ModelRegistry:
    """Get global model registry instance"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(db_path, models_dir)
    return _registry

# Example usage and testing
def test_model_registry():
    """Test the model registry functionality"""
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import time
    
    logger.info("Testing model registry...")
    
    # Create test registry
    registry = ModelRegistry("test_registry.db", "test_models")
    
    # Create dummy model and scaler
    model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    scaler = StandardScaler()
    
    # Dummy training data
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randn(100)
    
    # "Train" the model
    start_time = time.time()
    X_scaled = scaler.fit_transform(X_dummy)
    model.fit(X_scaled, y_dummy)
    training_duration = time.time() - start_time
    
    # Create dummy cross-validation scores
    cv_scores = [0.85, 0.83, 0.87, 0.84, 0.86]
    
    # Compute dataset hash
    feature_names = [f"feature_{i}" for i in range(X_dummy.shape[1])]
    dataset_hash = compute_dataset_hash(X_dummy, y_dummy, feature_names)
    
    # Register model
    model_id = registry.register_model(
        model_obj=model,
        scaler_obj=scaler,
        symbol="TEST",
        model_type="lightgbm",
        hyperparameters={"n_estimators": 100, "random_state": 42},
        preprocessing_config={"scaler": "StandardScaler"},
        validation_score=0.85,
        performance_metrics={"mae": 0.15, "rmse": 0.22},
        dataset_hash=dataset_hash,
        training_samples=len(X_dummy),
        feature_count=X_dummy.shape[1],
        random_seed=42,
        cross_val_scores=cv_scores,
        training_duration_seconds=training_duration,
        test_score=0.83,
        feature_importance={"feature_0": 0.3, "feature_1": 0.2},
        notes="Test model for registry validation"
    )
    
    print(f"✅ Registered model: {model_id}")
    
    # Test retrieval
    latest_model = registry.get_latest_model("TEST")
    assert latest_model is not None
    assert latest_model.model_id == model_id
    
    print(f"✅ Retrieved latest model: {latest_model.model_id}")
    
    # Test loading
    loaded_model, loaded_scaler, loaded_metadata = registry.load_model(model_id)
    assert loaded_metadata.model_id == model_id
    
    print(f"✅ Loaded model artifacts for: {loaded_metadata.model_id}")
    
    # Test statistics
    stats = registry.get_model_statistics()
    print(f"Registry stats: {stats}")
    
    # Cleanup test files
    import shutil
    Path("test_registry.db").unlink(missing_ok=True)
    shutil.rmtree("test_models", ignore_errors=True)
    
    print("✅ Model registry test completed successfully!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_model_registry()
