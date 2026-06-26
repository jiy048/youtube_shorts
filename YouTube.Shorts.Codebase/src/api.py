"""
Production-ready FastAPI service for YouTube Shorts recommendation
"""
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
import logging
import json
import redis
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import pickle
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
import uvicorn
import os
from feature_engineering import FeatureEngineer

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for models and artifacts
model = None
feature_engineer = None
feature_columns = None
redis_client = None

class PredictionRequest(BaseModel):
    user_id: str = Field(..., description="User ID (e.g., user_123)")
    video_id: str = Field(..., description="Video ID (e.g., video_456)")
    watch_time: float = Field(..., ge=0, le=300, description="Watch time in seconds")
    hour_of_day: Optional[int] = Field(None, ge=0, le=23, description="Hour of day (0-23)")
    session_context: Optional[Dict[str, Any]] = Field(None, description="Additional session context")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v.startswith('user_'):
            raise ValueError('User ID must start with user_')
        return v
    
    @validator('video_id')
    def validate_video_id(cls, v):
        if not v.startswith('video_'):
            raise ValueError('Video ID must start with video_')
        return v

class PredictionResponse(BaseModel):
    user_id: str
    video_id: str
    probability: float = Field(..., ge=0, le=1)
    confidence: str
    model_version: str
    response_time_ms: float
    timestamp: datetime

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., max_items=100)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    redis_connected: bool
    uptime_seconds: float

# Rate limiting
class RateLimiter:
    def __init__(self, redis_client, max_requests: int = 100, window_seconds: int = 60):
        self.redis_client = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    async def is_allowed(self, client_id: str) -> bool:
        if not self.redis_client:
            return True
        
        try:
            pipe = self.redis_client.pipeline()
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove old entries
            pipe.zremrangebyscore(f"rate_limit:{client_id}", 0, window_start)
            # Count current requests
            pipe.zcard(f"rate_limit:{client_id}")
            # Add current request
            pipe.zadd(f"rate_limit:{client_id}", {str(now): now})
            # Set expiry
            pipe.expire(f"rate_limit:{client_id}", self.window_seconds)
            
            results = pipe.execute()
            current_requests = results[1]
            
            return current_requests < self.max_requests
        except Exception as e:
            logger.warning(f"Rate limiting error: {e}")
            return True

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_models()
    await connect_redis()
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    if redis_client:
        redis_client.close()
    logger.info("Application shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Shorts Recommendation API",
    description="Production-ready recommendation system for YouTube Shorts",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Global variables for startup time
startup_time = time.time()

async def load_models():
    """Load ML models and artifacts"""
    global model, feature_engineer, feature_columns
    
    try:
        models_dir = Path('models')
        
        # Find the best model
        model_files = list(models_dir.glob('best_model_*.pkl'))
        if not model_files:
            raise FileNotFoundError("No trained model found")
        
        model_path = model_files[0]  # Use the first best model found
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load feature columns
        with open(models_dir / 'feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        logger.info(f"Loaded {len(feature_columns)} feature columns")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        feature_engineer.load_artifacts('models')
        logger.info("Loaded feature engineering artifacts")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

async def connect_redis():
    """Connect to Redis for caching and rate limiting"""
    global redis_client
    
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}")
        redis_client = None

def get_client_id(request: Request) -> str:
    """Extract client ID for rate limiting"""
    # Use IP address as client ID (you might want to use API keys in production)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    return request.client.host

async def check_rate_limit(request: Request):
    """Check rate limiting"""
    if redis_client:
        client_id = get_client_id(request)
        rate_limiter = RateLimiter(redis_client)
        
        if not await rate_limiter.is_allowed(client_id):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )

def preprocess_request(request: PredictionRequest) -> pd.DataFrame:
    """Preprocess a single prediction request"""
    # Create a minimal dataframe with the request data
    data = {
        'user_id': [request.user_id],
        'video_id': [request.video_id],
        'watch_time': [request.watch_time],
        'timestamp': [datetime.now()],
        'liked': [0],  # Placeholder for feature engineering
        'shared': [0]  # Placeholder for feature engineering
    }
    
    if request.hour_of_day is not None:
        data['hour_of_day'] = [request.hour_of_day]
    else:
        data['hour_of_day'] = [datetime.now().hour]
    
    df = pd.DataFrame(data)
    
    # Add some synthetic features for prediction
    # In production, you'd fetch these from your database
    df['age'] = 25  # Default user age
    df['gender'] = 'M'  # Default gender
    df['preferred_category'] = 'Gaming'  # Default preference
    df['category'] = 'Gaming'  # Default video category
    df['duration_sec'] = 60  # Default video duration
    df['day_of_week'] = datetime.now().weekday()
    df['is_weekend'] = 1 if datetime.now().weekday() >= 5 else 0
    
    # Basic feature engineering
    df['watch_time_normalized'] = df['watch_time'] / 160  # Normalize to max Shorts length
    df['video_popularity'] = 100  # Default popularity
    df['completion_rate'] = np.clip(df['watch_time'] / df['duration_sec'], 0, 1)
    df['session_length'] = 1  # Default session length
    df['recency_days'] = 0  # Current interaction
    df['recency_normalized'] = 0
    
    # Create additional features that the model expects
    for col in feature_columns:
        if col not in df.columns:
            if 'gender_' in col or 'category_' in col or 'day_period_' in col:
                df[col] = 0  # One-hot encoded features default to 0
            elif 'emb_' in col:
                df[col] = 0  # Embedding features default to 0
            else:
                df[col] = 0  # Other features default to 0
    
    # Ensure we have all required features in the right order
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    return df

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability"""
    if probability > 0.8 or probability < 0.2:
        return "high"
    elif probability > 0.6 or probability < 0.4:
        return "medium"
    else:
        return "low"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=model is not None,
        redis_connected=redis_client is not None,
        uptime_seconds=time.time() - startup_time
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit)
):
    """Make a single prediction"""
    start_time = time.time()
    
    try:
        # Log request
        logger.info(f"Prediction request: {request.user_id} -> {request.video_id}")
        
        # Check cache first
        cache_key = f"pred:{request.user_id}:{request.video_id}:{request.watch_time}"
        cached_result = None
        
        if redis_client:
            try:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    cached_result = json.loads(cached_result)
                    logger.info("Returning cached prediction")
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        if cached_result:
            response_time = (time.time() - start_time) * 1000
            return PredictionResponse(
                user_id=request.user_id,
                video_id=request.video_id,
                probability=cached_result['probability'],
                confidence=cached_result['confidence'],
                model_version="1.0.0",
                response_time_ms=response_time,
                timestamp=datetime.now()
            )
        
        # Preprocess request
        df = preprocess_request(request)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(df)[0][1])
        else:  # LightGBM
            probability = float(model.predict(df)[0])
        
        # Ensure probability is in valid range
        probability = np.clip(probability, 0.0, 1.0)
        
        confidence = get_confidence_level(probability)
        response_time = (time.time() - start_time) * 1000
        
        # Cache result
        if redis_client:
            try:
                cache_data = {
                    'probability': probability,
                    'confidence': confidence
                }
                redis_client.setex(cache_key, 300, json.dumps(cache_data))  # 5 min cache
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        # Log metrics in background
        background_tasks.add_task(
            log_prediction_metrics,
            request.user_id,
            request.video_id,
            probability,
            response_time
        )
        
        return PredictionResponse(
            user_id=request.user_id,
            video_id=request.video_id,
            probability=probability,
            confidence=confidence,
            model_version="1.0.0",
            response_time_ms=response_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(
    request: BatchPredictionRequest,
    _: None = Depends(check_rate_limit)
):
    """Make batch predictions"""
    start_time = time.time()
    
    try:
        logger.info(f"Batch prediction request: {len(request.requests)} items")
        
        if len(request.requests) > 100:
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
        
        results = []
        
        for req in request.requests:
            try:
                # Preprocess individual request
                df = preprocess_request(req)
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    probability = float(model.predict_proba(df)[0][1])
                else:  # LightGBM
                    probability = float(model.predict(df)[0])
                
                probability = np.clip(probability, 0.0, 1.0)
                confidence = get_confidence_level(probability)
                
                results.append({
                    'user_id': req.user_id,
                    'video_id': req.video_id,
                    'probability': probability,
                    'confidence': confidence
                })
                
            except Exception as e:
                logger.warning(f"Failed to process request {req.user_id}->{req.video_id}: {e}")
                results.append({
                    'user_id': req.user_id,
                    'video_id': req.video_id,
                    'error': str(e)
                })
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            'results': results,
            'batch_size': len(request.requests),
            'response_time_ms': response_time,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        'model_type': type(model).__name__ if model else None,
        'feature_count': len(feature_columns) if feature_columns else 0,
        'model_version': "1.0.0",
        'features': feature_columns[:10] if feature_columns else [],  # First 10 features
        'status': 'loaded' if model else 'not_loaded'
    }

@app.get("/metrics/features")
async def feature_importance():
    """Get feature importance (if available)"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = [
                {'feature': feature_columns[i], 'importance': float(importance[i])}
                for i in range(len(feature_columns))
            ]
            # Sort by importance
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            return {'feature_importance': feature_importance[:20]}  # Top 20
        else:
            return {'message': 'Feature importance not available for this model type'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not get feature importance: {str(e)}")

async def log_prediction_metrics(user_id: str, video_id: str, probability: float, response_time: float):
    """Log prediction metrics for monitoring"""
    try:
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'video_id': video_id,
            'probability': probability,
            'response_time_ms': response_time,
            'model_version': '1.0.0'
        }
        
        # Log to structured logger
        logger.info(f"PREDICTION_METRICS: {json.dumps(metrics)}")
        
        # Store in Redis for analytics (optional)
        if redis_client:
            redis_client.lpush('prediction_metrics', json.dumps(metrics))
            redis_client.ltrim('prediction_metrics', 0, 9999)  # Keep last 10k metrics
            
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': exc.detail,
            'timestamp': datetime.now().isoformat(),
            'path': str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat(),
            'path': str(request.url)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        access_log=True
    )