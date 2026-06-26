"""
Data validation and quality checks for the recommendation system
"""
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Schema definitions
users_schema = DataFrameSchema({
    "user_id": Column(str, checks=[
        Check(lambda x: x.str.startswith('user_'), error="User ID must start with 'user_'"),
        Check(lambda x: x.nunique() == len(x), error="User IDs must be unique")
    ]),
    "age": Column(int, checks=[
        Check(lambda x: x >= 13, error="Age must be >= 13"),
        Check(lambda x: x <= 100, error="Age must be <= 100")
    ]),
    "gender": Column(str, checks=[
        Check.isin(['M', 'F', 'Other'])
    ]),
    "preferred_category": Column(str, checks=[
        Check.isin(['Gaming', 'Music', 'Comedy', 'Sports', 'Education', 'Travel', 'Food', 'Fashion'])
    ])
})

videos_schema = DataFrameSchema({
    "video_id": Column(str, checks=[
        Check(lambda x: x.str.startswith('video_'), error="Video ID must start with 'video_'"),
        Check(lambda x: x.nunique() == len(x), error="Video IDs must be unique")
    ]),
    "creator_id": Column(int, checks=[
        Check(lambda x: x >= 0, error="Creator ID must be non-negative")
    ]),
    "category": Column(str, checks=[
        Check.isin(['Gaming', 'Music', 'Comedy', 'Sports', 'Education', 'Travel', 'Food', 'Fashion'])
    ]),
    "duration_sec": Column(float, checks=[
        Check(lambda x: x >= 1, error="Duration must be >= 1 second"),
        Check(lambda x: x <= 180, error="Duration must be <= 180 seconds for Shorts")
    ])
})

interactions_schema = DataFrameSchema({
    "user_id": Column(str, checks=[
        Check(lambda x: x.str.startswith('user_'), error="User ID must start with 'user_'")
    ]),
    "video_id": Column(str, checks=[
        Check(lambda x: x.str.startswith('video_'), error="Video ID must start with 'video_'")
    ]),
    "watch_time": Column(float, checks=[
        Check(lambda x: x >= 0, error="Watch time cannot be negative"),
        Check(lambda x: x <= 200, error="Watch time seems unreasonably high")
    ]),
    "liked": Column(int, checks=[
        Check.isin([0, 1])
    ]),
    "shared": Column(int, checks=[
        Check.isin([0, 1])
    ]),
    "timestamp": Column('datetime64[ns]')
})

# Business rule validations
def validate_business_rules(interactions_df, users_df, videos_df):
    """Validate business-specific rules"""
    errors = []
    
    # Check engagement rates
    like_rate = interactions_df['liked'].mean()
    if like_rate < 0.01:
        errors.append(f"Like rate too low: {like_rate:.4f} < 0.01")
    elif like_rate > 0.5:
        errors.append(f"Like rate suspiciously high: {like_rate:.4f} > 0.5")
    
    share_rate = interactions_df['shared'].mean()
    if share_rate < 0.005:
        errors.append(f"Share rate too low: {share_rate:.4f} < 0.005")
    elif share_rate > 0.2:
        errors.append(f"Share rate suspiciously high: {share_rate:.4f} > 0.2")
    
    # Check for data leakage - future interactions
    latest_allowed = pd.Timestamp.now()
    future_interactions = interactions_df[interactions_df['timestamp'] > latest_allowed]
    if len(future_interactions) > 0:
        errors.append(f"Found {len(future_interactions)} interactions in the future")
    
    # Check foreign key constraints
    missing_users = set(interactions_df['user_id']) - set(users_df['user_id'])
    if missing_users:
        errors.append(f"Found {len(missing_users)} interactions with missing users")
    
    missing_videos = set(interactions_df['video_id']) - set(videos_df['video_id'])
    if missing_videos:
        errors.append(f"Found {len(missing_videos)} interactions with missing videos")
    
    # Check for suspicious patterns
    user_interaction_counts = interactions_df['user_id'].value_counts()
    if user_interaction_counts.max() > 1000:
        errors.append(f"User with {user_interaction_counts.max()} interactions (possible bot)")
    
    # Check watch time vs video duration
    merged = interactions_df.merge(videos_df[['video_id', 'duration_sec']], on='video_id')
    excessive_watch = merged[merged['watch_time'] > merged['duration_sec'] * 1.1]  # 10% tolerance
    if len(excessive_watch) > len(merged) * 0.05:  # More than 5% of interactions
        errors.append(f"High rate of watch times exceeding video duration: {len(excessive_watch)/len(merged):.3f}")
    
    return errors

def validate_data(data_dir='data'):
    """Main validation function"""
    data_path = Path(data_dir)
    
    try:
        # Load data
        logger.info("Loading datasets...")
        users_df = pd.read_csv(data_path / 'users.csv')
        videos_df = pd.read_csv(data_path / 'videos.csv')
        interactions_df = pd.read_csv(data_path / 'interactions_cleaned.csv')
        
        # Convert timestamp
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        
        logger.info(f"Loaded {len(users_df)} users, {len(videos_df)} videos, {len(interactions_df)} interactions")
        
        # Schema validation
        logger.info("Validating schemas...")
        
        try:
            users_schema.validate(users_df)
            logger.info("✓ Users schema validation passed")
        except pa.errors.SchemaError as e:
            logger.error(f"Users schema validation failed: {e}")
            return False
        
        try:
            videos_schema.validate(videos_df)
            logger.info("✓ Videos schema validation passed")
        except pa.errors.SchemaError as e:
            logger.error(f"Videos schema validation failed: {e}")
            return False
        
        try:
            interactions_schema.validate(interactions_df)
            logger.info("✓ Interactions schema validation passed")
        except pa.errors.SchemaError as e:
            logger.error(f"Interactions schema validation failed: {e}")
            return False
        
        # Business rules validation
        logger.info("Validating business rules...")
        business_errors = validate_business_rules(interactions_df, users_df, videos_df)
        
        if business_errors:
            for error in business_errors:
                logger.warning(f"Business rule violation: {error}")
            # Don't fail on warnings, just log them
        
        # Data quality checks
        logger.info("Performing data quality checks...")
        
        # Check for duplicates
        interaction_duplicates = interactions_df.duplicated(subset=['user_id', 'video_id', 'timestamp']).sum()
        if interaction_duplicates > 0:
            logger.warning(f"Found {interaction_duplicates} duplicate interactions")
        
        # Check for missing values
        for df_name, df in [('users', users_df), ('videos', videos_df), ('interactions', interactions_df)]:
            missing = df.isnull().sum().sum()
            if missing > 0:
                logger.warning(f"Found {missing} missing values in {df_name}")
        
        # Summary statistics
        logger.info("=== Data Quality Summary ===")
        logger.info(f"Like rate: {interactions_df['liked'].mean():.4f}")
        logger.info(f"Share rate: {interactions_df['shared'].mean():.4f}")
        logger.info(f"Avg watch time: {interactions_df['watch_time'].mean():.1f}s")
        logger.info(f"Date range: {interactions_df['timestamp'].min()} to {interactions_df['timestamp'].max()}")
        
        # Category alignment
        merged = interactions_df.merge(users_df[['user_id', 'preferred_category']], on='user_id')
        merged = merged.merge(videos_df[['video_id', 'category']], on='video_id')
        alignment = (merged['preferred_category'] == merged['category']).mean()
        logger.info(f"User-video category alignment: {alignment:.3f}")
        
        logger.info("✓ All validations passed!")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return False

def validate_processed_data(df):
    """Validate processed/engineered features"""
    required_features = [
        'watch_time_normalized', 'video_popularity', 'hour_of_day',
        'day_of_week', 'session_length', 'is_weekend'
    ]
    
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check feature ranges
    if not (0 <= df['watch_time_normalized'].min() and df['watch_time_normalized'].max() <= 1):
        raise ValueError("watch_time_normalized should be in [0, 1]")
    
    if not (0 <= df['hour_of_day'].min() and df['hour_of_day'].max() <= 23):
        raise ValueError("hour_of_day should be in [0, 23]")
    
    if not (0 <= df['day_of_week'].min() and df['day_of_week'].max() <= 6):
        raise ValueError("day_of_week should be in [0, 6]")
    
    logger.info("✓ Processed data validation passed")
    return True

if __name__ == "__main__":
    success = validate_data()
    if not success:
        sys.exit(1)