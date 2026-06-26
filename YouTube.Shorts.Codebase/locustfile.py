"""
Load testing for YouTube Shorts recommendation API using Locust
"""
from locust import HttpUser, task, between
import random
import json

class RecommendationAPIUser(HttpUser):
    """Locust user class for load testing the recommendation API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.user_id = f"user_{random.randint(1, 10000)}"
        self.video_pool = [f"video_{i}" for i in range(1, 1000)]
        
        # Test health endpoint first
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"Health check failed: {response.status_code}")
    
    @task(10)
    def predict_single(self):
        """Test single prediction endpoint (most common)"""
        payload = {
            "user_id": self.user_id,
            "video_id": random.choice(self.video_pool),
            "watch_time": random.uniform(5.0, 120.0),
            "hour_of_day": random.randint(0, 23)
        }
        
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 0 <= data.get("probability", -1) <= 1:
                    response.success()
                else:
                    response.failure("Invalid probability value")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def predict_batch_small(self):
        """Test small batch prediction (less common)"""
        batch_size = random.randint(2, 5)
        requests_list = []
        
        for _ in range(batch_size):
            requests_list.append({
                "user_id": f"user_{random.randint(1, 10000)}",
                "video_id": random.choice(self.video_pool),
                "watch_time": random.uniform(10.0, 90.0)
            })
        
        payload = {"requests": requests_list}
        
        with self.client.post("/predict/batch", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("batch_size") == batch_size:
                    response.success()
                else:
                    response.failure("Incorrect batch size in response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def predict_batch_large(self):
        """Test large batch prediction (occasional)"""
        batch_size = random.randint(20, 50)
        requests_list = []
        
        for _ in range(batch_size):
            requests_list.append({
                "user_id": f"user_{random.randint(1, 10000)}",
                "video_id": random.choice(self.video_pool),
                "watch_time": random.uniform(5.0, 160.0)
            })
        
        payload = {"requests": requests_list}
        
        with self.client.post("/predict/batch", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("batch_size") == batch_size:
                    response.success()
                else:
                    response.failure("Incorrect batch size in response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_model_info(self):
        """Test model info endpoint (occasional)"""
        with self.client.get("/model/info", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "model_type" in data and "feature_count" in data:
                    response.success()
                else:
                    response.failure("Missing required fields in model info")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_feature_importance(self):
        """Test feature importance endpoint (occasional)"""
        with self.client.get("/metrics/features", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health endpoint (monitoring)"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Service not healthy")
            else:
                response.failure(f"Status code: {response.status_code}")

class HighVolumeUser(HttpUser):
    """Simulates high-volume API users"""
    
    wait_time = between(0.1, 0.5)  # Faster requests
    weight = 1  # Lower weight (fewer of these users)
    
    def on_start(self):
        self.user_id = f"power_user_{random.randint(1, 100)}"
        self.video_pool = [f"video_{i}" for i in range(1, 10000)]
    
    @task
    def rapid_predictions(self):
        """Make rapid prediction requests"""
        payload = {
            "user_id": self.user_id,
            "video_id": random.choice(self.video_pool),
            "watch_time": random.uniform(1.0, 60.0)
        }
        
        self.client.post("/predict", json=payload)

class ErrorTestingUser(HttpUser):
    """Tests error handling and edge cases"""
    
    wait_time = between(5, 10)  # Slower, less frequent
    weight = 1  # Even fewer of these users
    
    @task(3)
    def valid_request(self):
        """Send valid requests most of the time"""
        payload = {
            "user_id": f"user_{random.randint(1, 1000)}",
            "video_id": f"video_{random.randint(1, 1000)}",
            "watch_time": random.uniform(5.0, 120.0)
        }
        self.client.post("/predict", json=payload)
    
    @task(1)
    def invalid_requests(self):
        """Send various invalid requests to test error handling"""
        invalid_payloads = [
            # Invalid user_id format
            {
                "user_id": "invalid_user",
                "video_id": "video_123",
                "watch_time": 30.0
            },
            # Negative watch time
            {
                "user_id": "user_123",
                "video_id": "video_123",
                "watch_time": -10.0
            },
            # Missing required fields
            {
                "user_id": "user_123",
                "watch_time": 30.0
            },
            # Invalid hour
            {
                "user_id": "user_123",
                "video_id": "video_123",
                "watch_time": 30.0,
                "hour_of_day": 25
            }
        ]
        
        payload = random.choice(invalid_payloads)
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 422:  # Expected validation error
                response.success()
            else:
                response.failure(f"Expected 422, got {response.status_code}")
    
    @task(1)
    def malformed_json(self):
        """Send malformed JSON"""
        with self.client.post("/predict", data="invalid json", 
                            headers={'Content-Type': 'application/json'},
                            catch_response=True) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"Expected 422, got {response.status_code}")