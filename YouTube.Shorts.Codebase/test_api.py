"""
Comprehensive test suite for the YouTube Shorts recommendation API
"""
import pytest
import asyncio
import requests
import json
import time
from typing import Dict, List
import pandas as pd
import numpy as np

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

class TestAPI:
    """Test suite for the recommendation API"""
    
    @pytest.fixture(scope="class")
    def api_client(self):
        """Create a test client"""
        return requests.Session()
    
    def test_health_check(self, api_client):
        """Test the health endpoint"""
        response = api_client.get(f"{API_BASE_URL}/health", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["redis_connected"], bool)
    
    def test_model_info(self, api_client):
        """Test the model info endpoint"""
        response = api_client.get(f"{API_BASE_URL}/model/info", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "model_type" in data
        assert "feature_count" in data
        assert "model_version" in data
        assert "status" in data
    
    def test_single_prediction_valid(self, api_client):
        """Test single prediction with valid input"""
        payload = {
            "user_id": "user_123",
            "video_id": "video_456",
            "watch_time": 45.5,
            "hour_of_day": 14
        }
        
        response = api_client.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["user_id"] == payload["user_id"]
        assert data["video_id"] == payload["video_id"]
        assert 0 <= data["probability"] <= 1
        assert data["confidence"] in ["low", "medium", "high"]
        assert data["model_version"] == "1.0.0"
        assert data["response_time_ms"] > 0
        assert "timestamp" in data
    
    def test_single_prediction_minimal(self, api_client):
        """Test single prediction with minimal required fields"""
        payload = {
            "user_id": "user_999",
            "video_id": "video_888",
            "watch_time": 30.0
        }
        
        response = api_client.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert 0 <= data["probability"] <= 1
    
    def test_single_prediction_edge_cases(self, api_client):
        """Test single prediction with edge case values"""
        edge_cases = [
            {
                "user_id": "user_0",
                "video_id": "video_0",
                "watch_time": 0.0,  # Minimum watch time
                "hour_of_day": 0
            },
            {
                "user_id": "user_9999",
                "video_id": "video_9999",
                "watch_time": 300.0,  # Maximum watch time
                "hour_of_day": 23
            }
        ]
        
        for payload in edge_cases:
            response = api_client.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            assert response.status_code == 200
            data = response.json()
            assert 0 <= data["probability"] <= 1
    
    def test_single_prediction_invalid_inputs(self, api_client):
        """Test single prediction with invalid inputs"""
        invalid_cases = [
            # Invalid user_id format
            {
                "user_id": "invalid_user",
                "video_id": "video_123",
                "watch_time": 30.0
            },
            # Invalid video_id format
            {
                "user_id": "user_123",
                "video_id": "invalid_video",
                "watch_time": 30.0
            },
            # Negative watch time
            {
                "user_id": "user_123",
                "video_id": "video_123",
                "watch_time": -10.0
            },
            # Invalid hour
            {
                "user_id": "user_123",
                "video_id": "video_123",
                "watch_time": 30.0,
                "hour_of_day": 25
            },
            # Missing required fields
            {
                "user_id": "user_123",
                "watch_time": 30.0
            }
        ]
        
        for payload in invalid_cases:
            response = api_client.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            assert response.status_code == 422  # Validation error
    
    def test_batch_prediction_valid(self, api_client):
        """Test batch prediction with valid inputs"""
        payload = {
            "requests": [
                {
                    "user_id": "user_1",
                    "video_id": "video_1",
                    "watch_time": 45.0
                },
                {
                    "user_id": "user_2",
                    "video_id": "video_2",
                    "watch_time": 60.0,
                    "hour_of_day": 15
                },
                {
                    "user_id": "user_3",
                    "video_id": "video_3",
                    "watch_time": 30.0
                }
            ]
        }
        
        response = api_client.post(
            f"{API_BASE_URL}/predict/batch",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["results"]) == 3
        assert data["batch_size"] == 3
        assert data["response_time_ms"] > 0
        
        for result in data["results"]:
            if "error" not in result:
                assert 0 <= result["probability"] <= 1
                assert result["confidence"] in ["low", "medium", "high"]
    
    def test_batch_prediction_large(self, api_client):
        """Test batch prediction with maximum allowed size"""
        requests_list = []
        for i in range(100):  # Maximum batch size
            requests_list.append({
                "user_id": f"user_{i}",
                "video_id": f"video_{i}",
                "watch_time": 30.0 + (i % 60)
            })
        
        payload = {"requests": requests_list}
        
        response = api_client.post(
            f"{API_BASE_URL}/predict/batch",
            json=payload,
            timeout=TEST_TIMEOUT * 2  # Longer timeout for large batch
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["batch_size"] == 100
    
    def test_batch_prediction_oversized(self, api_client):
        """Test batch prediction with oversized batch"""
        requests_list = []
        for i in range(101):  # Over maximum batch size
            requests_list.append({
                "user_id": f"user_{i}",
                "video_id": f"video_{i}",
                "watch_time": 30.0
            })
        
        payload = {"requests": requests_list}
        
        response = api_client.post(
            f"{API_BASE_URL}/predict/batch",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 400  # Bad request
    
    def test_feature_importance(self, api_client):
        """Test feature importance endpoint"""
        response = api_client.get(f"{API_BASE_URL}/metrics/features", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        # Response format depends on model type
        assert "feature_importance" in data or "message" in data
    
    def test_prediction_consistency(self, api_client):
        """Test that identical requests return consistent results"""
        payload = {
            "user_id": "user_consistency_test",
            "video_id": "video_consistency_test",
            "watch_time": 45.0,
            "hour_of_day": 12
        }
        
        # Make multiple requests
        responses = []
        for _ in range(3):
            response = api_client.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            assert response.status_code == 200
            responses.append(response.json())
            time.sleep(0.1)  # Small delay between requests
        
        # Check consistency (allowing for small floating point differences)
        first_prob = responses[0]["probability"]
        for response in responses[1:]:
            assert abs(response["probability"] - first_prob) < 1e-6
    
    def test_response_time_performance(self, api_client):
        """Test API response time performance"""
        payload = {
            "user_id": "user_perf_test",
            "video_id": "video_perf_test",
            "watch_time": 45.0
        }
        
        # Measure response time
        start_time = time.time()
        response = api_client.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be under 1 second
        response_time = end_time - start_time
        assert response_time < 1.0, f"Response time {response_time:.3f}s exceeds 1.0s threshold"
        
        # Check internal response time metric
        data = response.json()
        assert data["response_time_ms"] > 0
        assert data["response_time_ms"] < 1000  # Under 1 second in milliseconds
    
    def test_concurrent_requests(self, api_client):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        def make_request(user_num):
            payload = {
                "user_id": f"user_concurrent_{user_num}",
                "video_id": f"video_concurrent_{user_num}",
                "watch_time": 30.0 + user_num
            }
            response = api_client.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            return response.status_code, response.json()
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for status_code, data in results:
            assert status_code == 200
            assert 0 <= data["probability"] <= 1
    
    def test_error_handling(self, api_client):
        """Test API error handling"""
        # Test invalid JSON
        response = api_client.post(
            f"{API_BASE_URL}/predict",
            data="invalid json",
            headers={'Content-Type': 'application/json'},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422
        
        # Test empty request
        response = api_client.post(
            f"{API_BASE_URL}/predict",
            json={},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422
    
    def test_cors_headers(self, api_client):
        """Test CORS headers are present"""
        response = api_client.options(f"{API_BASE_URL}/predict", timeout=TEST_TIMEOUT)
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

def test_load_performance():
    """Load test using requests (simple version)"""
    import concurrent.futures
    import statistics
    
    def make_request():
        payload = {
            "user_id": "user_load_test",
            "video_id": "video_load_test", 
            "watch_time": 45.0
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        end_time = time.time()
        
        return response.status_code == 200, end_time - start_time
    
    # Make 50 requests with 10 concurrent workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Analyze results
    success_count = sum(1 for success, _ in results if success)
    response_times = [rt for success, rt in results if success]
    
    print(f"\nLoad Test Results:")
    print(f"Success rate: {success_count/50:.1%}")
    print(f"Average response time: {statistics.mean(response_times):.3f}s")
    print(f"95th percentile: {statistics.quantiles(response_times, n=20)[18]:.3f}s")
    print(f"Max response time: {max(response_times):.3f}s")
    
    # Assertions
    assert success_count >= 45  # At least 90% success rate
    assert statistics.mean(response_times) < 1.0  # Average under 1 second

if __name__ == "__main__":
    # Run individual tests
    pytest.main([__file__, "-v", "--tb=short"])