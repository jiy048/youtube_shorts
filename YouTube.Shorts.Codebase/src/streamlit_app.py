"""
Streamlit UI for YouTube Shorts Recommendation System
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import numpy as np

# Configure page
st.set_page_config(
    page_title="YT Shorts Recommender",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4b4b;
    }
    .success-card {
        background-color: #f0f9f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00ff00;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def call_api(endpoint, payload=None):
    """Call API with caching"""
    try:
        if payload:
            response = requests.post(f"{API_BASE_URL}/{endpoint}", json=payload, timeout=30)
        else:
            response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "API server is offline"
    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📱 YouTube Shorts Recommendation System")
    st.markdown("**Production-ready ML system for predicting user engagement**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # API Health Check
        health_data, health_error = call_api("health")
        if health_data:
            if health_data["status"] == "healthy":
                st.success("✅ API Status: Healthy")
                st.metric("Uptime", f"{health_data['uptime_seconds']:.0f}s")
            else:
                st.error("❌ API Status: Unhealthy")
        else:
            st.error(f"❌ API Offline: {health_error}")
        
        st.divider()
        
        # Navigation
        page = st.selectbox(
            "Select Page",
            ["Single Prediction", "Batch Prediction", "Analytics Dashboard", "Model Info"]
        )
    
    # Main content based on selected page
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Analytics Dashboard":
        analytics_dashboard_page()
    elif page == "Model Info":
        model_info_page()

def single_prediction_page():
    """Single prediction interface"""
    st.header("🎯 Single Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        # User inputs
        user_id = st.text_input(
            "User ID", 
            value="user_123",
            help="Format: user_XXX"
        )
        
        video_id = st.text_input(
            "Video ID", 
            value="video_456",
            help="Format: video_XXX"
        )
        
        watch_time = st.slider(
            "Watch Time (seconds)", 
            min_value=0.0, 
            max_value=180.0, 
            value=45.0,
            step=0.5
        )
        
        hour_of_day = st.selectbox(
            "Hour of Day",
            options=list(range(24)),
            index=14
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            session_context = st.text_area(
                "Session Context (JSON)",
                value='{"device": "mobile", "location": "US"}',
                help="Additional context as JSON"
            )
    
    with col2:
        st.subheader("Prediction")
        
        if st.button("🚀 Predict", type="primary"):
            # Validate inputs
            if not user_id.startswith('user_'):
                st.error("User ID must start with 'user_'")
                return
            
            if not video_id.startswith('video_'):
                st.error("Video ID must start with 'video_'")
                return
            
            # Prepare payload
            payload = {
                "user_id": user_id,
                "video_id": video_id,
                "watch_time": watch_time,
                "hour_of_day": hour_of_day
            }
            
            # Add session context if provided
            try:
                if session_context.strip():
                    payload["session_context"] = json.loads(session_context)
            except json.JSONDecodeError:
                st.warning("Invalid JSON in session context, ignoring...")
            
            # Make prediction
            with st.spinner("Making prediction..."):
                result, error = call_api("predict", payload)
            
            if result:
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = result["probability"],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Engagement Probability"},
                    delta = {'reference': 0.5},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgray"},
                            {'range': [0.3, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.8
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Probability", f"{result['probability']:.3f}")
                with col2:
                    st.metric("Confidence", result['confidence'])
                with col3:
                    st.metric("Response Time", f"{result['response_time_ms']:.1f}ms")
                
                # Details
                with st.expander("Response Details"):
                    st.json(result)
            else:
                st.error(f"❌ Prediction failed: {error}")

def batch_prediction_page():
    """Batch prediction interface"""
    st.header("📊 Batch Prediction")
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Upload CSV file with user interactions",
        type=['csv'],
        help="CSV should have columns: user_id, video_id, watch_time"
    )
    
    if uploaded_file is not None:
        # Read and validate CSV
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
            
            # Validate required columns
            required_cols = ['user_id', 'video_id', 'watch_time']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Batch size limitation
            if len(df) > 100:
                st.warning(f"File has {len(df)} rows. Only first 100 will be processed.")
                df = df.head(100)
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                # Prepare batch payload
                requests_list = []
                for _, row in df.iterrows():
                    req = {
                        "user_id": str(row['user_id']),
                        "video_id": str(row['video_id']),
                        "watch_time": float(row['watch_time'])
                    }
                    
                    # Add optional fields if present
                    if 'hour_of_day' in row and pd.notna(row['hour_of_day']):
                        req['hour_of_day'] = int(row['hour_of_day'])
                    
                    requests_list.append(req)
                
                payload = {"requests": requests_list}
                
                # Make batch prediction
                with st.spinner("Processing batch prediction..."):
                    result, error = call_api("predict/batch", payload)
                
                if result:
                    st.success("✅ Batch prediction complete!")
                    
                    # Process results
                    results_df = pd.DataFrame(result['results'])
                    
                    # Show summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Requests", result['batch_size'])
                    with col2:
                        success_count = len(results_df[~results_df['probability'].isna()])
                        st.metric("Successful", success_count)
                    with col3:
                        if 'probability' in results_df.columns:
                            avg_prob = results_df['probability'].mean()
                            st.metric("Avg Probability", f"{avg_prob:.3f}")
                    with col4:
                        st.metric("Response Time", f"{result['response_time_ms']:.1f}ms")
                    
                    # Show results
                    st.subheader("Results")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    if 'probability' in results_df.columns:
                        fig = px.histogram(
                            results_df, 
                            x='probability',
                            title="Prediction Probability Distribution",
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"❌ Batch prediction failed: {error}")
        
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
    
    else:
        # Manual batch input
        st.subheader("Manual Batch Input")
        st.info("💡 Upload a CSV file above for bulk processing, or add individual requests below")
        
        # Initialize session state for batch requests
        if 'batch_requests' not in st.session_state:
            st.session_state.batch_requests = []
        
        # Add request form
        with st.form("add_request"):
            col1, col2, col3 = st.columns(3)
            with col1:
                user_id = st.text_input("User ID", placeholder="user_123")
            with col2:
                video_id = st.text_input("Video ID", placeholder="video_456")
            with col3:
                watch_time = st.number_input("Watch Time", min_value=0.0, max_value=180.0, value=45.0)
            
            if st.form_submit_button("➕ Add Request"):
                if user_id and video_id:
                    st.session_state.batch_requests.append({
                        "user_id": user_id,
                        "video_id": video_id,
                        "watch_time": watch_time
                    })
                    st.success("Request added!")
        
        # Show current requests
        if st.session_state.batch_requests:
            st.subheader(f"Current Batch ({len(st.session_state.batch_requests)} requests)")
            batch_df = pd.DataFrame(st.session_state.batch_requests)
            st.dataframe(batch_df)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All"):
                    st.session_state.batch_requests = []
                    st.rerun()
            
            with col2:
                if st.button("🚀 Process Batch"):
                    # Process manual batch
                    payload = {"requests": st.session_state.batch_requests}
                    
                    with st.spinner("Processing batch..."):
                        result, error = call_api("predict/batch", payload)
                    
                    if result:
                        st.success("✅ Batch complete!")
                        st.json(result)
                    else:
                        st.error(f"❌ Failed: {error}")

def analytics_dashboard_page():
    """Analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    # Simulated analytics data (in production, this would come from your metrics store)
    st.info("📊 This would show real-time analytics from your monitoring system")
    
    # Generate sample data for demonstration
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'requests_per_hour': np.random.poisson(50, len(dates)),
        'avg_response_time': np.random.normal(100, 20, len(dates)),
        'error_rate': np.random.exponential(0.02, len(dates)),
        'avg_probability': np.random.normal(0.3, 0.1, len(dates))
    })
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests", f"{sample_data['requests_per_hour'].sum():,}")
    with col2:
        st.metric("Avg Response Time", f"{sample_