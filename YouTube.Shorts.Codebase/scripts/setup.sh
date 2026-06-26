#!/bin/bash
# YouTube Shorts Recommendation System - Setup Script
# This script sets up the entire project structure and dependencies

set -e  # Exit on any error

echo "🚀 Setting up YouTube Shorts Recommendation System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Python version
    if command -v python3.11 &> /dev/null; then
        print_status "✓ Python 3.11 found"
    elif command -v python3 &> /dev/null; then
        python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ "$python_version" == "3.11" ]] || [[ "$python_version" == "3.10" ]] || [[ "$python_version" == "3.9" ]]; then
            print_status "✓ Python $python_version found (compatible)"
            PYTHON_CMD="python3"
        else
            print_error "❌ Python 3.9+ required, found $python_version"
            exit 1
        fi
    else
        print_error "❌ Python not found. Please install Python 3.9+"
        exit 1
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_status "✓ Docker found"
    else
        print_warning "⚠️ Docker not found (optional for local development)"
    fi
    
    # Check Make
    if command -v make &> /dev/null; then
        print_status "✓ Make found"
    else
        print_warning "⚠️ Make not found (will use manual commands)"
    fi
}

# Create directory structure
create_directories() {
    print_header "Creating Directory Structure"
    
    directories=(
        "src/utils"
        "tests"
        "scripts"
        "monitoring/grafana/dashboards"
        "monitoring/grafana/datasources"
        ".github/workflows"
        "docs"
        "data/raw"
        "data/processed"
        "data/synthetic"
        "models/trained"
        "models/artifacts"
        "models/metadata"
        "logs/training"
        "logs/api"
        "logs/monitoring"
        "config"
        "deployment/kubernetes"
        "deployment/terraform"
        "deployment/helm/yt-shorts-recommender/templates"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
}

# Setup Python environment
setup_python_env() {
    print_header "Setting up Python Environment"
    
    # Use python3.11 if available, otherwise python3
    PYTHON_CMD=${PYTHON_CMD:-python3.11}
    
    print_status "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
    else
        print_warning "requirements.txt not found, skipping dependency installation"
    fi
}

# Generate environment files
generate_env_files() {
    print_header "Generating Environment Files"
    
    # Create .env.example
    cat > .env.example << 'EOF'
# Development Environment Variables
ENVIRONMENT=development
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Database URLs
REDIS_URL=redis://localhost:6379
MLFLOW_TRACKING_URI=http://localhost:5000

# Model Configuration
MODEL_PATH=models/trained
FEATURE_STORE_PATH=models/artifacts

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000

# Security (generate your own in production)
SECRET_KEY=your-secret-key-here
RATE_LIMIT_PER_MINUTE=100

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
EOF

    # Create .gitignore
    cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# macOS
.DS_Store

# Project specific
data/
models/
logs/
mlruns/
.env
*.log

# Docker
.dockerignore

# Temporary files
*.tmp
*.temp
EOF

    print_status "Generated .env.example and .gitignore"
}

# Create basic configuration files
create_config_files() {
    print_header "Creating Configuration Files"
    
    # Create development config
    cat > config/development.yml << 'EOF'
# Development Configuration
api:
  host: "0.0.0.0"
  port: 8000
  reload: true
  workers: 1

database:
  redis_url: "redis://localhost:6379"

model:
  path: "models/trained"
  cache_predictions: true
  
monitoring:
  enable_metrics: true
  log_level: "INFO"

features:
  use_embeddings: true
  embedding_dim: 50
  
training:
  optimize_hyperparams: true
  n_trials: 50
  test_size: 0.2
EOF

    # Create production config
    cat > config/production.yml << 'EOF'
# Production Configuration
api:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 4

database:
  redis_url: "${REDIS_URL}"

model:
  path: "${MODEL_PATH}"
  cache_predictions: true
  
monitoring:
  enable_metrics: true
  log_level: "WARNING"

features:
  use_embeddings: true
  embedding_dim: 50
  
training:
  optimize_hyperparams: true
  n_trials: 100
  test_size: 0.2
EOF

    print_status "Created configuration files"
}

# Create deployment scripts
create_deployment_scripts() {
    print_header "Creating Deployment Scripts"
    
    # Create deploy script
    cat > scripts/deploy.sh << 'EOF'
#!/bin/bash
# Deployment script for YouTube Shorts Recommendation System

set -e

echo "🚀 Deploying YouTube Shorts Recommendation System..."

# Build Docker images
echo "Building Docker images..."
docker-compose build

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Health check
echo "Performing health check..."
curl -f http://localhost:8000/health || (echo "❌ Health check failed" && exit 1)

echo "✅ Deployment completed successfully!"
echo "🌐 API: http://localhost:8000"
echo "📱 UI: http://localhost:8501"
echo "📊 MLflow: http://localhost:5000"
echo "📈 Grafana: http://localhost:3000"
EOF

    # Create backup script
    cat > scripts/backup.sh << 'EOF'
#!/bin/bash
# Backup script for models and data

set -e

BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backup_${TIMESTAMP}.tar.gz"

echo "📦 Creating backup..."

mkdir -p $BACKUP_DIR

# Create backup
tar -czf "${BACKUP_DIR}/${BACKUP_FILE}" \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    data/ models/ config/

echo "✅ Backup created: ${BACKUP_DIR}/${BACKUP_FILE}"
EOF

    # Make scripts executable
    chmod +x scripts/deploy.sh
    chmod +x scripts/backup.sh
    chmod +x scripts/setup.sh
    
    print_status "Created deployment scripts"
}

# Create empty __init__.py files
create_init_files() {
    print_header "Creating Python Package Files"
    
    # Create __init__.py files for proper Python packages
    touch src/__init__.py
    touch src/utils/__init__.py
    touch tests/__init__.py
    
    print_status "Created Python package files"
}

# Main setup function
main() {
    print_header "YouTube Shorts Recommendation System Setup"
    
    check_prerequisites
    create_directories
    generate_env_files
    create_config_files
    create_deployment_scripts
    create_init_files
    setup_python_env
    
    print_header "Setup Complete! 🎉"
    
    echo ""
    echo -e "${GREEN}✅ Project setup completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Copy all the provided Python files to their respective locations"
    echo "2. Activate the virtual environment: source venv/bin/activate"
    echo "3. Run the pipeline: make full-pipeline"
    echo "4. Or run manually:"
    echo "   - python src/data_generator.py"
    echo "   - python src/train.py" 
    echo "   - uvicorn src.api:app --reload"
    echo ""
    echo -e "${BLUE}Access points:${NC}"
    echo "🌐 API: http://localhost:8000"
    echo "📱 UI: http://localhost:8501"
    echo "📚 Docs: http://localhost:8000/docs"
    echo ""
}

# Run main function
main "$@"