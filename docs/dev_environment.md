# Development Environment Setup

This document outlines the development environment setup for the SentinelNLP project.

## Local Development Environment

### Prerequisites

- Python 3.9+ with pip
- Git
- Docker and Docker Compose
- Neo4j Desktop (for local development)
- Visual Studio Code or PyCharm (recommended IDEs)

### Setup Steps

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/sentinelnlp.git
cd sentinelnlp
```

2. **Create and activate a virtual environment**

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Setup Neo4j database**

```bash
# Using Docker
docker run \
    --name sentinelnlp-neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4JLABS_PLUGINS=["apoc"] \
    neo4j:4.4
```

5. **Download and prepare language models**

```bash
python -m spacy download en_core_web_lg
# Download any other required models
python scripts/download_models.py
```

6. **Set up environment variables**

Create a `.env` file in the project root:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
API_SECRET_KEY=your_secret_key_here
```

## CI/CD Pipeline

We use GitHub Actions for continuous integration and deployment. The pipeline includes:

### Workflow Components

1. **Code Linting and Testing**
   - Runs on every push and pull request
   - Executes unit tests and integration tests
   - Enforces code style with flake8, black, and isort

2. **Dependency Security Scanning**
   - Checks for vulnerable dependencies
   - Runs on a weekly schedule

3. **Docker Image Building**
   - Builds Docker images for each component
   - Pushes to container registry
   - Runs on main branch changes and tag creation

4. **Documentation Generation**
   - Generates API documentation
   - Publishes to GitHub Pages
   - Runs on main branch changes

### Workflow Configuration

Add these files to `.github/workflows/`:

#### ci.yml
```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8 black isort
          pip install -r requirements.txt
      - name: Lint with flake8
        run: flake8 src tests
      - name: Check formatting with black
        run: black --check src tests
      - name: Check imports with isort
        run: isort --check-only --profile black src tests

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Test with pytest
        run: pytest --cov=src tests/
```

#### docker.yml
```yaml
name: Docker Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - name: Build and push
        uses: docker/build-push-action@v3
        with:
          context: .
          push: true
          tags: yourusername/sentinelnlp:latest
```

## Local Testing

### Running Tests

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage report
pytest --cov=src tests/
```

### Running the Application

```bash
# Run the API server
uvicorn src.api.main:app --reload

# Run the NLP pipeline on a sample document
python src/run_pipeline.py --input docs/samples/sample1.txt
```

## Code Quality Tools

We use the following tools to maintain code quality:

1. **black** - Code formatter
2. **flake8** - Linter
3. **isort** - Import sorter
4. **mypy** - Type checker
5. **pre-commit** - Git hooks

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Database Management

### Neo4j Procedures

1. Access the Neo4j browser at http://localhost:7474
2. Connect with username `neo4j` and password `password`
3. Run the following to verify connection:

```cypher
MATCH (n) RETURN count(n)
```

### Database Backup

To backup the Neo4j database:

```bash
docker exec sentinelnlp-neo4j neo4j-admin backup \
  --backup-dir=/backups \
  --name=sentinelnlp-backup
```

## Troubleshooting

Common issues and solutions:

1. **Package installation errors**
   - Ensure you have the latest pip version
   - Install Microsoft Visual C++ Build Tools if on Windows

2. **Neo4j connection issues**
   - Check if Docker container is running
   - Verify port 7687 is available
   - Check credentials in .env file

3. **Spacy model errors**
   - Download the correct model version
   - Check for compatibility with your spaCy version 