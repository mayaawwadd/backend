# Tests

This folder contains unit and integration tests for the AI Pulse Backend application.

## Running Tests

### Run all tests
```bash
python -m pytest
```

### Run tests in a specific file
```bash
python -m pytest tests/test_qdrant_vector_db_client.py
```

### Run tests with coverage report
```bash
python -m pytest --cov=app --cov-report=html
```

## Test Structure

- `test_qdrant_vector_db_client.py` - Tests for Qdrant vector database client functionality
- Add more test files following the `test_*.py` naming convention

## Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov
```

## Writing Tests

Follow these conventions:
- Name test files with `test_` prefix
- Name test functions with `test_` prefix
- Use fixtures for common setup/teardown
- Mock external dependencies
- Keep tests isolated and independent