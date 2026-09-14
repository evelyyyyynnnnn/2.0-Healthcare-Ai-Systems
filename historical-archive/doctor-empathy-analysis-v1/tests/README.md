# Test Suite Guide

## 📋 Overview

This directory contains all automated tests used to verify the correctness and stability of the project.

## 🧪 Test Files

### Core Tests
- **[test_analysis.py](test_analysis.py)** – Validates primary analysis workflows
- **[test_wordcloud_fix.py](test_wordcloud_fix.py)** – Ensures the word cloud regression fix remains stable

### Test Runner
- **[run_all_tests.py](run_all_tests.py)** – Entry point to execute the full suite
- **[__init__.py](__init__.py)** – Package initializer

## 🚀 Running Tests

### Run the Entire Suite
```bash
python tests/run_all_tests.py
```

### Run Individual Tests
```bash
# Core analysis tests
python tests/test_analysis.py

# Word cloud regression
python tests/test_wordcloud_fix.py
```

### Run from the Project Root with -m
```bash
# Full suite
python -m tests.run_all_tests

# Individual modules
python -m tests.test_analysis
python -m tests.test_wordcloud_fix
```

## 📊 Coverage Focus

### Functional Tests
- Empathy feature extraction
- Text preprocessing
- Scoring calculation
- Data export

### Regression Tests
- Word cloud generation
- Chinese font rendering
- File path handling

## 🔧 Environment

- Python 3.7+
- Dependencies listed in `requirements.txt`
- Access to sample data files

## 📝 Adding New Tests

To extend the suite:
1. Create a new test module (`test_*.py`)
2. Inherit from the appropriate test base class
3. Import the module in `run_all_tests.py`
4. Verify coverage before submitting changes

---

*Last updated: 2024*
