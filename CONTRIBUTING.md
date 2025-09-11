# Contributing to MarketRegimeML

Thank you for your interest in contributing to MarketRegimeML!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/firestrand/marketregimeml.git
cd marketregimeml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Development Commands

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=marketregimeml --cov-report=term

# Run specific test file
pytest tests/test_models_hmm.py -v

# Run with detailed output
pytest tests/ -v --tb=short
```

### Code Quality

```bash
# Format code with black
black marketregimeml/ tests/

# Check code style
flake8 marketregimeml/ tests/

# Sort imports
isort marketregimeml/ tests/

# Type checking (optional)
mypy marketregimeml/
```

### Before Committing

1. **Run tests**: `pytest tests/`
2. **Format code**: `black marketregimeml/ tests/`
3. **Check style**: `flake8 marketregimeml/ tests/`

## Contribution Guidelines

### ✅ We Accept

- New regime detection algorithms
- Improved evaluation metrics
- Better feature engineering for regime detection
- Performance optimizations
- Bug fixes
- Documentation improvements

### ❌ We Don't Accept

- Trading strategies or backtesting features
- Portfolio management functionality
- Complex data storage systems
- Features unrelated to regime detection

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all public functions
- Keep methods simple (cyclomatic complexity < 10)
- Follow SOLID principles

## Testing

- Write tests for new features
- Maintain >40% test coverage
- Use pytest for all tests
- Mock external API calls in unit tests

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Format your code with black
6. Commit with a clear message
7. Push to your fork
8. Open a pull request with:
   - Clear description of changes
   - Any related issue numbers
   - Test results

## Questions?

Open an issue on GitHub if you have questions or need help.