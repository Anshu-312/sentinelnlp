# Coding Standards and Contribution Guidelines

This document outlines the coding standards and contribution guidelines for the SentinelNLP project.

## Code Style

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some project-specific adaptations:

1. **Formatting**
   - Line length: 88 characters (Black default)
   - Indentation: 4 spaces (no tabs)
   - UTF-8 encoding for all Python files

2. **Naming Conventions**
   - Classes: `CamelCase`
   - Functions/Methods: `snake_case`
   - Variables: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private attributes/methods: `_leading_underscore`

3. **Imports**
   - Group imports in the following order:
     1. Standard library imports
     2. Related third-party imports
     3. Local application/library specific imports
   - Use absolute imports when possible
   - Use `isort` to automatically sort imports

4. **Documentation**
   - All modules, classes, and functions must have docstrings
   - Use Google style docstrings:

```python
def function_with_types_in_docstring(param1: int, param2: str) -> bool:
    """Example function with parameters and return value.
    
    Args:
        param1: The first parameter.
        param2: The second parameter.
        
    Returns:
        True if successful, False otherwise.
        
    Raises:
        ValueError: If param1 is negative.
    """
    if param1 < 0:
        raise ValueError("param1 can't be negative.")
    return True
```

5. **Type Annotations**
   - Use type hints for all function parameters and return values
   - Use `Optional[T]` for parameters that can be None
   - Use `Union[T1, T2]` for parameters that can be multiple types

## Code Quality Tools

All code must pass the following quality checks:

1. **Linting with Flake8**
   - Run: `flake8 src tests`
   - Configuration in `.flake8`

2. **Formatting with Black**
   - Run: `black src tests`
   - No custom configuration (use defaults)

3. **Import Sorting with isort**
   - Run: `isort src tests`
   - Configuration in `pyproject.toml`

4. **Type Checking with mypy**
   - Run: `mypy src`
   - Configuration in `mypy.ini`

5. **Pre-commit hooks**
   - Configuration in `.pre-commit-config.yaml`
   - Run: `pre-commit run --all-files`

## Testing Standards

### Test Organization

- All tests should be in the `tests/` directory
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Test data: `tests/data/`

### Test Requirements

1. **Code Coverage**
   - Minimum coverage requirement: 80%
   - Aim for 90%+ coverage for core components

2. **Unit Tests**
   - Use pytest for all tests
   - One test file per module being tested
   - Follow naming convention: `test_<module_name>.py`
   - Keep tests independent and atomic

3. **Integration Tests**
   - Test component interactions
   - Include database and API tests
   - Use fixtures and mocks appropriately

4. **Test Documentation**
   - Each test should have a clear docstring
   - Include test purpose and expected outcome

## Git Workflow

### Branching Strategy

We follow a modified Git Flow approach:

1. **Main Branches**
   - `main`: Production-ready code
   - `develop`: Latest development code

2. **Supporting Branches**
   - Feature branches: `feature/<feature-name>`
   - Bug fixes: `fix/<bug-description>`
   - Releases: `release/<version>`
   - Hotfixes: `hotfix/<issue-description>`

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Where `<type>` is one of:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

Example:
```
feat(ontology): add support for STIX 2.1 objects

Implement classes and relationships from STIX 2.1 specification.
Includes mapping to existing ontology classes.

Closes #123
```

### Pull Request Process

1. Create a feature/fix branch from `develop`
2. Implement your changes
3. Ensure all tests pass and code quality tools are satisfied
4. Submit a pull request to `develop`
5. Request review from at least one team member
6. Address review comments
7. Once approved, maintainers will merge the PR

## Documentation Standards

### Code Documentation

- **Module Documentation**: Each module should have a docstring explaining its purpose and usage
- **Class Documentation**: Each class should have a docstring explaining its purpose, attributes, and usage
- **Function Documentation**: Each function should have a docstring explaining what it does, its parameters, return values, and exceptions

### API Documentation

- API endpoints must be documented using FastAPI's built-in documentation
- Include example requests and responses
- Document all parameters and response structures

### Project Documentation

- Technical documentation should be written in Markdown
- User documentation should be comprehensive and include examples
- API reference should be generated from code comments

## Security Guidelines

1. **Dependency Management**
   - Regularly update dependencies
   - Use dependabot for automated updates
   - Check for vulnerabilities with `safety`

2. **Data Protection**
   - Never commit sensitive data (API keys, passwords)
   - Use environment variables for secrets
   - Sanitize user inputs

3. **Authentication**
   - Use industry standard authentication methods
   - Implement proper token validation
   - Set appropriate token expiration times

## Contribution Process

1. **Finding an Issue**
   - Check the GitHub issues tab for open issues
   - Look for issues marked as "good first issue" if you're new
   - Comment on an issue to express interest

2. **Proposing a Change**
   - For major changes, open an issue first to discuss
   - Describe the problem and proposed solution
   - Get feedback before implementing

3. **Implementation**
   - Follow the coding standards
   - Write tests for your changes
   - Update documentation as needed

4. **Submission**
   - Submit a pull request following the PR process
   - Be responsive to review feedback
   - Be prepared to make requested changes

5. **After Acceptance**
   - Your code will be merged by maintainers
   - You'll be credited as a contributor
   - Consider helping review other PRs 