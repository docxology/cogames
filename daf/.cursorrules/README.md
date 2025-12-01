# DAF Modular Cursorrules

This directory contains comprehensive, modular cursorrules files for the DAF (Distributed Agent Framework) sidecar to CoGames.

## File Organization

### Core Files

| File | Purpose | Read Time |
|------|---------|-----------|
| **00-ARCHITECTURE.md** | Core design principles and module organization | 15 min |
| **01-OUTPUT.md** | Output management patterns and APIs | 12 min |
| **02-LOGGING.md** | Logging infrastructure and patterns | 12 min |
| **03-TESTING.md** | Testing infrastructure and best practices | 12 min |
| **04-CODESTYLE.md** | Python style, naming, and code organization | 15 min |
| **05-INTEGRATION.md** | Integration with CoGames policies and missions | 12 min |

## Using These Files

### In Cursor/IDE

Use `@` mentions to load specific rules:

```
@.cursorrules-architecture    # Load architecture rules
@.cursorrules-output          # Load output management rules
@.cursorrules-logging         # Load logging rules
@.cursorrules-testing         # Load testing rules
@.cursorrules-codestyle       # Load code style rules
@.cursorrules-integration     # Load integration rules
```

### For Specific Tasks

**Writing a new DAF module?**
1. Start with `@00-ARCHITECTURE.md` for design
2. Add `@04-CODESTYLE.md` for code standards
3. Reference `@01-OUTPUT.md` and `@02-LOGGING.md` for common patterns

**Working on tests?**
- Use `@03-TESTING.md` for test infrastructure
- Reference `@04-CODESTYLE.md` for test style

**Integrating with CoGames?**
- Use `@05-INTEGRATION.md` for policy and mission patterns
- Reference architecture as needed

**New to DAF?**
1. Read `@00-ARCHITECTURE.md` (overview)
2. Skim `@01-OUTPUT.md` and `@02-LOGGING.md` (key concepts)
3. Reference as needed for implementation

## Quick Reference

### Key Principles

1. **Non-Duplicating**: DAF wraps CoGames, never duplicates
2. **Modular**: Each concern in separate module
3. **Organized**: Unified output structure by operation type
4. **Logged**: All operations tracked with metrics
5. **Quality**: Production infrastructure built-in
6. **Compatible**: Backward compatible with existing code

### Core Patterns

**Output Organization**:
```python
from daf.output_manager import get_output_manager
output_mgr = get_output_manager()
output_mgr.save_json_results(data, operation="sweep", filename="results")
```

**Logging**:
```python
from daf.logging_config import create_daf_logger
logger = create_daf_logger("operation_name")
with logger.track_operation("task"):
    pass
```

**Testing**:
```python
from daf.test_runner import TestRunner
runner = TestRunner(output_base_dir="./daf_output")
runner.run_test_batch([...])
runner.save_test_report()
```

### Directory Structure

```
daf_output/
├── sweeps/              # Hyperparameter sweeps
├── comparisons/         # Policy comparisons
├── training/            # Training runs
├── deployment/          # Deployments
├── evaluations/tests/   # Test results
├── visualizations/      # Generated plots/HTML
├── logs/                # Session logs
├── reports/             # Reports
└── artifacts/           # Artifacts
```

## Learning Path

### Beginner (First Time)
1. Read `README.md` in parent directory
2. Read `00-ARCHITECTURE.md` (principles)
3. Skim `01-OUTPUT.md` and `02-LOGGING.md`
4. Run `daf/examples/output_management_example.py`

### Developer (Adding Features)
1. Reference `00-ARCHITECTURE.md` for design
2. Follow `04-CODESTYLE.md` for code standards
3. Use `01-OUTPUT.md` and `02-LOGGING.md` for integration
4. Write tests following `03-TESTING.md`
5. Check integration with `05-INTEGRATION.md`

### Advanced (Extending DAF)
1. Review `00-ARCHITECTURE.md` thoroughly
2. Study module organization patterns
3. Reference all cursorrules as needed
4. Design with `01-OUTPUT.md` patterns
5. Implement with `04-CODESTYLE.md` standards
6. Test with `03-TESTING.md` patterns

## Common Questions

### "How do I save results?"
→ See `01-OUTPUT.md: Saving Results`

### "How do I log an operation?"
→ See `02-LOGGING.md: Logging Patterns`

### "What should my function signature look like?"
→ See `04-CODESTYLE.md: Type Hints and Docstrings`

### "How do I write tests?"
→ See `03-TESTING.md: Writing Tests`

### "How should I integrate with CoGames?"
→ See `05-INTEGRATION.md: Policy Loading`

### "What's the project structure?"
→ See `00-ARCHITECTURE.md: Module Organization`

## Quick Checklist

### Before Starting Work

- [ ] Read relevant cursorrules file
- [ ] Understand core pattern
- [ ] Check existing code examples
- [ ] Review related tests

### When Implementing

- [ ] Follow naming conventions (`04-CODESTYLE.md`)
- [ ] Add type hints to all functions
- [ ] Write Google-style docstrings
- [ ] Use OutputManager for outputs
- [ ] Use DAFLogger for logging
- [ ] Keep functions < 50 lines

### Before Committing

- [ ] Check linting: `ruff check daf/src/`
- [ ] Add unit tests
- [ ] Run tests: `pytest daf/tests/ -v`
- [ ] Update docstrings
- [ ] Update relevant documentation
- [ ] Check backward compatibility

## File Details

### 00-ARCHITECTURE.md

**Topics**:
- Core design principles
- Module organization and responsibilities
- Operation flow patterns
- Data flow
- Integration with CoGames
- Configuration patterns
- Session ID convention
- Error handling strategy
- Testing architecture
- Documentation architecture

**Read when**: Starting new module, understanding project structure

---

### 01-OUTPUT.md

**Topics**:
- OutputManager API
- Directory structure
- Session organization
- Standardized JSON format
- Usage patterns (sweeps, comparisons)
- Output utilities
- Best practices
- Configuration
- Session ID usage
- Error handling

**Read when**: Working with results, saving outputs, organizing operations

---

### 02-LOGGING.md

**Topics**:
- DAFLogger API
- Logging patterns
- Log levels
- Metrics tracking
- Console output formatting
- Configuration
- Best practices
- Integration with OutputManager
- Log files
- Troubleshooting

**Read when**: Adding logging, tracking operations, debugging

---

### 03-TESTING.md

**Topics**:
- TestRunner API
- Report generation
- Test execution
- Writing tests
- Test organization
- Test patterns
- Fixtures
- Best practices
- Coverage
- Continuous integration
- Troubleshooting

**Read when**: Writing tests, running tests, generating reports

---

### 04-CODESTYLE.md

**Topics**:
- Python standards
- Imports organization
- Type hints
- Docstrings
- Formatting
- Naming conventions
- Error handling
- Code organization
- Comments
- Functions guidelines
- Linting
- Common patterns
- Version control

**Read when**: Writing code, reviewing code, fixing style issues

---

### 05-INTEGRATION.md

**Topics**:
- Core integration principle
- Policy interfaces (MultiAgentPolicy, AgentPolicy)
- Policy loading (PolicySpec)
- Mission configuration
- Environment interaction
- Evaluation pattern
- Training integration
- Device management
- Verbose/logging coordination
- Seeding for reproducibility
- Dependency management
- Version compatibility
- No-duplication checklist
- Integration testing
- Common patterns
- Troubleshooting

**Read when**: Working with policies, missions, training, evaluation

## Contributing

When adding to cursorrules:

1. **Choose the right file** - Put content in most specific file
2. **Follow structure** - Keep format consistent with existing content
3. **Add examples** - Include code examples for patterns
4. **Link references** - Cross-reference related sections
5. **Update this README** - Add to File Details section

## Standards

### All Files Should Have

- [ ] Clear purpose statement at top
- [ ] Well-organized sections with headers
- [ ] Code examples for patterns
- [ ] Best practices (DO ✓ / DON'T ✗)
- [ ] Troubleshooting section
- [ ] Status and last updated date

### All Sections Should Have

- [ ] Clear title
- [ ] Explanation of concept
- [ ] Code examples
- [ ] Links to related sections

### All Code Examples Should

- [ ] Be syntactically correct
- [ ] Follow style guidelines
- [ ] Include comments for clarity
- [ ] Show both ✓ good and ✗ bad patterns

## See Also

- **Parent README**: `../README.md` - Module overview
- **Getting Started**: `../GETTING_STARTED.md` - Beginner guide
- **Documentation Index**: `../docs/OUTPUT_AND_LOGGING_INDEX.md` - Full docs
- **Main Docs**: `../docs/` - Complete documentation

## Status

✅ **Production Ready**
- All files complete
- All patterns documented
- All examples tested
- Zero inconsistencies

**Last Updated**: November 21, 2024
**Version**: 1.0







