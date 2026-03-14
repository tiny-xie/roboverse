# RoboVerse Development Roadmap

This document provides a high-level overview of the RoboVerse improvement roadmap. For detailed analysis and implementation guidelines, see the [Architecture Review](docs/source/metasim/developer_guide/architecture_review.md).

## Current Status

| Aspect | Status | Notes |
|--------|--------|-------|
| Core Framework | Stable | Multi-simulator support working |
| Test Coverage | Needs Work | ~15-20% estimated coverage |
| Documentation | Good | Comprehensive tutorials available |
| API Stability | Beta | Breaking changes possible |

## Priority Issues

### P0 - Critical (Immediate)

- [ ] **State Cache Consistency** - Fix data corruption when switching state formats
- [ ] **Test Coverage** - Add validation tests for robot configs and core tasks
- [ ] **CI Coverage Reporting** - Integrate pytest-cov with Codecov

### P1 - High (Next Release)

- [ ] **Abstract Method Declarations** - Complete @abstractmethod annotations
- [ ] **Unified Environment Factory** - Create `roboverse.make_env()` API
- [ ] **Configuration System** - Document and standardize config approaches
- [ ] **Error Handling** - Improve parallel simulation error propagation

### P2 - Medium (Future)

- [ ] **Code Quality** - Remove commented code, extract magic numbers
- [ ] **Type Annotations** - Complete type hints across codebase
- [ ] **Performance Benchmarks** - Add regression testing for performance

### P3 - Long-term

- [ ] **Plugin Architecture** - Modular simulator registration
- [ ] **Async Simulation** - Support for async environment stepping
- [ ] **Unified Config System** - Single configuration approach across modules

## Contributing

We welcome contributions! Here's how to get started:

1. **Read the detailed analysis**: [Architecture Review](docs/source/metasim/developer_guide/architecture_review.md)
2. **Pick an issue**: Start with P2 items for low-risk contributions
3. **Follow the protocol**: See "Safe Modification Protocol" in the review doc
4. **Submit a PR**: Include tests and update documentation as needed

### Recommended First Contributions

| Task | Effort | File |
|------|--------|------|
| Add @abstractmethod to `_set_dof_targets` | Very Low | `metasim/sim/base.py` |
| Remove commented-out code | Very Low | Multiple files |
| Add robot config validation tests | Medium | New test file |

## Release Timeline

| Version | Target | Focus |
|---------|--------|-------|
| 0.2.x | Current | Bug fixes, stability |
| 0.3.0 | TBD | Test coverage, API stabilization |
| 0.4.0 | TBD | Unified interfaces, deprecation cleanup |
| 1.0.0 | TBD | Stable API, comprehensive tests |

## Questions?

- [GitHub Discussions](https://github.com/RoboVerseOrg/RoboVerse/discussions)
- [Discord](https://discord.gg/6e2CPVnAD3)
- [Documentation](https://roboverse.wiki)
