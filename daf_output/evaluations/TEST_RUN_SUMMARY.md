# CoGames Full Test Suite Report

**Generated:** 2025-11-22 12:39:02

**Overall Status:** ✅ SUCCESS

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 285 |
| **Passed** | 280 ✅ |
| **Failed** | 0 ❌ |
| **Skipped** | 5 ⏭️ |
| **Total Duration** | 365.69s |
| **Pass Rate** | 98.2% |

---

## Test Plan

### Expected vs Actual Test Counts

| Phase | Expected | Actual | Variance |
|-------|----------|--------|----------|
| Phase 1 (CoGames) | 185 | 185 | +0 |
| Phase 2 (DAF) | 100 | 100 | +0 |
| **Total** | **285** | **285** | **+0** |

**Note:** Positive variance indicates more tests collected than planned, negative indicates fewer.


---

## Phase 1: CoGames Core Tests

**Status:** ✅ PASSED

### Summary

| Metric | Value |
|--------|-------|
| Tests Run | 185 |
| Passed | 185 ✅ |
| Failed | 0 ❌ |
| Skipped | 0 ⏭️ |
| Duration | 232.30s |
| Pass Rate | 100.0% |

### Test Suites

| Status | Suite | Results | Time |
|--------|-------|---------|------|
| ✅ | All Games Describe | 47/47 | 60.72s |
| ✅ | All Games Eval | 48/48 | 77.10s |
| ✅ | All Games Play | 47/47 | 65.37s |
| ✅ | Cli | 6/6 | 8.63s |
| ✅ | Cogs Vs Clips | 4/4 | 0.21s |
| ✅ | Cvc Assembler Hearts | 2/2 | 0.15s |
| ✅ | Procedural Maps | 11/11 | 2.63s |
| ✅ | Scripted Policies | 13/13 | 1.05s |
| ✅ | Train Integration | 2/2 | 15.67s |
| ✅ | Train Vector Alignment | 5/5 | 0.77s |

---

## Phase 2: DAF Module Tests

**Status:** ✅ PASSED

### Summary

| Metric | Value |
|--------|-------|
| Tests Run | 100 |
| Passed | 95 ✅ |
| Failed | 0 ❌ |
| Skipped | 5 ⏭️ |
| Duration | 133.39s |
| Pass Rate | 95.0% |

### Test Suites

| Status | Suite | Results | Time |
|--------|-------|---------|------|
| ✅ | Comparison | 10/12 | 128.13s |
| ✅ | Config | 15/15 | 0.06s |
| ✅ | Deployment | 9/11 | 1.69s |
| ✅ | Distributed Training | 11/12 | 0.82s |
| ✅ | Environment Checks | 13/13 | 0.70s |
| ✅ | Mission Analysis | 12/12 | 0.24s |
| ✅ | Sweeps | 16/16 | 0.71s |
| ✅ | Visualization | 9/9 | 1.04s |

---

## Performance Analysis

### Slowest Test Suites

1. **Comparison** - 128.13s
2. **All Games Eval** - 77.10s
3. **All Games Play** - 65.37s
4. **All Games Describe** - 60.72s
5. **Train Integration** - 15.67s

### Test Distribution

- **Phase 1 (CoGames):** 63.5% of total time (232.30s)
- **Phase 2 (DAF):** 36.5% of total time (133.39s)

---

## Detailed Results

### Phase 1: CoGames Test Suites

#### ✅ All Games Describe

- **Status:** PASSED
- **Tests:** 47 passed, 0 failed, 0 skipped (47 total)
- **Pass Rate:** 100.0%
- **Duration:** 60.72s
- **Output Files:**
  - [📄 Text](cogames_test_all_games_describe_output.txt)
  - [📊 JSON](cogames_test_all_games_describe.json)
  - [✔️ JUnit XML](cogames_test_all_games_describe.xml)

#### ✅ All Games Eval

- **Status:** PASSED
- **Tests:** 48 passed, 0 failed, 0 skipped (48 total)
- **Pass Rate:** 100.0%
- **Duration:** 77.10s
- **Output Files:**
  - [📄 Text](cogames_test_all_games_eval_output.txt)
  - [📊 JSON](cogames_test_all_games_eval.json)
  - [✔️ JUnit XML](cogames_test_all_games_eval.xml)

#### ✅ All Games Play

- **Status:** PASSED
- **Tests:** 47 passed, 0 failed, 0 skipped (47 total)
- **Pass Rate:** 100.0%
- **Duration:** 65.37s
- **Output Files:**
  - [📄 Text](cogames_test_all_games_play_output.txt)
  - [📊 JSON](cogames_test_all_games_play.json)
  - [✔️ JUnit XML](cogames_test_all_games_play.xml)

#### ✅ Cli

- **Status:** PASSED
- **Tests:** 6 passed, 0 failed, 0 skipped (6 total)
- **Pass Rate:** 100.0%
- **Duration:** 8.63s
- **Output Files:**
  - [📄 Text](cogames_test_cli_output.txt)
  - [📊 JSON](cogames_test_cli.json)
  - [✔️ JUnit XML](cogames_test_cli.xml)

#### ✅ Cogs Vs Clips

- **Status:** PASSED
- **Tests:** 4 passed, 0 failed, 0 skipped (4 total)
- **Pass Rate:** 100.0%
- **Duration:** 0.21s
- **Output Files:**
  - [📄 Text](cogames_test_cogs_vs_clips_output.txt)
  - [📊 JSON](cogames_test_cogs_vs_clips.json)
  - [✔️ JUnit XML](cogames_test_cogs_vs_clips.xml)

#### ✅ Cvc Assembler Hearts

- **Status:** PASSED
- **Tests:** 2 passed, 0 failed, 0 skipped (2 total)
- **Pass Rate:** 100.0%
- **Duration:** 0.15s
- **Output Files:**
  - [📄 Text](cogames_test_cvc_assembler_hearts_output.txt)
  - [📊 JSON](cogames_test_cvc_assembler_hearts.json)
  - [✔️ JUnit XML](cogames_test_cvc_assembler_hearts.xml)

#### ✅ Procedural Maps

- **Status:** PASSED
- **Tests:** 11 passed, 0 failed, 0 skipped (11 total)
- **Pass Rate:** 100.0%
- **Duration:** 2.63s
- **Output Files:**
  - [📄 Text](cogames_test_procedural_maps_output.txt)
  - [📊 JSON](cogames_test_procedural_maps.json)
  - [✔️ JUnit XML](cogames_test_procedural_maps.xml)

#### ✅ Scripted Policies

- **Status:** PASSED
- **Tests:** 13 passed, 0 failed, 0 skipped (13 total)
- **Pass Rate:** 100.0%
- **Duration:** 1.05s
- **Output Files:**
  - [📄 Text](cogames_test_scripted_policies_output.txt)
  - [📊 JSON](cogames_test_scripted_policies.json)
  - [✔️ JUnit XML](cogames_test_scripted_policies.xml)

#### ✅ Train Integration

- **Status:** PASSED
- **Tests:** 2 passed, 0 failed, 0 skipped (2 total)
- **Pass Rate:** 100.0%
- **Duration:** 15.67s
- **Output Files:**
  - [📄 Text](cogames_test_train_integration_output.txt)
  - [📊 JSON](cogames_test_train_integration.json)
  - [✔️ JUnit XML](cogames_test_train_integration.xml)

#### ✅ Train Vector Alignment

- **Status:** PASSED
- **Tests:** 5 passed, 0 failed, 0 skipped (5 total)
- **Pass Rate:** 100.0%
- **Duration:** 0.77s
- **Output Files:**
  - [📄 Text](cogames_test_train_vector_alignment_output.txt)
  - [📊 JSON](cogames_test_train_vector_alignment.json)
  - [✔️ JUnit XML](cogames_test_train_vector_alignment.xml)

### Phase 2: DAF Test Suites

#### ✅ Comparison

- **Status:** PASSED
- **Tests:** 10 passed, 0 failed, 2 skipped (12 total)
- **Pass Rate:** 83.3%
- **Duration:** 128.13s
- **Output Files:**
  - [📄 Text](daf_test_comparison_output.txt)
  - [📊 JSON](daf_test_comparison.json)
  - [✔️ JUnit XML](daf_test_comparison.xml)

#### ✅ Config

- **Status:** PASSED
- **Tests:** 15 passed, 0 failed, 0 skipped (15 total)
- **Pass Rate:** 100.0%
- **Duration:** 0.06s
- **Output Files:**
  - [📄 Text](daf_test_config_output.txt)
  - [📊 JSON](daf_test_config.json)
  - [✔️ JUnit XML](daf_test_config.xml)

#### ✅ Deployment

- **Status:** PASSED
- **Tests:** 9 passed, 0 failed, 2 skipped (11 total)
- **Pass Rate:** 81.8%
- **Duration:** 1.69s
- **Output Files:**
  - [📄 Text](daf_test_deployment_output.txt)
  - [📊 JSON](daf_test_deployment.json)
  - [✔️ JUnit XML](daf_test_deployment.xml)

#### ✅ Distributed Training

- **Status:** PASSED
- **Tests:** 11 passed, 0 failed, 1 skipped (12 total)
- **Pass Rate:** 91.7%
- **Duration:** 0.82s
- **Output Files:**
  - [📄 Text](daf_test_distributed_training_output.txt)
  - [📊 JSON](daf_test_distributed_training.json)
  - [✔️ JUnit XML](daf_test_distributed_training.xml)

#### ✅ Environment Checks

- **Status:** PASSED
- **Tests:** 13 passed, 0 failed, 0 skipped (13 total)
- **Pass Rate:** 100.0%
- **Duration:** 0.70s
- **Output Files:**
  - [📄 Text](daf_test_environment_checks_output.txt)
  - [📊 JSON](daf_test_environment_checks.json)
  - [✔️ JUnit XML](daf_test_environment_checks.xml)

#### ✅ Mission Analysis

- **Status:** PASSED
- **Tests:** 12 passed, 0 failed, 0 skipped (12 total)
- **Pass Rate:** 100.0%
- **Duration:** 0.24s
- **Output Files:**
  - [📄 Text](daf_test_mission_analysis_output.txt)
  - [📊 JSON](daf_test_mission_analysis.json)
  - [✔️ JUnit XML](daf_test_mission_analysis.xml)

#### ✅ Sweeps

- **Status:** PASSED
- **Tests:** 16 passed, 0 failed, 0 skipped (16 total)
- **Pass Rate:** 100.0%
- **Duration:** 0.71s
- **Output Files:**
  - [📄 Text](daf_test_sweeps_output.txt)
  - [📊 JSON](daf_test_sweeps.json)
  - [✔️ JUnit XML](daf_test_sweeps.xml)

#### ✅ Visualization

- **Status:** PASSED
- **Tests:** 9 passed, 0 failed, 0 skipped (9 total)
- **Pass Rate:** 100.0%
- **Duration:** 1.04s
- **Output Files:**
  - [📄 Text](daf_test_visualization_output.txt)
  - [📊 JSON](daf_test_visualization.json)
  - [✔️ JUnit XML](daf_test_visualization.xml)

---

## Recommendations

✅ **All tests passed successfully!**

- Review test outputs for any warnings or performance concerns
- Monitor trends in test execution time
- Consider adding performance regression tests if needed


---

## Output Files

Test output files are saved in: `daf_output/evaluations/tests/`

Each test suite produces:
- Individual output file: `*_output.txt`
- Detailed test logs with assertion details
- Full pytest output including timing

## Running Tests

To run the full test suite:

```bash
cd /Users/4d/Documents/GitHub/cogames
./daf/tests/run_daf_tests.sh
```

To run individual test suites:

```bash
# CoGames tests
uv run pytest tests/test_cli.py -v

# DAF tests
uv run pytest daf/tests/test_config.py -v
```

---

*Report generated by CoGames Test Infrastructure*
