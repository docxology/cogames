# DAF Mission Analysis Module

**DAF sidecar utility: Mission discovery by invoking `cogames.cli.mission` methods.**

## Overview

Mission meta-analysis, discovery from README.md, and validation.

## CoGames Integration

**Primary Method**: `cogames.cli.mission.get_mission_name_and_config()`

Extracts:
- Number of agents
- Max steps
- Map size
- Map builder type
- Loadability status

## Key Functions

- `daf_analyze_mission()` - Single mission analysis
- `daf_analyze_mission_set()` - Multiple missions
- `daf_validate_mission_set()` - Mission validation
- `daf_discover_missions_from_readme()` - README parsing
- `daf_get_mission_metadata()` - Metadata extraction

## Testing

See `daf/tests/test_mission_analysis.py` for coverage.

## See Also

- `daf/docs/INTEGRATION_MAP.md` - Full integration details
