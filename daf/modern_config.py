#!/usr/bin/env python3
"""
Modern configuration management for DAF using Pydantic and YAML.

Provides type-safe, validated configurations with environment variable overrides.
Replaces legacy JSON configs with human-editable YAML format.
"""

from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import os

try:
    import yaml
except ImportError:
    yaml = None


class TestPhase(str, Enum):
    """Test execution phase."""
    COGAMES = "cogames"
    DAF = "daf"
    ALL = "all"


class TestSuiteConfig(BaseModel):
    """Configuration for individual test suite."""
    name: str = Field(..., description="Suite name")
    file: str = Field(..., description="Test file path")
    phase: int = Field(..., ge=1, le=2, description="Phase number (1 or 2)")
    expected_count: int = Field(0, ge=0, description="Expected test count")
    skip: bool = Field(False, description="Skip this suite")
    timeout_seconds: int = Field(600, ge=60, description="Test timeout")
    
    class Config:
        use_enum_values = True


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    max_workers: Optional[int] = Field(None, description="Max concurrent suites (None=auto)")
    parallel: bool = Field(True, description="Run suites in parallel")
    verbose: bool = Field(False, description="Verbose output")
    collect_only: bool = Field(False, description="Only collect tests, don't execute")
    stop_on_failure: bool = Field(False, description="Stop on first failure")
    retry_failed: bool = Field(True, description="Retry failed tests")
    max_retries: int = Field(3, ge=0, le=10, description="Max retry attempts")
    
    @validator("max_workers")
    def validate_workers(cls, v):
        if v is not None and v < 1:
            raise ValueError("max_workers must be > 0")
        return v


class OutputConfig(BaseModel):
    """Output configuration."""
    base_dir: str = Field("./daf_output", description="Base output directory")
    formats: List[str] = Field(["json", "markdown", "html"], description="Report formats")
    keep_old_runs: int = Field(10, ge=1, description="Number of old runs to keep")
    auto_cleanup: bool = Field(True, description="Auto-cleanup old runs")
    
    class Config:
        use_enum_values = True


class ReportingConfig(BaseModel):
    """Reporting configuration."""
    enabled: bool = Field(True, description="Enable reporting")
    generate_charts: bool = Field(True, description="Generate performance charts")
    include_insights: bool = Field(True, description="Include automated insights")
    performance_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"p95_seconds": 60.0, "p99_seconds": 120.0},
        description="Performance thresholds for alerts"
    )


class DAFConfig(BaseModel):
    """Main DAF configuration."""
    version: str = Field("1.0", description="Config version")
    
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    
    cogames_suites: List[TestSuiteConfig] = Field(default_factory=list)
    daf_suites: List[TestSuiteConfig] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True
    
    @classmethod
    def from_yaml(cls, yaml_file: Path) -> "DAFConfig":
        """Load config from YAML file.
        
        Args:
            yaml_file: Path to YAML config file
            
        Returns:
            DAFConfig instance
            
        Raises:
            ValueError: If YAML library not available or file not found
        """
        if yaml is None:
            raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
        
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_file}")
        
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f) or {}
        
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "DAFConfig":
        """Load config from environment variables.
        
        Supports override of any config field via DAF_* environment variables.
        Example: DAF_EXECUTION__MAX_WORKERS=4
        
        Returns:
            DAFConfig instance
        """
        config = cls()
        
        # Parse environment variables
        for key, value in os.environ.items():
            if key.startswith("DAF_"):
                # Convert DAF_EXECUTION__MAX_WORKERS to execution.max_workers
                parts = key[4:].lower().split("__")
                if len(parts) == 2:
                    section, field = parts
                    if section == "execution":
                        setattr(config.execution, field, _parse_env_value(value))
                    elif section == "output":
                        setattr(config.output, field, _parse_env_value(value))
                    elif section == "reporting":
                        setattr(config.reporting, field, _parse_env_value(value))
        
        return config
    
    @classmethod
    def from_defaults(cls) -> "DAFConfig":
        """Create config with CoGames and DAF test suites.
        
        Returns:
            DAFConfig instance with all default test suites
        """
        config = cls()
        
        config.cogames_suites = [
            TestSuiteConfig(name="CLI Tests", file="tests/test_cli.py", phase=1, expected_count=6),
            TestSuiteConfig(name="Core Game Tests", file="tests/test_cogs_vs_clips.py", phase=1, expected_count=4),
            TestSuiteConfig(name="CVC Assembler Hearts Tests", file="tests/test_cvc_assembler_hearts.py", phase=1, expected_count=2),
            TestSuiteConfig(name="Procedural Maps Tests", file="tests/test_procedural_maps.py", phase=1, expected_count=11),
            TestSuiteConfig(name="Scripted Policies Tests", file="tests/test_scripted_policies.py", phase=1, expected_count=13),
            TestSuiteConfig(name="Train Integration Tests", file="tests/test_train_integration.py", phase=1, expected_count=2),
            TestSuiteConfig(name="Train Vector Alignment Tests", file="tests/test_train_vector_alignment.py", phase=1, expected_count=5),
            TestSuiteConfig(name="All Games Describe Tests", file="tests/test_all_games_describe.py", phase=1, expected_count=47),
            TestSuiteConfig(name="All Games Eval Tests", file="tests/test_all_games_eval.py", phase=1, expected_count=48),
            TestSuiteConfig(name="All Games Play Tests", file="tests/test_all_games_play.py", phase=1, expected_count=47),
        ]
        
        config.daf_suites = [
            TestSuiteConfig(name="Configuration Tests", file="daf/tests/test_config.py", phase=2, expected_count=15),
            TestSuiteConfig(name="Environment Check Tests", file="daf/tests/test_environment_checks.py", phase=2, expected_count=13),
            TestSuiteConfig(name="Sweep Tests", file="daf/tests/test_sweeps.py", phase=2, expected_count=16),
            TestSuiteConfig(name="Comparison Tests", file="daf/tests/test_comparison.py", phase=2, expected_count=12),
            TestSuiteConfig(name="Deployment Tests", file="daf/tests/test_deployment.py", phase=2, expected_count=11),
            TestSuiteConfig(name="Distributed Training Tests", file="daf/tests/test_distributed_training.py", phase=2, expected_count=12),
            TestSuiteConfig(name="Visualization Tests", file="daf/tests/test_visualization.py", phase=2, expected_count=9),
            TestSuiteConfig(name="Mission Analysis Tests", file="daf/tests/test_mission_analysis.py", phase=2, expected_count=12),
        ]
        
        return config
    
    def to_yaml(self) -> str:
        """Convert config to YAML string.
        
        Returns:
            YAML formatted configuration
            
        Raises:
            ValueError: If YAML library not available
        """
        if yaml is None:
            raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
        
        return yaml.dump(
            self.dict(),
            default_flow_style=False,
            sort_keys=False,
        )
    
    def save_yaml(self, output_file: Path) -> None:
        """Save config to YAML file.
        
        Args:
            output_file: Path to output file
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(self.to_yaml())


def _parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate type.
    
    Args:
        value: Environment variable value string
        
    Returns:
        Parsed value (int, bool, str, etc.)
    """
    value_lower = value.lower()
    
    # Boolean
    if value_lower in ("true", "yes", "1"):
        return True
    if value_lower in ("false", "no", "0"):
        return False
    
    # Integer
    if value.isdigit():
        return int(value)
    
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    
    # String
    return value


# Export default config
DEFAULT_CONFIG = DAFConfig.from_defaults()

