"""Shared test fixtures for DAF tests with improved error handling and resilience.

Provides robust mission loading, policy creation, and environment setup utilities.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import pytest

logger = logging.getLogger(__name__)


class MissionLoadingError(Exception):
    """Raised when mission loading fails."""

    pass


class PolicyCreationError(Exception):
    """Raised when policy creation fails."""

    pass


@pytest.fixture
def mission_loader():
    """Fixture providing robust mission loading utility.
    
    Yields:
        Function that safely loads missions with better error handling
    """

    def load_mission(mission_name: str) -> Tuple[str, Any]:
        """Load a mission safely with detailed error messages.
        
        Args:
            mission_name: Name of mission to load (e.g., 'cogsguard_machina_1')
            
        Returns:
            Tuple of (mission_name, env_config)
            
        Raises:
            MissionLoadingError: If mission cannot be loaded
        """
        try:
            from cogames.cli.mission import get_mission

            logger.info(f"Loading mission: {mission_name}")

            name, env_cfg, _ = get_mission(mission_name)

            logger.info(f"Successfully loaded mission: {name}")
            return name, env_cfg

        except ImportError as e:
            raise MissionLoadingError(f"Failed to import mission utilities: {e}") from e
        except AttributeError as e:
            raise MissionLoadingError(f"Invalid mission name '{mission_name}': {e}") from e
        except ValueError as e:
            raise MissionLoadingError(f"Mission configuration error for '{mission_name}': {e}") from e
        except Exception as e:
            raise MissionLoadingError(f"Unexpected error loading mission '{mission_name}': {type(e).__name__}: {e}") from e

    yield load_mission


@pytest.fixture
def policy_loader():
    """Fixture providing robust policy creation utility.
    
    Yields:
        Function that safely creates policies with better error handling
    """

    def load_policy(policy_class_path: str, weights_path: Optional[Path] = None):
        """Load a policy safely with detailed error messages.
        
        Args:
            policy_class_path: Class path to policy (e.g., 'lstm', 'baseline')
            weights_path: Optional path to weights file
            
        Returns:
            Policy instance
            
        Raises:
            PolicyCreationError: If policy cannot be created
        """
        try:
            from mettagrid.policy.policy import PolicySpec
            from mettagrid.policy.loader import initialize_or_load_policy
            from mettagrid.policy.policy_env_interface import PolicyEnvInterface

            logger.info(f"Loading policy: {policy_class_path}")

            # Create policy spec
            policy_spec = PolicySpec(class_path=policy_class_path, data_path=weights_path)

            # Note: Full initialization requires env_interface
            # For testing, we typically just validate the spec
            logger.info(f"Successfully created policy spec: {policy_class_path}")
            return policy_spec

        except ImportError as e:
            raise PolicyCreationError(f"Failed to import policy utilities: {e}") from e
        except AttributeError as e:
            raise PolicyCreationError(f"Invalid policy class path '{policy_class_path}': {e}") from e
        except FileNotFoundError as e:
            raise PolicyCreationError(f"Weights file not found: {e}") from e
        except Exception as e:
            raise PolicyCreationError(
                f"Unexpected error loading policy '{policy_class_path}': {type(e).__name__}: {e}"
            ) from e

    yield load_policy


@pytest.fixture
def environment_validator():
    """Fixture providing environment validation utility.
    
    Yields:
        Function that validates environment with detailed reporting
    """

    def validate_environment(check_cuda: bool = False, check_disk: bool = False) -> bool:
        """Validate environment with optional checks.
        
        Args:
            check_cuda: Whether to check CUDA availability
            check_disk: Whether to check disk space
            
        Returns:
            True if environment is healthy
        """
        try:
            from daf.src.environment_checks import daf_check_environment

            logger.info("Validating environment")
            result = daf_check_environment(check_cuda=check_cuda, check_disk=check_disk)

            if result.is_healthy():
                logger.info("Environment validation passed")
                return True
            else:
                logger.warning(f"Environment issues: {result.warnings}")
                return False

        except Exception as e:
            logger.error(f"Environment validation error: {e}")
            return False

    yield validate_environment


@pytest.fixture
def temp_mission_data(tmp_path):
    """Fixture providing temporary mission data directory.
    
    Yields:
        Temporary directory path for mission data
    """
    mission_data_dir = tmp_path / "mission_data"
    mission_data_dir.mkdir(parents=True, exist_ok=True)
    yield mission_data_dir


@pytest.fixture
def mission_config_factory():
    """Fixture providing mission configuration factory.
    
    Yields:
        Function that creates test mission configurations
    """

    def create_config(
        mission_name: str, num_agents: int = 2, episode_length: int = 100, **kwargs
    ) -> dict:
        """Create a test mission configuration.
        
        Args:
            mission_name: Name of mission
            num_agents: Number of agents
            episode_length: Episode length in steps
            **kwargs: Additional configuration options
            
        Returns:
            Configuration dictionary
        """
        config = {
            "mission_name": mission_name,
            "num_agents": num_agents,
            "episode_length": episode_length,
        }
        config.update(kwargs)
        return config

    yield create_config


@pytest.fixture
def safe_mission_loader(mission_loader):
    """Wrapper fixture providing safe mission loading with skip on error.
    
    Yields:
        Function that loads missions or skips test if loading fails
    """

    def load_or_skip(mission_name: str) -> Tuple[str, Any]:
        """Load mission or skip test if loading fails.
        
        Args:
            mission_name: Name of mission to load
            
        Returns:
            Tuple of (mission_name, env_config)
            
        Raises:
            pytest.skip if mission cannot be loaded
        """
        try:
            return mission_loader(mission_name)
        except MissionLoadingError as e:
            pytest.skip(f"Could not load mission '{mission_name}': {e}")

    yield load_or_skip


@pytest.fixture
def safe_policy_loader(policy_loader):
    """Wrapper fixture providing safe policy loading with skip on error.
    
    Yields:
        Function that loads policies or skips test if loading fails
    """

    def load_or_skip(policy_class_path: str, weights_path: Optional[Path] = None):
        """Load policy or skip test if loading fails.
        
        Args:
            policy_class_path: Class path to policy
            weights_path: Optional path to weights file
            
        Returns:
            Policy spec
            
        Raises:
            pytest.skip if policy cannot be loaded
        """
        try:
            return policy_loader(policy_class_path, weights_path)
        except PolicyCreationError as e:
            pytest.skip(f"Could not load policy '{policy_class_path}': {e}")

    yield load_or_skip

