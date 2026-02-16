"""Tests for daf.src.train.auth_integration module.

Tests the authentication wrapper functions that integrate with
cogames.auth.BaseCLIAuthenticator.
"""

import logging

import pytest

logger = logging.getLogger(__name__)


class TestAuthIntegration:
    """Tests for authentication integration functions."""

    def test_check_auth_status_returns_bool(self):
        """daf_check_auth_status returns a boolean."""
        from daf.src.train.auth_integration import daf_check_auth_status

        result = daf_check_auth_status()
        assert isinstance(result, bool)
        logger.info(f"Auth status: {result}")

    def test_get_auth_token_returns_string_or_none(self):
        """daf_get_auth_token returns a string token or None."""
        from daf.src.train.auth_integration import daf_get_auth_token

        token = daf_get_auth_token()
        assert token is None or isinstance(token, str)

    def test_custom_server_url(self):
        """Auth functions accept custom server URLs without error."""
        from daf.src.train.auth_integration import daf_check_auth_status

        # Should not raise, even if cogames.auth is unavailable
        result = daf_check_auth_status(server_url="https://custom.example.com")
        assert isinstance(result, bool)

    @pytest.mark.auth
    def test_login_requires_interaction(self):
        """daf_login is importable and callable."""
        from daf.src.train.auth_integration import daf_login

        assert callable(daf_login)

    def test_get_auth_config_path(self):
        """daf_get_auth_config_path returns a Path."""
        from daf.src.train.auth_integration import daf_get_auth_config_path
        from pathlib import Path

        try:
            result = daf_get_auth_config_path()
            assert isinstance(result, Path)
        except (ImportError, AttributeError):
            # cogames.auth may not be fully available
            pytest.skip("cogames.auth not fully available")
