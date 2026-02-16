"""Authentication integration for DAF.

DAF sidecar utility: Wraps `cogames.auth.BaseCLIAuthenticator` for DAF workflows.

Provides login, token status checking, and token retrieval for
tournament and leaderboard interactions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console

logger = logging.getLogger("daf.auth_integration")


def daf_login(
    server_url: str = "https://cogames.ai",
    force: bool = False,
    console: Optional[Console] = None,
) -> bool:
    """Perform CoGames login via OAuth2 flow.

    Wraps cogames.auth.BaseCLIAuthenticator to handle login.

    Args:
        server_url: CoGames server URL
        force: Force re-login even if token exists
        console: Optional Rich console for output

    Returns:
        True if login succeeded, False otherwise
    """
    if console is None:
        console = Console()

    try:
        from cogames.auth import CoGamesAuthenticator

        authenticator = CoGamesAuthenticator(server_url=server_url)

        if not force and authenticator.has_saved_token():
            logger.info("Already logged in (token exists)")
            console.print("[green]Already logged in.[/green]")
            return True

        logger.info(f"Initiating login to {server_url}")
        authenticator.login()
        logger.info("Login successful")
        console.print("[green]Login successful.[/green]")
        return True

    except ImportError:
        logger.error("cogames.auth not available")
        console.print("[red]cogames.auth module not available[/red]")
        return False
    except Exception as e:
        logger.error(f"Login failed: {e}", exc_info=True)
        console.print(f"[red]Login failed: {e}[/red]")
        return False


def daf_check_auth_status(
    server_url: str = "https://cogames.ai",
) -> bool:
    """Check if a valid auth token exists.

    Args:
        server_url: CoGames server URL

    Returns:
        True if valid token exists
    """
    try:
        from cogames.auth import CoGamesAuthenticator

        authenticator = CoGamesAuthenticator(server_url=server_url)
        has_token = authenticator.has_saved_token()
        logger.info(f"Auth status for {server_url}: {'active' if has_token else 'no token'}")
        return has_token

    except ImportError:
        logger.warning("cogames.auth not available")
        return False
    except Exception as e:
        logger.warning(f"Auth status check failed: {e}")
        return False


def daf_get_auth_token(
    server_url: str = "https://cogames.ai",
) -> Optional[str]:
    """Retrieve stored auth token.

    Args:
        server_url: CoGames server URL

    Returns:
        Auth token string if available, None otherwise
    """
    try:
        from cogames.auth import CoGamesAuthenticator

        authenticator = CoGamesAuthenticator(server_url=server_url)
        if authenticator.has_saved_token():
            token = authenticator.get_token()
            logger.info("Retrieved auth token")
            return token

        logger.info("No auth token available")
        return None

    except ImportError:
        logger.warning("cogames.auth not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to retrieve auth token: {e}")
        return None


def daf_get_auth_config_path() -> Path:
    """Get the path where auth configuration is stored.

    Returns:
        Path to auth config file
    """
    try:
        from cogames.auth import AuthConfigReaderWriter

        return AuthConfigReaderWriter.default_config_path()
    except (ImportError, AttributeError):
        # Fallback to default location
        return Path.home() / ".cogames" / "auth.yaml"
