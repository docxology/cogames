"""Utility functions for DAF output management.

Provides helper functions for working with organized DAF outputs:
- Finding outputs
- Cleanup and archival
- Migration and consolidation
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table


def find_latest_session(output_dir: Path | str, operation: str = "") -> Optional[Path]:
    """Find the most recent session directory.

    Args:
        output_dir: DAF output base directory
        operation: Optional operation type to filter (sweep, comparison, etc)

    Returns:
        Path to most recent session directory or None if not found

    Example:
        >>> latest = find_latest_session("./daf_output", "sweep")
        >>> print(latest)
        ./daf_output/sweeps/20241121_143022
    """
    output_dir = Path(output_dir)

    if operation:
        search_dir = output_dir / operation
    else:
        search_dir = output_dir

    # Find all session directories (format: YYYYMMDD_HHMMSS)
    sessions = sorted(
        [
            d
            for d in search_dir.glob("*")
            if d.is_dir() and len(d.name) == 15 and d.name[8] == "_"
        ],
        reverse=True,
    )

    return sessions[0] if sessions else None


def list_sessions(
    output_dir: Path | str,
    operation: str = "",
    max_results: int = 10,
) -> List[dict]:
    """List available sessions in chronological order.

    Args:
        output_dir: DAF output base directory
        operation: Optional operation type to filter
        max_results: Maximum number of sessions to return

    Returns:
        List of session info dictionaries

    Example:
        >>> sessions = list_sessions("./daf_output", "sweep")
        >>> for session in sessions:
        ...     print(f"{session['id']}: {session['created']}")
    """
    output_dir = Path(output_dir)

    if operation:
        search_dir = output_dir / operation
    else:
        # Search all operation directories
        search_dirs = [
            output_dir / op
            for op in [
                "sweeps",
                "comparisons",
                "training",
                "deployment",
                "evaluations",
            ]
        ]

    sessions = []

    search_paths = [search_dir] if operation else search_dirs

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for session_dir in sorted(search_path.glob("*"), reverse=True):
            if not session_dir.is_dir() or not (len(session_dir.name) == 15):
                continue

            stat = session_dir.stat()
            sessions.append(
                {
                    "id": session_dir.name,
                    "created": datetime.fromtimestamp(stat.st_ctime),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "path": str(session_dir),
                    "operation": search_path.name if operation else operation or "unknown",
                }
            )

            if len(sessions) >= max_results:
                return sorted(sessions, key=lambda x: x["created"], reverse=True)

    return sorted(sessions, key=lambda x: x["created"], reverse=True)


def cleanup_old_sessions(
    output_dir: Path | str,
    operation: str = "",
    days: int = 30,
    dry_run: bool = True,
    console: Optional[Console] = None,
) -> int:
    """Remove old session directories.

    Args:
        output_dir: DAF output base directory
        operation: Optional operation type to filter
        days: Number of days to keep
        dry_run: If True, show what would be deleted without deleting
        console: Optional Rich console for output

    Returns:
        Number of sessions removed

    Example:
        >>> # Remove sessions older than 30 days
        >>> cleanup_old_sessions("./daf_output", days=30, dry_run=False)
        Removed 5 old sessions
    """
    if console is None:
        console = Console()

    output_dir = Path(output_dir)
    cutoff_date = datetime.now() - timedelta(days=days)
    removed = 0

    if operation:
        search_dir = output_dir / operation
        search_dirs = [search_dir] if search_dir.exists() else []
    else:
        search_dirs = [
            output_dir / op
            for op in [
                "sweeps",
                "comparisons",
                "training",
                "deployment",
                "evaluations",
            ]
        ]

    for search_path in search_dirs:
        if not search_path.exists():
            continue

        for session_dir in search_path.iterdir():
            if not session_dir.is_dir():
                continue

            stat = session_dir.stat()
            modified_date = datetime.fromtimestamp(stat.st_mtime)

            if modified_date < cutoff_date:
                size_mb = sum(
                    f.stat().st_size for f in session_dir.rglob("*") if f.is_file()
                ) / (1024 * 1024)

                if dry_run:
                    console.print(
                        f"[yellow]Would remove[/yellow] {session_dir.name} ({size_mb:.1f}MB)"
                    )
                else:
                    shutil.rmtree(session_dir)
                    console.print(
                        f"[green]Removed[/green] {session_dir.name} ({size_mb:.1f}MB)"
                    )

                removed += 1

    return removed


def print_session_info(
    session_dir: Path | str,
    console: Optional[Console] = None,
) -> None:
    """Print formatted information about a session.

    Args:
        session_dir: Path to session directory
        console: Optional Rich console for output

    Example:
        >>> print_session_info("./daf_output/sweeps/20241121_143022")
    """
    if console is None:
        console = Console()

    session_dir = Path(session_dir)

    if not session_dir.exists():
        console.print(f"[red]Session directory not found: {session_dir}[/red]")
        return

    # Get session metadata if available
    metadata_file = session_dir / "metadata.json"
    metadata = {}

    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    # Calculate directory statistics
    total_files = sum(1 for _ in session_dir.rglob("*") if _.is_file())
    total_size = sum(
        f.stat().st_size for f in session_dir.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    stat = session_dir.stat()
    created = datetime.fromtimestamp(stat.st_ctime)

    # Print header
    console.print(f"\n[bold cyan]Session: {session_dir.name}[/bold cyan]")
    console.print(f"[cyan]Path: {session_dir}[/cyan]\n")

    # Print info table
    table = Table(show_header=False)
    table.add_row("Created", created.strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("Total Files", str(total_files))
    table.add_row("Total Size", f"{total_size:.1f} MB")

    if metadata:
        for key, value in metadata.items():
            if key not in ["operations"]:  # Skip large fields
                table.add_row(key, str(value)[:50])

    console.print(table)

    # List subdirectories
    subdirs = sorted([d for d in session_dir.iterdir() if d.is_dir()])
    if subdirs:
        console.print("\n[bold]Subdirectories:[/bold]")
        for subdir in subdirs:
            file_count = sum(1 for _ in subdir.rglob("*") if _.is_file())
            console.print(f"  • {subdir.name} ({file_count} files)")


def export_session(
    session_dir: Path | str,
    export_path: Path | str,
    compress: bool = True,
    console: Optional[Console] = None,
) -> Path:
    """Export a session to archive file or directory.

    Args:
        session_dir: Source session directory
        export_path: Destination path
        compress: If True, create .tar.gz archive
        console: Optional Rich console for output

    Returns:
        Path to exported file/directory

    Example:
        >>> export_session(
        ...     "./daf_output/sweeps/20241121_143022",
        ...     "./backups/sweep_20241121.tar.gz"
        ... )
    """
    if console is None:
        console = Console()

    session_dir = Path(session_dir)
    export_path = Path(export_path)

    if not session_dir.exists():
        console.print(f"[red]Source directory not found: {session_dir}[/red]")
        raise FileNotFoundError(f"Source directory not found: {session_dir}")

    export_path.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        # Create tar.gz archive
        archive_base = str(export_path).replace(".tar.gz", "")
        console.print(f"[yellow]Compressing...[/yellow]")
        shutil.make_archive(
            archive_base,
            "gztar",
            session_dir.parent,
            session_dir.name,
        )
        result_path = Path(f"{archive_base}.tar.gz")
        console.print(f"[green]✓ Exported to: {result_path}[/green]")
    else:
        # Copy directory
        console.print(f"[yellow]Copying...[/yellow]")
        shutil.copytree(session_dir, export_path)
        result_path = export_path
        console.print(f"[green]✓ Exported to: {result_path}[/green]")

    size_mb = sum(
        f.stat().st_size for f in result_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    console.print(f"[cyan]Size: {size_mb:.1f} MB[/cyan]")

    return result_path


def print_output_summary(
    output_dir: Path | str = "./daf_output",
    console: Optional[Console] = None,
) -> None:
    """Print summary of all DAF outputs.

    Args:
        output_dir: DAF output base directory
        console: Optional Rich console for output

    Example:
        >>> print_output_summary("./daf_output")
    """
    if console is None:
        console = Console()

    output_dir = Path(output_dir)

    if not output_dir.exists():
        console.print(f"[red]Output directory not found: {output_dir}[/red]")
        return

    console.print(f"\n[bold cyan]DAF Output Summary[/bold cyan]")
    console.print(f"[cyan]Directory: {output_dir}[/cyan]\n")

    # Scan operations
    operations = [
        "sweeps",
        "comparisons",
        "training",
        "deployment",
        "evaluations",
        "visualizations",
        "logs",
        "reports",
        "artifacts",
    ]

    table = Table(title="Output Organization", show_header=True, header_style="bold")
    table.add_column("Operation Type")
    table.add_column("Sessions", justify="right")
    table.add_column("Total Files", justify="right")
    table.add_column("Size (MB)", justify="right")

    for op in operations:
        op_dir = output_dir / op
        if not op_dir.exists():
            continue

        # Count sessions (YYYYMMDD_HHMMSS format)
        sessions = len(
            [
                d
                for d in op_dir.iterdir()
                if d.is_dir() and len(d.name) == 15 and d.name[8] == "_"
            ]
        )

        # Count files and size
        try:
            files = sum(1 for _ in op_dir.rglob("*") if _.is_file())
            size_mb = sum(
                f.stat().st_size for f in op_dir.rglob("*") if f.is_file()
            ) / (1024 * 1024)
        except (OSError, PermissionError):
            files = 0
            size_mb = 0.0

        if sessions > 0 or files > 0:
            table.add_row(op, str(sessions), str(files), f"{size_mb:.1f}")

    console.print(table)

    # Overall stats
    try:
        total_files = sum(1 for _ in output_dir.rglob("*") if _.is_file())
        total_size = sum(
            f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
        ) / (1024 * 1024)
    except (OSError, PermissionError):
        total_files = 0
        total_size = 0.0

    console.print(f"\n[cyan]Total: {total_files} files, {total_size:.1f} MB[/cyan]\n")






