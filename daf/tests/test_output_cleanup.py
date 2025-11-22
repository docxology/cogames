#!/usr/bin/env python3
"""
Test output cleanup utility with retention policy.

Manages test output storage by removing old test runs while keeping
a specified number of recent runs.
"""

import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def get_session_dirs(parent_dir: Path) -> List[tuple[Path, datetime]]:
    """
    Get all session directories with their creation times.
    
    Args:
        parent_dir: Parent directory containing session subdirectories
        
    Returns:
        List of (directory, creation_datetime) tuples sorted by time (oldest first)
    """
    sessions = []
    
    if not parent_dir.exists():
        return sessions
    
    for item in parent_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Try to parse directory name as timestamp (YYYYMMDD_HHMMSS)
        try:
            dir_name = item.name
            if len(dir_name) == 15 and dir_name[8] == '_':  # YYYYMMDD_HHMMSS
                dt = datetime.strptime(dir_name, "%Y%m%d_%H%M%S")
                sessions.append((item, dt))
        except (ValueError, AttributeError):
            # Skip directories that don't match timestamp format
            continue
    
    # Sort by timestamp (oldest first)
    sessions.sort(key=lambda x: x[1])
    return sessions


def cleanup_old_runs(
    parent_dir: Path,
    keep_count: int = 10,
    dry_run: bool = False,
    min_age_days: Optional[int] = None,
) -> dict:
    """
    Clean up old test output runs keeping only the most recent ones.
    
    Args:
        parent_dir: Parent directory containing session subdirectories
        keep_count: Number of recent runs to keep
        dry_run: If True, don't actually delete, just report what would be deleted
        min_age_days: Only delete runs older than this many days
        
    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "total_runs": 0,
        "runs_deleted": 0,
        "size_freed_mb": 0.0,
        "skipped_runs": [],
        "deleted_runs": [],
    }
    
    sessions = get_session_dirs(parent_dir)
    total_runs = len(sessions)
    stats["total_runs"] = total_runs
    
    if total_runs <= keep_count:
        logger.info(f"Only {total_runs} run(s) found, keeping all (keep_count={keep_count})")
        return stats
    
    # Determine which sessions to delete
    to_delete = sessions[:-keep_count]  # All except the last keep_count
    
    # Apply minimum age filter if specified
    if min_age_days is not None:
        cutoff_time = datetime.now() - timedelta(days=min_age_days)
        to_delete = [
            (path, dt) for path, dt in to_delete
            if dt < cutoff_time
        ]
    
    # Delete old runs
    for session_dir, session_time in to_delete:
        size_mb = get_directory_size(session_dir) / (1024 * 1024)
        
        if dry_run:
            logger.info(f"[DRY RUN] Would delete: {session_dir.name} ({size_mb:.1f} MB)")
            stats["deleted_runs"].append({
                "name": session_dir.name,
                "size_mb": size_mb,
                "deleted": False,
            })
        else:
            try:
                shutil.rmtree(session_dir)
                logger.info(f"Deleted: {session_dir.name} ({size_mb:.1f} MB)")
                stats["runs_deleted"] += 1
                stats["size_freed_mb"] += size_mb
                stats["deleted_runs"].append({
                    "name": session_dir.name,
                    "size_mb": size_mb,
                    "deleted": True,
                })
            except Exception as e:
                logger.error(f"Failed to delete {session_dir.name}: {e}")
    
    # Track skipped runs (kept)
    kept_runs = sessions[-keep_count:] if keep_count < total_runs else sessions
    for session_dir, session_time in kept_runs:
        size_mb = get_directory_size(session_dir) / (1024 * 1024)
        stats["skipped_runs"].append({
            "name": session_dir.name,
            "size_mb": size_mb,
        })
    
    return stats


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def cleanup_test_outputs(
    output_base: Path = Path("./daf_output"),
    keep_count: int = 10,
    dry_run: bool = False,
    min_age_days: Optional[int] = None,
) -> dict:
    """
    Clean up all test output categories.
    
    Args:
        output_base: Base output directory
        keep_count: Number of recent runs to keep per category
        dry_run: If True, don't actually delete
        min_age_days: Only delete runs older than this many days
        
    Returns:
        Dictionary mapping cleanup categories to statistics
    """
    cleanup_dirs = [
        output_base / "sweeps",
        output_base / "comparisons",
        output_base / "training",
        output_base / "deployment",
        output_base / "evaluations",
    ]
    
    results = {}
    total_freed_mb = 0.0
    total_deleted = 0
    
    for cleanup_dir in cleanup_dirs:
        category = cleanup_dir.name
        logger.info(f"\nCleaning up {category}...")
        
        stats = cleanup_old_runs(
            cleanup_dir,
            keep_count=keep_count,
            dry_run=dry_run,
            min_age_days=min_age_days,
        )
        results[category] = stats
        
        total_freed_mb += stats["size_freed_mb"]
        total_deleted += stats["runs_deleted"]
    
    results["summary"] = {
        "total_deleted": total_deleted,
        "total_freed_mb": total_freed_mb,
        "dry_run": dry_run,
    }
    
    return results


def print_cleanup_summary(results: dict) -> None:
    """Print cleanup results summary."""
    summary = results.get("summary", {})
    
    print("\n" + "=" * 70)
    print("Test Output Cleanup Summary")
    print("=" * 70)
    
    if summary.get("dry_run"):
        print("[DRY RUN MODE - No actual deletions]")
    
    print(f"Total runs deleted: {summary.get('total_deleted', 0)}")
    print(f"Total space freed:  {summary.get('total_freed_mb', 0.0):.1f} MB")
    
    print("\nBreakdown by category:")
    for category, stats in results.items():
        if category == "summary":
            continue
        print(f"\n  {category.upper()}:")
        print(f"    Total runs: {stats['total_runs']}")
        print(f"    Runs kept:  {len(stats['skipped_runs'])}")
        print(f"    Runs deleted: {stats['runs_deleted']}")
        print(f"    Space freed: {stats['size_freed_mb']:.1f} MB")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean up old test output runs"
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=10,
        help="Number of recent runs to keep per category (default: 10)"
    )
    parser.add_argument(
        "--min-age-days",
        type=int,
        help="Only delete runs older than this many days"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--output-dir",
        default="./daf_output",
        help="Base output directory (default: ./daf_output)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)
    
    # Run cleanup
    results = cleanup_test_outputs(
        output_base=output_dir,
        keep_count=args.keep,
        dry_run=args.dry_run,
        min_age_days=args.min_age_days,
    )
    
    print_cleanup_summary(results)
    
    sys.exit(0)

