#!/usr/bin/env python3
"""
Project cleanup script for Penn Treebank Language Modeling project.

This script prepares the project for efficient uploads to Git repositories
or Google Drive by removing large extracted files, temporary data, and 
cached content while preserving essential project files.

Usage:
    python scripts/cleanup_project.py [options]

Options:
    --dry-run       Show what would be deleted without actually deleting
    --aggressive    More aggressive cleanup (removes all generated files)
    --keep-models   Keep trained model checkpoints (default: remove)
    --interactive   Ask before deleting each category
"""

import os
import shutil
import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total_size


def format_size(size_bytes: int) -> str:
    """Format size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def find_cleanup_targets() -> List[Tuple[str, Path, str]]:
    """
    Find all directories and files that can be cleaned up.
    
    Returns:
        List of tuples: (category, path, description)
    """
    targets = []
    
    # Large extracted data directories
    extracted_data_dirs = [
        "data/ptb/LDC99T42",
        "data/ptb/treebank3",  # Alternative extraction location
    ]
    
    for data_dir in extracted_data_dirs:
        path = Path(data_dir)
        if path.exists():
            size = get_directory_size(path)
            targets.append((
                "extracted_data", 
                path, 
                f"Extracted Penn Treebank data ({format_size(size)})"
            ))
    
    # Python cache directories
    cache_dirs = []
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            cache_path = Path(root) / "__pycache__"
            size = get_directory_size(cache_path)
            targets.append((
                "python_cache", 
                cache_path, 
                f"Python cache ({format_size(size)})"
            ))
    
    # Jupyter notebook checkpoints
    for root, dirs, files in os.walk("."):
        if ".ipynb_checkpoints" in dirs:
            checkpoint_path = Path(root) / ".ipynb_checkpoints"
            size = get_directory_size(checkpoint_path)
            targets.append((
                "jupyter_cache", 
                checkpoint_path, 
                f"Jupyter checkpoints ({format_size(size)})"
            ))
    
    # TensorBoard logs
    runs_dir = Path("runs")
    if runs_dir.exists():
        size = get_directory_size(runs_dir)
        targets.append((
            "tensorboard_logs", 
            runs_dir, 
            f"TensorBoard logs ({format_size(size)})"
        ))
    
    # Model checkpoints (optional cleanup)
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        size = get_directory_size(checkpoints_dir)
        targets.append((
            "model_checkpoints", 
            checkpoints_dir, 
            f"Model checkpoints ({format_size(size)})"
        ))
    
    # Results and temporary files
    temp_files = [
        "results",
        "temp",
        "tmp",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
    ]
    
    for temp_dir in temp_files:
        path = Path(temp_dir)
        if path.exists():
            size = get_directory_size(path)
            targets.append((
                "temp_files", 
                path, 
                f"Temporary files in {temp_dir}/ ({format_size(size)})"
            ))
    
    # Large log files
    for log_file in Path(".").rglob("*.log"):
        if log_file.stat().st_size > 1024 * 1024:  # > 1MB
            size = log_file.stat().st_size
            targets.append((
                "log_files", 
                log_file, 
                f"Large log file ({format_size(size)})"
            ))
    
    return targets


def cleanup_category(targets: List[Tuple[str, Path, str]], category: str, 
                    dry_run: bool = False, interactive: bool = False) -> Tuple[int, int]:
    """
    Clean up all targets in a specific category.
    
    Returns:
        Tuple of (files_removed, total_size_freed)
    """
    category_targets = [t for t in targets if t[0] == category]
    if not category_targets:
        return 0, 0
    
    files_removed = 0
    total_size_freed = 0
    
    print(f"\n🗑️  Cleaning up {category.replace('_', ' ')}:")
    
    for _, path, description in category_targets:
        if not path.exists():
            continue
            
        size = get_directory_size(path) if path.is_dir() else path.stat().st_size
        
        if interactive:
            response = input(f"   Delete {description}? (y/N): ").lower().strip()
            if response not in ['y', 'yes']:
                print(f"   ⏭️  Skipped: {path}")
                continue
        
        if dry_run:
            print(f"   [DRY RUN] Would delete: {path} ({format_size(size)})")
        else:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"   ✅ Deleted: {path} ({format_size(size)})")
                files_removed += 1
                total_size_freed += size
            except Exception as e:
                print(f"   ❌ Failed to delete {path}: {e}")
    
    return files_removed, total_size_freed


def main():
    parser = argparse.ArgumentParser(
        description="Clean up Penn Treebank project for efficient uploads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/cleanup_project.py --dry-run    # Preview what would be deleted
    python scripts/cleanup_project.py              # Standard cleanup
    python scripts/cleanup_project.py --aggressive # Remove everything including models
    python scripts/cleanup_project.py --keep-models # Keep model checkpoints
        """
    )
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--aggressive', action='store_true',
                       help='More aggressive cleanup (removes all generated files)')
    parser.add_argument('--keep-models', action='store_true',
                       help='Keep trained model checkpoints')
    parser.add_argument('--interactive', action='store_true',
                       help='Ask before deleting each category')
    
    args = parser.parse_args()
    
    print("🧹 Penn Treebank Project Cleanup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("config").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   (Directory should contain 'src' and 'config' folders)")
        sys.exit(1)
    
    # Find all cleanup targets
    print("🔍 Scanning for cleanup targets...")
    targets = find_cleanup_targets()
    
    if not targets:
        print("✨ Project is already clean! No cleanup targets found.")
        return
    
    # Calculate total potential savings
    total_potential_size = sum(
        get_directory_size(path) if path.is_dir() else path.stat().st_size 
        for _, path, _ in targets if path.exists()
    )
    
    print(f"📊 Found {len(targets)} cleanup targets")
    print(f"💾 Potential space savings: {format_size(total_potential_size)}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be deleted")
    
    # Define cleanup categories and their order
    cleanup_order = [
        ("python_cache", "Python cache files", True),
        ("jupyter_cache", "Jupyter notebook checkpoints", True),
        ("extracted_data", "Extracted Penn Treebank data", True),
        ("tensorboard_logs", "TensorBoard logs", True),
        ("temp_files", "Temporary files", True),
        ("log_files", "Large log files", True),
        ("model_checkpoints", "Model checkpoints", not args.keep_models and args.aggressive),
    ]
    
    total_files_removed = 0
    total_size_freed = 0
    
    # Process each category
    for category, display_name, should_clean in cleanup_order:
        if not should_clean:
            category_targets = [t for t in targets if t[0] == category]
            if category_targets:
                category_size = sum(
                    get_directory_size(path) if path.is_dir() else path.stat().st_size 
                    for _, path, _ in category_targets if path.exists()
                )
                print(f"\n⏭️  Skipping {display_name} ({format_size(category_size)})")
            continue
            
        files_removed, size_freed = cleanup_category(
            targets, category, args.dry_run, args.interactive
        )
        total_files_removed += files_removed
        total_size_freed += size_freed
    
    # Summary
    print("\n" + "=" * 50)
    if args.dry_run:
        print("🔍 DRY RUN COMPLETE")
        print(f"📁 Would remove: {total_files_removed} items")
        print(f"💾 Would free: {format_size(total_size_freed)}")
        print("\nRun without --dry-run to perform actual cleanup")
    else:
        print("✨ CLEANUP COMPLETE")
        print(f"📁 Removed: {total_files_removed} items")
        print(f"💾 Freed: {format_size(total_size_freed)}")
        
        if total_size_freed > 0:
            print(f"\n🚀 Project is now {format_size(total_size_freed)} smaller!")
            print("   Ready for efficient Git commits and Google Drive uploads")
    
    # Show what remains
    print(f"\n📋 Essential files retained:")
    essential_files = [
        "src/ (source code)",
        "config/ (configuration files)", 
        "scripts/ (utility scripts)",
        "requirements.txt (dependencies)",
        "README.md (documentation)",
        "data/ptb/*.tar.zst (compressed data)",
        "data/ptb/*.txt (processed text files)",
    ]
    
    for item in essential_files:
        print(f"   ✅ {item}")
    
    if args.keep_models or not args.aggressive:
        print(f"   ✅ checkpoints/ (model checkpoints - kept)")


if __name__ == "__main__":
    main()
