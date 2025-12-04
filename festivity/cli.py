"""
Main CLI entry point for festivity command.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for festivity CLI."""
    parser = argparse.ArgumentParser(
        prog='festivity',
        description='Festivity Map - Map festive decorations using DINOv3 features',
        epilog='Use "festivity <command> --help" for help on specific commands'
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Import command modules (lazy import to avoid circular dependencies)
    from festivity.commands import (
        init, train, add_images, extract_gps, score, map_cmd, trajectory_stats, view
    )
    
    # Register each command
    init.register_command(subparsers)
    train.register_command(subparsers)
    add_images.register_command(subparsers)
    extract_gps.register_command(subparsers)
    score.register_command(subparsers)
    map_cmd.register_command(subparsers)
    trajectory_stats.register_command(subparsers)
    view.register_command(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, print help
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Execute the command
    args.func(args)


if __name__ == '__main__':
    main()
