"""
Calculate trajectory statistics.
"""

import argparse


def register_command(subparsers):
    """Register the trajectory-stats command."""
    parser = subparsers.add_parser(
        'trajectory-stats',
        help='Calculate trajectory statistics',
        description='Calculate duration and distance for mapping trajectory'
    )
    
    parser.add_argument(
        '--workspace',
        type=str,
        help='Workspace root path (or use FESTIVITY_WORKSPACE env var)'
    )
    
    parser.set_defaults(func=execute)


def execute(args):
    """Execute the trajectory-stats command."""
    print("Trajectory-stats command - Coming soon!")
