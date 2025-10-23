#!/usr/bin/env python3
"""
Enhanced CLI utilities for PhenoHunter.
Provides colored output, progress bars, and formatting helpers.
"""

import sys
from typing import Optional


class Colors:
    """ANSI color codes for terminal output."""
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    # Reset
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY environments)."""
        for attr in dir(cls):
            if not attr.startswith('_') and attr not in ['disable', 'is_enabled']:
                setattr(cls, attr, '')

    @classmethod
    def is_enabled(cls):
        """Check if colors are enabled."""
        return sys.stdout.isatty()


# Disable colors if not in TTY
if not Colors.is_enabled():
    Colors.disable()


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = min(70, len(text) + 4)
    print(f"\n{Colors.BOLD}{Colors.BLUE}{char * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{char * width}{Colors.RESET}\n")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.RED}✗{Colors.RESET} {text}", file=sys.stderr)


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")


def print_section(title: str):
    """Print a section title."""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{title}{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * len(title)}{Colors.RESET}")


def print_table_row(label: str, value: str, width: int = 20):
    """Print a formatted table row."""
    print(f"  {Colors.CYAN}{label:<{width}}{Colors.RESET}: {value}")


def print_progress(current: int, total: int, prefix: str = "Progress"):
    """Print a simple progress indicator."""
    percent = (current / total) * 100 if total > 0 else 0
    bar_length = 40
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r{prefix}: [{bar}] {percent:.1f}% ({current}/{total})", end='', flush=True)
    if current == total:
        print()  # New line when complete


def format_chemical(name: str, value: float, std: Optional[float] = None, unit: str = "%") -> str:
    """Format chemical compound output."""
    if std is not None:
        return f"{Colors.CYAN}{name:<20}{Colors.RESET}: {value:6.2f} ± {std:4.2f} {unit}"
    return f"{Colors.CYAN}{name:<20}{Colors.RESET}: {value:6.2f} {unit}"


def format_metric(name: str, value: float, format_spec: str = ".3f") -> str:
    """Format a metric output."""
    formatted_value = f"{value:{format_spec}}"
    return f"{Colors.CYAN}{name:<20}{Colors.RESET}: {formatted_value}"


def print_banner():
    """Print the PhenoHunter banner."""
    banner = f"""
{Colors.BOLD}{Colors.MAGENTA}╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   {Colors.BRIGHT_CYAN}🧬  C.C.R.O.P - PhenoHunt{Colors.MAGENTA}                                      ║
║   {Colors.BRIGHT_GREEN}Cannabis Computational Research & Optimization Platform{Colors.MAGENTA}      ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{Colors.YELLOW}?{Colors.RESET} {message} {suffix}: ").strip().lower()

    if not response:
        return default
    return response in ['y', 'yes']


class ProgressBar:
    """A simple progress bar context manager."""

    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print()  # New line when done

    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        print_progress(self.current, self.total, self.desc)
