"""
SLO (Service Level Objective) checking utilities.

Defines SLO thresholds and provides result checking functions.
"""

from rich.table import Table

from backend.eval.formatting import console, format_check_mark


def create_slo_table(
    slo_checks: list[tuple[str, bool, str, str]],
    title: str = "SLO Summary",
) -> Table:
    """
    Create a standard SLO results table.

    Args:
        slo_checks: List of (name, passed, actual_value, target_value) tuples
        title: Table title

    Returns:
        Formatted Rich Table
    """
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("SLO", style="bold")
    table.add_column("Actual", justify="right")
    table.add_column("Target", justify="right", style="dim")
    table.add_column("Status", justify="center")

    for name, passed, actual, target in slo_checks:
        table.add_row(name, actual, target, format_check_mark(passed))

    return table


def print_slo_result(slo_checks: list[tuple[str, bool, str, str]]) -> bool:
    """
    Print SLO summary table and pass/fail status.

    Args:
        slo_checks: List of (name, passed, actual_value, target_value) tuples

    Returns:
        True if all SLOs passed, False otherwise
    """
    console.print()
    slo_table = create_slo_table(slo_checks)
    console.print(slo_table)

    passed_slos = sum(1 for _, passed, _, _ in slo_checks if passed)
    total_slos = len(slo_checks)
    failed_slo_names = [name for name, passed, _, _ in slo_checks if not passed]

    if failed_slo_names:
        console.print(f"\n[red bold][!] {len(failed_slo_names)} SLO(s) FAILED:[/red bold]")
        for slo_name in failed_slo_names:
            console.print(f"    [red]X[/red] {slo_name}")
        return False
    else:
        console.print(f"\n[green bold][OK] All {total_slos} SLOs passed[/green bold]")
        return True


def get_failed_slos(slo_checks: list[tuple[str, bool, str, str]]) -> list[str]:
    """
    Extract names of failed SLOs from slo_checks list.

    Args:
        slo_checks: List of (name, passed, actual_value, target_value) tuples

    Returns:
        List of SLO names that failed
    """
    return [name for name, passed, _, _ in slo_checks if not passed]


def determine_exit_code(
    all_slos_passed: bool,
    is_regression: bool,
) -> int:
    """
    Determine exit code from SLO and regression status.

    Args:
        all_slos_passed: Whether all SLOs passed
        is_regression: Whether a regression was detected

    Returns:
        0 for success, 1 for failure
    """
    if not all_slos_passed or is_regression:
        return 1
    return 0


__all__ = [
    "create_slo_table",
    "print_slo_result",
    "get_failed_slos",
    "determine_exit_code",
]
