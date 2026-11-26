"""A command a release removed answers with one line, not argparse's usage dump.

THE BREAKAGE THIS PREVENTS: upstream WOOF 1.1.7 removed ``woof local-da`` and
answers it from ``woof.removed_commands``.  This fork keeps ``woof local-da``
(its data assimilation work builds on it), so the table must not list it: a
listed command would be answered as removed before argparse ever reached it.
"""

from __future__ import annotations

from woof import cli
from woof.removed_commands import REMOVED_COMMANDS, removed_command


def test_the_fork_keeps_local_da_registered_and_unlisted():
    assert "local-da" not in REMOVED_COMMANDS
    parser = cli.build_parser()
    commands = next(action for action in parser._actions
                    if action.__class__.__name__ == "_SubParsersAction")
    assert "local-da" in commands.choices


def test_a_command_that_exists_is_not_answered_as_removed():
    assert removed_command(["go", "local-da"]) is None
    assert removed_command(["local-da"]) is None
    assert removed_command(["--backend", "cuda"]) is None
    assert removed_command([]) is None
