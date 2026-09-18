"""Removed text-overlay planning flags stay unavailable to the CLI."""

import pytest

from videofactory import cli


@pytest.mark.parametrize("flag", [
    ["--plan-emphasis"],
    ["--emphasis-selector", "rule"],
    ["--emphasis-model", "test-model"],
    ["--emphasis-timeout", "300"],
    ["--emphasis-max-count", "3"],
    ["--emphasis-min-gap-seconds", "5"],
])
def test_removed_emphasis_options_are_rejected(flag, capsys):
    with pytest.raises(SystemExit) as error:
        cli.parser().parse_args(["--project", "demo", *flag])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
