"""A crashed builder must not leave its partial `.build-` directory behind.

2026-08-25, found by 067cdc63 while retracting a different claim. Cleanup lived only in the
`except OSError` around `build.replace(final)`, so a builder that died INSIDE `exporter` --
ENOSPC mid-export, OOM, a killed worker -- left its partial directory on disk. That is how
159 GB sat on an already-full disk after the crash: the export reached member 0061 of 96 and
died, and nothing removed what it had written.

Same reasoning as the flock docstring, which already spells out that the holder can die.
Cleanup cannot live only on the happy path on a box where workers get killed.
"""

from __future__ import annotations

from pathlib import Path



def test_both_build_sites_have_the_finally():
    """One site fixed and the other left leaking is the recurring trap in this file."""
    import aiida_pino

    src = (Path(aiida_pino.__file__).parent / "calculations" / "base.py").read_text()
    assert src.count("finally:\n                shutil.rmtree(build, ignore_errors=True)") == 1
    assert src.count("finally:\n            shutil.rmtree(build, ignore_errors=True)") == 1
