"""GAOT must be recognised as its own architecture, and stamped as a keyword.

Two bugs, both found 2026-07-30 on the live DB:

1. `console/service.py:_model_architecture` token-matched against a private list --
   rigno, unet, fcn, deeponet -- with no GAOT entry, so every GAOT run displayed as the
   "FNO / PINO" fall-through.
2. Nothing auto-tagged a model node. 86 of 136 `PinoModelData` carried keywords, all
   set explicitly by the self-improve driver; every GAOT model and the imported
   `rigno_ti64ds3` had none, so they could not be filtered by family.
"""
import pytest

from aiida_pino.utils.architecture import (
    DEFAULT_ARCHITECTURE,
    architecture_for_summary,
    architecture_from_names,
    architecture_keyword_for_metadata,
    architecture_keyword_from_names,
    merge_keywords,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("config_flow3D_GAOT_onecycle", "GAOT"),
        ("gaot_sp_skip_20k", "GAOT"),
        ("config_flow3D_RIGNO_randompath", "RIGNO"),
        ("rigno_ti64ds3", "RIGNO"),
        ("unet_head_v2", "U-Net"),
        ("deeponet_x", "DeepONet"),
        ("config_flow3D_FNO", DEFAULT_ARCHITECTURE),
    ],
)
def test_architecture_label(name, expected):
    assert architecture_from_names(name) == expected


def test_gaot_wins_over_an_inherited_rigno_name():
    """The GAOT configs inherit from the RIGNO ones -- `config_flow3D_GAOT_randompath`
    has `defaults: [config_flow3D_RIGNO_randompath_onecycle]`. A name or chain
    mentioning both must resolve to the actual architecture, not the parent."""
    assert architecture_from_names("config_flow3D_GAOT_randompath", "rigno parent") == "GAOT"
    assert architecture_keyword_from_names("gaot_rp", "config_flow3D_RIGNO_x") == "gaot"


def test_summary_reads_the_usual_name_fields():
    assert architecture_for_summary({"config_name": "config_flow3D_GAOT_material"}) == "GAOT"
    assert architecture_for_summary({"run_name": "gaot_cond150_fullres_20k"}) == "GAOT"
    assert architecture_for_summary({}) == DEFAULT_ARCHITECTURE


def test_keyword_matches_the_existing_convention():
    """Self-improve runs already carry 'fno' in their keyword list; the stamped
    architecture keyword uses the same lowercase form."""
    assert architecture_keyword_for_metadata({"run_name": "gaot_sp_20k"}) == "gaot"
    assert architecture_keyword_for_metadata({"run_name": "rigno_x"}) == "rigno"
    assert architecture_keyword_for_metadata({}) == "fno"


def test_merge_keywords_never_drops_caller_tags():
    """The self-improve driver's own run tags must survive the architecture keyword."""
    got = merge_keywords(["ALSI", "d2", "n40", "fno"], "fno")
    assert got == ["ALSI", "d2", "n40", "fno"]
    assert merge_keywords(None, "gaot") == ["gaot"]
    assert merge_keywords(["gaot"], "gaot") == ["gaot"]
    assert merge_keywords(["  ", ""], "gaot") == ["gaot"]


def test_merge_keywords_dedups_case_insensitively():
    """Older demo models are tagged ["FNO", "demo", "straight path"]; a case-sensitive
    union would append a second lowercase "fno" to every one of them. The existing
    tag's case is preserved."""
    assert merge_keywords(["FNO", "demo", "straight path"], "fno") == [
        "FNO", "demo", "straight path"
    ]
    assert merge_keywords(["GAOT"], "gaot") == ["GAOT"]


def test_importing_aiida_pino_data_does_not_cycle():
    """`aiida_pino.utils.__init__` pulls in `model_registry`, which imports
    `aiida_pino.data`. A module-scope `from ..utils.architecture import ...` inside
    `data/pino_model.py` therefore closes a cycle and makes a bare
    `import aiida_pino.data` fail with "partially initialized module" -- which is
    exactly what happened on 2026-07-30. The import must stay deferred.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", "import aiida_pino.data; print('ok')"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ok" in proc.stdout


def test_architecture_import_is_not_at_module_scope_in_pino_model():
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "src" / "aiida_pino" / "data" / "pino_model.py"
    ).read_text()
    module_scope = src.split("class PinoModelData")[0]
    assert "utils.architecture" not in module_scope, (
        "a module-scope import of utils.architecture here re-creates the import cycle"
    )
