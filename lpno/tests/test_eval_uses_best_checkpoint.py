"""The archived best checkpoint must be reachable, and its absence announced.

`best_checkpoint.py` writes `u_network.best.pth` and `best_checkpoint.json` every
run, precisely because physicsnemo's `u_network.0.pth` holds the LAST weights and a
run that diverges ends with the wreckage. Reported 2026-08-25: nothing ever read
either file. No eval path loaded the best weights and no evaluation_profile key
selected them, so the safety net was write-only.

Live cost: model 68163 (`gaot_preheat107_cond_32k_s43`) records mae_T 0.012049
at step 26000 = 36.1 K in its own best_checkpoint.json, and scored 558 K
whole-domain / 1203 K beam box / fl IoU 0.062 from the last checkpoint -- exit 0,
populated metrics.json, nothing flagged. The two files are different tensors.

Scope measured by the reporter across seven preheat arms: three have best step <
max_steps, only 68163 is degraded, so "best < max" is necessary but not sufficient.
The discriminating tell is an eval far worse than the model's own recorded best --
hence the warning carries the recorded number.
"""
import json

import pytest

from lpno.models.ns_energy_pipeline import _assert_eval_checkpoint_loadable


class _Model:
    checkpoint_filename = "u_network.0.pth"

    def __init__(self):
        self.loaded_from = None

    def load(self, ckpt_dir, map_location=None):
        self.loaded_from = self.checkpoint_filename


def _run_dir(tmp_path, *, best=True, meta=True):
    (tmp_path / "u_network.0.pth").write_bytes(b"last")
    if best:
        (tmp_path / "u_network.best.pth").write_bytes(b"best")
    if meta:
        (tmp_path / "best_checkpoint.json").write_text(json.dumps(
            {"step": 26000, "metric": "mae_T", "value": 0.012049, "weights": "u_network.best.pth"}))
    return tmp_path


def test_opt_in_loads_the_best_checkpoint(tmp_path):
    m = _Model()
    _assert_eval_checkpoint_loadable(m, _run_dir(tmp_path), use_best_checkpoint=True)
    assert m.loaded_from == "u_network.best.pth"
    # the model's own filename must be restored, not left mutated
    assert m.checkpoint_filename == "u_network.0.pth"


def test_default_still_loads_the_last_checkpoint(tmp_path):
    """Default must not silently change historical numbers."""
    m = _Model()
    _assert_eval_checkpoint_loadable(m, _run_dir(tmp_path))
    assert m.loaded_from == "u_network.0.pth"


def test_default_warns_that_a_better_checkpoint_was_archived(tmp_path, capsys):
    """Silence is the actual defect: 68163 scored 558 K against a recorded 36.1 K."""
    _assert_eval_checkpoint_loadable(_Model(), _run_dir(tmp_path))
    out = capsys.readouterr().out
    assert "u_network.best.pth" in out
    assert "26000" in out, "name the step, so best<max is visible"
    assert "0.012049" in out, "name the recorded metric -- the discriminating tell"
    assert "use_best_checkpoint" in out, "name the switch that changes it"


def test_no_best_checkpoint_is_silent(tmp_path, capsys):
    _assert_eval_checkpoint_loadable(_Model(), _run_dir(tmp_path, best=False, meta=False))
    assert capsys.readouterr().out == ""


def test_opt_in_without_a_best_file_falls_back_and_says_so(tmp_path, capsys):
    """Asking for best when the run archived none must not crash the eval."""
    m = _Model()
    _assert_eval_checkpoint_loadable(
        m, _run_dir(tmp_path, best=False, meta=False), use_best_checkpoint=True)
    assert m.loaded_from == "u_network.0.pth"
    assert "u_network.best.pth" in capsys.readouterr().out


def test_missing_checkpoint_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _assert_eval_checkpoint_loadable(_Model(), tmp_path)


def test_the_warning_names_an_override_form_that_actually_works(tmp_path, capsys):
    """The first version of this message told the reader to do something refused.

    Reported 2026-08-25: there is no `use_best_checkpoint` evaluation-profile key,
    and aiida-pino refuses unknown keys at SUBMIT time -- so following the original
    advice produced a refusal, not a no-op. The working form is
    `hydra_overrides=["++custom.use_best_checkpoint=true"]`; `++` because the key
    is declared in no config, so a plain `custom.…=true` fails struct validation.

    A diagnostic whose stated remedy does not work is the same defect class this
    guard exists to catch, so pin the remedy.
    """
    _assert_eval_checkpoint_loadable(_Model(), _run_dir(tmp_path))
    out = capsys.readouterr().out
    assert "++custom.use_best_checkpoint=true" in out, "name the form that works"
    assert "hydra_overrides" in out, "name where it goes"


def test_the_illustrative_pk_is_marked_as_an_illustration(tmp_path, capsys):
    """The example pk prints verbatim for every model, so it must say it is one.

    Reported 2026-08-25: someone scoring model 12345 saw "model 68163" in their own
    log with nothing marking it as an example, which reads as a statement about the
    model in hand.
    """
    _assert_eval_checkpoint_loadable(_Model(), _run_dir(tmp_path))
    out = capsys.readouterr().out
    assert "68163" in out, "the worked example is worth keeping"
    lowered = out.lower()
    assert "for example" in lowered and "different model" in lowered, (
        "must not read as a claim about the model being scored")
