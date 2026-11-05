"""A VRAM smoke that never got a card must not be read as "does not fit".

2026-08-23. `harvest_smoke` offered two readings of a failed smoke -- a
configuration problem (footprint unknown) or an allocation error (does not fit). The case
that actually dominates on a busy box is neither: omitting `gpu_vram_mb` makes the smoke
request the WHOLE card, and on a saturated box that request never places, so the run is
killed or times out WAITING. No config error, no CUDA error.

A reader then picks the second reading, because "the VRAM smoke failed" primes "does not
fit". That is measured, not hypothetical: it put "RIGNO steady at 32k does not fit the
primary card" into a written note, while another model trained 32000/32000 steps of the
same configuration inside 13310 MiB. Seven arms were affected.

The gpu-seat shim already prints a marker the moment a run is handed a card, so this is
answerable from evidence rather than by inference.
"""

from __future__ import annotations

from aiida_pino.workflows.train_model import GPU_SEAT_MARKER, smoke_ever_got_a_gpu


class _Repo:
    def __init__(self, files):
        self._files = files

    def list_object_names(self):
        return list(self._files)

    def get_object_content(self, name):
        value = self._files[name]
        if value is None:  # a directory in the retrieved folder
            raise IsADirectoryError(name)
        return value


class _Retrieved:
    def __init__(self, files):
        self.base = type("B", (), {"repository": _Repo(files)})()


def test_the_marker_is_the_one_the_shim_actually_prints():
    """If these drift the check silently answers False for every run."""
    from pathlib import Path

    import aiida_pino

    adapter = (Path(aiida_pino.__file__).parent / "scripts" / "train_adapter.py").read_text()
    assert GPU_SEAT_MARKER in adapter


def test_a_run_that_acquired_a_seat_is_detected():
    log = f"some setup\n{GPU_SEAT_MARKER} 1\n[step: 1]\n"
    assert smoke_ever_got_a_gpu(_Retrieved({"aiida.out": log})) is True


def test_a_run_that_only_ever_waited_is_detected():
    """The reported case: the last line is the spool still waiting for a whole card."""
    log = (
        "[gpu-spool] waiting for the whole card on any managed GPU + 32768 MiB host RAM\n"
        "(GPU 0: 20000/22528 MiB, GPU 1: 20000/22528 MiB reserved)\n"
    )
    assert smoke_ever_got_a_gpu(_Retrieved({"aiida.out": log})) is False


def test_an_unreadable_log_is_None_not_False():
    """Absence of evidence must not be reported as evidence of absence.

    Returning False here would assert "it never got a GPU" about a run whose log simply
    could not be read -- swapping one confident wrong diagnosis for another.
    """
    assert smoke_ever_got_a_gpu(None) is None
    assert smoke_ever_got_a_gpu(_Retrieved({})) is None
    assert smoke_ever_got_a_gpu(_Retrieved({"training_output": None})) is None


def test_a_directory_member_does_not_hide_a_readable_log():
    """`training_output` is a directory and raises; the real log must still be reached."""
    files = {"training_output": None, "aiida.out": f"{GPU_SEAT_MARKER} 0\n"}
    assert smoke_ever_got_a_gpu(_Retrieved(files)) is True


def test_a_run_whose_shim_was_never_installed_is_unknown_not_false():
    """`wrap_with_gpu_seat` returns the command UNWRAPPED on two paths.

    A `cuda_device` that is not a single GPU index, and `gpu_spool` not being importable:
    both print a `[gpu-seat] WARNING:` and run WITHOUT the shim. Such a run uses a GPU and
    emits no acquired-marker, so reading its absence as "never got a card" would be this
    check inverting into the same confident wrong diagnosis it was written to close.

    Flagged in review as a drift risk in the marker coupling;
    the sharper version is this one, and it is a live code path rather than a future edit.
    """
    from aiida_pino.workflows.train_model import GPU_SEAT_SKIPPED_MARKER

    spool_missing = (
        f"{GPU_SEAT_SKIPPED_MARKER} aiida_pino.gpu_spool is not importable; running "
        "without the GPU seat lock.\n[step: 1]\n"
    )
    assert smoke_ever_got_a_gpu(_Retrieved({"aiida.out": spool_missing})) is None

    bad_device = (
        f"{GPU_SEAT_SKIPPED_MARKER} cuda_device='0,1' is not a single GPU index; "
        "running without the gpu_spool seat lock.\n"
    )
    assert smoke_ever_got_a_gpu(_Retrieved({"aiida.out": bad_device})) is None


def test_both_skip_paths_still_announce_themselves():
    """If a skip path stops printing the prefix, the guard above silently stops working."""
    from pathlib import Path

    import aiida_pino
    from aiida_pino.workflows.train_model import GPU_SEAT_SKIPPED_MARKER

    adapter = (Path(aiida_pino.__file__).parent / "scripts" / "train_adapter.py").read_text()
    body = adapter[adapter.index("def wrap_with_gpu_seat") :]
    body = body[: body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
    unwrapped = body.count("return command")
    announced = body.count(GPU_SEAT_SKIPPED_MARKER)
    assert unwrapped <= announced, (
        f"{unwrapped} unwrapped-return path(s) in wrap_with_gpu_seat but only {announced} "
        f"announce {GPU_SEAT_SKIPPED_MARKER!r}; an unannounced one is read as 'never got a card'"
    )
