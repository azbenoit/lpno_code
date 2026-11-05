"""Automatic GPU VRAM reservations from measured peaks.

Reservations (`training_profile.gpu_vram_mb`) were human guesses; a wrong
guess either wastes a card or OOMs co-tenants (the 2026-07-25 triple OOM).
Protocol (since 2026-07-27):

- every training run records its measured peak (`gpu_vram_peak_mb` extra on
  the calcjob, sampled by the train adapter via nvidia-smi);
- runs are keyed by the memory-relevant subset of their profile plus the
  dataset layout (`vram_key` extra), so the estimate database grows for free;
- a submission without a manual reservation looks the key up and reserves
  `peak + max(1 GiB, 10%)`;
- with no recorded peak, the workchain first runs a short exclusive smoke
  (`smoke` keyword, hidden from the Runs list) to measure one.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

VRAM_KEY_EXTRA = "vram_key"
VRAM_PEAK_EXTRA = "gpu_vram_peak_mb"
SMOKE_KEYWORD = "smoke"
SMOKE_MAX_STEPS = 100
# How many most-recent recorded peaks to consider; max of them is the estimate.
LOOKUP_RECENT = 5

# Profile keys that change the GPU memory footprint. Deliberately excludes
# run_name/wandb/learning-rate/scheduler knobs: they never change allocation.
MEMORY_RELEVANT_PROFILE_KEYS = (
    "config_name",
    "batch_size_grid",
    "batch_size_validation",
    "inference_batch_size",
    "include_time",
    "time_as_batch",
    "time_as_dim",
    "use_2d",
    "time_avg_window",
    "moving_ref_frame",
    "moving_frame_steps",
    "pde_loss",
    "vof_pde_loss",
    "diff_loss",
    "fast_training",
    "sharp_alpha",
    "alpha_masking",
    "autoregressive",
)
# Env switches that change the grid the model actually sees (a missing stride
# was the real cause of the 2026-07-25 OOMs) or the allocator behaviour.
MEMORY_RELEVANT_ENV_KEYS = ("PINO_LAZY_STRIDE", "PINO_LAZY_CACHE", "PYTORCH_CUDA_ALLOC_CONF")


VRAM_CONFIG_DIGEST_EXTRA = "vram_config_digest"


def config_capacity_digest(config_name) -> str | None:
    """12-hex digest of the packaged config CHAIN's file contents, or None if unreadable.

    `vram_footprint_key` covers `config_name`, `hydra_overrides`, the memory-relevant env
    and the dataset signature -- but never the CONTENTS of the named config. Raising
    `arch.rigno.processor_steps` 8 -> 18 and `num_graph_variants` 1 -> 8 INSIDE
    `config_flow3D_RIGNO_randompath_onecycle.yaml` therefore left the key unchanged, and
    the lookup returned a depth-8 peak to size a depth-18 job -- reported to the user as
    "auto-sized from measured peak", so nobody re-checks it (2026-08-23).

    The comment on `hydra_overrides` in the key anticipates exactly this -- "a peak
    measured amp-on must never size an amp-off reservation" -- but a config-file edit
    reaches the same capacity knobs while bypassing it. Under-booking is the direction
    that costs a card AND the result: the job is admitted, then OOMs.

    Whole-file contents rather than an `arch.*` subtree, deliberately. A key list of
    "memory-relevant" config paths is the same maintenance trap as the key list this bug
    slipped through; a comment-only edit forcing one extra 100-step smoke is far cheaper
    than one under-booked run.
    """
    if not config_name:
        return None
    try:
        from pathlib import Path

        import yaml

        import lpno

        conf_dir = Path(lpno.__file__).parent / "conf" / "hydra"
    except Exception:
        return None

    digest = hashlib.sha1()
    seen: set[str] = set()
    stack = [str(config_name)]
    read_any = False
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        path = conf_dir / f"{name}.yaml"
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
            cfg = yaml.safe_load(raw.decode("utf-8")) or {}
        except Exception:
            return None
        read_any = True
        digest.update(name.encode("utf-8"))
        digest.update(raw)
        for entry in cfg.get("defaults", []) or []:
            if isinstance(entry, dict):
                for group, choice in entry.items():
                    group_path = conf_dir / str(group) / f"{choice}.yaml"
                    if group_path.is_file():
                        try:
                            digest.update(group_path.read_bytes())
                        except OSError:
                            return None
            elif isinstance(entry, str) and entry != "_self_":
                stack.append(entry)
    return digest.hexdigest()[:12] if read_any else None


def vram_footprint_key(
    training_profile: dict[str, Any],
    dataset_metadata: dict[str, Any] | None,
    *,
    role: str = "train",
    computer_label: str | None = None,
) -> str:
    """Stable key for 'same architecture on the same data layout', per role.

    An eval runs the same architecture on the same data as its training but
    holds no backward pass, so its peak is a fraction of the training's (4.2 GB
    vs 10.2 GB measured here). Sharing one key would size evals from training
    peaks — the over-booking this discriminator exists to stop. ``train`` omits
    the field so every peak recorded before this existed stays reachable.
    """
    env = training_profile.get("env") or {}
    metadata = dataset_metadata or {}
    payload = {
        "profile": {key: training_profile.get(key) for key in MEMORY_RELEVANT_PROFILE_KEYS},
        # hydra_overrides carry memory-relevant switches (amp.enabled, arch/
        # capacity ablations), so distinct override sets must not share a key —
        # a peak measured amp-on must never size an amp-off reservation. Costs
        # one extra smoke per override set.
        "hydra_overrides": sorted(training_profile.get("hydra_overrides") or []),
        "env": {key: env[key] for key in MEMORY_RELEVANT_ENV_KEYS if env.get(key)},
        # compatibility_signature encodes fields x spatial layout; time_steps
        # bound the batch the loaders build in time_as_batch mode.
        "dataset": {
            "compatibility_signature": metadata.get("compatibility_signature"),
            "canonical_layout": metadata.get("canonical_layout"),
            "time_steps_max": metadata.get("time_steps_max"),
        },
    }
    if role != "train":
        payload["role"] = role
    # A peak measured on one GPU does not size a booking on another: TITAN RTX
    # (Turing, sm_75) and RTX 4090 (Ada, sm_89) are different silicon, and
    # pooling them under one key books the wrong number on whichever machine
    # did not set it. `local` is spelled as absence, exactly like `role` above,
    # so every peak recorded before a second machine existed stays reachable and
    # local keys are byte-identical.
    if computer_label and str(computer_label) != "local":
        payload["computer"] = str(computer_label)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def vram_reserve_mb(peak_mb: int) -> int:
    """Reservation from a measured peak: peak + max(1 GiB, 10%)."""
    peak_mb = int(peak_mb)
    return peak_mb + max(1024, (peak_mb + 9) // 10)


#: Peaks below this are treated as noise, not measurements. Crashed or truncated
#: evals recorded 164/858/986 MiB here; sizing from 164 would reserve 1188 MiB
#: and cause exactly the OOM this machinery prevents.
EVAL_PEAK_FLOOR_MB = 1000


#: Percentile of recorded eval peaks used for the cross-footprint fallback.
#: NOT the max: the recorded peaks span 2566-4608 MiB except for one RIGNO
#: unstrided outlier at 14328, and taking the max let that single job set the
#: reservation for every eval. Measured consequence 2026-08-05: an eval booked
#: 15761 MiB (= reserve(14328)) while using 6268, leaving 6767 free on a 22528
#: card and blocking a queued 12500 MiB training for 13 hours. p90 sits at 4580
#: -> a 5604 MiB reservation, above every non-outlier eval ever measured.
EVAL_FALLBACK_PERCENTILE = 90.0


def fallback_eval_peak_mb(
    floor_mb: int = EVAL_PEAK_FLOOR_MB, percentile: float = EVAL_FALLBACK_PERCENTILE
) -> int | None:
    """A high percentile of recorded eval peaks, for a first-of-a-kind footprint.

    `vram_footprint_key` includes `hydra_overrides`, so an eval of a new config
    has no recorded peak and would otherwise inherit its parent training's
    reservation -- an unrelated number, typically 4-5x too large.

    An eval holds no backward pass, so its footprint is dominated by the
    inference batch rather than model capacity, and the recorded peaks cluster
    tightly. A percentile rather than the max keeps one atypical job from setting
    the floor for all of them; callers still clamp with `min(inherited, ...)`, so
    this can only ever LOWER a booking.
    """
    from aiida import orm

    query = orm.QueryBuilder().append(
        orm.CalcJobNode,
        filters={
            "attributes.process_label": "PinoEvalCalculation",
            f"extras.{VRAM_PEAK_EXTRA}": {">": int(floor_mb)},
        },
        project=[f"extras.{VRAM_PEAK_EXTRA}"],
    )
    peaks = sorted(int(row[0]) for row in query.all() if row[0])
    if not peaks:
        return None
    # Nearest-rank percentile: no interpolation, so the value is always one that
    # was actually observed.
    idx = min(len(peaks) - 1, max(0, int(round(percentile / 100.0 * len(peaks))) - 1))
    return peaks[idx]


#: Extra written by the eval adapter holding the job's measured peak resident set.
#: Host RAM is the axis with no `host_ram_peak_mb` on a CalcJob -- that key lives on
#: model nodes, which an eval never produces -- so this is the only eval measurement.
EVAL_RSS_EXTRA = "runtime_max_rss_mb"

#: Below this an eval's recorded RSS is treated as a truncated job, not a measurement,
#: mirroring `EVAL_PEAK_FLOOR_MB` on the VRAM axis.
EVAL_RSS_FLOOR_MB = 512


def lookup_eval_host_ram_mb(key: str) -> int | None:
    """Largest measured resident set among recent evals of this exact footprint.

    The eval host-RAM booking was a flat cap, and a flat cap is wrong in the direction
    that matters. Measured over the 497 eval calcjobs carrying a measured RSS: p50 6918,
    p90 17671, p99 37132, max 47949 MiB. The cap is 16384, so **44 of 335 footprints
    exceed it** -- those evals are admitted against a number a third of what they use,
    which is the same invisible over-run that OOM-kills co-tenants on the training side.
    (It is also why the reported "2848-3130 MiB measured against 12000 forced" is not a
    reason to LOWER the cap: those figures are one footprint, and half of all evals use
    more than 6.9 GB.)

    Keyed rather than pooled because the key predicts this well: across the 34 footprints
    with three or more measurements the within-key max/min spread is 1.00x at the median,
    1.58x at p90, 3.14x at worst. `max` over the window for the reason in
    `lookup_vram_peak_mb` -- within one key the peaks estimate the same number, and
    under-booking is the direction that endangers co-tenants.

    Returns None when nothing is recorded, leaving the flat cap in place.
    """
    from aiida import orm

    query = (
        orm.QueryBuilder()
        .append(
            orm.CalcJobNode,
            filters={
                "attributes.process_label": "PinoEvalCalculation",
                f"extras.{VRAM_KEY_EXTRA}": key,
                f"extras.{EVAL_RSS_EXTRA}": {">": EVAL_RSS_FLOOR_MB},
            },
            project=[f"extras.{EVAL_RSS_EXTRA}"],
        )
        .order_by({orm.CalcJobNode: {"ctime": "desc"}})
        .limit(LOOKUP_RECENT)
    )
    peaks = [int(row[0]) for row in query.all() if row[0]]
    return max(peaks) if peaks else None


def lookup_vram_peak_mb(key: str, config_name: str | None = None) -> int | None:
    """Highest of the most recent recorded peaks for this footprint key.

    `max` here is deliberate and is NOT the defect `fallback_eval_peak_mb`
    fixed with p90. That fallback aggregates across UNRELATED footprints, where
    one atypical job poisoning the pool was real. Within one key the workload
    is the same architecture, stride, batch and dataset layout, so the recent
    peaks are estimates of the same number and the highest is the safe pick —
    a lower percentile would systematically under-book, and under-booking is
    the direction that endangers co-tenants (one run under-booked by 2.4 GB
    even from the max, because peaks grow late in training). The window of
    `LOOKUP_RECENT` ages out any genuinely wrong outlier in a few runs, and
    every finished run records its true peak, so the estimate self-corrects
    upward the run after it is first exceeded.
    """
    from aiida import orm

    query = (
        orm.QueryBuilder()
        .append(
            orm.CalcJobNode,
            filters={f"extras.{VRAM_KEY_EXTRA}": key, f"extras.{VRAM_PEAK_EXTRA}": {">": 0}},
            project=[f"extras.{VRAM_PEAK_EXTRA}", f"extras.{VRAM_CONFIG_DIGEST_EXTRA}"],
        )
        .order_by({orm.CalcJobNode: {"ctime": "desc"}})
        .limit(LOOKUP_RECENT)
    )
    # A peak whose config chain has CHANGED since it was measured does not size this run
    # -- the key cannot see inside the named config, so a capacity edit leaves it
    # identical (see `config_capacity_digest`). A peak with NO recorded digest is
    # accepted: it predates this field, and there is no way to tell a stale one from a
    # fine one, so refusing them all would re-smoke every footprint on the box at once.
    # Everything measured from now on carries the digest and is checked.
    current = config_capacity_digest(config_name) if config_name else None
    peaks = [
        int(peak)
        for peak, digest in query.all()
        if not (current and digest and digest != current)
    ]
    return max(peaks) if peaks else None


def code_computer_label(code) -> str | None:
    """The label of the machine a code runs on, or None when it is this one.

    None for local so the VRAM key stays byte-identical to every peak recorded
    before a second machine existed.
    """
    computer = getattr(code, "computer", None)
    label = getattr(computer, "label", None)
    transport = getattr(computer, "transport_type", "core.local")
    if not label or transport == "core.local":
        return None
    return str(label)
