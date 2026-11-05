"""First-class workflow for standard PINO training that emits a registered model node."""

from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain, calcfunction

from aiida_pino.calculations.train import PinoTrainCalculation
from aiida_pino.data import PinoDatasetCollectionData, PinoModelData
from aiida_pino.utils import (
    apply_dataset_layout_to_training_profile,
    build_derivation_profile_from_training_profile,
    demote_for_code,
    reconcile_crop_env,
    report_materialisation_cost,
    build_model_metadata,
    canonical_base_layout_profile,
    dataset_layout_profile_from_metadata,
    find_reusable_derived_dataset,
    native_source_incompatibility_reason,
    normalize_dataset_collection_node,
    resolve_base_dataset,
    validate_derivation_profile,
    validate_training_profile,
)
from aiida_pino.utils.vram import (
    SMOKE_KEYWORD,
    SMOKE_MAX_STEPS,
    VRAM_CONFIG_DIGEST_EXTRA,
    VRAM_KEY_EXTRA,
    VRAM_PEAK_EXTRA,
    config_capacity_digest,
    code_computer_label,
    lookup_vram_peak_mb,
    vram_footprint_key,
    vram_reserve_mb,
)
from aiida_pino.workflows.derive_dataset import materialize_derived_dataset
from aiida_pino.workflows.labels import InheritsCallerLabel

# Split labels that must never appear in a training set. `duplicate_excluded`
# and friends are matched by suffix so a future `outlier_excluded` is covered.
_FORBIDDEN_TRAIN_SPLITS = ("test",)
_FORBIDDEN_TRAIN_SPLIT_SUFFIX = "excluded"


def split_violation_message(train_uuids, val_uuids, split_by_uuid) -> str | None:
    """Why this train/val pair must not be trained on, or None if it is clean.

    Pure so it is testable without a database: `split_by_uuid` maps member uuid
    -> the simulation's `split` extra (missing/None = unlabeled, always allowed;
    the guard fires only on positive evidence, so legacy datasets with no split
    extras are untouched).

    Exists because nothing on the training path read the split extras: on
    2026-08-11 one unfiltered 229-member dataset was passed as both train and
    val, and four arms trained on their own 29-case test split. The eval-side
    guard caught it 11.6 GPU-hours late; this is the train-side twin.
    """
    train_uuids = list(train_uuids)
    val_uuids = list(val_uuids)
    problems = []
    overlap = set(train_uuids) & set(val_uuids)
    if overlap:
        problems.append(
            f"train and val share {len(overlap)} of {len(train_uuids)} members"
            + (" (identical sets)" if len(overlap) == len(train_uuids) == len(val_uuids) else "")
        )
    bad: dict[str, int] = {}
    for uuid in train_uuids:
        split = split_by_uuid.get(uuid)
        if split is None:
            continue
        split = str(split)
        if split in _FORBIDDEN_TRAIN_SPLITS or split.endswith(_FORBIDDEN_TRAIN_SPLIT_SUFFIX):
            bad[split] = bad.get(split, 0) + 1
    if bad:
        detail = ", ".join(f"{n} split={s!r}" for s, n in sorted(bad.items()))
        problems.append(f"training set contains {detail} members")
    if not problems:
        return None
    return (
        "; ".join(problems) + ". Pass split-filtered collections (train-only / "
        "val-only members), or allow_split_violation=Bool(True) if training on "
        "these members is deliberate."
    )


def _member_uuids(collection) -> list[str]:
    uuids = collection.base.attributes.get("member_simulation_uuids", None)
    if uuids:
        return [str(u) for u in uuids]
    return [sim.uuid for sim in collection.iter_member_simulations()]


def _splits_by_uuid(uuids) -> dict[str, str]:
    """One query for the `split` extra of every member; unlabeled members are absent."""
    if not uuids:
        return {}
    query = orm.QueryBuilder().append(
        orm.Node, filters={"uuid": {"in": list(uuids)}}, project=["uuid", "extras.split"]
    )
    return {str(uuid): split for uuid, split in query.all() if split is not None}


@calcfunction
def materialize_model_node(
    training_artifacts: orm.FolderData,
    train_summary: orm.Dict,
    training_profile: orm.Dict,
    training_source,
    derivation_profile: orm.Dict,
    model_name: orm.Str | None = None,
    validation_source=None,
) -> PinoModelData:
    """Create a compact registered model from training artifacts plus provenance metadata."""
    metadata = build_model_metadata(
        training_artifacts,
        train_summary,
        training_profile,
        training_source,
        derivation_profile.get_dict(),
        model_name=model_name.value.strip() if model_name is not None and model_name.value.strip() else None,
        validation_source=validation_source,
    )
    return PinoModelData.from_training_artifacts(training_artifacts, metadata=metadata)


#: Emitted by the gpu-seat shim (`train_adapter.py`) the moment a run is handed a card.
#: Its ABSENCE is direct evidence that a run never reached a GPU -- but ONLY when the shim
#: was actually installed, which is what `GPU_SEAT_SKIPPED_MARKER` establishes.
GPU_SEAT_MARKER = "[gpu-seat] acquired gpu"

#: `wrap_with_gpu_seat` returns the command UNWRAPPED on two paths -- a `cuda_device` that
#: is not a single GPU index, and `gpu_spool` not being importable -- and both announce
#: themselves with this prefix. Such a run uses a GPU and emits no acquired-marker, so
#: reading its absence as "never got a card" would be the same confident wrong diagnosis
#: this check exists to prevent, pointed the other way.
GPU_SEAT_SKIPPED_MARKER = "[gpu-seat] WARNING:"


def smoke_ever_got_a_gpu(retrieved) -> bool | None:
    """True/False if the retrieved log settles it, None if it cannot be read.

    None, not False, when nothing is readable: absence of evidence must not be reported
    as evidence of absence, which is the whole failure this exists to prevent.
    """
    if retrieved is None:
        return None
    try:
        names = list(retrieved.base.repository.list_object_names())
    except Exception:  # noqa: BLE001
        return None
    seen_any = False
    for name in ("aiida.out", "_scheduler-stdout.txt", "_scheduler-stderr.txt"):
        if name not in names:
            continue
        try:
            text = retrieved.base.repository.get_object_content(name)
        except Exception:  # noqa: BLE001 - directories and binaries live here too
            continue
        seen_any = True
        if GPU_SEAT_MARKER in text:
            return True
        if GPU_SEAT_SKIPPED_MARKER in text:
            # The shim was never installed, so there is no marker to miss. This run may
            # well have used a GPU; we simply cannot tell from the seat log.
            return None
    return False if seen_any else None


def smoke_failure_detail(retrieved) -> str:
    """The last error line from a failed smoke's retrieved output, or "".

    `exit_message` is generic -- the real 2026-08-22 failure carried
    "Training did not produce train_summary.json" while the actual cause,
    `ValueError: training.rec_results_freq=8000 > max_steps=100`, sat in
    `_scheduler-stderr.txt`. Reporting the exit message alone downgrades a
    confidently wrong diagnosis to a blank one; the error line is what a reader
    needs to tell "never reached the GPU" from "does not fit the GPU".

    Best-effort by construction: this runs on an already-failed run, so it must
    never raise. Note `training_output` is a DIRECTORY in the retrieved folder
    and reading it raises `IsADirectoryError`, which is why every member is
    tried individually rather than aborting on the first failure.
    """
    if retrieved is None:
        return ""
    markers = ("Error:", "error:", "Traceback", "Exception")
    try:
        names = list(retrieved.base.repository.list_object_names())
    except Exception:  # noqa: BLE001
        return ""
    for name in ("_scheduler-stderr.txt", "aiida.out", "_scheduler-stdout.txt"):
        if name not in names:
            continue
        try:
            text = retrieved.base.repository.get_object_content(name)
        except Exception:  # noqa: BLE001 - directories and binaries live here too
            continue
        hits = [
            line.strip()
            for line in str(text).splitlines()
            if any(marker in line for marker in markers)
            and line.strip()
            and not line.strip().startswith("Traceback")
        ]
        # FIRST, not last. Python prints a chained traceback cause-first, and the
        # outer frame here is always a `subprocess.CalledProcessError` wrapping
        # the gpu_spool seat command -- which names the command and hides the
        # reason. Measured on smoke 64216: the informative
        # `ValueError: training.rec_results_freq=8000 > max_steps=100` is at
        # line 30 and the wrapper at line 40, so taking the last hit reported
        # the wrapper and buried the cause.
        if hits:
            return hits[0][:300]
    return ""


#: Derivation-profile keys deliberately EXCLUDED from the submission-path layout
#: comparison, each for a stated reason. A key that is neither compared nor listed
#: here refuses every submission that sets it, which is how `derived_fields`
#: shipped unreachable -- so `tests/test_layout_guard.py` fails for any key that
#: is in neither set, forcing the decision instead of defaulting to strict
#: equality.
LAYOUT_COMPARISON_EXEMPT_KEYS = frozenset({
    # WHAT the staged member contains, not what FRAME it is in. A training
    # profile has no field expressing it, so the implied layout can only ever
    # carry the default and any request would be unsatisfiable.
    "derived_fields",
    # Same defect, found 2026-08-25 the day it shipped: `crop_z_top` describes the
    # stored arrays' EXTENT, and `build_derivation_profile_from_training_profile`
    # reads eight named fields and nothing else, so the comparison was permanently
    # 0 vs 10 and refused every training on the gas-cropped datasets -- the run the
    # user had asked for. Two profiles identical key-for-key, submission rejected.
    "crop_z_top",
    # THIRD instance of the same defect, reported the day it shipped (2026-09-07): a training
    # profile has no field expressing `materialize` either, so a derivation carrying it could
    # never equal the implied layout and every submission using the flag was refused as an
    # incompatible variant before anything staged. It is a STAGING directive -- whether the
    # transforms are baked or applied by the loader -- and says nothing about the FRAME.
    "materialize",
    # Never hit, and would fail identically the day a dataset carries a non-null
    # value: the builder cannot produce it either. Found by asking the question the
    # original guard did not -- "can a training profile PRODUCE this key" rather
    # than "is it compared".
    "time_frames",
})


def layouts_compatible(requested: dict, selected: dict, *, env_stride: str = "") -> bool:
    """Whether the training profile's implied layout matches the selected variant.

    `env.PINO_LAZY_STRIDE` is applied by the LOADER, on top of whatever the
    dataset holds -- it does not require an already-strided dataset. Demanding a
    match refused 77 historical trainings whose config asked for stride
    [2, 2, 1] against a full-resolution [1, 1, 1] dataset, which is not a
    contradiction but the normal way that switch is used. Everything else is
    compared exactly, minus the content keys above.
    """
    comparable = dict(requested or {})
    chosen = dict(selected or {})
    for key in LAYOUT_COMPARISON_EXEMPT_KEYS:
        comparable.pop(key, None)
        chosen.pop(key, None)
    if str(env_stride or "").strip():
        comparable.pop("stride", None)
        chosen.pop("stride", None)
    # FOURTH instance of the key-for-key defect, and the one the `materialize` comment above
    # predicted. A BAKED stride leaves the env empty by design, so the pop above never fires
    # and the comparison is [1,1,1] against [2,2,1) -- three arms refused 303 on 2026-09-09
    # while two with a baked CROP went through, a split entirely explained by which key was
    # baked. A training profile cannot express a baked stride at all:
    # `build_derivation_profile_from_training_profile` reads `env.PINO_LAZY_STRIDE`, so the
    # implied layout has no route to carry one.
    #
    # Gated on `materialize` rather than unconditional. This guard's own history is that
    # demanding a match refused 77 historical trainings, and dropping it wholesale would give
    # up the check that catches an ACCIDENTAL strided dataset. Under `materialize` the bake is
    # the stated intent, which is the whole difference.
    if bool((selected or {}).get("materialize")):
        comparable.pop("stride", None)
        chosen.pop("stride", None)
    # The gas crop is loader-applied too, for exactly the reason the stride is: it rides as
    # `PINO_LAZY_CROP_Z_TOP` and is demoted out of the profile before anything is staged, so
    # a cropped CONFIG against an uncropped DATASET is the normal case, not a substitution.
    # Since the crop became the default on 2026-08-26 that is EVERY training against every
    # stored dataset, and comparing it exactly refused them all with "config=10 dataset=0" --
    # the same false positive that once refused 77 trainings over the stride.
    comparable.pop("crop_z_top", None)
    chosen.pop("crop_z_top", None)
    return comparable == chosen


def smoke_training_profile(profile: dict) -> dict:
    """The training profile for the short VRAM smoke, derived from the real one.

    Shortening a run is not just `max_steps`. `validate_recording_cadence`
    (lpno `config_loading.py`) refuses a run whose recording cadence
    is coarser than its length, and it checks THREE things against `max_steps`:
    `rec_results_freq`, `rec_validation_freq`, and `stop_criterion.freq` being a
    multiple of the validation cadence. Clamping fewer than all three only moves
    the error.

    2026-08-22: the smoke inherited the config's `rec_results_freq=8000` against
    `max_steps=100` and was refused at config validation, so **any config with a
    cadence coarser than 100 steps could never be VRAM-smoked** -- precisely the
    long-running configs the auto-sizing path exists for. It cost ~1 h of an
    EXCLUSIVE card to fail on something knowable at submit time.

    2026-08-22, later the same day: the clamp used to be emitted HERE as three
    `++` overrides, and it never reached the run. pino's
    `remap_plus_args_to_custom` rewrites every `+`/`++` arg into the `custom.`
    namespace, so `++training.rec_results_freq=100` became
    `++custom.training.rec_results_freq=100` -- a dead key, while the real
    cadence kept the config's 2000 and the smoke was refused exactly as before
    (three RIGNO seed replicates lost). The clamp now lives in
    `train_adapter.append_recording_cadence_clamp`, which emits PLAIN overrides
    only for keys the target config actually declares (plus physicsnemo's
    `training.save_network_freq`, which every config has). It belongs there: the
    adapter runs on the machine that runs the training, so it reads that
    machine's config rather than this one's.
    """
    smoke = dict(profile)
    # Never inherit seat priority: a priority smoke parks a `.want` marker that
    # blocks all new spool admissions on the card until it drains -- hours of
    # forced idle for a 100-step measurement.
    smoke.pop("gpu_seat_priority", None)
    smoke["max_steps"] = SMOKE_MAX_STEPS
    smoke["run_name"] = (profile.get("run_name") or "run") + "_vramsmoke"
    smoke["gpu_vram_mb"] = None
    # No cadence overrides here: the adapter emits them, as plain overrides and
    # only for declared keys. An inherited coarse cadence is fine -- the adapter
    # appends its clamp after these, so the clamp wins.
    smoke["hydra_overrides"] = list(profile.get("hydra_overrides") or [])
    return smoke


def warm_targets(sources: dict, *, transport_type: str, runtime_fields=None) -> dict:
    """The LOCAL sources whose runtime cache must be built before a job can read it.

    Pure, so it is testable without a workchain: `{name: collection}` in, the subset that
    `local_build_report` says needs a build out, de-duplicated by uuid (train and val are
    often one collection). Remote targets return nothing -- their caches are staged
    explicitly and presubmit refuses when they are absent.
    """
    if transport_type != "core.local":
        return {}
    from aiida_pino.utils.presubmit import local_build_report

    picked: dict = {}
    seen: set[str] = set()
    for name, node in sources.items():
        if node is None or node.uuid in seen:
            continue
        seen.add(node.uuid)
        if local_build_report(node, runtime_fields=runtime_fields).get("needed"):
            picked[name] = node
    return picked


class TrainModelWorkChain(InheritsCallerLabel, WorkChain):
    """Train a PINO model from native datasets or dataset unions."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("train_dataset", valid_type=orm.Data)
        spec.input("val_dataset", valid_type=orm.Data)
        spec.input("training_profile", valid_type=orm.Dict)
        spec.input("derivation_profile", valid_type=orm.Dict, required=False)
        spec.input(
            "init_model",
            # The calcjob's `init_artifacts` has always taken all three; only
            # this workchain narrowed it to PinoModelData, which is why resuming
            # a killed run from its scratch training_output/ needed an
            # import-model round trip first (2026-08-09).
            valid_type=(PinoModelData, orm.FolderData, orm.RemoteData),
            required=False,
            help="Warm start: a model node, or a FolderData/RemoteData holding "
            "checkpoint files, seeds the trainer's network_dir.",
        )
        spec.input("python_code", valid_type=orm.AbstractCode)
        spec.input(
            "allow_split_violation",
            valid_type=orm.Bool,
            required=False,
            help="Explicitly permit a training set that contains test-split or "
            "excluded members, or that overlaps the validation set. Without this "
            "the workchain refuses: on 2026-08-11 four arms were handed one "
            "unfiltered 229-member dataset as both train and val and trained on "
            "their own 29-case test split for 11.6 GPU-hours before the EVAL "
            "guard caught it.",
        )
        spec.outline(
            cls.validate_inputs,
            cls.resolve_sources,
            cls.warm_runtime_caches,
            cls.harvest_warm,
            cls.prepare_vram_reservation,
            cls.harvest_smoke,
            cls.run_training,
            cls.build_model,
            cls.finalize,
        )
        spec.output("model", valid_type=PinoModelData)
        spec.output("train_summary", valid_type=orm.Dict)
        spec.output("training_artifacts", valid_type=orm.FolderData)
        spec.exit_code(300, "ERROR_INVALID_TRAINING_PROFILE", message="Invalid `training_profile`.")
        spec.exit_code(301, "ERROR_INVALID_DERIVATION_PROFILE", message="Invalid `derivation_profile`.")
        spec.exit_code(302, "ERROR_TRAINING_FAILED", message="PINO training failed.")
        spec.exit_code(
            303,
            "ERROR_INCOMPATIBLE_DATASET_VARIANT",
            message="Selected dataset cannot satisfy the requested native dataset variant.",
        )
        spec.exit_code(304, "ERROR_INVALID_DATASET_INPUT", message="Invalid dataset input type.")
        spec.exit_code(307, "ERROR_RUNTIME_CACHE_WARM_FAILED", message="Building a local runtime cache failed; see the warm job's aiida.out.")
        spec.exit_code(
            305,
            "ERROR_UNDECLARED_LAYOUT_SUBSTITUTION",
            message="Training config implies a different dataset layout than the one selected.",
        )
        spec.exit_code(
            306,
            "ERROR_TRAINING_SPLIT_VIOLATION",
            message="Training set contains test/excluded members or overlaps the "
            "validation set. Pass split-filtered collections, or "
            "`allow_split_violation=Bool(True)` to train anyway.",
        )

    #: Presubmit problem kinds mapped onto this workchain's exit codes. A kind
    #: missing here raises KeyError at the report site -- loud, never a silent pass.
    _PROBLEM_EXIT = {
        "invalid_dataset_input": "ERROR_INVALID_DATASET_INPUT",
        "invalid_training_profile": "ERROR_INVALID_TRAINING_PROFILE",
        "invalid_derivation_profile": "ERROR_INVALID_DERIVATION_PROFILE",
        "split_violation": "ERROR_TRAINING_SPLIT_VIOLATION",
        "incompatible_variant": "ERROR_INCOMPATIBLE_DATASET_VARIANT",
        "undeclared_substitution": "ERROR_UNDECLARED_LAYOUT_SUBSTITUTION",
    }

    def validate_inputs(self):
        """Delegates to `resolve_training_submission` -- the ONE copy of the sequence.

        The full validation/demotion/resolution sequence used to live inline here and
        in `resolve_sources`, which made a presubmit dry-run impossible without
        duplicating it -- and a duplicated builder is how the preheat bug survived a
        week. The shared function stops at the first problem exactly as the inline
        steps did; this wrapper only maps problems onto exit codes and stores the
        results on ctx.
        """
        from aiida_pino.utils.presubmit import resolve_training_submission

        outcome = resolve_training_submission(
            self.inputs.train_dataset,
            self.inputs.val_dataset,
            self.inputs.training_profile.get_dict(),
            self.inputs.python_code,
            self.inputs.derivation_profile.get_dict() if "derivation_profile" in self.inputs else None,
            allow_split_violation=(
                "allow_split_violation" in self.inputs and self.inputs.allow_split_violation.value
            ),
            materialize=True,
            note=self.report,
        )
        self.ctx.presubmit = outcome
        for problem in outcome["problems"]:
            self.report(problem["message"])
            return getattr(self.exit_codes, self._PROBLEM_EXIT[problem["kind"]])
        self.ctx.train_dataset = outcome["train_dataset"]
        self.ctx.val_dataset = outcome["val_dataset"]
        self.ctx.training_profile = outcome["training_profile"]
        self.ctx.derivation_profile = outcome["derivation_profile"]

    def resolve_sources(self):
        outcome = self.ctx.presubmit
        train_plan = outcome["plans"].get("train") or {}
        val_plan = outcome["plans"].get("val") or {}
        if train_plan.get("resolved") is None or val_plan.get("resolved") is None:
            # materialize=True resolves everything or raises inside validate_inputs;
            # reaching here without a node is a presubmit bug, not a user error.
            self.report("Internal error: presubmit returned no resolved collection.")
            return self.exit_codes.ERROR_INCOMPATIBLE_DATASET_VARIANT
        self.ctx.resolved_train_source = train_plan["resolved"]
        self.ctx.resolved_val_source = val_plan["resolved"]

    def _substituted(self, asked: PinoDatasetCollectionData, served: PinoDatasetCollectionData):
        """Say so, every time the node trained on is not the node handed in.

        The substitution is deliberate — the layout must follow the model — but
        it was invisible. The whole log of one workchain was one line naming
        the submitted job; nothing said that a moving-frame
        30-frame-averaged 121x100x40 crop, had been built 2 s after submit and
        used instead of the 150x100x40 split the user picked.
        """
        if served.pk != asked.pk:
            self.report(
                f"Dataset substitution: asked for {asked.pk} ({asked.label!r}), "
                f"training on {served.pk} ({served.label!r}) to match the requested layout."
            )
        return served

    @staticmethod
    def _dataset_layout_metadata(source: PinoDatasetCollectionData) -> dict:
        return source.get_dataset_metadata()

    def warm_runtime_caches(self):
        """Build any missing LOCAL runtime cache as a scheduled job, not inside upload.

        `do_upload` runs `prepare_for_submission` synchronously on the daemon worker's
        loop; a derived export there froze that worker for its whole duration (an hour
        behind one 419-member export on 2026-09-11, kills timing out, a restart orphaning
        the pool). As a CalcJob the export is a process the scheduler owns.
        """
        from aiida_pino.calculations.warm_runtime_cache import WarmRuntimeCacheCalculation
        from aiida_pino.calculations.base import runtime_fields_from_training_profile

        computer = getattr(self.inputs.python_code, "computer", None)
        transport_type = getattr(computer, "transport_type", "core.local") if computer else "core.local"
        fields = runtime_fields_from_training_profile(self.ctx.training_profile)
        targets = warm_targets(
            {"train": self.ctx.resolved_train_source, "val": self.ctx.resolved_val_source},
            transport_type=transport_type, runtime_fields=fields,
        )
        futures = {}
        for name, node in targets.items():
            self.report(f"warming the local runtime cache for {name} ds{node.pk} as its own job")
            inputs = {"code": self.inputs.python_code, "dataset": node,
                      "metadata": {"label": f"warm runtime cache ds{node.pk}"}}
            if fields:
                inputs["runtime_fields"] = orm.List(list=list(fields))
            futures[f"warm_{name}"] = self.submit(WarmRuntimeCacheCalculation, **inputs)
        if futures:
            return ToContext(**futures)

    def harvest_warm(self):
        from aiida_pino.calculations.warm_runtime_cache import WarmRuntimeCacheCalculation

        for key in ("warm_train", "warm_val"):
            job = self.ctx.get(key)
            if job is None:
                continue
            names = job.outputs.retrieved.list_object_names() if "retrieved" in job.outputs else []
            if not job.is_finished_ok or WarmRuntimeCacheCalculation.RESULT_NAME not in names:
                self.report(f"runtime-cache warm job {job.pk} did not publish a cache; see its aiida.out")
                return self.exit_codes.ERROR_RUNTIME_CACHE_WARM_FAILED

    def prepare_vram_reservation(self):
        self.ctx.train_metadata = self._dataset_layout_metadata(self.ctx.resolved_train_source)
        val_metadata = self._dataset_layout_metadata(self.ctx.resolved_val_source)
        if dataset_layout_profile_from_metadata(self.ctx.train_metadata) != dataset_layout_profile_from_metadata(
            val_metadata
        ):
            self.report("Training and validation datasets do not resolve to the same runtime layout.")
            return self.exit_codes.ERROR_INCOMPATIBLE_DATASET_VARIANT
        # The crop lives in ONE place, and which place is decided by what the dataset
        # resolved to -- a fact only available here, after resolution. A plain training gets
        # an uncropped collection and needs `PINO_LAZY_CROP_Z_TOP`; a training that sets
        # `PINO_LAZY_STRIDE` resolves to a collection that already bakes the crop and must
        # not carry it. Reconciled rather than refused: the two agree, so the geometry is
        # unambiguous, and the refusal used to arrive one submission too late.
        _env, _note = reconcile_crop_env(
            (self.ctx.training_profile or {}).get("env"), self.ctx.train_metadata
        )
        if _note:
            _tp = dict(self.ctx.training_profile or {})
            _tp["env"] = _env
            self.ctx.training_profile = _tp
            self.report(_note)
        self.ctx.resolved_training_profile = apply_dataset_layout_to_training_profile(
            self.ctx.training_profile,
            self.ctx.train_metadata,
        )
        profile = self.ctx.resolved_training_profile
        # A warm start inherits its donor's ARCHITECTURE. physicsnemo loads the donor
        # checkpoint under a bare `except`, so a state-dict mismatch logs one line and then
        # trains from random init to completion behind a normal-looking curve -- 6000 steps
        # lost that way on 2026-08-23, triggered by a default that moved between the
        # donor's run and this one, with no mistake by either submitter.
        #
        # Applied HERE, before the VRAM footprint key is computed below, because the
        # architecture is exactly what sets the footprint: inheriting after the key would
        # size a depth-18 run from a depth-1 peak.
        if "init_model" in self.inputs:
            from aiida_pino.utils.warm_start import architecture_overrides_from_donor

            additions, conflicts = architecture_overrides_from_donor(
                self.inputs.init_model, profile.get("hydra_overrides") or []
            )
            if additions:
                profile["hydra_overrides"] = list(profile.get("hydra_overrides") or []) + additions
                self.report(
                    f"warm start: inherited {len(additions)} architecture key(s) from donor "
                    f"{self.inputs.init_model.pk}: {', '.join(additions)}"
                )
            for conflict in conflicts:
                # An explicit choice wins, but never silently: matching the donor is what
                # makes a warm start load at all, so deviating is worth saying out loud.
                self.report(f"warm start: KEEPING your explicit {conflict}")
            if not additions and not conflicts:
                self.report(
                    f"warm start: donor {self.inputs.init_model.pk} archived no `.hydra/config.yaml`, "
                    "so its architecture could not be inherited -- pin `arch.*` by hand if the "
                    "packaged defaults have moved since it trained"
                )
        key = vram_footprint_key(
            profile,
            self.ctx.train_metadata,
            computer_label=code_computer_label(self.inputs.python_code),
        )
        # NOT self.ctx.vram_key yet: setting it before the manual-reservation
        # return would enrol a hand-set booking in the estimate database, which
        # is the one thing a manual reservation must never do (it is a human's
        # number, not a measurement).

        # A booking measured under a DIFFERENT footprint is worse than none.
        # "Clone a profile that finished" is the reliable way to submit, but a
        # profile is not architecture-neutral: the RIGNO varspeed profile paired
        # PINO_LAZY_STRIDE=2,2,1 with host_ram_mb=32768, and those figures were
        # only safe BECAUSE of the stride. Cloned for a full-resolution GAOT run
        # the stride was correctly dropped, every cached tensor grew 4x, and
        # gaot_varspeed_multitau_v2 (31328) was OOM-killed at its first
        # validation with the box at 85 GB. gpu_spool admits on the DECLARED
        # figure, so an under-declaration is not a tight budget but an invisible
        # one -- the job is admitted and then exceeds what it claimed,
        # endangering co-tenants.
        #
        # `vram_key` proves the mismatch: it is written beside every automated
        # booking below, so a clone carries it. A profile with a booking and NO
        # key was hand-set, and is still honoured -- this refuses to inherit,
        # never to obey.
        inherited_key = profile.get("vram_key")
        if profile.get("gpu_vram_mb") and inherited_key and inherited_key != key:
            self.report(
                f"Discarding inherited memory booking: it was measured for footprint "
                f"{inherited_key}, this run is {key} (architecture, stride, batch or "
                f"dataset layout differs). Re-measuring with a smoke rather than "
                f"trusting {profile.get('gpu_vram_mb')} MiB / host_ram_mb="
                f"{profile.get('host_ram_mb')} from a different run."
            )
            profile["gpu_vram_mb"] = None
            profile["host_ram_mb"] = None
            profile["vram_key"] = None

        if profile.get("gpu_vram_mb") or profile.get("gpu_exclusive"):
            # Manual reservation or deliberate whole-card booking: no automation.
            return
        self.ctx.vram_key = key
        peak = lookup_vram_peak_mb(key, profile.get("config_name"))
        if peak:
            profile["gpu_vram_mb"] = vram_reserve_mb(peak)
            # Stamped beside the booking so the next clone can prove whether it
            # still applies.
            profile["vram_key"] = key
            self.report(
                f"auto VRAM: reserving {profile['gpu_vram_mb']} MiB from measured peak {peak} MiB (footprint {key})"
            )
            return
        # No recorded peak for this footprint: measure one with a short smoke.
        # gpu_vram_mb stays None so the smoke books the card exclusively — an
        # uncontaminated measurement and a fast does-not-fit verdict.
        smoke_profile = smoke_training_profile(profile)
        # No init_artifacts: a warm start does not change the memory footprint.
        future = self.submit(
            PinoTrainCalculation,
            code=self.inputs.python_code,
            train_dataset=self.ctx.resolved_train_source,
            val_dataset=self.ctx.resolved_val_source,
            training_profile=orm.Dict(dict=smoke_profile),
        )
        future.base.extras.set("keywords", [SMOKE_KEYWORD])
        future.base.extras.set(VRAM_KEY_EXTRA, key)
        # Name the ASK, not just the action. "exclusive-card" is easy to read as a detail
        # of how the smoke runs; it is in fact a demand for a WHOLE GPU, which is a very
        # different request from the run's own footprint and does not place at all on a
        # saturated box. Seven arms queued into that state before
        # anyone noticed, because nothing said so at submit time (2026-08-23).
        self.report(
            f"no recorded VRAM peak for footprint {key}: submitted {SMOKE_MAX_STEPS}-step "
            f"smoke {future.pk} to measure one. It requests an EXCLUSIVE WHOLE CARD, so it "
            "will not start until one is free -- on a busy box set gpu_vram_mb from a "
            "measured relative instead of waiting for this."
        )
        return ToContext(smoke_job=future)

    def harvest_smoke(self):
        if "smoke_job" not in self.ctx:
            return
        smoke = self.ctx.smoke_job
        if not smoke.is_finished_ok:
            # Report what the child ACTUALLY said. The old message guessed
            # "usually means the configuration does not fit this GPU at all",
            # which on 2026-08-22 sent a diagnosis hunting a memory problem for a
            # run that never reached the GPU -- it was refused at config
            # validation. A guess stated confidently is worse than no diagnosis,
            # and this one is expensive: the smoke holds an exclusive card.
            detail = smoke_failure_detail(getattr(getattr(smoke, "outputs", None), "retrieved", None))
            if not detail:
                detail = str(getattr(smoke, "exit_message", "") or "").strip()
            status = getattr(smoke, "exit_status", None)
            said = f": {detail}" if detail else ""
            code = f" (exit {status})" if status is not None else ""
            # THIRD reading, and it dominates the other two on a busy box: the smoke
            # never got a card at all. Omitting `gpu_vram_mb` makes the smoke ask for the
            # WHOLE card, and on a saturated box that request never places -- so the run
            # is killed or times out WAITING, with no config error and no CUDA error.
            # Neither of the readings below then applies, and a reader picks the second
            # because "the VRAM smoke failed" primes "does not fit". That is not
            # hypothetical: it put "RIGNO steady at 32k does not fit the primary card"
            # into a written note, while another model was training 32000/32000 steps of
            # the same thing inside 13310 MiB (2026-08-23, seven arms). Checked FIRST, and from the gpu-seat marker rather than by
            # inference, so it is evidence and not another guess.
            got_gpu = smoke_ever_got_a_gpu(getattr(getattr(smoke, "outputs", None), "retrieved", None))
            if got_gpu is False:
                self.report(
                    f"VRAM smoke {smoke.pk} failed{code}{said}, and it NEVER ACQUIRED A GPU "
                    f"SEAT (no {GPU_SEAT_MARKER!r} in its log). It asked for an exclusive "
                    "card, the box had none free, and it was killed or timed out WAITING. "
                    "The footprint is still unknown and this says NOTHING about whether the "
                    "configuration fits: re-run when a card is free, or set gpu_vram_mb from "
                    "a measured relative."
                )
                return self.exit_codes.ERROR_TRAINING_FAILED
            unknown = (
                " (whether it ever reached a GPU could not be determined from the retrieved "
                "log, so weigh both readings)"
                if got_gpu is None
                else ""
            )
            self.report(
                f"VRAM smoke {smoke.pk} failed{code}{said}. It DID acquire a GPU seat, so: "
                "if it names a configuration or validation problem the config never reached "
                "the GPU and the memory footprint is still unknown; if it names an allocation "
                f"or CUDA error the {SMOKE_MAX_STEPS}-step run genuinely does not fit this "
                f"card.{unknown}"
                if got_gpu
                else
                f"VRAM smoke {smoke.pk} failed{code}{said}. If it names a configuration "
                "or validation problem the config never reached the GPU and the memory "
                "footprint is still unknown; if it names an allocation or CUDA error the "
                f"{SMOKE_MAX_STEPS}-step run genuinely does not fit this card.{unknown}"
            )
            return self.exit_codes.ERROR_TRAINING_FAILED
        peak = smoke.outputs.train_summary.get_dict().get("gpu_vram_peak_mb")
        if isinstance(peak, int) and not isinstance(peak, bool) and peak > 0:
            smoke.base.extras.set(VRAM_PEAK_EXTRA, peak)
            # What the config chain looked like when this peak was measured. The
            # footprint key cannot see inside the named config, so without this a later
            # capacity edit reuses this number to size a bigger model.
            _digest = config_capacity_digest(self.ctx.resolved_training_profile.get("config_name"))
            if _digest:
                smoke.base.extras.set(VRAM_CONFIG_DIGEST_EXTRA, _digest)
            self.ctx.resolved_training_profile["gpu_vram_mb"] = vram_reserve_mb(peak)
            self.report(
                f"auto VRAM: reserving {self.ctx.resolved_training_profile['gpu_vram_mb']} MiB "
                f"from smoke-measured peak {peak} MiB"
            )
        else:
            # Leave gpu_vram_mb unset so the real run books the card
            # exclusively — the safe direction.
            self.report(f"VRAM smoke {smoke.pk} measured no peak (nvidia-smi unavailable); booking the card exclusively")

    def run_training(self):
        if "train_job" in self.ctx:
            # Belt and braces for checkpoint replays: plumpy persists the
            # outline position as a bare index, so a daemon restart across the
            # 5->7 step outline change can land a resumed chain here after its
            # training already ran. Never resubmit a second full training job.
            return
        calc_inputs: dict = {
            "code": self.inputs.python_code,
            "train_dataset": self.ctx.resolved_train_source,
            "val_dataset": self.ctx.resolved_val_source,
            "training_profile": orm.Dict(dict=self.ctx.resolved_training_profile),
        }
        if "init_model" in self.inputs:
            calc_inputs["init_artifacts"] = self.inputs.init_model
        future = self.submit(PinoTrainCalculation, **calc_inputs)
        if "vram_key" in self.ctx:
            future.base.extras.set(VRAM_KEY_EXTRA, self.ctx.vram_key)
        self.report(f"Submitted PINO training job: {future.pk}")
        return ToContext(train_job=future)

    def build_model(self):
        if not self.ctx.train_job.is_finished_ok:
            return self.exit_codes.ERROR_TRAINING_FAILED
        # Record the real run's measured peak: this is what makes the estimate
        # database grow with every run.
        peak = self.ctx.train_job.outputs.train_summary.get_dict().get("gpu_vram_peak_mb")
        if isinstance(peak, int) and not isinstance(peak, bool) and peak > 0 and getattr(self.ctx, "vram_key", None):
            extras = self.ctx.train_job.base.extras
            extras.set(VRAM_PEAK_EXTRA, peak)
            if extras.get(VRAM_KEY_EXTRA, None) is None:
                extras.set(VRAM_KEY_EXTRA, self.ctx.vram_key)
            _digest = config_capacity_digest(
                (getattr(self.ctx, "resolved_training_profile", None) or {}).get("config_name")
            )
            if _digest:
                extras.set(VRAM_CONFIG_DIGEST_EXTRA, _digest)
            # An overrun is silent otherwise: gpu_spool admits on the booked
            # figure, so a run that exceeded it was invisibly endangering
            # co-tenants the whole time (one run: booked 15761, resident
            # 18150). The recorded peak self-corrects the NEXT booking of this
            # footprint; this line is so the overrun is seen, not just healed.
            booked = self.ctx.resolved_training_profile.get("gpu_vram_mb")
            if booked and peak > int(booked):
                self.report(
                    f"VRAM overrun: measured peak {peak} MiB exceeded the {booked} MiB "
                    f"booking (footprint {self.ctx.vram_key}). Admission control saw the "
                    f"smaller number for this whole run; the next booking of this "
                    f"footprint will size from {peak} MiB."
                )
        self.ctx.model = materialize_model_node(
            self.ctx.train_job.outputs.training_artifacts,
            self.ctx.train_job.outputs.train_summary,
            orm.Dict(dict=self.ctx.resolved_training_profile),
            self.ctx.resolved_train_source,
            orm.Dict(dict=self.ctx.derivation_profile),
            # The validation split decides which checkpoint is kept, and it was
            # on no model node in any form -- only in the workchain inputs and
            # in an archived `custom.val_data_dir=.../runtime/<uuid>` override.
            validation_source=self.ctx.resolved_val_source,
        )
        # "Where did this run" in one hop. `queryable_model_extras` cannot do
        # this — it is a pure function of metadata and the pointer needs the
        # graph — so without stamping it here only backfilled models had it:
        # measured, the five newest models carried 2 of 5 new extras while the
        # backfilled ones carried all of them.
        job = self.ctx.train_job
        self.ctx.model.base.extras.set("calcjob_pk", int(job.pk))
        self.ctx.model.base.extras.set("calcjob_uuid", str(job.uuid))
        workdir = job.base.attributes.get("remote_workdir", None)
        if workdir:
            self.ctx.model.base.extras.set("calcjob_workdir", str(workdir))
        # Cost and graph size the parser already stamped on the calcjob.
        for key in (
            "graph_nodes", "graph_encoder_edges", "graph_decoder_edges",
            "graph_grid", "graph_latent", "graph_latent_tokens",
            "ms_per_iter", "runtime_wall_seconds",
        ):
            value = job.base.extras.get(key, None)
            if value is not None and self.ctx.model.base.extras.get(key, None) is None:
                self.ctx.model.base.extras.set(
                    "wall_seconds" if key == "runtime_wall_seconds" else key, value
                )

    def finalize(self):
        self.out("model", self.ctx.model)
        self.out("train_summary", self.ctx.train_job.outputs.train_summary)
        self.out("training_artifacts", self.ctx.train_job.outputs.training_artifacts)
