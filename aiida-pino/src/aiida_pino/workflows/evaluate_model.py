"""First-class workflow for PINO evaluation against a registered model node."""

from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain, calcfunction

from aiida_pino.calculations.eval import PinoEvalCalculation
from aiida_pino.data import PinoDatasetCollectionData, PinoModelData, PinoModelEvaluationData
from aiida_pino.utils.model_registry import _member_uuids, payload_keys
from aiida_pino.utils import (
    build_derivation_profile_from_training_profile,
    domain_extent_mismatch,
    demote_for_code,
    reconcile_crop_env,
    report_materialisation_cost,
    build_evaluation_metadata,
    canonical_base_layout_profile,
    dataset_layout_profile_from_metadata,
    find_reusable_derived_dataset,
    merge_model_training_with_evaluation_profile,
    native_source_incompatibility_reason,
    normalize_dataset_collection_node,
    resolve_base_dataset,
    validate_derivation_profile,
    validate_evaluation_profile,
)
from aiida_pino.utils.vram import (
    VRAM_KEY_EXTRA,
    code_computer_label,
    fallback_eval_peak_mb,
    lookup_eval_host_ram_mb,
    lookup_vram_peak_mb,
    vram_footprint_key,
    vram_reserve_mb,
)

# Host-RAM booking for an evaluation, by JOB KIND rather than inheritance. Evals read
# cases sequentially (one decompressed case + model + activations); measured evals on
# the 150x100x40 grids run comfortably in 8-16 GB. 16384 is deliberately generous --
# the guardrail direction is that a derived number may only LOWER an inherited booking.
# Measured cost of the inheritance it replaces (2026-08-06): a varspeed training
# legitimately declares 49152 MiB for its 57 GB graph-build peak; its eval inherited
# that number while using ~5.6 GB of VRAM, and the booking blocked the paired CONTROL
# training for hours.
EVAL_HOST_RAM_MB = 16384


# Auto-subset budget for evaluations (2026-08-06): an eval projected to
# exceed ~40 minutes runs on every Nth frame instead, using the training pipeline's
# PINO_EVAL_FRAME_STRIDE (which stamps `eval_frame_stride` into the stored metrics, so
# a subset score can never be silently read as a full-split score). Rationale: the
# varspeed test split measured 2.9 h full-split at 1.18 s/frame while a strided read
# answers the same health question in minutes. An explicit PINO_EVAL_FRAME_STRIDE in
# the evaluation_profile env always wins.
EVAL_TIME_BUDGET_S = 2400
# Measured 2026-08-06 on the 150x100x40 grid (600k cells): 301 frames in 5.9 min.
# Smaller grids run faster, so using this constant everywhere only ever makes the
# budget more conservative.
EVAL_SEC_PER_FRAME = 1.18


# Default temporal density for an automatic eval (2026-08-06): ~25
# frames per simulation instead of the stored ~400. The stored density exists for
# training and archival; for a health readout the per-frame metrics are heavily
# autocorrelated within a case, so 25 spread frames carry nearly the same information
# at 1/16th the cost (val split: 8 cases x 25 = 200 frames ~ 4 min). Full frames stay
# one parameter away (`full_frames: true`), and the stride is stamped into the stored
# metrics either way, so a subset score can never silently pass as a full one.
EVAL_FRAMES_PER_CASE = 25


def auto_eval_stride(frames_per_case: int | None,
                     total_frames: int | None,
                     target_per_case: int = EVAL_FRAMES_PER_CASE,
                     budget_s: float = EVAL_TIME_BUDGET_S,
                     sec_per_frame: float = EVAL_SEC_PER_FRAME) -> int:
    """Frame stride for an automatic eval; 1 = full split.

    Primary rule: land near `target_per_case` frames per simulation. Secondary rule:
    never exceed the wall-clock budget even if the per-case rule would (many cases).
    Unknown sizes return 1 on their axis: never subset on a guess about SIZE (a wrong
    stride silently degrades a score), only on a known one.
    """
    per_case_stride = 1
    if frames_per_case and frames_per_case > target_per_case:
        per_case_stride = -(-frames_per_case // target_per_case)  # ceil
    budget_stride = 1
    if total_frames and total_frames > 0:
        max_frames = max(1, int(budget_s / sec_per_frame))
        if total_frames > max_frames:
            budget_stride = -(-total_frames // max_frames)
    return max(per_case_stride, budget_stride)


def served_frames_per_case(derivation_profile, stored_per_case: int | None) -> int | None:
    """Frames per case the model will actually be SHOWN, not the number stored.

    `dataset_frames_per_case` reads the collection's stored frame count, which is
    the count before derivation. A steady derivation time-averages in the beam
    frame and collapses each case to ONE frame, so a stride computed from the
    stored count is applied to a series that no longer has those frames. On the
    calibrated straight-track split that meant a stride of 41 over a 13-frame
    series: one case scored out of thirteen, with nothing in the metrics saying so
    except `fl.iou_support_frames`.

    Returns `stored_per_case` unchanged whenever the derivation does not collapse
    time, so ordinary lab and time-as-batch evals keep subsetting as before.
    """
    d = derivation_profile or {}
    collapses_time = (
        not d.get("time_as_batch")
        and not d.get("time_as_dim")
        and d.get("moving_ref_frame")
        and int(d.get("time_avg_window") or 1) > 1
    )
    return 1 if collapses_time else stored_per_case


def dataset_frames_per_case(node) -> int | None:
    """Best-effort frames-per-simulation from durable collection attributes."""
    try:
        frames = node.base.attributes.all.get("time_steps_max") \
            or node.base.attributes.all.get("time_steps")
        return int(frames) if frames else None
    except Exception:  # noqa: BLE001
        return None


def dataset_total_frames(node) -> int | None:
    """Best-effort frame count for a dataset collection, from durable attributes."""
    try:
        attrs = node.base.attributes.all
        cases = attrs.get("simulation_count")
        frames = attrs.get("time_steps_max") or attrs.get("time_steps")
        if cases and frames:
            return int(cases) * int(frames)
    except Exception:  # noqa: BLE001 - unknown size means stride 1, never a guess
        pass
    return None

from aiida_pino.workflows.derive_dataset import materialize_derived_dataset
from aiida_pino.workflows.labels import InheritsCallerLabel


@calcfunction
def materialize_evaluation_node(
    model: PinoModelData,
    dataset_source,
    evaluation_profile: orm.Dict,
    derivation_profile: orm.Dict,
    metrics: orm.Dict,
    eval_summary: orm.Dict,
    resolved_profile: orm.Dict,
) -> PinoModelEvaluationData:
    """Create a structured evaluation record for a model/dataset pair.

    `resolved_profile` is the merged profile the job RAN with; `evaluation_profile` stays the
    user's own, so provenance keeps the request while the stride label describes the run.
    """
    metadata = build_evaluation_metadata(
        model,
        dataset_source,
        evaluation_profile.get_dict(),
        metrics,
        eval_summary,
        derivation_profile.get_dict(),
        effective_env=(resolved_profile.get_dict().get("env") or {}),
    )
    return PinoModelEvaluationData.from_payload(metadata=metadata)


class EvaluateModelWorkChain(InheritsCallerLabel, WorkChain):
    """Evaluate a native PINO model on a native dataset or union."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("dataset", valid_type=orm.Data)
        spec.input("model", valid_type=PinoModelData)
        spec.input("evaluation_profile", valid_type=orm.Dict, required=False)
        spec.input(
            "allow_training_data",
            valid_type=orm.Bool,
            required=False,
            help="Explicitly permit scoring cases the model was trained on. "
            "Without this the workchain refuses: 4 of 208 historical evals were "
            "contaminated (one at 78%) and indistinguishable from held-out "
            "scores until 2026-08-09.",
        )
        spec.input(
            "allow_domain_change",
            valid_type=orm.Bool,
            required=False,
            help="Explicitly permit scoring the model on a different DOMAIN EXTENT than it "
            "trained on (a different `crop_z_top`). Without this the workchain refuses: a "
            "train_eval passes its TRAINING derivation to the evaluation, and on 2026-08-27 "
            "that served 40 z-cells to a model trained on 30 -- whole-domain MAE 816 K "
            "against 91 K for a comparable arm, substrate MAE within 2 %, no error raised. "
            "A resolution (stride) difference is NOT a domain change and is never refused.",
        )
        spec.input("derivation_profile", valid_type=orm.Dict, required=False)
        spec.input("python_code", valid_type=orm.AbstractCode)
        spec.outline(
            cls.validate_inputs,
            cls.resolve_dataset_source,
            cls.run_evaluation,
            cls.build_evaluation_record,
            cls.finalize,
        )
        spec.output("evaluation", valid_type=PinoModelEvaluationData)
        spec.output("eval_artifacts", valid_type=orm.FolderData)
        spec.output("metrics", valid_type=orm.Dict)
        spec.output("eval_summary", valid_type=orm.Dict)
        spec.exit_code(300, "ERROR_INVALID_EVALUATION_PROFILE", message="Invalid `evaluation_profile`.")
        spec.exit_code(301, "ERROR_INVALID_DERIVATION_PROFILE", message="Invalid `derivation_profile`.")
        spec.exit_code(302, "ERROR_EVALUATION_FAILED", message="PINO evaluation failed.")
        spec.exit_code(
            303,
            "ERROR_INCOMPATIBLE_DATASET_VARIANT",
            message="Selected dataset cannot satisfy the requested native dataset variant.",
        )
        spec.exit_code(304, "ERROR_INVALID_DATASET_INPUT", message="Invalid dataset input type.")
        spec.exit_code(
            307,
            "ERROR_DATASET_NOT_STAGED",
            message="Dataset is not staged on the target computer; run `aiida-pino stage-remote`.",
        )
        spec.exit_code(
            306,
            "ERROR_DOMAIN_EXTENT_MISMATCH",
            message="The evaluation would serve a different domain extent than the model "
            "trained on. Pass `allow_domain_change=Bool(True)` to score it anyway.",
        )
        spec.exit_code(
            305,
            "ERROR_TRAINING_DATA_OVERLAP",
            message="The dataset shares cases with the model's training set. "
            "Pass `allow_training_data=Bool(True)` to score it anyway; the "
            "record will be stamped held_out=False.",
        )

    def validate_inputs(self):
        try:
            self.ctx.dataset = normalize_dataset_collection_node(self.inputs.dataset, label="dataset")
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_DATASET_INPUT
        if "evaluation_profile" in self.inputs:
            try:
                self.ctx.evaluation_profile = validate_evaluation_profile(self.inputs.evaluation_profile.get_dict())
            except ValueError as exc:
                self.report(str(exc))
                return self.exit_codes.ERROR_INVALID_EVALUATION_PROFILE
        else:
            self.ctx.evaluation_profile = {}
        if "derivation_profile" in self.inputs:
            try:
                self.ctx.derivation_profile = validate_derivation_profile(self.inputs.derivation_profile.get_dict())
            except ValueError as exc:
                self.report(str(exc))
                return self.exit_codes.ERROR_INVALID_DERIVATION_PROFILE
        else:
            # The layout must follow the MODEL, not the dataset: a model trained
            # time_as_batch cannot be scored against a time_as_dim variant of its
            # test set. Defaulting to the canonical layout asked exactly that, and
            # a ragged dataset has no time_as_dim variant at all (exit 303), while
            # a uniform one silently rebuilt a mismatched arch (exit 302).
            # The dataset variant only decides which staged arrays to read.
            try:
                # `unrecorded_crop_z_top=0`: this reconstructs what the MODEL saw. A model
                # whose profile names no crop was trained on the full z extent, and must be
                # scored on it -- inheriting today's default would silently re-score 449 of
                # 451 stored models on a grid they never trained on.
                # The archive decides `derived_fields` for a model whose input-key lists
                # were never overridden in its profile; None when there is no archive.
                from aiida_pino.console.model_metrics import model_hydra_config

                recorded = (self.inputs.model.get_model_metadata() or {}).get("derivation_profile") or {}
                self.ctx.derivation_profile = build_derivation_profile_from_training_profile(
                    self.inputs.model.get_training_profile(),
                    unrecorded_crop_z_top=0,
                    hydra_config=model_hydra_config(self.inputs.model),
                    recorded_derived_fields=list(recorded.get("derived_fields") or []),
                )
            except (AttributeError, ValueError):
                self.ctx.derivation_profile = canonical_base_layout_profile()
        # Before the domain check, and before the demotion. Whether the dataset can satisfy the
        # requested variant at all is the more fundamental failure, and the domain check
        # reporting first hid it -- `test_evaluate_model_fails_cleanly_for_incompatible_native_variant`
        # got 306 where it must get 303. It has to stay AHEAD of `demote_for_code` too: the
        # demotion moves `crop_z_top` out of the derivation and into `self.ctx.demoted_env`,
        # which the domain check does not read, so a domain check placed after it would score
        # a correctly-cropped eval as uncropped and refuse it.
        reason = native_source_incompatibility_reason(self.ctx.dataset, self.ctx.derivation_profile)
        if reason:
            self.report(f"Evaluation dataset incompatibility: {reason}")
            return self.exit_codes.ERROR_INCOMPATIBLE_DATASET_VARIANT

        # The derivation decides the DOMAIN, and an explicitly supplied one is checked
        # nowhere else: the reconstruct-what-the-model-saw branch above runs ONLY when no
        # `derivation_profile` input is given, and a train_eval always supplies one -- its
        # TRAINING derivation, handed to both halves.
        _domain_msg = domain_extent_mismatch(
            self.inputs.model.get_training_profile(),
            self.ctx.derivation_profile,
            self.ctx.evaluation_profile,
            training_derivation_profile=(
                self.inputs.model.get_model_metadata() or {}).get("derivation_profile"),
        )
        if _domain_msg and not (
            "allow_domain_change" in self.inputs and self.inputs.allow_domain_change.value
        ):
            self.report(_domain_msg)
            return self.exit_codes.ERROR_DOMAIN_EXTENT_MISMATCH
        if _domain_msg:
            self.report(f"allow_domain_change set, proceeding anyway. {_domain_msg}")

        # Same demotion as the training workchain, through the SAME helper: move
        # read-time-capable transforms to the loader so the evaluation dataset stages
        # pack-indexed rather than as a copy. An 8-member mb40 eval was writing 46 GB of
        # uncompressed .npy purely for a `stride: [2,2,1]`.
        _profile, _env, _notes = demote_for_code(
            self.ctx.derivation_profile, self.ctx.evaluation_profile, self.inputs.python_code
        )
        if _notes:
            self.ctx.derivation_profile = _profile
            # The demoted vars are held ASIDE, not written onto the evaluation profile.
            # `merge_model_training_with_evaluation_profile` deep-copies the model's TRAINING
            # profile and lets the evaluation profile override it, and it never merges `env`
            # key-by-key -- so an evaluation profile carrying an `env` replaces the inherited
            # one wholesale. Writing the two demoted keys here therefore DELETED the training
            # env from every train_eval: `PINO_MATERIAL_OVERLAY` went missing (loud -- the
            # overlay raises) and `PINO_LAZY_MOVING_REF` went missing SILENTLY, which scores
            # moving-reference weights on lab-frame data and returns a plausible number that
            # reads as a model failure.
            # They are applied after the merge instead, in `run_evaluation`.
            self.ctx.demoted_env = {
                name: value for name, value in (_env or {}).items()
                if name in ("PINO_LAZY_STRIDE", "PINO_LAZY_CROP_Z_TOP")
            }
            self.report(
                f"staging zero-copy: {', '.join(_notes)} moved to the loader, "
                "so the evaluation dataset stages pack-indexed instead of being copied"
            )
        # A score on the model's own training data is not an evaluation, and a
        # number computed that way is indistinguishable from a held-out result
        # once it lands on the leaderboard. Refuse by default; the bypass exists
        # for deliberate training-fit checks and stamps the record held_out=False.
        # Transitive: a warm-started model must also be protected against the
        # cases its ANCESTOR trained on. `_member_uuids` sees only the last
        # stage, which left 8 of 13 warm-start chains unguarded.
        from aiida_pino.utils.model_registry import lineage_member_uuids

        _summary = self.inputs.model.get_model_metadata().get("training_source_summary")
        train_uuids = set(_member_uuids(_summary)) | lineage_member_uuids(_summary)
        members = set(self.ctx.dataset.base.attributes.get("member_simulation_uuids") or [])
        # Twin-collapsed: the same case present through a different node with an
        # identical payload still counts as overlap.
        overlap_uuids = payload_keys(train_uuids) & payload_keys(members)
        if overlap_uuids and not (
            "allow_training_data" in self.inputs and self.inputs.allow_training_data.value
        ):
            self.report(
                f"REFUSED: {len(overlap_uuids)} of {len(members)} cases in "
                f"dataset {self.ctx.dataset.pk} are in model "
                f"{self.inputs.model.pk}'s training set. This is a training-fit "
                "measurement, not an evaluation. Pass "
                "allow_training_data=Bool(True) to run it deliberately."
            )
            return self.exit_codes.ERROR_TRAINING_DATA_OVERLAP

    def _resolve_source(self, source: PinoDatasetCollectionData):
        profile = self.ctx.derivation_profile
        metadata = source.get_dataset_metadata()
        if dataset_layout_profile_from_metadata(metadata) == profile:
            return source
        base = resolve_base_dataset(source)
        # Only serve the base directly when the base's OWN stamped layout is what
        # was asked for. Testing `profile == canonical` instead assumed every base
        # is stamped canonical, which silently served a time_as_dim base for a
        # time_as_batch request once the canonical default moved.
        if dataset_layout_profile_from_metadata(base.get_dataset_metadata()) == profile:
            return base
        existing = find_reusable_derived_dataset(base, profile)
        if existing is not None:
            return existing
        report_materialisation_cost(self.report, base, profile)
        return materialize_derived_dataset(base, orm.Dict(dict=profile))

    def resolve_dataset_source(self):
        try:
            self.ctx.resolved_dataset_source = self._resolve_source(self.ctx.dataset)
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INCOMPATIBLE_DATASET_VARIANT

    def _dataset_layout_metadata(self) -> dict:
        return self.ctx.resolved_dataset_source.get_dataset_metadata()

    def run_evaluation(self):
        merged_profile = merge_model_training_with_evaluation_profile(
            self.inputs.model,
            self.ctx.evaluation_profile,
            dataset_metadata=self._dataset_layout_metadata(),
        )
        # Overlay the demoted read-time transforms ON TOP of the inherited env, never in
        # place of it. These two describe how the dataset was STAGED for this eval, so they
        # win over an inherited value; everything else the training set -- the material
        # overlay, the loader window, cache and worker counts -- has to survive.
        demoted = getattr(self.ctx, "demoted_env", None)
        if demoted:
            env = dict(merged_profile.get("env") or {})
            env.update(demoted)
            merged_profile["env"] = env
        # Then drop a read-time crop the RESOLVED collection already bakes. An eval usually
        # resolves to a baked-crop variant, where carrying the env var too makes the loader
        # refuse -- correctly, but only after the submission. See `reconcile_crop_env`.
        _env, _note = reconcile_crop_env(
            merged_profile.get("env"), self._dataset_layout_metadata()
        )
        if _note:
            merged_profile["env"] = _env
            self.report(_note)
        # An eval that EXPLICITLY asks for full resolution means it. The model's inherited
        # `PINO_LAZY_STRIDE` applies at READ TIME, on top of whatever the collection holds,
        # so it silently decimates the very grid the request just named -- and unlike the
        # bake path (where the stride is materialised into a variant and the env is popped by
        # `apply_dataset_layout_to_training_profile`), nothing else clears it. That is why a
        # strided-trained model can evaluate full-res or not depending only on which way its
        # stride happened to be applied at training time, which nobody chose. Only an
        # EXPLICIT `derivation_profile` counts: one built from the training profile carries
        # the model's own stride and means "score it as it trained".
        if "derivation_profile" in self.inputs:
            requested = list((self.ctx.derivation_profile or {}).get("stride") or [1, 1, 1])
            env = dict(merged_profile.get("env") or {})
            carried = str(env.get("PINO_LAZY_STRIDE", "") or "").strip()
            # Only an INHERITED stride is dropped. One the evaluation profile set itself is a
            # deliberate read-time decimation and must win, the same way an explicit override
            # wins everywhere else.
            own = "PINO_LAZY_STRIDE" in ((self.ctx.evaluation_profile or {}).get("env") or {})
            if not own and requested == [1, 1, 1] and carried and carried != "1,1,1":
                env.pop("PINO_LAZY_STRIDE", None)
                merged_profile["env"] = env
                self.report(
                    f"evaluation asked for full resolution, so the model's inherited "
                    f"PINO_LAZY_STRIDE={carried} was dropped; without this the loader would "
                    f"decimate the grid the derivation_profile just requested"
                )
        # The merged profile inherited the model's training reservation, which is
        # an accident of the deep-copy, not a choice about this eval: a 20000 MiB
        # training made its evals book 20000 MiB against a 22528 MiB card while
        # they peak at ~4 GB. Size from measured eval peaks instead; a deliberate
        # `gpu_vram_mb` in evaluation_profile still wins.
        key = vram_footprint_key(
            merged_profile,
            self._dataset_layout_metadata(),
            role="eval",
            computer_label=code_computer_label(self.inputs.python_code),
        )
        self.ctx.vram_key = key
        if "gpu_vram_mb" not in (self.ctx.evaluation_profile or {}):
            inherited = merged_profile.get("gpu_vram_mb")
            # merged_profile is what the key was computed from, so its config is the
            # one whose contents the digest must match.
            peak = lookup_vram_peak_mb(key, merged_profile.get("config_name"))
            if peak:
                # Exact footprint: a real measurement, free to raise or lower.
                merged_profile["gpu_vram_mb"] = vram_reserve_mb(peak)
                self.report(
                    f"auto VRAM: reserving {merged_profile['gpu_vram_mb']} MiB from measured "
                    f"eval peak {peak} MiB (footprint {key})"
                )
            else:
                # First-of-a-kind footprint. Inheriting the training's booking is
                # an unrelated number and typically 4-5x too large, so fall back
                # to the largest peak any eval has recorded. Guardrail: a guess
                # may only LOWER a booking -- under-booking is the dangerous
                # direction, since the ledger then hands out room that is not
                # there, while over-booking merely wastes a seat.
                fallback = fallback_eval_peak_mb()
                if fallback:
                    sized = vram_reserve_mb(fallback)
                    if isinstance(inherited, int) and inherited > 0:
                        sized = min(inherited, sized)
                    merged_profile["gpu_vram_mb"] = sized
                    self.report(
                        f"auto VRAM: no peak for footprint {key}; reserving {sized} MiB from the "
                        f"p90 of recorded eval peaks {fallback} MiB (clamped by inherited {inherited})"
                    )
        # Same accident as VRAM above, other axis: `host_ram_mb` arrives via the
        # deep-copy of the TRAINING profile and nothing corrected it. Derive from the
        # job kind, never inherit; a derived value may only LOWER a booking. An
        # explicit `host_ram_mb` in evaluation_profile still wins.
        if "host_ram_mb" not in (self.ctx.evaluation_profile or {}):
            inherited_ram = merged_profile.get("host_ram_mb")
            # A measurement of THIS footprint beats the flat cap in both directions, and
            # the cap was wrong in the direction that matters: 44 of 335 recorded eval
            # footprints exceed EVAL_HOST_RAM_MB, one by 3x (47949 MiB), so those evals
            # were admitted against a third of what they use. The lower-only guardrail
            # below still applies to the GUESS; a measurement of the same footprint is
            # allowed to raise, because under-booking is what takes co-tenants down.
            measured = lookup_eval_host_ram_mb(key) if key else None
            if measured:
                capped = vram_reserve_mb(measured)
                source = f"measured eval RSS {measured} MiB for footprint {key}"
            else:
                capped = EVAL_HOST_RAM_MB
                if isinstance(inherited_ram, int) and 0 < inherited_ram < capped:
                    capped = inherited_ram
                source = "the flat eval cap (no RSS recorded for this footprint)"
            if inherited_ram != capped:
                self.report(
                    f"auto host RAM: booking {capped} MiB for the eval instead of the "
                    f"training profile's {inherited_ram} MiB, from {source} -- evals read "
                    f"sequentially and never pay the training-side peaks"
                )
            merged_profile["host_ram_mb"] = capped

        # Auto-subset: keep the eval under EVAL_TIME_BUDGET_S unless the profile set
        # its own stride. The stride stamps `eval_frame_stride` into the metrics, so
        # the result is visibly a subset score.
        env = dict(merged_profile.get("env") or {})
        full_frames = bool((self.ctx.evaluation_profile or {}).get("full_frames"))
        if full_frames:
            self.report("full_frames=true: evaluating every stored frame")
        elif "PINO_EVAL_FRAME_STRIDE" not in env:
            ds_node = self.ctx.dataset if hasattr(self.ctx, "dataset") else self.inputs.dataset
            stored = dataset_frames_per_case(ds_node)
            per_case = served_frames_per_case(self.ctx.derivation_profile, stored)
            if per_case != stored:
                n_cases = len(ds_node.get_member_simulation_uuids())
                total = per_case * n_cases
                self.report(
                    f"steady derivation: {stored} frames/case are stored but one is "
                    f"served, so the stride is computed on {total} frames and not "
                    f"{dataset_total_frames(ds_node)}"
                )
            else:
                total = dataset_total_frames(ds_node)
            stride = auto_eval_stride(per_case, total)
            if stride > 1:
                env["PINO_EVAL_FRAME_STRIDE"] = str(stride)
                merged_profile["env"] = env
                self.report(
                    f"auto stride: ~{per_case} frames/case stored, targeting "
                    f"~{EVAL_FRAMES_PER_CASE}/case -> every {stride}th frame "
                    f"(~{(per_case or 0)//stride}/case, stamped in the metrics as a "
                    f"subset score). Pass full_frames=true in evaluation_profile for "
                    f"every stored frame."
                )
        self.ctx.resolved_evaluation_profile = orm.Dict(dict=merged_profile)
        # Check the staging BEFORE creating the job. The check itself already existed and its
        # message was already right -- it even says "this is a DERIVED collection ... it could
        # not have been staged in advance" -- but it lived in CalcJob `presubmit`, so it
        # arrived as an engine traceback on a job that should never have been created, and the
        # parent reported the generic ERROR_EVALUATION_FAILED that hides 234 heterogeneous
        # failures. 22 evals died this way since 2026-08-15.
        #
        # For a remote eval that DERIVES its collection the old ordering was unsatisfiable:
        # the collection was created at 16:29:03 by the submission itself and demanded pre-staged at
        # 16:29:04. Checked here the collection exists, so the `stage-remote <pk>` line the
        # message prints is a command that can actually be run and the submission retried.
        #
        # Deliberately the CalcJob's OWN check with fields from the CalcJob's OWN helper: a
        # cheaper local re-implementation would answer a weaker question than submission asks,
        # which is exactly the trap in `aiida-pino staged --computer X` (it lists BASE
        # datasets, so it passes while the derivation+fields key is absent).
        from aiida_pino.calculations.base import (
            pack_index_allowed_for_transport,
            remote_dataset_symlink_entry,
            runtime_fields_from_training_profile,
        )

        # Warn -- never refuse -- when the scoring grid is not the training grid. Recorded on
        # the eval too (`eval_resolution_matches_training`), but a reader of the metrics is
        # too late: the run has already produced numbers that look real.
        from aiida_pino.utils.model_registry import (
            _eval_effective_stride,
            _model_training_stride,
            resolution_mismatch_warning,
        )

        _mismatch = resolution_mismatch_warning(
            _model_training_stride(self.inputs.model),
            _eval_effective_stride(
                self.ctx.evaluation_profile,
                self.inputs.model,
                self.ctx.derivation_profile,
                self.ctx.resolved_dataset_source,
                (merged_profile.get("env") or {}),
            ),
        )
        if _mismatch:
            self.report(_mismatch)

        computer = getattr(self.inputs.python_code, "computer", None)
        transport = getattr(computer, "transport_type", "core.local") if computer else "core.local"
        if computer is not None and not pack_index_allowed_for_transport(transport):
            try:
                remote_dataset_symlink_entry(
                    self.ctx.resolved_dataset_source,
                    computer,
                    runtime_fields=runtime_fields_from_training_profile(merged_profile),
                    target="dataset",
                )
            except ValueError as exc:
                self.report(str(exc))
                return self.exit_codes.ERROR_DATASET_NOT_STAGED
        future = self.submit(
            PinoEvalCalculation,
            code=self.inputs.python_code,
            dataset=self.ctx.resolved_dataset_source,
            training_artifacts=self.inputs.model,
            training_profile=self.ctx.resolved_evaluation_profile,
        )
        # Key it at submit time: the parser writes `gpu_vram_peak_mb` on every
        # calcjob for free, but without this key no lookup could ever find it —
        # which is why all 129 peaks recorded before this carried vram_key=None.
        future.base.extras.set(VRAM_KEY_EXTRA, key)
        self.report(f"Submitted PINO evaluation job: {future.pk}")
        return ToContext(eval_job=future)

    def build_evaluation_record(self):
        if not self.ctx.eval_job.is_finished_ok:
            return self.exit_codes.ERROR_EVALUATION_FAILED
        self.ctx.evaluation = materialize_evaluation_node(
            self.inputs.model,
            self.ctx.resolved_dataset_source,
            orm.Dict(dict=self.ctx.evaluation_profile),
            orm.Dict(dict=self.ctx.derivation_profile),
            self.ctx.eval_job.outputs.metrics,
            self.ctx.eval_job.outputs.eval_summary,
            self.ctx.resolved_evaluation_profile,
        )

    def finalize(self):
        self.out("evaluation", self.ctx.evaluation)
        self.out("eval_artifacts", self.ctx.eval_job.outputs.eval_artifacts)
        self.out("metrics", self.ctx.eval_job.outputs.metrics)
        self.out("eval_summary", self.ctx.eval_job.outputs.eval_summary)
