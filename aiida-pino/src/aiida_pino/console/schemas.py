from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- error envelope (identical shape to the web demo backend so the JS client that
# the frontend is modeled on handles it unchanged) -------------------------
class ErrorDetail(StrictModel):
    code: str
    message: str
    retryable: bool
    retry_after_ms: int | None = None


class ErrorResponse(StrictModel):
    status: Literal["error"] = "error"
    error: ErrorDetail
    server_request_id: str


def make_error_response(
    *,
    server_request_id: str,
    code: str,
    message: str,
    retryable: bool,
    retry_after_ms: int | None = None,
) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorDetail(code=code, message=message, retryable=retryable, retry_after_ms=retry_after_ms),
        server_request_id=server_request_id,
    )


# --- submit request bodies. Profile fields are free-form dicts validated
# server-side by aiida_pino.utils.validate_* (the real contract); the models
# only pin the envelope so obviously-malformed requests fail fast with 422. ---
class PresubmitRequest(StrictModel):
    """A TrainEvalRequest minus the commitment: resolve, check, report, submit nothing."""

    train_dataset_pk: int
    val_dataset_pk: int | None = None
    python_code: str
    training_profile: dict[str, Any] = Field(default_factory=dict)
    derivation_profile: dict[str, Any] | None = None
    allow_split_violation: bool = False


class TrainEvalRequest(StrictModel):
    # WHY this run exists and WHO launched it, carried with the submission
    # rather than set by a second `describe` call afterwards -- measured
    # 2026-08-22, 12 of the last 12 trainings had a label and none had a
    # description. `submitted_by` is free text on purpose: several people or tools
    # may submit as the same OS user, so an auto-detected value would name nobody.
    description: str | None = None
    submitted_by: str | None = None
    mode: Literal["train", "train_eval"] = "train_eval"
    train_dataset_pk: int
    val_dataset_pk: int | None = None
    python_code: str
    training_profile: dict[str, Any] = Field(default_factory=dict)
    derivation_profile: dict[str, Any] | None = None
    evaluation_profile: dict[str, Any] | None = None
    init_model_pk: int | None = None  # warm start from this model's checkpoints
    # Score the dataset's TEST split rather than val. True restores the behaviour
    # every console submission had before 2026-08-06, when `evaluate_on_test` was
    # added to the workchain (f1739fc) and the console was never updated to set it
    # -- so the test subset it resolves has been silently discarded ever since.
    # Set False for a full-resolution set where the test split is expensive: 22
    # cases x ~400 frames measured at 2.9 h on varspeed.
    evaluate_on_test: bool = True


class TrainingPresetCreateRequest(StrictModel):
    name: str
    description: str | None = None
    training_profile: dict[str, Any] = Field(default_factory=dict)


class EvalOnlyRequest(StrictModel):
    # The same WHY and WHO as `TrainEvalRequest`, for the same reason. The CLI advertised
    # both flags here from the start and this model had neither, so `_submit_eval` dropped
    # them silently -- an eval "signed" with `--submitted-by` carried no signature anywhere
    # in its 18-node provenance graph (random paths, 2026-09-14), and six evals were killed
    # by their own owner for want of a description. Accepted-and-ignored is worse than
    # unsupported.
    description: str | None = None
    submitted_by: str | None = None
    dataset_pk: int
    model_pk: int
    python_code: str
    evaluation_profile: dict[str, Any] | None = None
    derivation_profile: dict[str, Any] | None = None
    #: Escape hatches, default OFF -- the guards are right by default and these exist so a
    #: refusal is never a dead end. Both were WorkChain inputs with no route from the CLI,
    #: which made the remedy each refusal NAMES unreachable through the sanctioned path.
    allow_domain_change: bool = False
    allow_training_data: bool = False


class KfoldRequest(StrictModel):
    dataset_pk: int
    python_code: str
    training_profile: dict[str, Any] = Field(default_factory=dict)
    derivation_profile: dict[str, Any] | None = None


class RawFlow3DImportRequest(StrictModel):
    source_path: str
    python_code: str
    dataset_profile: dict[str, Any] = Field(default_factory=dict)
    materials_root: str | None = None


class ImportModelRequest(StrictModel):
    python_code: str
    training_source_pk: int
    validation_source_pk: int | None = None
    artifacts_pk: int | None = None
    artifacts_path: str | None = None
    training_profile: dict[str, Any] = Field(default_factory=dict)
    derivation_profile: dict[str, Any] | None = None
    model_name: str | None = None
    dry_run: bool = False


class DeriveDatasetRequest(StrictModel):
    dataset_pk: int
    derivation_profile: dict[str, Any] = Field(default_factory=dict)


class CollectionCreateRequest(StrictModel):
    simulation_pks: list[int] = Field(default_factory=list)
    collection_name: str
    description: str | None = None
    # A name is not a description; blank ones are refused unless this is set.
    skip_description: bool = False
    keywords: list[str] | None = None
    dataset_pks: list[int] = Field(default_factory=list)
    # "intersect" admits members whose field sets differ (same source type) and
    # trains on the shared fields; None/"strict" refuses them, naming the odd
    # fields out.
    field_policy: str | None = None


class Flow3DGenerateRequest(StrictModel):
    python_code: str
    run_config: dict[str, Any]
    execution_profile: dict[str, Any] = Field(default_factory=dict)
    dataset_profile: dict[str, Any] = Field(default_factory=dict)
    materials_root: str | None = None
    # Where the finished case is filed. Without it the run joins (or creates) a
    # base dataset named after dataset_profile.batch_keyword, or "flow3d_runs".
    target_dataset_pk: int | None = None


class Flow3DBatchRequest(StrictModel):
    target_dataset_pk: int | None = None
    python_code: str
    run_configs: list[dict[str, Any]]
    execution_profile: dict[str, Any] = Field(default_factory=dict)
    dataset_profile: dict[str, Any] = Field(default_factory=dict)
    materials_root: str | None = None
    max_concurrent_runs: int = 1


class AssignSplitsRequest(StrictModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    only_unassigned: bool = True
    seed: int = 0


# --- annotations (split/keywords/description on stored nodes) ----------------
class AnnotationsRequest(StrictModel):
    title: str | None = None
    description: str | None = None
    split: str | None = None
    keywords_set: list[str] | None = None
    keywords_add: list[str] | None = None
    keywords_remove: list[str] | None = None

    @model_validator(mode="after")
    def _require_one_field(self) -> "AnnotationsRequest":
        if all(
            value is None
            for value in (self.title, self.description, self.split, self.keywords_set, self.keywords_add, self.keywords_remove)
        ):
            raise ValueError(
                "at least one of title, description, split, keywords_set, keywords_add, keywords_remove is required"
            )
        return self


class FolderRequest(StrictModel):
    """Move datasets into a folder; an empty path unfiles them."""

    pks: list[int]
    path: str | None = None


class EvictDataRequest(StrictModel):
    """Delete a dataset's array data, keeping metadata and provenance."""

    dry_run: bool = True
    reason: str | None = None


class FolderEditRequest(StrictModel):
    """Rename a folder (`new` set) or delete it, unfiling its datasets."""

    label: str
    new: str | None = None


# --- preview / helper request bodies (all broker-free, cheap) ----------------
class CollectionPreviewRequest(StrictModel):
    simulation_pks: list[int]


class DeleteRequest(StrictModel):
    pks: list[int]
    dry_run: bool = True
