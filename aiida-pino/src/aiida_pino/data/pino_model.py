"""Native AiiDA data node for compact PINO model bundles."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from aiida import orm

from .pino_dataset import safe_attribute_get


# .csv and the validator/monitor directories: the per-seed tables are a few kB
# and are the input to any later ensemble-sigma computation. They used to
# survive only as long as the calcjob's /tmp workdir did -- mapping a run back
# to them meant crawling every scratch directory past dead requeue duplicates
# (2026-08-09).
MODEL_KEEP_EXTENSIONS = {".ckpt", ".pt", ".pth", ".yaml", ".yml", ".json", ".txt", ".csv"}
MODEL_KEEP_DIRECTORIES = {".hydra", "validators", "monitors"}


def _checkpoint_sort_key(path: str) -> tuple[int, int, int, str]:
    relative = PurePosixPath(path)
    suffix_rank = {".pth": 0, ".pt": 1, ".ckpt": 2}.get(relative.suffix.lower(), 9)
    preferred_name = 0 if relative.name.startswith("u_network") else 1
    return (suffix_rank, preferred_name, len(relative.parts), path)


def checkpoint_choice_note(chosen: str, candidates: list[str]) -> str | None:
    """Say when a `best` checkpoint exists but is NOT the one being used.

    `best_checkpoint.py` writes `u_network.best.pth` to protect against late
    divergence, and nothing reads it: the candidates tie on every ranked field,
    so the sort falls through to the raw path string and `'0'` sorts under
    `'b'`. The final checkpoint therefore wins by an accident of ASCII.

    Changing that preference would silently alter what every existing model
    resolves to, including ones whose published numbers were measured on the
    final checkpoint — so this makes the choice VISIBLE rather than making it
    for anyone.

    **And visibility really is all that is warranted here.** A first reading of
    this called the final checkpoint a 1.88x regression; that number was
    retracted (2026-08-20) once the validation series' own noise
    floor was measured. Adjacent validation points swing 20-35% and the "best"
    point sits 1.4-1.7x BELOW its run's plateau median — so `best` is a minimum
    over ~19 noisy draws, i.e. selection on noise, and two of the three arms
    originally called regressing were at or better than typical. Both estimators
    are biased; on current evidence the final checkpoint is the less biased one.
    Do not "fix" this by preferring best.
    """
    best = [path for path in candidates if "best" in PurePosixPath(path).name.lower()]
    if not best or chosen in best:
        return None
    return (
        f"resolved {chosen}; this model also bundles {best[0]}. The choice is made by sort "
        "order, not by score — but NEITHER is obviously the better estimator. A 'best' "
        "checkpoint is the minimum of many noisy validations (measured: adjacent points "
        "swing 20-35%, and the best point sits 1.4-1.7x below its own run's plateau "
        "median), so preferring it reports a number better than the model typically "
        "achieves. A mean over the last N validations beats either endpoint."
    )


def _iter_checkpoint_candidates(node: orm.FolderData) -> list[str]:
    repository = node.base.repository
    candidates: list[str] = []
    for root, _directories, files in repository.walk():
        root_path = PurePosixPath(root)
        for filename in files:
            relative = root_path / filename if str(root_path) != "." else PurePosixPath(filename)
            if relative.suffix.lower() in {".ckpt", ".pt", ".pth"}:
                candidates.append(str(relative))
    return sorted(candidates, key=_checkpoint_sort_key)


def _config_content_hash(node) -> str | None:
    """sha256 of the archived resolved config's CANONICAL content.

    Parsed and re-serialised (sorted keys) rather than raw bytes, so formatting
    and key order cannot fork the identity of equal configs.
    """
    import hashlib
    import json

    try:
        import yaml

        raw = node.base.repository.get_object_content(".hydra/config.yaml")
        parsed = yaml.safe_load(raw)
    except Exception:  # noqa: BLE001 - unarchived config: no hash, honestly
        return None
    if not isinstance(parsed, dict):
        return None
    # Drop per-run identity before hashing: with custom.name and the output
    # paths included, the first backfill produced 212 "distinct config states"
    # across 214 models -- one per run, useless for the cohort question the
    # hash exists to answer ("which models trained under the same config").
    pruned = {k: v for k, v in parsed.items() if k not in ("network_dir",
              "initialization_network_dir")}
    custom = parsed.get("custom")
    if isinstance(custom, dict):
        pruned["custom"] = {k: v for k, v in custom.items()
                            if k not in ("name", "data_dir", "val_data_dir")}
    canonical = json.dumps(pruned, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _copy_selected_model_payload(source: orm.FolderData, target_root: Path) -> list[str]:
    """Copy the compact model payload from a training artifact folder."""
    copied: list[str] = []
    repository = source.base.repository
    for root, directories, files in repository.walk():
        root_path = PurePosixPath(root)
        for directory in directories:
            if directory not in MODEL_KEEP_DIRECTORIES:
                continue
            relative = root_path / directory if str(root_path) != "." else PurePosixPath(directory)
            destination = target_root / Path(str(relative))
            destination.mkdir(parents=True, exist_ok=True)
        for filename in files:
            relative = root_path / filename if str(root_path) != "." else PurePosixPath(filename)
            if relative.parts and relative.parts[0] in MODEL_KEEP_DIRECTORIES:
                destination = target_root / Path(str(relative))
                destination.parent.mkdir(parents=True, exist_ok=True)
                with repository.open(str(relative), "rb") as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                copied.append(str(relative))
                continue
            if Path(filename).suffix.lower() not in MODEL_KEEP_EXTENSIONS:
                continue
            destination = target_root / Path(str(relative))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with repository.open(str(relative), "rb") as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            copied.append(str(relative))
    return copied



#: Hydra override keys that carry the training seed, in the order they win.
_SEED_OVERRIDE_KEYS = ("seed", "training.seed", "custom.seed", "trainer.seed")

#: What an omitted seed resolves to, verified in the sibling repo at
#: `lpno/models/pipeline/config_loading.py:784`:
#: `seed = int(_get_cfg_value(cfg, "seed", 42))`.
PINO_DEFAULT_SEED = 42


def _env_stride(profile: dict) -> list[int] | None:
    """The loader-level spatial stride, which lives in an env var, not the profile."""
    raw = str(((profile or {}).get("env") or {}).get("PINO_LAZY_STRIDE", "") or "").strip()
    if not raw:
        return None
    try:
        values = [int(part) for part in raw.replace(" ", "").split(",")]
    except ValueError:
        return None
    return values if len(values) == 3 else None


def _env_crop(profile: dict) -> int | None:
    """The loader-level gas crop, which lives in an env var, not the profile."""
    raw = str(((profile or {}).get("env") or {}).get("PINO_LAZY_CROP_Z_TOP", "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _effective_crop_z_top(profile: dict, metadata: dict) -> int | None:
    """The z-layers removed from the top of the domain, from either place it can ride.

    The env var wins over the recorded derivation for the same reason the env stride does:
    it is applied by the datasource itself. A defaulted import asserts nothing (unknown
    stays unknown), mirroring `_effective_stride`.
    """
    env = _env_crop(profile)
    if env is not None:
        return env
    summary = (metadata or {}).get("training_summary") or {}
    if summary.get("derivation_profile_defaulted"):
        return None
    recorded = ((metadata or {}).get("derivation_profile") or {}).get("crop_z_top")
    try:
        return int(recorded) if recorded is not None else None
    except (TypeError, ValueError):
        return None


def _effective_stride(profile: dict, metadata: dict) -> list[int] | None:
    """What the loader really served: the env stride wins over the recorded one.

    `PINO_LAZY_STRIDE` is applied by the datasource itself, after and
    independently of the dataset's derivation, so when both are present the env
    value is what the model saw.
    """
    env = _env_stride(profile)
    if env:
        return env
    # An IMPORT whose derivation profile was defaulted asserts nothing about the
    # data: `canonical_base_layout_profile()` says stride [1,1,1], which reads as
    # "trained at full resolution" when the run may have read a pre-strided copy
    # (random2579_ds_split is 75x50x40 of a 150x100x40 grid -- 2,2,1 baked into
    # the DATA DIRECTORY, which no profile field names). Unknown must stay
    # unknown: the import workchain marks the default, and nothing is stamped.
    summary = (metadata or {}).get("training_summary") or {}
    if summary.get("derivation_profile_defaulted"):
        return None
    recorded = ((metadata or {}).get("derivation_profile") or {}).get("stride")
    try:
        values = [int(v) for v in (recorded or [])]
    except (TypeError, ValueError):
        return None
    return values if len(values) == 3 else None


def pk_for_uuid(uuid: object, recorded_pk: object = None) -> int | None:
    """The pk of the node this uuid names — never a pk recorded in the past.

    A pk is a local row number and the 2026-07-21 PostgreSQL migration renumbered
    every one of them; a uuid is stable. Records written before that date carry
    pks that still *load*, as a different node: 10 of 158 models pointed at a
    `PinoSimulationData` where the dataset collection they trained on should be,
    and one evaluation's `model_pk` resolved to a `RemoteData`. That is worse
    than no answer, so an unresolvable uuid returns None rather than falling back
    to the stale number. `recorded_pk` is used only when there is no uuid at all.
    """
    text = str(uuid or "").strip()
    if text:
        try:
            return orm.load_node(text).pk
        except Exception:  # noqa: BLE001 - no profile loaded, or the node is gone
            return None
    if recorded_pk in (None, ""):
        return None
    try:
        return int(recorded_pk)
    except (TypeError, ValueError):
        return None


def queryable_model_extras(metadata: dict) -> dict:
    """Config, seed and training-data pks as flat extras.

    Without these, answering "which config, which seed, which dataset?" costs
    three link traversals per model — model -> creator workchain -> profile Dict
    -> parse `hydra_overrides` strings — and a 60-model census takes ~7 minutes
    where one `QueryBuilder` filter should do it. Everything here is already in
    `model_metadata`; this only lifts it somewhere queryable. Extras are
    mutable, so a backfill can add them to models stored before this existed
    (a separate backfill, not in this repository).
    """
    profile = dict(metadata.get("training_profile") or {})
    extras: dict = {}

    config_name = str(metadata.get("config_name") or profile.get("config_name") or "").strip()
    if config_name:
        extras["config_name"] = config_name

    # The seed lives in an override string (`seed=17`) far more often than in a
    # profile key, which is exactly why it was unqueryable.
    seed = profile.get("seed")
    seed_source = "training_profile.seed" if seed is not None else None
    if seed is None:
        for override in profile.get("hydra_overrides") or []:
            text = str(override)
            if "=" not in text:
                continue
            key, _, raw = text.partition("=")
            # Hydra's append/force prefixes are part of the override syntax, not
            # of the key: the seed is written `+seed=1003` on 93 models here.
            key = key.strip().lstrip("+~")
            if key in _SEED_OVERRIDE_KEYS:
                try:
                    seed = int(str(raw).strip())
                except ValueError:
                    seed = str(raw).strip()
                seed_source = "hydra_override"
                break
    if seed is None:
        # An omitted seed is not "unseeded": `config_loading.py:784` resolves it
        # to the literal 42 (`int(_get_cfg_value(cfg, "seed", 42))`), a value no
        # aiida-pino record mentioned. 65 of 166 models were silent about a seed
        # that was perfectly determinate. An archived Hydra config can still
        # override this — a backfill reads it where one exists.
        seed, seed_source = PINO_DEFAULT_SEED, "lpno_default"
    extras["training_seed"] = seed
    extras["training_seed_source"] = seed_source

    source_pk = pk_for_uuid(metadata.get("training_source_uuid"), metadata.get("training_source_pk"))
    if source_pk is not None:
        extras["training_source_pk"] = source_pk
    if metadata.get("training_source_label") not in (None, ""):
        extras["training_source_label"] = metadata["training_source_label"]
    # The summary key is `pk`, not `train_dataset_pk`, so this lookup missed on
    # every model past and future — 166 of 166 carried neither extra. Resolved
    # from the uuid for the same reason `training_source_pk` is: the recorded pk
    # predates the migration that renumbered every node.
    for extra_name, source in (
        ("train_dataset_pk", metadata.get("training_source_summary")),
        ("val_dataset_pk", metadata.get("validation_source_summary")),
    ):
        summary = source or {}
        resolved = pk_for_uuid(summary.get("uuid"), summary.get("pk"))
        if resolved is not None:
            extras[extra_name] = resolved
            uuid = str(summary.get("uuid") or "").strip()
            if uuid:
                extras[extra_name.replace("_pk", "_uuid")] = uuid

    # `max_steps` is the ceiling that was REQUESTED. The trainer's stopping
    # criterion routinely stops short of it — 31 of 158 models trained fewer
    # steps than their record claims, worst 4000 of a declared 48000 — and the
    # console labelled the ceiling "Training steps". Both numbers are kept, and
    # the one the run actually reached is the one that means anything.
    steps = (profile.get("max_steps") if isinstance(profile, dict) else None)
    if isinstance(steps, int):
        extras["max_steps"] = steps
    # Cost, from the summary the adapter wrote into this node. The stdout-only
    # half (graph size, measured throughput) is added by the calcjob parser and
    # by a backfill of older nodes.
    from ..utils.run_metrics import performance_extras

    extras.update(performance_extras(metadata.get("training_summary")))

    trained = (metadata.get("training_summary") or {}).get("steps_trained")
    if isinstance(trained, int) and trained > 0:
        extras["steps_trained"] = trained
        extras["steps_trained_source"] = "checkpoint"

    # The resolution the model ACTUALLY trained at. `env.PINO_LAZY_STRIDE`
    # downsamples the grid inside the loader and never reached the record, so
    # 18 of 158 models trained at 2x2x1 while `derivation_profile.stride` says
    # [1,1,1] or nothing — including the holder of the best T MAE badge
    # (13.138 K) on a half-resolution grid. An error measured on a coarser grid
    # is not comparable with one measured on the full grid, and nothing on the
    # model said which it was.
    effective = _effective_stride(profile, metadata)
    if effective:
        extras["training_stride"] = effective
        extras["training_stride_source"] = (
            "env.PINO_LAZY_STRIDE" if _env_stride(profile) else "derivation_profile.stride"
        )
        # Stride ONLY, by contract: `full_resolution` predates the crop and its consumers
        # (comparability, the peak lookups) read it as "no stride". The crop gets its own
        # extra below rather than being folded in here, so neither meaning shifts under
        # the other's consumers.
        extras["full_resolution"] = effective == [1, 1, 1]
    # The domain extent the model ACTUALLY trained on -- the stride story, repeating on a
    # second axis. `PINO_LAZY_CROP_Z_TOP` removes the gas layer inside the loader and never
    # reached the record: since the gas crop became the default, every new model trains on
    # 30 of 40 z-layers while nothing queryable on the node says so. The env var is stored
    # only inside `training_profile` (an attribute, invisible to a QueryBuilder extras
    # filter), and the only other trace is `trained_grid.json` in the artifacts. Same
    # unknown-stays-unknown rule as the stride: a defaulted import stamps nothing.
    crop = _effective_crop_z_top(profile, metadata)
    if crop is not None:
        extras["training_crop_z_top"] = crop
        extras["training_crop_source"] = (
            "env.PINO_LAZY_CROP_Z_TOP" if _env_crop(profile) is not None
            else "derivation_profile.crop_z_top"
        )
    return extras


class PinoModelData(orm.FolderData):
    """Compact model bundle for training outputs used by evaluation and browsing."""

    FORMAT_VERSION = 1

    @classmethod
    def from_training_artifacts(
        cls,
        training_artifacts: orm.FolderData,
        *,
        metadata: dict,
    ) -> "PinoModelData":
        """Build a model node from a retrieved training-artifacts folder."""
        with tempfile.TemporaryDirectory() as tmp_path:
            tmpdir = Path(tmp_path)
            copied_files = _copy_selected_model_payload(training_artifacts, tmpdir)
            node = cls(tree=str(tmpdir))
        node.base.attributes.set("format_version", cls.FORMAT_VERSION)
        node.base.attributes.set("model_metadata", dict(metadata))
        node.base.attributes.set("copied_files", copied_files)
        # Content identity of the RESOLVED config, so "trained under the same
        # config state" is one filter and same-name yaml drift is queryable.
        # config_name is a pointer; this is the content it pointed at.
        digest = _config_content_hash(node)
        if digest:
            node.base.extras.set("config_hash", digest)
        node.label = str(metadata.get("model_name") or metadata.get("run_name") or "model")
        node.base.extras.set("aiida_pino_kind", "model")
        node.base.extras.set("model_name", node.label)
        # Stamp the architecture as a keyword so every model is filterable by family
        # without the console having to re-derive it. Nothing auto-tagged models before
        # this: 86 of 136 carried keywords, all of them set explicitly by the
        # self-improve driver, so every GAOT run had none. `merge_keywords` unions
        # rather than overwrites, so a caller's own tags survive.
        # Imported inside the function, NOT at module scope: `aiida_pino.utils.__init__`
        # pulls in `model_registry`, which imports `aiida_pino.data`, so a top-level
        # import here closes a cycle and makes `import aiida_pino.data` fail outright
        # with "partially initialized module" (verified 2026-07-30).
        from ..utils.architecture import (
            architecture_keyword_for_metadata,
            merge_keywords,
        )

        node.base.extras.set(
            "keywords",
            merge_keywords(
                metadata.get("keywords"), architecture_keyword_for_metadata(metadata)
            ),
        )
        if metadata.get("training_source_uuid"):
            node.base.extras.set("training_source_uuid", metadata["training_source_uuid"])
        if metadata.get("training_source_kind"):
            node.base.extras.set("training_source_kind", metadata["training_source_kind"])
        for key, value in queryable_model_extras(metadata).items():
            node.base.extras.set(key, value)
        return node

    @staticmethod
    def queryable_extras_from_metadata(metadata: dict) -> dict:
        """The identity a census asks about, as extras a QueryBuilder can filter."""
        return queryable_model_extras(metadata)

    def get_model_metadata(self) -> dict:
        """Return model metadata."""
        return dict(safe_attribute_get(self, "model_metadata", {}))

    def get_training_profile(self) -> dict:
        """Return the frozen training profile stored with the model."""
        return dict(self.get_model_metadata().get("training_profile", {}))

    def get_model_name(self) -> str:
        """Return the model/run name."""
        metadata = self.get_model_metadata()
        return str(metadata.get("model_name") or metadata.get("run_name") or self.label or "model")

    def resolve_primary_checkpoint(self) -> str:
        """Return the best available checkpoint path inside the stored model bundle."""
        metadata = self.get_model_metadata()
        primary_checkpoint = str(metadata.get("primary_checkpoint", "") or "")
        if primary_checkpoint:
            try:
                self.base.repository.get_object(primary_checkpoint)
                return primary_checkpoint
            except FileNotFoundError:
                pass

        candidates = _iter_checkpoint_candidates(self)
        if candidates:
            return candidates[0]
        return primary_checkpoint


def serialize_model_metadata(metadata: dict) -> str:
    """Render model metadata for repository/debug payloads."""
    return json.dumps(metadata, indent=2, sort_keys=True)
