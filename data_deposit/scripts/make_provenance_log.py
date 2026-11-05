#!/usr/bin/env python3
"""Write provenance_log.csv: one row per evaluation behind a printed number, with every UUID.

The manuscript's supplementary material promises "the evaluations listed row by row in the
provenance log released with them", and README_AiiDA.md §6 starts from an evaluation's UUID;
this is that list, read straight from the deposited archive (opened read-only, no import).
Each row names the table or figure (from the `lpno/evaluations/<label>` group), the
evaluation, the model it scored and that model's seed, the scored dataset, the training
dataset and the training workflow (the node `aiida-pino resubmit` clones), and the headline
metrics in kelvin: beam-box MAE is stored normalised and is multiplied here by the model's own
`ref_scales.T` from its archived `.hydra/config.yaml` (`model/.hydra/` for an imported model, whose
seed is also read there when the node carries no `training_seed` extra).

    python data_deposit/scripts/make_provenance_log.py <deposit dir> > provenance_log.csv
    python data_deposit/scripts/make_provenance_log.py <deposit dir> lpno_provenance_rounds.aiida > provenance_log_rounds.csv
"""
import csv
import os
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> int:
    archive = Path(sys.argv[1]) / (sys.argv[2] if len(sys.argv) > 2 else "lpno_provenance.aiida")
    os.environ.setdefault("AIIDA_PATH", tempfile.mkdtemp(prefix="provlog_aiida_"))   # never ~/.aiida
    from aiida import load_profile, orm
    from aiida.storage.sqlite_zip.backend import SqliteZipBackend

    load_profile(SqliteZipBackend.create_profile(archive), allow_switch=True)
    groups = orm.QueryBuilder().append(orm.Group, filters={"label": {"like": "lpno/evaluations/%"}}).all(flat=True)
    ref_t: dict[str, float] = {}
    seeds: dict[str, object] = {}
    writer = csv.writer(sys.stdout)
    writer.writerow(["table", "evaluation", "evaluation_uuid", "model", "model_uuid", "seed",
                     "scored_dataset", "scored_dataset_uuid", "training_dataset_uuid", "training_workflow_uuid",
                     "substrate_mae_K", "substrate_rmse_K", "beambox_mae_K", "meltpool_iou", "scored_n"])
    for group in sorted(groups, key=lambda g: g.label):
        table = group.label.split("/", 2)[2]
        for ev in sorted(group.nodes, key=lambda n: n.label):
            md = ev.base.attributes.all.get("evaluation_metadata") or {}
            kelvin = (md.get("metric_units") or {}).get("kelvin") or {}
            metrics = md.get("metrics") or {}
            model = orm.load_node(md["model_uuid"])
            if model.uuid not in ref_t:
                custom = _model_config(model).get("custom") or {}
                ref_t[model.uuid] = float((custom.get("ref_scales") or {}).get("T", "nan"))
                seeds[model.uuid] = model.base.extras.get("training_seed", custom.get("seed", ""))
            dataset = orm.load_node(md["dataset_source_uuid"])
            creators = [link.node for link in model.base.links.get_incoming().all() if link.link_type.value == "create"]
            workflow = creators[0] if creators else None
            while workflow is not None and workflow.caller is not None:
                workflow = workflow.caller
            beambox = (metrics.get("T") or {}).get("beambox_abs_err_mean")
            writer.writerow([
                table, ev.label, ev.uuid, model.label, model.uuid, seeds[model.uuid],
                dataset.label, dataset.uuid, model.base.extras.get("train_dataset_uuid", ""),
                workflow.uuid if workflow is not None else "",
                _fmt(kelvin.get("T.substrate_abs_err_mean")), _fmt(kelvin.get("T.substrate_rmse")),
                _fmt(beambox * ref_t[model.uuid] if beambox is not None else None),
                _fmt((metrics.get("fl") or {}).get("iou_mean")), md.get("scored_n", ""),
            ])
    return 0


def _fmt(value):
    return "" if value is None else f"{float(value):.6g}"


def _model_config(model) -> dict:
    """The model's archived Hydra config. A model trained in AiiDA keeps it at `.hydra/`; one trained
    elsewhere and registered with `aiida-pino submit import-model` keeps its files under `model/`."""
    for path in (".hydra/config.yaml", "model/.hydra/config.yaml"):
        try:
            return yaml.safe_load(model.base.repository.get_object_content(path)) or {}
        except FileNotFoundError:
            continue
    return {}


if __name__ == "__main__":
    sys.exit(main())
