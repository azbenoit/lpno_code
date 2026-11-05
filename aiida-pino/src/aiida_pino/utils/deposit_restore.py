"""Put a published deposit's simulation arrays back into an imported AiiDA profile.

The LP-NO deposit ships `lpno_provenance.aiida` with every simulation's payload evicted, and
the payloads as one uncompressed tar per simulation family (`payloads_<family>.tar`, members
`<uuid>.npz`), indexed by `payload_manifest.csv` (`uuid, family, split, case_name, bytes,
sha256`). After `verdi archive import`, `restore_from_deposit` reads each member in place from
its tar, checks it against the manifest's sha256 and writes it back into the deposited node
itself (`PinoSimulationData.restore_data`), so every collection, derivation, model and
evaluation that names the node works again, and a re-run is recorded against the original
provenance graph.
"""
from __future__ import annotations

import csv
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


class DepositPayloads:
    """The deposit directory's payload index and tars."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        index = self.root / "payload_manifest.csv"
        if not index.is_file():
            raise FileNotFoundError(f"{index} not found: point --deposit at the deposit directory")
        with open(index, newline="") as handle:
            self.rows = {row["uuid"]: row for row in csv.DictReader(handle)}

    def tar_for(self, uuid: str) -> Path:
        return self.root / f"payloads_{self.rows[uuid]['family']}.tar"


def restore_from_deposit(
    deposit: DepositPayloads,
    simulations: Iterable,
    *,
    dry_run: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Restore every evicted simulation among `simulations` from the deposit's tars.

    A dry run reports what would be restored, the bytes it adds to the profile's repository
    and any tar it needs that is not in the deposit directory; nothing is written.
    """
    sims = list(simulations)
    todo = [s for s in sims if s.is_data_evicted()]
    unknown = sorted(str(s.uuid) for s in todo if str(s.uuid) not in deposit.rows)
    known = [s for s in todo if str(s.uuid) in deposit.rows]
    by_tar: dict[Path, list] = defaultdict(list)
    for sim in known:
        by_tar[deposit.tar_for(str(sim.uuid))].append(sim)
    missing_tars = sorted(str(p.name) for p in by_tar if not p.is_file())
    summary: dict[str, Any] = {
        "simulations": len(sims),
        "already_present": len(sims) - len(todo),
        "to_restore": len(known),
        "bytes": sum(int(deposit.rows[str(s.uuid)]["bytes"]) for s in known),
        "tars": sorted(p.name for p in by_tar),
        "missing_tars": missing_tars,
        "not_in_deposit": unknown,
    }
    if dry_run:
        return {**summary, "dry_run": True}
    if unknown or missing_tars:
        raise FileNotFoundError(
            f"cannot restore: {len(unknown)} simulation(s) are not in payload_manifest.csv and "
            f"{len(missing_tars)} tar(s) are missing ({', '.join(missing_tars[:5])})"
        )
    restored = 0
    for tar_path, members in sorted(by_tar.items()):
        with tarfile.open(tar_path, "r:") as archive:   # uncompressed: members are read in place
            for sim in members:
                handle = archive.extractfile(f"{sim.uuid}.npz")
                if handle is None:
                    raise FileNotFoundError(f"{tar_path.name} has no member {sim.uuid}.npz")
                sim.restore_data(handle, expected_sha256=deposit.rows[str(sim.uuid)]["sha256"])
                restored += 1
                if progress:
                    progress(f"restored {restored}/{len(known)}: {sim.uuid} from {tar_path.name}")
    return {**summary, "dry_run": False, "restored": restored}
