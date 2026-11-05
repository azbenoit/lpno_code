"""How much gas can be cropped off the top WITHOUT cutting any metal.

The gas crop exists to drop the void above the substrate, which is dead input for every
model. Until now it was a constant -- `crop_z_top: 10` -- chosen from the deck geometry:
the surface sits at z=0 with 95 um of headroom at dz=10 um, so ten cells "is exactly the
gas on every real deck".

**That reasoning is geometrically right and physically wrong, and it has been cutting
metal.** Material does not stay below z=0: the melt bulges, the keyhole ejects, and spatter
lands in the headroom. Measured 2026-09-01 over members of `ti64_preheat60`'s live splits,
every one on a 50-cell z axis with exactly 10 cells above z=0:

    sim 40757  metal-free top slices  8
    sim 40819                         7
    sim 40838                         9
    sim 40788                         5     <- a crop of 10 removes 5 slices holding metal
    sim 41066                         9
    sim 41218                         9

So the safe crop is a property of the DATA, not of the mesh, and it varies case by case.
This module measures it. `Fraction Of Fluid` is the material indicator (0 in void, >0
wherever there is material), so the deepest safe crop is the number of top z-slices in
which no frame, anywhere in x or y, holds material.

The per-simulation answer is cached on the node (`metal_free_top_slices`): a scan costs
about 0.13 s, which is cheap once and needless twice.
"""

from __future__ import annotations

from typing import Any, Iterable

#: Extra holding a simulation's measured metal-free depth, in z-cells from the top.
METAL_FREE_EXTRA = "metal_free_top_slices"

#: The field that says where material is. Not `Solid Fraction` (which is 0 in the melt
#: pool, so a molten bulge would read as croppable void) and not `Melt Region` (0 in the
#: cold substrate). Fraction Of Fluid is >0 wherever material exists in any state, which
#: is the only question being asked here.
MATERIAL_FIELD = "Fraction Of Fluid"


def _frames(material):
    """Iterate the per-frame spatial volumes, whatever the leading layout."""
    import numpy as np

    array = np.asarray(material)
    if array.ndim >= 4:
        for index in range(array.shape[0]):
            yield array[index]
    else:
        yield array


def _substrate_connected(mask):
    """The part of `mask` joined to the substrate, or None when it cannot be determined.

    The substrate is the z=0 face: the bottom of the domain is solid metal in every deck
    here. Labelling is 3-D and face-connected, so a droplet touching the rim diagonally is
    still counted as attached -- the conservative direction, since calling rim "spatter"
    would authorise cropping into the surface.

    Falls back to the whole mask when SciPy is unavailable: without labelling, "connected"
    cannot be determined, and treating everything as rim keeps the crop safe rather than
    optimistic.
    """
    import numpy as np

    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - SciPy present in this environment
        return mask
    labels, count = ndimage.label(mask)
    if not count:
        return None
    floor = set(np.unique(labels[..., 0])) - {0}
    if not floor:
        # Nothing touches the substrate at all: no rim to protect, so every attached
        # component is absent and only free-flying material remains.
        return np.zeros_like(mask)
    return np.isin(labels, list(floor))


def metal_free_top_slices(simulation, *, use_cache: bool = True) -> int | None:
    """Top z-slices holding no SUBSTRATE-CONNECTED material, or None if unmeasurable.

    None means "do not know" and must never be read as "crop freely": a simulation whose
    payload cannot be opened, or that stores no material field, has not been measured.
    """
    import numpy as np

    if use_cache:
        cached = simulation.base.extras.all.get(METAL_FREE_EXTRA)
        if isinstance(cached, int):
            return cached
    try:
        with simulation._open_payload_archive() as archive:  # noqa: SLF001 - measurement
            if MATERIAL_FIELD not in archive.files:
                return None
            material = np.asarray(archive[MATERIAL_FIELD])
    except Exception:  # noqa: BLE001 - unreadable payload is unmeasured, not unlimited
        return None
    if material.ndim < 2:
        return None
    nz = int(material.shape[-1])
    # Only material CONNECTED TO THE SUBSTRATE counts. A detached droplet in flight is
    # spatter: cropping through it loses a few voxels of an object the model cannot track
    # anyway, whereas cropping through the rim truncates the surface the melt pool sits in.
    # Measured on `ti64_preheat60`: excluding spatter recovers 59896 (2 -> 4 free slices,
    # a blob at x~80 detached from a mound reaching z=46) and 41104 (3 -> 4). It does NOT
    # rescue the dataset, because the binding case 40971 is a genuine continuous mound
    # rising to z=48 across a third of the domain -- verified visually, both curves
    # identical. So "ignore spatter" is the right DEFINITION and a small effect here.
    # Connected-component labelling costs ~0.1 s per frame, and a 107-frame case run
    # naively took ~10 s -- 14 minutes for an 81-member dataset, far too slow for a guard
    # that runs at submit time. The prune: a frame can only RAISE the answer if its
    # any-material top already exceeds the best rim top found so far, and any-material is
    # a cheap reduction. Visiting frames tallest-first then lets the rest exit immediately.
    frames = list(_frames(material))
    tops = []
    for index, frame in enumerate(frames):
        mask_any = frame > 0.0
        if not mask_any.any():
            continue
        cols = mask_any.any(axis=tuple(range(mask_any.ndim - 1)))
        tops.append((int(np.max(np.nonzero(cols)[0])), index))
    tops.sort(reverse=True)

    occupied_max = -1
    for any_top, index in tops:
        if any_top <= occupied_max:
            break  # nothing below this can lift the rim any higher
        rim = _substrate_connected(frames[index] > 0.0)
        if rim is None or not rim.any():
            continue
        top = int(np.max(np.nonzero(rim.any(axis=tuple(range(rim.ndim - 1))))[0]))
        occupied_max = max(occupied_max, top)
    safe = nz if occupied_max < 0 else nz - 1 - occupied_max
    try:
        simulation.base.extras.set(METAL_FREE_EXTRA, int(safe))
    except Exception:  # noqa: BLE001 - a cache write must never fail a measurement
        pass
    return int(safe)


def survey_members(members: Iterable[Any]) -> dict[str, Any]:
    """Per-case metal-free depth for a member set, plus the dataset-wide safe crop.

    The dataset answer is the MINIMUM: a crop is applied to every member, so the shallowest
    case is the constraint. `unmeasured` is reported separately rather than folded into the
    minimum -- an unknown is not a zero, and pretending otherwise would silently disable
    the crop for a whole dataset because one payload failed to open.
    """
    per_case: dict[int, int] = {}
    unmeasured: list[int] = []
    for member in members:
        value = metal_free_top_slices(member)
        if value is None:
            unmeasured.append(getattr(member, "pk", -1))
        else:
            per_case[getattr(member, "pk", -1)] = value
    return {
        "per_case": per_case,
        "unmeasured": unmeasured,
        "safe_crop": min(per_case.values()) if per_case else None,
        "measured": len(per_case),
    }


def crop_advisory(survey: dict[str, Any], requested: int, *, baked: bool = False) -> str | None:
    """A LOUD warning when `requested` would cut metal, naming the count and the fix.

    Returns None when the crop is safe, so the caller can `if msg: report(msg)`.

    A warning rather than a refusal, deliberately, and for the reason the FLOW-3D disk
    floor records: a refusing gate turns a recoverable situation into dead work, and a
    submission blocked at 3 a.m. is worse than one that runs with a stated caveat. The
    message therefore has to carry everything needed to act on it without re-deriving:
    HOW MANY cases are affected, how badly, and WHAT to use instead.
    """
    safe = survey.get("safe_crop")
    if safe is None or int(requested) <= int(safe):
        return None
    affected = {pk: value for pk, value in survey["per_case"].items() if value < int(requested)}
    worst = min(affected.values()) if affected else safe
    total = survey.get("measured", 0)
    examples = ", ".join(
        f"{pk} (keeps {value})" for pk, value in sorted(affected.items(), key=lambda kv: kv[1])[:3]
    )
    # A BAKED crop is a different warning from a read-time one, and the text has to say so.
    # Applied by the loader, the stored arrays keep the material and only the served view
    # drops it -- reversible by clearing an env var. Baked, the metal is physically absent
    # from the staged variant and every model trained on it saw truncated geometry, with no
    # recovery short of re-deriving (2026-09-07).
    baked_tail = (
        " THIS CROP WILL BE BAKED into the staged arrays (materialize), so the material is "
        "physically removed rather than hidden: it cannot be recovered by changing an env "
        "var, only by re-deriving the dataset."
        if baked else ""
    )
    unmeasured = survey.get("unmeasured") or []
    tail = (f" {len(unmeasured)} member(s) could not be measured and are NOT counted."
            if unmeasured else "")
    return (
        f"CROP CUTS METAL: crop_z_top={int(requested)} removes material from "
        f"{len(affected)} of {total} measured cases. The shallowest case tolerates only "
        f"{worst} slice(s); worst offenders: {examples}. Material rises above z=0 (melt "
        f"bulge, ejecta, spatter), so the geometric gas height is NOT the safe crop. "
        f"RECOMMENDED crop_z_top={int(safe)} -- the deepest crop that touches no metal in "
        f"any member.{tail}{baked_tail}"
    )
