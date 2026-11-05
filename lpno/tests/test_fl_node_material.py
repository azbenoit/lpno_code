"""fl_Node is in EVERY run's graph, so it is a third melt-window path.

`preprocessing_datasets` (target) and `postprocessing.derive_fl_masked` (eval metric)
were made per-case on 2026-08-26; this node was not, so the graph kept deriving
predicted fl at Ti64's 1873/1923 for all 300 ck300 materials.
"""
import torch

from lpno.equations.FNO_postprocess import fl_Node


def test_falls_back_to_scalars_without_channels():
    node = fl_Node(T_s=1873.0 / 3000.0, T_l=1923.0 / 3000.0)
    out = node({"T": torch.full((1, 1, 2, 2, 2), 1000.0 / 3000.0)})
    assert out["fl"].mean().item() < 0.1, "1000 K is solid for Ti64"


def test_uses_per_case_channels_when_present():
    """1000 K is molten for an 868 K liquidus, solid for Ti64's 1923 K."""
    node = fl_Node(T_s=1873.0 / 3000.0, T_l=1923.0 / 3000.0)
    shp = (1, 1, 2, 2, 2)
    out = node({
        "T": torch.full(shp, 1000.0 / 3000.0),
        "mat_Ts": torch.full(shp, 825.0 / 1800.0),   # channels arrive on their own scale
        "mat_Tl": torch.full(shp, 868.0 / 1800.0),
    }, ts_scale=1800.0 / 3000.0, tl_scale=1800.0 / 3000.0)
    assert out["fl"].mean().item() > 0.9, f"expected molten, got {out['fl'].mean().item():.3f}"


def test_per_sample_windows_in_one_batch():
    """A batch mixes materials; each sample must use its own window."""
    node = fl_Node(T_s=1873.0 / 3000.0, T_l=1923.0 / 3000.0)
    shp = (2, 1, 2, 2, 2)
    T = torch.full(shp, 1000.0 / 3000.0)
    ts = torch.empty(shp); ts[0] = 825.0 / 1800.0; ts[1] = 1873.0 / 1800.0
    tl = torch.empty(shp); tl[0] = 868.0 / 1800.0; tl[1] = 1923.0 / 1800.0
    out = node({"T": T, "mat_Ts": ts, "mat_Tl": tl},
               ts_scale=1800.0 / 3000.0, tl_scale=1800.0 / 3000.0)["fl"]
    assert out[0].mean().item() > 0.9 and out[1].mean().item() < 0.1
