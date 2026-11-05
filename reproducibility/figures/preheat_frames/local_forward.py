"""Local, verified forward of m65230 (RIGNO preheat107 cond) on chosen steady
preheat test cases. Every convention is checked against the pipeline's own
eval preview (calc 65233) before any new case is trusted."""
import io, sys, json, os, tempfile
from pathlib import Path

import numpy as np
import torch
from aiida import load_profile, orm
load_profile()
from aiida_pino.data.pino_dataset_collection import derive_simulation_arrays
from omegaconf import OmegaConf

# Output directory for the intermediate NPZ/pickle this script writes. It also reads
# hard-coded AiiDA node ids from the private store, so it is provenance for the panels,
# not a public reproduction path.
OUT = os.environ.get("PREHEAT_FIG_DIR", str(Path(__file__).resolve().parent))
pv = np.load(f"{OUT}/preview65233.npz")

ds = orm.load_node(65012)
profile = ds.base.attributes.get("derivation_profile")
print("derivation profile:", json.dumps(profile))

# ---- model ----
m = orm.load_node(65230)
tmp = tempfile.mkdtemp(prefix="m65230_")
m.base.repository.copy_tree(tmp)
cfg = OmegaConf.load(f"{tmp}/.hydra/config.yaml")
ik = list(cfg.custom.FNO_input_keys)
ok = list(cfg.custom.FNO_output_keys) if "FNO_output_keys" in cfg.custom else ["T", "alpha", "fl"]
print("input keys:", ik)
print("output keys:", ok)
rc = cfg.arch.rigno
from physicsnemo.sym.key import Key
from lpno.models.rigno_arch import RIGNOArch
model = RIGNOArch(
    input_keys=[Key(k) for k in ik],
    output_keys=[Key(k) for k in ok],
    dimension=int(getattr(rc, "dimension", 3)),
    node_latent_size=int(getattr(rc, "node_latent_size", 64)),
    edge_latent_size=int(getattr(rc, "edge_latent_size", 64)),
    processor_steps=int(getattr(rc, "processor_steps", 6)),
    mlp_hidden_layers=int(getattr(rc, "mlp_hidden_layers", 1)),
    subsample_factor=float(getattr(rc, "subsample_factor", 8.0)),
    rmesh_levels=int(getattr(rc, "rmesh_levels", 4)),
    k_p2r=int(getattr(rc, "k_p2r", 4)),
    k_r2p=int(getattr(rc, "k_r2p", 4)),
    p_edge_masking=float(getattr(rc, "p_edge_masking", 0.5)),
    graph_seed=int(getattr(rc, "graph_seed", 0)),
    cache_dir=f"{tmp}/graphs",
    grad_checkpointing=False,
    num_graph_variants=int(getattr(rc, "num_graph_variants", 1)),
    inference_edge_masking=float(getattr(rc, "inference_edge_masking", 0.0)),
    rnode_count=int(getattr(rc, "rnode_count", 0)),
    film_keys=list(getattr(rc, "film_keys", []) or []),
)
sd = torch.load(f"{tmp}/u_network.0.pth", map_location="cpu")
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)
print("state dict: missing", len(missing), "unexpected", len(unexpected))
if missing: print(" missing:", missing[:5])
model.eval()

# ---- channel builder ----
# semantics pinned against the preview: P/10, V*10, T0/3000, r_m/1e-4,
# coords*1e4 (metres -> 1e-4 m units), H_norm = (P/sqrt(V))/denom, denom from
# the preview's own scored cases, conduction_field passed through as stored.
def exact(v): return float(v)
def case_inputs(sim_pk):
    n = orm.load_node(sim_pk)
    sm = n.base.attributes.get("simulation_metadata")
    man = sm["flow3d_case_manifest"]["input"]
    arrays, tvals = derive_simulation_arrays(n, profile)
    # canonical arrays: (nt=1, 1, nx, ny, nz) after derivation
    g = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    def A(name):
        a = g[name]
        return a.reshape(a.shape[-3:]) if a.ndim >= 3 else a
    shp = A("Temperature").shape
    P = exact(man["laser"]["power_w"]); V = exact(man["laser"]["scan_speed_m_per_s"])
    T0 = exact(man["simulation"]["initial_temperature_k"]); r = exact(man["laser"]["beam_radius_m"])
    heat = str(man["laser"].get("heat_distribution", "gaussian")).lower()
    full = lambda v: np.full(shp, v, dtype=np.float32)
    ch = {
        "V_scan": full(V * 10.0),
        "P": full(P / 10.0),
        "x": (A("x") * 1e4).astype(np.float32),
        "y": (A("y") * 1e4).astype(np.float32),
        "z": (A("z") * 1e4).astype(np.float32),
        "preheat_T0": full(T0 / 3000.0),
        "beam_shape_uniform": full(1.0 if heat == "uniform" else 0.0),
        "beam_radius": full(r / 1e-4),
        "conduction_field": A("conduction_field").astype(np.float32),
    }
    ch["H_norm"] = full((P / np.sqrt(abs(V))) / H_DENOM)
    meta = dict(P=P, V=V, T0=T0, r_um=r*1e6, name=sm.get("case_name"))
    gt = dict(T=A("Temperature"), alpha=A("Cell Volume Fraction"),
              fl=A("Fraction Of Fluid"))
    return ch, gt, meta

# denom fitted at train time; recover it exactly from BOTH preview cases and demand agreement
def preview_denom(row, P, V):
    return (P / np.sqrt(V)) / float(pv["in_H_norm"][row].flat[0])
d018 = preview_denom(0, 214.9229, 0.721615)
d084_meta = orm.load_node(62859).base.attributes.get("simulation_metadata")["flow3d_case_manifest"]["input"]
d084 = preview_denom(1, d084_meta["laser"]["power_w"], d084_meta["laser"]["scan_speed_m_per_s"])
print(f"H denom from ph_018: {d018:.6f}  from ph_084: {d084:.6f}")
assert abs(d018 - d084) / d018 < 1e-4, "H denominators disagree"
H_DENOM = d018

# ---- verification against the preview ----
def verify(sim_pk, row):
    ch, gt, meta = case_inputs(sim_pk)
    worst = {}
    for k in ch:
        ref = pv[f"in_{k}"][row, 0]
        worst[k] = float(np.abs(ch[k] - ref).max())
    print(f"verify {meta['name']}: channel max diffs:", {k: round(v, 6) for k, v in worst.items()})
    bad = {k: v for k, v in worst.items() if v > 2e-3}
    assert not bad, f"channel mismatch: {bad}"
    return ch, gt, meta

ch018, gt018, _ = verify(40788, 0)
ch084, gt084, _ = verify(62859, 1)

# ---- forward check ----
def forward(ch):
    invar = {k: torch.from_numpy(v[None, None]).float() for k, v in ch.items()}
    with torch.inference_mode():
        out = model(invar)
    return {k: v[0, 0].numpy() for k, v in out.items()}

p018 = forward(ch018)
ref = pv["pred_T"][0, 0]
d = np.abs(p018["T"] - ref)
print(f"forward check ph_018: pred_T max diff vs preview {d.max():.5f}  mean {d.mean():.6f}")
np.savez_compressed(f"{OUT}/verified_forward.npz", **{f"chk_{k}": v for k, v in p018.items()})

# ---- the three cases ----
PICKS = {"high preheat": 59239, "small beam": 62821, "big beam": 61852}
results = {}
for tag, pk in PICKS.items():
    ch, gt, meta = case_inputs(pk)
    pred = forward(ch)
    results[tag] = (pred, gt, meta)
    print(f"{tag}: {meta['name']} T0={meta['T0']:.0f}K r={meta['r_um']:.1f}um "
          f"GT Tmax {gt['T'].max():.0f}K pred Tmax {pred['T'].max()*3000:.0f}K")

# alpha naming check vs preview (true_alpha)
a018 = gt018["alpha"]
ra = pv["true_alpha"][0, 0]
print("alpha check ph_018: max diff", float(np.abs(a018 - ra).max()))

import pickle
store = {}
for tag, (pred, gt, meta) in results.items():
    store[tag] = dict(pred={k: v for k, v in pred.items()},
                      gt=gt, meta=meta)
with open(f"{OUT}/three_cases.pkl", "wb") as f:
    pickle.dump(store, f)
print("saved", f"{OUT}/three_cases.pkl")
