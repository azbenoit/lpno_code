#!/usr/bin/env python3
"""Ti-6Al-4V-only FNO prediction on the Fig 12 Ti-6Al-4V case, for panel (e) of fig:twoalloy-pred.

The model is fno_steady_edge75 (model 86651), the "FNO, Ti-6Al-4V only" row of tab:twoalloy. Its
tabled scores use final-step weights and H_norm on the training denominator 425.87 (eval df4cc30c),
while the only stored prediction of this case (eval 4db21084, CalcJob 6d9aca2c, in the 2026-09-15
export of the provenance store; the current deposit does not hold it) used best-checkpoint weights
and a denominator refit on the scored split. So this script reruns the forward on CPU from that
eval's preview inputs:

  1. best-checkpoint weights on the stored inputs must reproduce the stored prediction (T and alpha
     to 1e-4 in normalized units), which checks the network build and the eval post-processing
     (T filled toward the boiling point through the smooth alpha mask, alpha gated by itself);
  2. final-step weights with H_norm = (P / sqrt(V)) / 425.87 give the tabled setting.

Writes the predicted fields (temperatures in K) and the case parameters into the ti64only_* entries of
data/twoalloy_prediction_frames.npz, keeping its other entries. Needs torch, physicsnemo and
lpno (the environment of tools/setup_env.py):

    LPNO_DEPOSIT=<deposit dir> LPNO_TI64ONLY_FNO=<model run dir with .hydra/ and u_network.*.pth> \\
        python3 reproducibility/figures/forward_ti64only_fno.py
"""
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from physicsnemo.sym.hydra import instantiate_arch
from physicsnemo.sym.key import Key
from lpno.custom_utils.utils import fill_where_mask_smooth_np, fill_where_mask_smooth_torch

os.nice(19)
torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
MDIR = Path(os.environ["LPNO_TI64ONLY_FNO"])
ARCHIVE = Path(os.environ["LPNO_DEPOSIT"]) / "lpno_provenance.aiida"
PREVIEW_SHA = "10cd36b0fa649ec2bc914e6dcafa13752dd775b14a8b7e1148e5f63a7ec08e57"  # eval 4db21084 preview
TRAIN_DENOM = 425.87          # TI64_ONLY_DENOM, the denominator the tabled eval df4cc30c used
TSCALE = 3000.0               # ref_scales T

cfg = OmegaConf.load(MDIR / ".hydra" / "config.yaml")
assert cfg.custom.alpha_masking and not OmegaConf.select(cfg, "custom.per_case_t_boil", default=False)
T_FILL = float(cfg.custom.T_boil_k) / TSCALE
ik, ok = list(cfg.custom.FNO_input_keys), list(cfg.custom.FNO_output_keys)
with zipfile.ZipFile(ARCHIVE) as zf:
    z = np.load(io.BytesIO(zf.read(f"repo/{PREVIEW_SHA}")))
    z = {k: z[k] for k in z.files}
row = 0
P, V = float(z["in_P"][row].flat[0]) * 10, float(z["in_V_scan"][row].flat[0]) / 10
assert abs(P - 225.0) < 1e-3 and abs(V - 0.4) < 1e-6, (P, V)
inv = {k: torch.from_numpy(np.asarray(z[f"in_{k}"][row:row + 1], np.float32)) for k in ik}


def build(weights):
    dec = instantiate_arch(cfg=cfg.arch.decoder, output_keys=[Key(k) for k in ok])
    model = instantiate_arch(cfg=cfg.arch.fno, decoder_net=dec, input_keys=[Key(k) for k in ik],
                             output_keys=[Key(k) for k in ok])
    sd = torch.load(MDIR / weights, map_location="cpu")
    model.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd)
    return model.eval()


def forward(model, invar):
    """Network output plus the eval's post-processing (_apply_prediction_postprocessing)."""
    with torch.inference_mode():
        o = model(invar)
        T = fill_where_mask_smooth_torch(o["T"], o["alpha"], threshold=0.5, d=T_FILL, channel_axis=1, k=20.0)
    a = o["alpha"][0, 0].numpy()
    return T[0, 0].numpy(), fill_where_mask_smooth_np(a, a, threshold=0.5, d=0.0, k=20, inplace=False)


T_b, a_b = forward(build("u_network.best.pth"), inv)
err = max(float(np.abs(T_b - z["pred_T"][row, 0]).max()), float(np.abs(a_b - z["pred_alpha"][row, 0]).max()))
print(f"best checkpoint vs stored prediction: max |d| {err:.1e} (normalized)")
assert err < 1e-4, "local forward does not reproduce the stored eval"

h = (P / np.sqrt(V)) / TRAIN_DENOM
inv["H_norm"] = torch.full_like(inv["H_norm"], h)
T_f, a_f = forward(build("u_network.0.pth"), inv)
out = HERE / "data" / "twoalloy_prediction_frames.npz"
with np.load(out) as zz:
    keep = {k: zz[k] for k in zz.files if not k.startswith("ti64only_")}
np.savez_compressed(out, **keep, ti64only_pred_T=(T_f * TSCALE).astype(np.float32),
                    ti64only_pred_a=a_f.astype(np.float32), ti64only_P=P, ti64only_V=V, ti64only_H_norm=h)
print(f"H_norm {h:.4f}; wrote {out}")
