import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from physicsnemo.sym.domain.inferencer import Inferencer
from physicsnemo.sym.utils.io.vtk import VTKUniformGrid  

# pyright: reportOptionalMemberAccess=false

ArrayLike = Union[np.ndarray, Tensor]

class FNOGridInferencer(Inferencer):
    """
    Grid-aware inferencer for PINO/FNO models with channel-first 2D/3D grids.
    Saves VTI with correct axis order (x fastest) and optional physical bounds.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_keys: Iterable[str],
        output_names: Iterable[str],
        results_dir: Union[str, os.PathLike] = "outputs/fno_infer",
        device: Optional[Union[str, torch.device]] = None,
        requires_grad: bool = False,
        mask_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        mask_value: float = np.nan,
        log_iter: bool = False,
        unscale_fns: Optional[Mapping[str, Callable[[np.ndarray], np.ndarray]]] = None,
        dtype: torch.dtype = torch.float32,
        bounds: Optional[List[List[float]]] = None,   # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    ) -> None:
        self.model = model
        self.input_keys: List[str] = list(input_keys)
        self.output_names: List[str] = list(output_names)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.requires_grad = requires_grad
        self.mask_fn = mask_fn
        self.mask_value = mask_value
        self.log_iter = log_iter
        self.unscale_fns = dict(unscale_fns) if unscale_fns is not None else {}
        self.dtype = dtype
        self.bounds = bounds  # if None → unit spacing with origin (0,0,0)

        self.model.to(self.device)
        self.model.eval()  

    @staticmethod
    def _to_numpy(a: ArrayLike) -> np.ndarray:
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    @staticmethod
    def _ensure_bc_grid(a: np.ndarray) -> np.ndarray:
        """
        Normalize array shapes to channel-first batch form:
        - 2D: [B, C, nx, nz]
        - 3D: [B, C, nx, ny, nz]
        Accepts:
            [nx, nz], [B, nx, nz], [B, C, nx, nz],
            [nx, ny, nz], [B, nx, ny, nz], [B, C, nx, ny, nz]
        """
        if a.ndim == 2:
            a = a[None, None, ...]          # [1,1,nx,nz]
        elif a.ndim == 3:
            a = a[:, None, ...]             # [B,1,nx,nz]
        elif a.ndim == 4:
            # Could be [B,C,nx,nz] (2D channel-first) or [B,nx,ny,nz] (3D no channel).
            if a.shape[1] not in (1, 2, 3, 4):
                a = a[:, None, ...]         # [B,1,nx,ny,nz]
        elif a.ndim == 5:
            pass
        else:
            raise ValueError(f"Unsupported grid tensor shape {a.shape}; expected 2–5 dims.")
        return a

    def _coerce_invar_grid(self, invar: Mapping[str, ArrayLike]) -> Dict[str, np.ndarray]:
        coerced: Dict[str, np.ndarray] = {}
        for k in self.input_keys:
            if k not in invar:
                raise KeyError(f"Missing required input key '{k}' in invar.")
            coerced[k] = self._ensure_bc_grid(self._to_numpy(invar[k]))

        first = next(iter(coerced.values()))
        B = first.shape[0]
        grid_shape = first.shape[2:]
        ndim = first.ndim
        for k, v in coerced.items():
            if v.ndim != ndim or v.shape[0] != B or v.shape[2:] != grid_shape:
                raise ValueError(
                    f"Input '{k}' has shape {v.shape} incompatible with batch {B} and grid {grid_shape}."
                )
        return coerced

    def _prepare_invar_grid(self, invar: Mapping[str, np.ndarray]) -> Dict[str, Tensor]:
        grid_tensors: Dict[str, Tensor] = {}
        for k, v in invar.items():
            t = torch.from_numpy(v).to(self.device, dtype=self.dtype)
            t.requires_grad_(self.requires_grad)
            grid_tensors[k] = t
        return grid_tensors

    def _forward(self, invar_t: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        if self.requires_grad:
            return self.forward_grad(invar_t)
        with torch.inference_mode():
            return self.model(invar_t)  # same as forward_nograd, inlined for speed

    def _pred_to_numpy(self, pred_out: Mapping[str, Tensor]) -> Dict[str, np.ndarray]:
        pred_np: Dict[str, np.ndarray] = {}
        for k in self.output_names:
            if k not in pred_out:
                raise KeyError(f"Model did not return expected output key '{k}'. Got {list(pred_out.keys())}")
            v = pred_out[k]
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Model output '{k}' must be a torch.Tensor, got {type(v)}")
            arr = v.detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr[:, None, ...]  # [B,nx,nz] -> [B,1,nx,nz]
            elif arr.ndim == 4 and arr.shape[1] not in (1, 2, 3, 4):
                arr = arr[:, None, ...]  # [B,nx,ny,nz] -> [B,1,nx,ny,nz]
            if arr.ndim not in (4, 5):
                raise ValueError(f"Output '{k}' has invalid shape {arr.shape}; expected 4D or 5D.")
            if k in self.unscale_fns:
                arr = self.unscale_fns[k](arr)
            pred_np[k] = arr
        return pred_np

    def run(
        self,
        invar: Mapping[str, ArrayLike],
        name: str = "infer",
        step: int = 0,
        writer=None,
        save_filetypes: Tuple[str, ...] = (".vti",),
        batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        invar_np = self._coerce_invar_grid(invar)
        total_batch = next(iter(invar_np.values())).shape[0]

        if batch_size is None:
            chunk_size = total_batch
        else:
            chunk_size = int(batch_size)
            if chunk_size <= 0:
                raise ValueError(f"batch_size must be positive when provided; got {batch_size}.")
            chunk_size = min(chunk_size, total_batch)

        if chunk_size == total_batch:
            invar_t = self._prepare_invar_grid(invar_np)
            pred_out = self._forward(invar_t)
            pred_np = self._pred_to_numpy(pred_out)
        else:
            chunked_preds: Dict[str, List[np.ndarray]] = {k: [] for k in self.output_names}
            for start in range(0, total_batch, chunk_size):
                stop = min(start + chunk_size, total_batch)
                chunk_np = {k: v[start:stop] for k, v in invar_np.items()}
                invar_t = self._prepare_invar_grid(chunk_np)
                pred_out = self._forward(invar_t)
                pred_chunk = self._pred_to_numpy(pred_out)
                for k in self.output_names:
                    chunked_preds[k].append(pred_chunk[k])

                del invar_t, pred_out, pred_chunk

            pred_np = {k: np.concatenate(v, axis=0) for k, v in chunked_preds.items()}

        self.save_results(name=name, results_dir=str(self.results_dir),
                          writer=writer, save_filetypes=save_filetypes, step=step, pred_np=pred_np)
        return pred_np

    def save_results(
        self,
        name: str,
        results_dir: Union[str, os.PathLike],
        writer=None,
        save_filetypes: Tuple[str, ...] = (".vti",),
        step: int = 0,
        pred_np: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        """
        Write one .vti per batch item using VTKUniformGrid with correct flattening:
        make x the fastest-varying index by transposing [1,C,nx,ny,nz] -> [1,C,nz,ny,nx] before ravel.
        """
        if not pred_np:
            return

        first = next(iter(pred_np.values()))
        if first.ndim == 4:
            B, C, nx, nz = first.shape
            ny = 1
            pred_np_3d = {k: v[:, :, :, None, :] for k, v in pred_np.items()}
        elif first.ndim == 5:
            B, C, nx, ny, nz = first.shape
            pred_np_3d = dict(pred_np)
        else:
            raise ValueError(f"Expected [B,C,nx,nz] or [B,C,nx,ny,nz], got {first.shape}")

        # Optional mask over [B, nx, ny, nz]
        mask: Optional[np.ndarray] = None
        if self.mask_fn is not None:
            # Build in x,y,z order: with indexing="ij" the outputs are (nx,ny,nz),
            # which is what the shape checks below and the [B,C,nx,ny,nz]
            # prediction expect. Building it (nz,ny,nx) made both branches raise
            # whenever nx != nz, and silently transposed xx/zz when they matched.
            xx, yy, zz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
            base = np.asarray(self.mask_fn(xx, yy, zz), dtype=bool)  # mask_fn(x,y,z)
            if base.shape == (nx, ny, nz):
                mask = np.broadcast_to(base[None, ...], (B, nx, ny, nz))
            elif base.shape == (B, nx, ny, nz):
                mask = base
            else:
                raise ValueError("mask_fn must return [nx,ny,nz] or [B,nx,ny,nz].")

        # Bounds / spacing
        if self.bounds is None:
            bounds = [[0, nx - 1], [0, ny - 1], [0, nz - 1]]
        else:
            bounds = self.bounds
        npoints = [nx, ny, nz]

        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for b in range(B):
            # Prepare flat point-data: dict[k] = [N, C], with N = nx*ny*nz, x fastest.
            flat_vars: Dict[str, np.ndarray] = {}
            for k, arr in pred_np_3d.items():
                arrb = arr[b : b + 1, ...]              # [1,C,nx,ny,nz]
                if mask is not None:
                    m = mask[b]                          # [nx,ny,nz]
                    arrb = arrb.copy()
                    arrb[:, :, m] = self.mask_value

                # --- critical ordering fix ---
                # transpose so that after C-order ravel, x is fastest, then y, then z
                arrb = np.transpose(arrb, (0, 1, 4, 3, 2))  # [1,C,nz,ny,nx]
                arrb = np.ascontiguousarray(arrb)

                N = nx * ny * nz
                flat = arrb.reshape(1, C, N).transpose(0, 2, 1)[0]  # [N, C]
                flat_vars[k] = flat

            stem = f"{name}_{b}" if not self.log_iter else f"{name}_{b}_{step:06d}"
            vtk = VTKUniformGrid(
                bounds=bounds, # pyright: ignore[reportArgumentType]
                npoints=npoints,
                export_map={k: [k] for k in flat_vars.keys()},
                file_name=stem,
                file_dir=str(out_dir),
            )
            vtk.var_to_vtk(flat_vars)
