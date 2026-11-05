import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# The difference column is drawn with "bwr" on symmetric error limits. The field
# contour (T = T_s, alpha = 0.5) then usually falls inside those limits, so the
# guard passed and traced an isoline of the *error* field -- which the legend
# labelled "{var} = {contour_val}", reading as the melt boundary. The contour
# belongs on the true/pred panels only.
_ERROR_CMAP = "bwr"
from matplotlib.colors import Normalize
from physicsnemo.sym.utils.io.plotter import GridValidatorPlotter

class CustomSlicePlotter(GridValidatorPlotter):
    """
    Returns (figure, name) tuples so NeMo saves into .../validators/<val_name>/step_<global_step>/.
    Produces three figures: XY@z=..., YZ@x=..., XZ@y=..., each with [true | pred | diff].
    """
    def __init__(self, xy_idx=0, yz_idx=0, xz_idx=0, variable="T", variables=None, contours=None, n_examples=1, debug=False, ref_scales=None):
        super().__init__(n_examples=n_examples)
        # Titles and filenames used to denormalize P and V_scan with
        # literal *10 and *100, which invert P/ref['P'] and V_scan/ref['V'] only
        # for ref P=10, V=0.1. Any other scale set (the Abaqus config runs
        # P=10000) got a label off by orders of magnitude. Defaults reproduce the
        # old numbers exactly for every config that ships those references.
        ref_scales = ref_scales or {}
        self.p_ref = float(ref_scales.get("P", 10.0))
        self.v_ref = float(ref_scales.get("V", 0.1))
        self.xy_idx = int(xy_idx)
        self.yz_idx = int(yz_idx)
        self.xz_idx = int(xz_idx)
        # Allow either a single variable or a list of variables to plot.
        if variables is None:
            self.variables = [variable]
        else:
            self.variables = list(variables)
        # Optional dict mapping variable name -> contour level
        self.contours = contours or {}
        self.debug = debug
        self.velocity_keys = ("u", "v", "w")
        self.velocity_condition_key = "fl"
        self.velocity_condition_threshold = 0.3
        self.velocity_cmap = "cividis_r"
        self.velocity_scale = None

    @staticmethod
    def _sample_scalar_param(param_arr, batch_idx):
        arr = np.asarray(param_arr)
        if arr.ndim < 2:
            return 0.0
        return float(arr[batch_idx, 0].reshape(-1)[0])

    def _label_power(self, param_arr, batch_idx) -> int:
        """Laser power in W for a title/filename, undoing P/ref['P']."""
        return int(round(self._sample_scalar_param(param_arr, batch_idx) * self.p_ref))

    def _label_speed(self, param_arr, batch_idx) -> int:
        """Scan speed in mm/s for a title/filename, undoing V_scan/ref['V']."""
        return int(round(self._sample_scalar_param(param_arr, batch_idx) * self.v_ref * 1000.0))

    def _velocity_triplet_available(self, true_outvar, pred_outvar):
        required_keys = all(
            key in self.variables and key in true_outvar and key in pred_outvar
            for key in self.velocity_keys
        )
        if not required_keys:
            return False
        # Velocity quiver plotting is currently implemented for 3D volumes only.
        return (
            true_outvar["u"].ndim == 5
            and true_outvar["v"].ndim == 5
            and true_outvar["w"].ndim == 5
            and pred_outvar["u"].ndim == 5
            and pred_outvar["v"].ndim == 5
            and pred_outvar["w"].ndim == 5
        )

    def _condition_volume(self, condition_dict, batch_idx, shape):
        condition = condition_dict.get(self.velocity_condition_key)
        if condition is None:
            return np.ones(shape, dtype=np.float32)
        if condition.ndim != 5 or condition.shape[1] != 1:
            return np.ones(shape, dtype=np.float32)
        if condition.shape[2:] != shape:
            return np.ones(shape, dtype=np.float32)
        return condition[batch_idx, 0]

    def _extract_velocity_plane(
        self,
        plane,
        index,
        u_true,
        v_true,
        w_true,
        u_pred,
        v_pred,
        w_pred,
        fl_true,
        fl_pred,
    ):
        threshold = self.velocity_condition_threshold
        if plane == "xy":
            x, y = np.meshgrid(np.arange(u_true.shape[0]), np.arange(u_true.shape[1]), indexing="ij")
            true_u = u_true[:, :, index]
            true_v = v_true[:, :, index]
            pred_u = u_pred[:, :, index]
            pred_v = v_pred[:, :, index]
            mask_true = fl_true[:, :, index] >= threshold
            mask_pred = fl_pred[:, :, index] >= threshold
            xlabel, ylabel = "x index", "y index"
            plane_title = f"XY @ z index {index}"
            name = f"velocity_xy_z{index}"
            return x, y, true_u, true_v, pred_u, pred_v, mask_true, mask_pred, xlabel, ylabel, plane_title, name

        if plane == "yz":
            x, y = np.meshgrid(np.arange(v_true.shape[1]), np.arange(v_true.shape[2]), indexing="ij")
            true_u = v_true[index, :, :]
            true_v = w_true[index, :, :]
            pred_u = v_pred[index, :, :]
            pred_v = w_pred[index, :, :]
            mask_true = fl_true[index, :, :] >= threshold
            mask_pred = fl_pred[index, :, :] >= threshold
            xlabel, ylabel = "y index", "z index"
            plane_title = f"YZ @ x index {index}"
            name = f"velocity_yz_x{index}"
            return x, y, true_u, true_v, pred_u, pred_v, mask_true, mask_pred, xlabel, ylabel, plane_title, name

        x, y = np.meshgrid(np.arange(u_true.shape[0]), np.arange(u_true.shape[2]), indexing="ij")
        true_u = u_true[:, index, :]
        true_v = w_true[:, index, :]
        pred_u = u_pred[:, index, :]
        pred_v = w_pred[:, index, :]
        mask_true = fl_true[:, index, :] >= threshold
        mask_pred = fl_pred[:, index, :] >= threshold
        xlabel, ylabel = "x index", "z index"
        plane_title = f"XZ @ y index {index}"
        name = f"velocity_xz_y{index}"
        return x, y, true_u, true_v, pred_u, pred_v, mask_true, mask_pred, xlabel, ylabel, plane_title, name

    def _crop_limits(self, x, y, mask):
        if np.any(mask):
            x_sel = x[mask]
            y_sel = y[mask]
            x_min = float(np.min(x_sel))
            x_max = float(np.max(x_sel))
            y_min = float(np.min(y_sel))
            y_max = float(np.max(y_sel))
            if np.isclose(x_min, x_max):
                x_min -= 0.5
                x_max += 0.5
            if np.isclose(y_min, y_max):
                y_min -= 0.5
                y_max += 0.5
            return x_min, x_max, y_min, y_max
        return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))

    def _plot_quiver_panel(self, fig, ax, x, y, u, v, magnitude, mask, norm, title, xlabel, ylabel):
        if np.any(mask):
            quiver = ax.quiver(
                x[mask],
                y[mask],
                u[mask],
                v[mask],
                magnitude[mask],
                cmap=self.velocity_cmap,
                angles="xy",
                scale_units="xy",
                scale=self.velocity_scale,
                norm=norm,
            )
            colorbar = fig.colorbar(quiver, ax=ax)
            colorbar.set_label("|U|")
        else:
            ax.text(
                0.5,
                0.5,
                f"No points satisfy {self.velocity_condition_key} >= {self.velocity_condition_threshold}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        x_min, x_max, y_min, y_max = self._crop_limits(x, y, mask)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal")

    def _plot_velocity_triplet(self, invar, true_outvar, pred_outvar):
        figs = []
        u_true = true_outvar["u"]
        v_true = true_outvar["v"]
        w_true = true_outvar["w"]
        u_pred = pred_outvar["u"]
        v_pred = pred_outvar["v"]
        w_pred = pred_outvar["w"]
        for key, arr in (("u", u_true), ("v", v_true), ("w", w_true)):
            assert arr.ndim == 5 and arr.shape[1] == 1, f"{key}: Expected [B,1,Nx,Ny,Nz], got {arr.shape}"
        for key, arr in (("u", u_pred), ("v", v_pred), ("w", w_pred)):
            assert arr.shape == u_true.shape, f"{key}: true/pred shape mismatch"

        B, _, Nx, Ny, Nz = u_true.shape
        n = min(self.n_examples, B)
        Pow = invar["P"]
        V_scan = invar.get("V_scan", np.zeros_like(Pow))

        for b in range(n):
            u_true_b = u_true[b, 0]
            v_true_b = v_true[b, 0]
            w_true_b = w_true[b, 0]
            u_pred_b = u_pred[b, 0]
            v_pred_b = v_pred[b, 0]
            w_pred_b = w_pred[b, 0]
            fl_true = self._condition_volume(true_outvar, b, (Nx, Ny, Nz))
            fl_pred = self._condition_volume(pred_outvar, b, (Nx, Ny, Nz))

            V_scan_b = self._label_speed(V_scan, b)
            Pow_b = self._label_power(Pow, b)

            plane_specs = [
                ("xy", int(np.clip(self.xy_idx, 0, Nz - 1))),
                ("yz", int(np.clip(self.yz_idx, 0, Nx - 1))),
                ("xz", int(np.clip(self.xz_idx, 0, Ny - 1))),
            ]
            for plane, index in plane_specs:
                (
                    x,
                    y,
                    true_u,
                    true_v,
                    pred_u,
                    pred_v,
                    mask_true,
                    mask_pred,
                    xlabel,
                    ylabel,
                    plane_title,
                    base_name,
                ) = self._extract_velocity_plane(
                    plane=plane,
                    index=index,
                    u_true=u_true_b,
                    v_true=v_true_b,
                    w_true=w_true_b,
                    u_pred=u_pred_b,
                    v_pred=v_pred_b,
                    w_pred=w_pred_b,
                    fl_true=fl_true,
                    fl_pred=fl_pred,
                )
                mask_diff = mask_true & mask_pred
                diff_u = pred_u - true_u
                diff_v = pred_v - true_v

                true_mag = np.sqrt(true_u**2 + true_v**2)
                pred_mag = np.sqrt(pred_u**2 + pred_v**2)
                diff_mag = np.sqrt(diff_u**2 + diff_v**2)

                mags = []
                if np.any(mask_true):
                    mags.append(float(np.nanmax(true_mag[mask_true])))
                if np.any(mask_pred):
                    mags.append(float(np.nanmax(pred_mag[mask_pred])))
                if np.any(mask_diff):
                    mags.append(float(np.nanmax(diff_mag[mask_diff])))
                vmax = max(0.25, max(mags) if mags else 0.25)
                norm = Normalize(vmin=0.0, vmax=vmax)

                fig = plt.figure(figsize=(15, 4), dpi=110, constrained_layout=True)
                panels = [
                    (
                        true_u,
                        true_v,
                        true_mag,
                        mask_true,
                        f"velocity P:{Pow_b} V:{V_scan_b} (true)",
                    ),
                    (
                        pred_u,
                        pred_v,
                        pred_mag,
                        mask_pred,
                        f"velocity P:{Pow_b} V:{V_scan_b} (pred)",
                    ),
                    (
                        diff_u,
                        diff_v,
                        diff_mag,
                        mask_diff,
                        f"velocity P:{Pow_b} V:{V_scan_b} error",
                    ),
                ]
                for col, (u_comp, v_comp, magnitude, mask, title) in enumerate(panels):
                    ax = fig.add_subplot(1, 3, col + 1)
                    self._plot_quiver_panel(
                        fig=fig,
                        ax=ax,
                        x=x,
                        y=y,
                        u=u_comp,
                        v=v_comp,
                        magnitude=magnitude,
                        mask=mask,
                        norm=norm,
                        title=f"{plane_title} - {title}",
                        xlabel=xlabel,
                        ylabel=ylabel,
                    )
                figs.append((fig, f"{base_name}_P{Pow_b}_V{V_scan_b}_b{b}"))
                plt.close(fig)
        return figs

    def _plot_scalar_variable(self, var, invar, true_outvar, pred_outvar):
        figs = []
        v_true = true_outvar[var]   # np.ndarray
        v_pred = pred_outvar[var]
        assert v_pred.shape == v_true.shape, f"{var}: true/pred shape mismatch for {var}"
        contour_val = self.contours.get(var, None)

        if v_true.ndim == 4 and v_true.shape[1] == 1:
            # 2D mode tensors: [B,1,Nx,Nz]
            B, _, Nx, Nz = v_true.shape
            n = min(self.n_examples, B)
            Pow = invar["P"]
            V_scan = invar.get("V_scan", np.zeros_like(Pow))
            xz_extent = (0, Nx - 1, 0, Nz - 1)

            for b in range(n):
                T = v_true[b, 0]  # (Nx,Nz)
                P = v_pred[b, 0]
                V_scan_b = self._label_speed(V_scan, b)
                Pow_b = self._label_power(Pow, b)

                SMALL = 1e-6
                T_avg = np.mean(np.abs(T)) + SMALL
                D = (T - P) / T_avg  # relative diff

                fig = plt.figure(figsize=(15, 4), dpi=110, constrained_layout=True)
                for col, (arr, title, cmap) in enumerate([
                    (T, f"{var} P:{Pow_b} V:{V_scan_b} (true)", "inferno"),
                    (P, f"{var} P:{Pow_b} V:{V_scan_b} (pred)", "inferno"),
                    (D, f"{var} P:{Pow_b} V:{V_scan_b} rel error", "bwr"),
                ]):
                    ax = fig.add_subplot(1, 3, col + 1)
                    if cmap == _ERROR_CMAP:
                        vmax_ = float(np.nanmax(np.abs(arr)))
                        vmin_ = -vmax_
                    else:
                        arr_min = float(np.nanmin(arr))
                        arr_max = float(np.nanmax(arr))
                        vmin_, vmax_ = arr_min, arr_max
                    im = ax.imshow(arr.T, origin="lower", extent=xz_extent, cmap=cmap, vmin=vmin_, vmax=vmax_, aspect="equal")
                    if contour_val is not None and cmap != _ERROR_CMAP and vmin_ <= contour_val <= vmax_:
                        ax.contour(arr.T, levels=[contour_val], colors="k", linewidths=1.0)
                        ax.legend([Line2D([0], [0], color="k", lw=1)], [f"{var} = {contour_val}"], loc="lower left")
                    ax.set_xlabel("x index")
                    ax.set_ylabel("z index")
                    ax.set_title(f"2D XZ - {title}")
                    cb = fig.colorbar(im, ax=ax)
                    cb.set_label(title)
                figs.append((fig, f"{var}_xz_2d_P{Pow_b}_V{V_scan_b}_b{b}"))
                plt.close(fig)
            return figs

        assert v_true.ndim == 5 and v_true.shape[1] == 1, f"{var}: Expected [B,1,Nx,Ny,Nz] or [B,1,Nx,Nz], got {v_true.shape}"
        B, _, Nx, Ny, Nz = v_true.shape
        n = min(self.n_examples, B)
        Pow = invar["P"]
        V_scan = invar.get("V_scan", np.zeros_like(Pow))

        # Use index-based extents for plotting (0..N-1 on each axis)
        xy_extent = (0, Nx - 1, 0, Ny - 1)
        yz_extent = (0, Ny - 1, 0, Nz - 1)
        xz_extent = (0, Nx - 1, 0, Nz - 1)

        if self.debug:
            for k in range(Nz):
                T = v_true[0, 0]  # (Nx,Ny,Nz)
                P = v_pred[0, 0]

                # ---------- XY slice (z = k) ----------
                fig = plt.figure(figsize=(15, 4), dpi=110, constrained_layout=True)
                # compute vmin/vmax per plotted region instead of global min/max
                for col, (arr, title, cmap) in enumerate([
                    (T[:, :, k],  f"{var} (true)", "inferno"),
                    (P[:, :, k],  f"{var} (pred)", "inferno"),
                ]):
                    ax = fig.add_subplot(1, 3, col + 1)
                    arr_min = float(np.nanmin(arr))
                    arr_max = float(np.nanmax(arr))
                    vmin_, vmax_ = arr_min, arr_max
                    im = ax.imshow(arr.T, origin="lower", extent=xy_extent, cmap=cmap, vmin=vmin_, vmax=vmax_, aspect="equal")
                    if contour_val is not None and arr_min <= contour_val <= arr_max:
                        ax.contour(arr.T, levels=[contour_val], colors="k", linewidths=1.0)
                        ax.legend([Line2D([0], [0], color="k", lw=1)], [f"{var} = {contour_val}"], loc="lower left")
                    ax.set_xlabel("x index")
                    ax.set_ylabel("y index")
                    ax.set_title(f"XY @ z index {k} - {title}")
                    fig.colorbar(im, ax=ax)
                figs.append((fig, f"{var}_xy_z{k}_b{0}"))
                plt.close(fig)
            return figs

        for b in range(n):  # n_examples
            T = v_true[b, 0]  # (Nx,Ny,Nz)
            P = v_pred[b, 0]
            V_scan_b = self._label_speed(V_scan, b)
            Pow_b = self._label_power(Pow, b)

            SMALL = 1e-6
            # Use average of T as denominator for relative error
            T_avg = np.mean(np.abs(T)) + SMALL
            D = (T - P) / T_avg  # relative diff

            # ---------- XY slice (z = k) ----------
            k = int(np.clip(self.xy_idx, 0, Nz - 1))

            fig = plt.figure(figsize=(15, 4), dpi=110, constrained_layout=True)
            for col, (arr, title, cmap) in enumerate([
                (T[:, :, k],  f"{var} P:{Pow_b} V:{V_scan_b} (true)", "inferno"),
                (P[:, :, k],  f"{var} P:{Pow_b} V:{V_scan_b} (pred)", "inferno"),
                (D[:, :, k],  f"{var} P:{Pow_b} V:{V_scan_b} rel error", "bwr"),
            ]):
                ax = fig.add_subplot(1, 3, col + 1)
                # use symmetric limits for difference colormap 'bwr'
                if cmap == _ERROR_CMAP:
                    vmax_ = float(np.nanmax(np.abs(arr)))
                    vmin_ = -vmax_
                else:
                    arr_min = float(np.nanmin(arr))
                    arr_max = float(np.nanmax(arr))
                    vmin_, vmax_ = arr_min, arr_max
                im = ax.imshow(arr.T, origin="lower", extent=xy_extent, cmap=cmap, vmin=vmin_, vmax=vmax_, aspect="equal")
                if contour_val is not None and cmap != _ERROR_CMAP and vmin_ <= contour_val <= vmax_:
                    ax.contour(arr.T, levels=[contour_val], colors="k", linewidths=1.0)
                    ax.legend([Line2D([0], [0], color="k", lw=1)], [f"{var} = {contour_val}"], loc="lower left")
                ax.set_xlabel("x index")
                ax.set_ylabel("y index")
                ax.set_title(f"XY @ z index {k} - {title}")
                fig.colorbar(im, ax=ax)
            figs.append((fig, f"{var}_xy_z{k}_P{Pow_b}_V{V_scan_b}_b{b}"))
            plt.close(fig)

            # ---------- YZ slice (x = i) ----------
            i = int(np.clip(self.yz_idx, 0, Nx - 1))

            fig = plt.figure(figsize=(15, 4), dpi=110, constrained_layout=True)
            for col, (arr, title, cmap) in enumerate([
                (T[i, :, :],  f"{var} P:{Pow_b} V:{V_scan_b} (true)", "inferno"),
                (P[i, :, :],  f"{var} P:{Pow_b} V:{V_scan_b} (pred)", "inferno"),
                (D[i, :, :],  f"{var} P:{Pow_b} V:{V_scan_b} rel error", "bwr"),
            ]):
                ax = fig.add_subplot(1, 3, col + 1)
                if cmap == _ERROR_CMAP:
                    vmax_ = float(np.nanmax(np.abs(arr)))
                    vmin_ = -vmax_
                else:
                    arr_min = float(np.nanmin(arr))
                    arr_max = float(np.nanmax(arr))
                    vmin_, vmax_ = arr_min, arr_max
                im = ax.imshow(arr.T, origin="lower", extent=yz_extent, cmap=cmap, vmin=vmin_, vmax=vmax_, aspect="equal")
                if contour_val is not None and cmap != _ERROR_CMAP and vmin_ <= contour_val <= vmax_:
                    ax.contour(arr.T, levels=[contour_val], colors="k", linewidths=1.0)
                    ax.legend([Line2D([0], [0], color="k", lw=1)], [f"{var} = {contour_val}"], loc="lower left")
                ax.set_xlabel("y index")
                ax.set_ylabel("z index")
                ax.set_title(f"YZ @ x index {i}- {title}")
                fig.colorbar(im, ax=ax)
            figs.append((fig, f"{var}_yz_x{i}_P{Pow_b}_V{V_scan_b}_b{b}"))
            plt.close(fig)

            # ---------- XZ slice (y = j) ----------
            j = int(np.clip(self.xz_idx, 0, Ny - 1))

            fig = plt.figure(figsize=(15, 4), dpi=110, constrained_layout=True)
            for col, (arr, title, cmap) in enumerate([
                (T[:, j, :],  f"{var} P:{Pow_b} V:{V_scan_b} (true)", "inferno"),
                (P[:, j, :],  f"{var} P:{Pow_b} V:{V_scan_b} (pred)", "inferno"),
                (D[:, j, :],  f"{var} P:{Pow_b} V:{V_scan_b} rel error", "bwr"),
            ]):
                ax = fig.add_subplot(1, 3, col + 1)
                if cmap == _ERROR_CMAP:
                    vmax_ = float(np.nanmax(np.abs(arr)))
                    vmin_ = -vmax_
                else:
                    arr_min = float(np.nanmin(arr))
                    arr_max = float(np.nanmax(arr))
                    vmin_, vmax_ = arr_min, arr_max
                im = ax.imshow(arr.T, origin="lower", extent=xz_extent, cmap=cmap, vmin=vmin_, vmax=vmax_, aspect="equal")
                if contour_val is not None and cmap != _ERROR_CMAP and vmin_ <= contour_val <= vmax_:
                    ax.contour(arr.T, levels=[contour_val], colors="k", linewidths=1.0)
                    ax.legend([Line2D([0], [0], color="k", lw=1)], [f"{var} = {contour_val}"], loc="lower left")
                ax.set_xlabel("x index")
                ax.set_ylabel("z index")
                ax.set_title(f"XZ @ y index {j} - {title}")
                cb = fig.colorbar(im, ax=ax)
                cb.set_label(title)
            figs.append((fig, f"{var}_xz_y{j}_P{Pow_b}_V{V_scan_b}_b{b}"))
            plt.close(fig)
        return figs

    def __call__(self, invar, true_outvar, pred_outvar):
        # Expects channel-first tensors:
        # 3D: [B,1,Nx,Ny,Nz]
        # 2D: [B,1,Nx,Nz]
        figs = []
        if not self.variables:
            return figs

        velocity_available = self._velocity_triplet_available(true_outvar, pred_outvar)
        velocity_rendered = False
        for var in self.variables:
            if velocity_available and var in self.velocity_keys:
                if not velocity_rendered:
                    figs.extend(self._plot_velocity_triplet(invar, true_outvar, pred_outvar))
                    velocity_rendered = True
                continue
            figs.extend(self._plot_scalar_variable(var, invar, true_outvar, pred_outvar))
        return figs


class TemporalPatchPlotter(GridValidatorPlotter):
    """Visualization helper for masked next-step temporal patch training."""

    def __init__(self, variables=None, state_prev_suffix="_prev", n_examples=1):
        super().__init__(n_examples=n_examples)
        self.variables = list(variables or ["T", "alpha"])
        self.state_prev_suffix = state_prev_suffix

    @staticmethod
    def _mid_slice(arr):
        if arr.ndim == 4:
            z_idx = arr.shape[-1] // 2
            return arr[0, :, :, z_idx]
        if arr.ndim == 3:
            return arr[0]
        raise ValueError(f"Unsupported patch tensor shape: {arr.shape}")

    def __call__(self, invar, true_outvar, pred_outvar):
        figs = []
        batch_size = min(self.n_examples, next(iter(true_outvar.values())).shape[0])
        for batch_idx in range(batch_size):
            for variable in self.variables:
                if variable not in true_outvar or variable not in pred_outvar:
                    continue
                prev_key = f"{variable}{self.state_prev_suffix}"
                mask_key = f"{prev_key}_mask"
                prev_panel = invar.get(prev_key)
                mask_panel = invar.get(mask_key)
                truth_panel = true_outvar[variable]
                pred_panel = pred_outvar[variable]
                if prev_panel is None:
                    continue

                prev_img = self._mid_slice(np.asarray(prev_panel[batch_idx]))
                truth_img = self._mid_slice(np.asarray(truth_panel[batch_idx]))
                pred_img = self._mid_slice(np.asarray(pred_panel[batch_idx]))
                error_img = pred_img - truth_img
                mask_img = None
                if mask_panel is not None:
                    mask_img = self._mid_slice(np.asarray(mask_panel[batch_idx]))

                fig = plt.figure(figsize=(16, 3.5), dpi=110, constrained_layout=True)
                panels = [
                    (prev_img, f"{variable} prev", "inferno"),
                    (mask_img if mask_img is not None else np.zeros_like(prev_img), f"{variable} mask", "gray"),
                    (truth_img, f"{variable} next (true)", "inferno"),
                    (pred_img, f"{variable} next (pred)", "inferno"),
                    (error_img, f"{variable} error", "bwr"),
                ]
                for col, (panel, title, cmap) in enumerate(panels, start=1):
                    ax = fig.add_subplot(1, 5, col)
                    vmax = float(np.nanmax(np.abs(panel))) if cmap == "bwr" else None
                    vmin = -vmax if vmax is not None else None
                    im = ax.imshow(panel.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(title)
                    ax.set_xlabel("x index")
                    ax.set_ylabel("y index")
                    fig.colorbar(im, ax=ax)
                figs.append((fig, f"temporal_patch_{variable}_b{batch_idx}"))
                plt.close(fig)
        return figs
