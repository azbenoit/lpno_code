#!/usr/bin/env python3
"""Render and optionally run a rectangular, moving-laser FLOW-3D WELD case.

This deliberately uses only the Python standard library.  Input values and
derived values are SI; the renderer writes a complete FLOW-3D `prepin` file.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import config_schema
import laser_path
import resource_guards
from spool import default_spool_dir


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
TEMPLATE = ROOT_DIR / "template" / "prepin.rectangular_moving_laser.template"
RUNS_DIR = ROOT_DIR / "runs"
MATERIALS_DIR = ROOT_DIR / "materials"
DEFAULT_MATERIAL = "ti_imi318_mills"
DEFAULT_ENV = Path.home() / ".local/opt/flow3d/FLOW3D_WELD_2025R1u1/local/flow3dvars.sh"
DEFAULT_POST_HOME = Path.home() / ".local/opt/flow3d/FLOW3D_POST_2025R1"
LOCAL_LICENSE_ENV = ROOT_DIR / "license.env"
CASE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
TOKEN = re.compile(r"__[A-Z0-9_]+__")
MATERIAL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ConfigError(ValueError):
    """A JSON value is missing, invalid, or physically inconsistent."""


def number(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{path} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise ConfigError(f"{path} must be greater than zero")
    if nonnegative and result < 0:
        raise ConfigError(f"{path} must not be negative")
    return result


def integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def point(value: Any, path: str, default: list[float] | None = None) -> tuple[float, float, float]:
    if value is None and default is not None:
        value = default
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError(f"{path} must be a three-element [x, y, z] list in metres")
    return tuple(number(component, f"{path}[{index}]") for index, component in enumerate(value))  # type: ignore[return-value]


def bounds_pair(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{path} must be a two-element [min, max] list in metres")
    low = number(value[0], f"{path}[0]")
    high = number(value[1], f"{path}[1]")
    if high <= low:
        raise ConfigError(f"{path}[1] must be greater than {path}[0]")
    return low, high


def section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def fmt(value: float) -> str:
    """Stable Fortran-compatible number spelling."""
    return format(value, ".12g")


def fresnel_epsilon_from_absorptivity(absorptivity: float) -> float:
    """Invert FLOW-3D's simplified Fresnel A(0) for the coefficient EREFLCT (epsilon).

    The irefopt=3 model uses (calculator form, c=cos(theta)):
      A(theta)=1-0.5*[(1+(1-e*c)^2)/(1+(1+e*c)^2)+(e^2-2*e*c+2*c^2)/(e^2+2*e*c+2*c^2)]
    At normal incidence A0(e)=1-(1+(1-e)^2)/(1+(1+e)^2) is monotonic on [0,1] from 0
    to 0.8, so bisect for e given a material's normal-incidence absorptivity at the
    laser wavelength. The keyhole's multiple reflections then raise the effective
    absorption above this single-hit value.
    """
    if not 0.0 < absorptivity < 0.8:
        raise ConfigError("material optical.normal_absorptivity must be between 0 and 0.8 (simplified Fresnel range)")

    def a0(epsilon: float) -> float:
        return 1.0 - (1.0 + (1.0 - epsilon) ** 2) / (1.0 + (1.0 + epsilon) ** 2)

    low, high = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if a0(mid) < absorptivity:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
SPEED_OF_LIGHT = 299792458.0  # m/s


def fresnel_epsilon_from_resistivity(resistivity_ohm_m: float, wavelength_m: float) -> float:
    """Derive the Fresnel coefficient EREFLCT (epsilon) from electrical resistivity.

    Drude / Hagen-Rubens free-electron form:
      epsilon = sqrt(2 * eps0 * omega * rho),  omega = 2*pi*c / wavelength
    with rho the electrical resistivity (ohm.m, ideally the liquid-metal value near
    the melt-pool surface) and wavelength the laser wavelength. This is the
    physically derived route (vs anchoring to an assumed absorptivity). It is sound
    for transition-metal alloys (Ti, steels, Ni/Co-based) at ~1 um, but the
    DC-resistivity Drude form overestimates absorption for Cu/Al/noble metals at
    1 um (interband effects, omega*tau ~ 1) — prefer measured absorptivity there.
    """
    if resistivity_ohm_m <= 0 or wavelength_m <= 0:
        raise ConfigError("electrical_resistivity_ohm_m and wavelength_m must be positive")
    omega = 2.0 * math.pi * SPEED_OF_LIGHT / wavelength_m
    return math.sqrt(2.0 * VACUUM_PERMITTIVITY * omega * resistivity_ohm_m)


def parse_env_file(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE or export KEY=VALUE lines from a local env file."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ConfigError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ConfigError(f"{path}:{line_number}: empty environment key")
        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            value = value[1:-1]
        values[key] = value
    return values


def local_license_endpoint(path: Path = LOCAL_LICENSE_ENV) -> str:
    """Return the local FLOW-3D license endpoint without printing it."""
    for key in ("F3D_LICENSE", "F3DTKNUX_LICENSE_FILE", "LM_LICENSE_FILE"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    env_file = parse_env_file(path)
    for key in ("F3D_LICENSE", "F3DTKNUX_LICENSE_FILE", "LM_LICENSE_FILE"):
        value = env_file.get(key, "").strip()
        if value:
            return value
    return ""


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def material_path_from_spec(spec: Any, config_path: Path) -> Path:
    def by_name(name: str) -> Path:
        # Each material lives in its own directory: materials/<name>/<name>.json
        # (scalars + optical) alongside CSV tables for temperature-dependent values.
        exact = MATERIALS_DIR / name / f"{name}.json"
        if exact.is_file():
            return exact
        lowered = name.lower()
        for sub in MATERIALS_DIR.iterdir() if MATERIALS_DIR.is_dir() else []:
            if sub.is_dir() and sub.name.lower() == lowered:
                candidate = sub / f"{sub.name}.json"
                if candidate.is_file():
                    return candidate
        return exact

    if spec is None:
        return by_name(DEFAULT_MATERIAL)
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            raise ConfigError("material must not be empty")
        if "/" not in text and "\\" not in text and not text.endswith(".json"):
            if not MATERIAL_NAME.fullmatch(text):
                raise ConfigError("material name may only use letters, digits, _, - and .")
            return by_name(text)
        candidate = Path(text)
    elif isinstance(spec, dict):
        has_name = "name" in spec
        has_file = "file" in spec
        if has_name == has_file:
            raise ConfigError("material object must contain exactly one of name or file")
        if has_name:
            name = spec["name"]
            if not isinstance(name, str) or not MATERIAL_NAME.fullmatch(name):
                raise ConfigError("material.name may only use letters, digits, _, - and .")
            return by_name(name)
        file_name = spec["file"]
        if not isinstance(file_name, str) or not file_name.strip():
            raise ConfigError("material.file must be a non-empty path")
        candidate = Path(file_name)
    else:
        raise ConfigError("material must be a material name, material path, or object")

    if candidate.is_absolute():
        return candidate
    config_relative = (config_path.parent / candidate).resolve()
    if config_relative.is_file():
        return config_relative
    return (ROOT_DIR / candidate).resolve()


def table_rows(value: Any, path: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty list of [temperature_k, value] pairs")
    rows: list[tuple[float, float]] = []
    previous_temperature: float | None = None
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 2:
            raise ConfigError(f"{path}[{index}] must be [temperature_k, value]")
        temperature = number(row[0], f"{path}[{index}][0]", positive=True)
        table_value = number(row[1], f"{path}[{index}][1]", positive=True)
        if previous_temperature is not None and temperature <= previous_temperature:
            raise ConfigError(f"{path} temperatures must be strictly increasing")
        previous_temperature = temperature
        rows.append((temperature, table_value))
    return rows


def read_csv_table(path: Path, label: str) -> list[list[float]]:
    """Read a 2-column temperature_k,value CSV (with header) into [[T, v], ...]."""
    if not path.is_file():
        raise ConfigError(f"material table CSV not found: {path}")
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for line_number, record in enumerate(reader, start=2):
            if not record or all(not cell.strip() for cell in record):
                continue
            if len(record) < 2:
                raise ConfigError(f"{path}:{line_number}: expected temperature_k,value")
            try:
                rows.append([float(record[0]), float(record[1])])
            except ValueError as error:
                raise ConfigError(f"{path}:{line_number}: non-numeric value ({error})") from error
    if not rows:
        raise ConfigError(f"{label} CSV has no data rows: {path}")
    return rows


def render_table(name: str, rows: list[tuple[float, float]]) -> str:
    lines = [f"#{name}"]
    lines.extend(f"{fmt(temperature)}  {fmt(value)}" for temperature, value in rows)
    lines.append(f"#end {name}")
    return "\n".join(lines)


def boundary_replacements(raw: dict[str, Any]) -> dict[str, str]:
    boundary = raw.get("boundary_conditions", {})
    if boundary is None:
        boundary = {}
    if not isinstance(boundary, dict):
        raise ConfigError("boundary_conditions must be an object when provided")
    names = {
        "x_min": "__BC_XMIN__",
        "x_max": "__BC_XMAX__",
        "y_min": "__BC_YMIN__",
        "y_max": "__BC_YMAX__",
        "z_min": "__BC_ZMIN__",
        "z_max": "__BC_ZMAX__",
    }
    codes = {
        "wall": "2",
        "solid_wall": "2",
        "pressure": "5",
        "specified_pressure": "5",
    }
    replacements: dict[str, str] = {}
    for name, token in names.items():
        value = boundary.get(name, "pressure")
        if not isinstance(value, str):
            raise ConfigError(f"boundary_conditions.{name} must be wall or pressure")
        key = value.strip().lower()
        if key not in codes:
            raise ConfigError(f"boundary_conditions.{name} must be wall or pressure")
        replacements[token] = codes[key]
    return replacements


def fluid_initialization_block(
    *,
    mode: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    surface_z: float,
    cell: float,
    temperature: float,
    pressure: float | None = None,
) -> str:
    pressure_value = fmt(pressure) if pressure is not None else None
    if mode == "substrate_with_gas":
        p = pressure_value or "101300.0"
        return f""" &fl
    nfls=2,
    fluidRegionName(1)='Rectangular substrate',
    ifdis(1)=0,
    fxl(1)={fmt(x_min)},
    fxh(1)={fmt(x_max)},
    fyl(1)={fmt(y_min)},
    fyh(1)={fmt(y_max)},
    fzl(1)={fmt(z_min)},
    fzh(1)={fmt(surface_z)},
    if_flreg_enabled(1)=1,
    fluidRegionName(2)='Gas clearance above substrate',
    fioh(2)=0,
    ifdis(2)=0,
    fxl(2)={fmt(x_min)},
    fxh(2)={fmt(x_max)},
    fyl(2)={fmt(y_min)},
    fyh(2)={fmt(y_max)},
    fzl(2)={fmt(surface_z)},
    fzh(2)={fmt(z_max)},
    if_flreg_enabled(2)=1,
    pointerName(1)='Ambient void pointer',
    ivht(1)=1,
    xvr(1)={fmt(x_min + cell / 2.0)},
    yvr(1)={fmt(y_min + cell / 2.0)},
    zvr(1)={fmt(surface_z + max(cell / 2.0, (z_max - surface_z) / 2.0))},
    pvrd(1,1)={p},
    tvrd(1,1)={fmt(temperature)},
    if_pointer_enabled(1)=1,
    timvrd(1)=0.0,
    ndrop=0,
    pvoid={p},
    presi={p},
    iflinittyp=0,
    tempi={fmt(temperature)},
    tvoid={fmt(temperature)},
    igivht=1,
 /"""
    if mode == "filled_mesh":
        p = pressure_value or "100000.0"
        return f""" &fl
    nfls=1,
    fluidRegionName(1)='Filled rectangular metal block',
    freg(1)=1.0,
    if_flreg_enabled(1)=1,
    ndrop=0,
    pvoid={p},
    presi={p},
    iflinittyp=0,
    tempi={fmt(temperature)},
    tvoid={fmt(temperature)},
    igivht=1,
 /"""
    if mode == "legacy_halfspace":
        p = pressure_value or "1.0e+05"
        return f""" &fl
    nfls=1,
    fluidRegionName(1)='Region 1',
    freg(1)=1.0,
    fcc(1)=0.0,
    fcrxy(1)=0,
    fcx2(1)=0,
    fcx(1)=0,
    fcxy(1)=0,
    fcxz(1)=0,
    fcy2(1)=0,
    fcy(1)=0,
    fcyz(1)=0,
    fcz2(1)=0,
    fcz(1)=0,
    fcdum(1)=0.0,
    fzh(1)={fmt(surface_z)},
    fxtrfc(1)=0,
    fytrfc(1)=0,
    fztrfc(1)=0,
    fmagn(1)=1.0,
    fmagnx(1)=1.0,
    fmagny(1)=1.0,
    fmagnz(1)=1.0,
    frotx(1)=0.0,
    froty(1)=0.0,
    frotz(1)=0.0,
    ftrnx(1)=0,
    ftrny(1)=0,
    ftrnz(1)=0,
    oxgnreg(1)=0,
    oxgn2reg(1)=0.0,
    sc2reg(1)=0.0,
    ndrop=0,
    ui=0,
    vi=0,
    wi=0,
    vlin=0,
    segi=0,
    pvoid={p},
    presi={p},
    rpsi=0,
    cnci=0.0,
    rsati=0.0,
    chrgi=0,
    valcgx=0.02,
    tempi={fmt(temperature)},
    tvoid={fmt(temperature)},
    airi=0.0,
    sc2i=0.0,
    udeep=0,
    vdeep=0,
    wdeep=0,
    oxgni=0,
    oxgn2i=0.0,
 /"""
    raise ConfigError("geometry.initialization must be substrate_with_gas, filled_mesh, or legacy_halfspace")


def solver_profile_replacements(profile: str) -> dict[str, str]:
    """Return FLOW-3D namelist defaults for the requested rendering profile."""
    if profile == "standard":
        return {
            "__FCLEAN__": "0.05",
            "__XPUT_LEGACY_LINES__": "",
            "__IFDYNCONV__": "1",
            "__GRAVITY_Z__": "-9.81",
            "__HFLV1__": "1000.0",
            "__GAMMA__": "1.67",
            "__PROPS_LEGACY_PREFIX__": "",
            "__PROPS_LEGACY_FLUID2__": "",
            "__CV2__": "449.0",
            "__RSIZE__": "0.1",
            "__RSIZE_COND_LINE__": "",
            "__TVEXP__": "0.0000494786",
            "__PV0__": "101300.0",
            "__SIGMA__": "__MATERIAL_SIGMA__",
            "__PROPS_LEGACY_GAS__": "",
            "__P_REFERENCE__": "101325.0",
            "__T_REFERENCE__": "0.0",
            "__WELD_LEGACY_PREFIX__": "",
            "__WELD_ILBEAM_LINE__": "    ilbeam=__LASER_RADIATION_FLAG__,",
            "__WELD_LEGACY_AFTER_REFLECTION__": "",
            "__WELD_ABSORPTIVITY_FLAG_LINE__": "    icbmengy=__LASER_ABSORPTIVITY_FLAG__,",
            "__WELD_LEGACY_ENERGY_LINES__": "",
            "__WELD_LEGACY_VAPOR_LINES__": "",
            "__WELD_LEGACY_RECOIL_LINES__": "",
            "__WELD_WIDTH_LINES__": "",
            "__WELD_HFLBM_LINE__": "",
            "__WELD_SPIN_LINES__": "",
            "__BC_ZMAX_LINES__": "\n".join(
                [
                    "    pbct(1,6)=101300.0,",
                    "    tbct(1,6)=__BOUNDARY_TEMPERATURE_K__,",
                ]
            ),
        }
    if profile in ("legacy_case_fine_1", "keyhole"):
        legacy = {
            "__FCLEAN__": "0.1",
            "__XPUT_LEGACY_LINES__": "\n".join(
                [
                    "    fclean2=0.0,",
                    "    epsadj=1.0,",
                    "    omega_swat=1.0,",
                    "    alphamb=0.25,",
                    "    epsels=1.0,",
                    "    epsvis=1.0,",
                    "    crho=0.0,",
                    "    accimp=1.0,",
                    "    btmshr=0.0026,",
                    "    wndshr=0.003,",
                    "    rmvisz=1.5,",
                    "    pltfrc=0.1,",
                    "    pltafrc=0.01,",
                    "    tpltd(1)=1,",
                    "    tapltd(1)=5e-06,",
                ]
            ),
            "__IFDYNCONV__": "0",
            "__GRAVITY_Z__": "-9.8",
            "__HFLV1__": "50",
            "__GAMMA__": "1.4",
            "__PROPS_LEGACY_PREFIX__": "\n".join(
                [
                    "    rfnc=0,",
                    "    cvnc=0,",
                    "    dfcd=0.5,",
                    "    rzmlt=1.0,",
                    "    f1mx=1.0,",
                    "    gasden=1.1614,",
                ]
            ),
            "__PROPS_LEGACY_FLUID2__": "    fluid2='Air at 100kPa, 300K',",
            "__CV2__": "718",
            "__RSIZE__": "1",
            "__RSIZE_COND_LINE__": "    rsize_cond=1,",
            "__TVEXP__": "5.989e-05",
            "__PV0__": "50650",
            "__SIGMA__": "0",
            "__PROPS_LEGACY_GAS__": "\n".join(
                [
                    "    sigma_df=0,",
                    "    csigma=0,",
                    "    mu2=1.846e-05,",
                    "    rhof2=1.1614,",
                    "    thc2=0.0263,",
                    "    rf2=287,",
                    "    mus=3.25,",
                ]
            ),
            "__P_REFERENCE__": "101300",
            "__T_REFERENCE__": "273.15",
            "__WELD_LEGACY_PREFIX__": "\n".join(
                [
                    "    dpenlbm=0,",
                    "    ipow_corr=1,",
                ]
            ),
            "__WELD_ILBEAM_LINE__": "",
            "__WELD_LEGACY_AFTER_REFLECTION__": "\n".join(
                [
                    "    rfplume=0,",
                    "    aplume=0.0,",
                    "    acondens=1.0,",
                ]
            ),
            "__WELD_ABSORPTIVITY_FLAG_LINE__": "",
            "__WELD_LEGACY_ENERGY_LINES__": "\n".join(
                [
                    "    cangfs=0,",
                    "    cbmengpt=1.0,",
                    "    cbmengy2=1.0,",
                    "    cbmengpt2=1.0,",
                    "    ereflct2=0.25,",
                    "    constfresnel=0.0,",
                    "    constfresnel2=0.0,",
                ]
            ),
            "__WELD_LEGACY_VAPOR_LINES__": "    pvprmx=1.0e10,",
            "__WELD_LEGACY_RECOIL_LINES__": "\n".join(
                [
                    "    rxvpr=0.0,",
                    "    ryvpr=0.0,",
                    "    rzvpr=1.0,",
                    "    rbeamvapor=0,",
                    "    vprmag=1.0,",
                ]
            ),
            "__WELD_WIDTH_LINES__": "\n".join(
                [
                    "    xwidth0(1)=0.0002,",
                    "    ywidth0(1)=0.0002,",
                    "    xwidthf(1)=0.0001,",
                    "    ywidthf(1)=0.0001,",
                ]
            ),
            "__WELD_HFLBM_LINE__": "    hflbm(1,1)=0,",
            "__WELD_SPIN_LINES__": "\n".join(
                [
                    "    zspinlbmt(1,1)=0,",
                    "    tperiod(1)=0,",
                ]
            ),
            "__BC_ZMAX_LINES__": "\n".join(
                [
                    "    rwall(6)=1.0,",
                    "    iwavebc(6)=0,",
                    "    hnwave(6)=0,",
                    "    gam(6)=3.3,",
                    "    fbct(1,6)=0,",
                    "    sc2bct(1,6)=0.0,",
                    "    cncbct(1,6)=0.0,",
                    "    rsatbct(1,6)=0.0,",
                    "    flhtbct(1,6)=-1e+08,",
                ]
            ),
        }
        # The keyhole profile is the legacy case_fine_1 tuning but keeps the
        # material's surface tension instead of zeroing it (A12 vs A24).
        if profile == "keyhole":
            legacy["__SIGMA__"] = "__MATERIAL_SIGMA__"
        return legacy
    raise ConfigError("simulation.solver_profile must be standard, legacy_case_fine_1, or keyhole")


def laser_time_tables(power: float, u: float, v: float, scan_time: float, laser_off_time: float, turn_off: bool) -> dict[str, str]:
    if turn_off:
        return {
            "__LASER_POWER_TABLE__": "\n".join(
                [
                    f"    powlbm(1,1)={fmt(power)},",
                    f"    powlbm(2,1)={fmt(power)},",
                    "    powlbm(3,1)=0.0,",
                ]
            ),
            "__LASER_VELOCITY_TABLE__": "\n".join(
                [
                    f"    utlbm(1,1)={fmt(u)},",
                    f"    utlbm(2,1)={fmt(u)},",
                    "    utlbm(3,1)=0.0,",
                    f"    vtlbm(1,1)={fmt(v)},",
                    f"    vtlbm(2,1)={fmt(v)},",
                    "    vtlbm(3,1)=0.0,",
                    "    wtlbm(1,1)=0.0,",
                    "    wtlbm(2,1)=0.0,",
                    "    wtlbm(3,1)=0.0,",
                ]
            ),
            "__LASER_TIME_TABLE__": "\n".join(
                [
                    "    timlbm(1)=0.0,",
                    f"    timlbm(2)={fmt(scan_time)},",
                    f"    timlbm(3)={fmt(laser_off_time)},",
                ]
            ),
        }
    return {
        "__LASER_POWER_TABLE__": f"    powlbm(1,1)={fmt(power)},",
        "__LASER_VELOCITY_TABLE__": "\n".join(
            [
                f"    utlbm(1,1)={fmt(u)},",
                f"    vtlbm(1,1)={fmt(v)},",
                "    wtlbm(1,1)=0.0,",
            ]
        ),
        "__LASER_TIME_TABLE__": "    timlbm(1)=0.0,",
    }


def beam_profile_beamlets(laser: dict[str, Any], radius: float) -> tuple[str, list[dict[str, float]]]:
    profile = laser.get("beam_profile")
    if profile is None:
        return "single", [{"x": 0.0, "y": 0.0, "radius": radius, "power_fraction": 1.0}]
    if not isinstance(profile, dict):
        raise ConfigError("laser.beam_profile must be an object")
    kind = str(profile.get("type", "radial_table")).strip().lower().replace("-", "_")
    if kind == "single":
        return "single", [{"x": 0.0, "y": 0.0, "radius": radius, "power_fraction": 1.0}]

    def count_value(value: Any, path: str, default: int) -> int:
        if value is None:
            return default
        count = integer(value, path)
        if count < 3:
            raise ConfigError(f"{path} must be at least 3")
        if count > 64:
            raise ConfigError(f"{path} must be at most 64")
        return count

    def beamlet_radius(value: Any, path: str, default_fraction: float) -> float:
        frac = number(value if value is not None else default_fraction, path, positive=True)
        if frac > 1.0:
            raise ConfigError(f"{path} must be <= 1")
        return frac * radius

    beamlets: list[dict[str, float]] = []
    if kind == "ring":
        count = count_value(profile.get("angular_samples", profile.get("beam_count")), "laser.beam_profile.angular_samples", 12)
        offset_fraction = number(profile.get("radius_fraction", 0.55), "laser.beam_profile.radius_fraction", nonnegative=True)
        if offset_fraction > 1.0:
            raise ConfigError("laser.beam_profile.radius_fraction must be <= 1")
        child_radius = beamlet_radius(profile.get("beamlet_radius_fraction"), "laser.beam_profile.beamlet_radius_fraction", 0.22)
        if offset_fraction * radius + child_radius > radius * 1.05:
            raise ConfigError("laser.beam_profile ring beamlets extend outside the nominal beam radius")
        for i in range(count):
            theta = 2.0 * math.pi * i / count
            beamlets.append(
                {
                    "x": offset_fraction * radius * math.cos(theta),
                    "y": offset_fraction * radius * math.sin(theta),
                    "radius": child_radius,
                    "power_fraction": 1.0 / count,
                }
            )
    elif kind in {"radial_table", "tabular"}:
        samples = profile.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ConfigError("laser.beam_profile.samples must be a non-empty list")
        angular_default = count_value(profile.get("angular_samples"), "laser.beam_profile.angular_samples", 8)
        child_radius = beamlet_radius(profile.get("beamlet_radius_fraction"), "laser.beam_profile.beamlet_radius_fraction", 0.20)
        rings: list[tuple[float, float, int]] = []
        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ConfigError(f"laser.beam_profile.samples[{idx}] must be an object")
            radius_fraction = number(sample.get("radius_fraction"), f"laser.beam_profile.samples[{idx}].radius_fraction", nonnegative=True)
            if radius_fraction > 1.0:
                raise ConfigError(f"laser.beam_profile.samples[{idx}].radius_fraction must be <= 1")
            weight = sample.get("weight", sample.get("power_fraction"))
            weight_value = number(weight, f"laser.beam_profile.samples[{idx}].weight", nonnegative=True)
            count = 1 if radius_fraction == 0 else count_value(sample.get("angular_samples"), f"laser.beam_profile.samples[{idx}].angular_samples", angular_default)
            if radius_fraction * radius + child_radius > radius * 1.05:
                raise ConfigError(f"laser.beam_profile.samples[{idx}] beamlets extend outside the nominal beam radius")
            rings.append((radius_fraction, weight_value, count))
        total_weight = sum(weight for _, weight, _ in rings)
        if total_weight <= 0.0:
            raise ConfigError("laser.beam_profile sample weights must sum to a positive value")
        for radius_fraction, weight, count in rings:
            ring_fraction = weight / total_weight
            if radius_fraction == 0:
                beamlets.append({"x": 0.0, "y": 0.0, "radius": child_radius, "power_fraction": ring_fraction})
                continue
            for i in range(count):
                theta = 2.0 * math.pi * i / count
                beamlets.append(
                    {
                        "x": radius_fraction * radius * math.cos(theta),
                        "y": radius_fraction * radius * math.sin(theta),
                        "radius": child_radius,
                        "power_fraction": ring_fraction / count,
                    }
                )
    else:
        raise ConfigError("laser.beam_profile.type must be single, ring, radial_table, or tabular")
    total_fraction = sum(b["power_fraction"] for b in beamlets)
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ConfigError("laser.beam_profile power fractions must sum to 1")
    return kind, beamlets


def indexed_time_tables(
    beamlets: list[dict[str, float]],
    power: float,
    u: float,
    v: float,
    scan_time: float,
    laser_off_time: float,
    turn_off: bool,
) -> tuple[str, str, str]:
    power_lines: list[str] = []
    velocity_lines: list[str] = []
    for idx, beamlet in enumerate(beamlets, start=1):
        p = power * beamlet["power_fraction"]
        if turn_off:
            power_lines.extend([f"    powlbm(1,{idx})={fmt(p)},", f"    powlbm(2,{idx})={fmt(p)},", f"    powlbm(3,{idx})=0.0,"])
            velocity_lines.extend(
                [
                    f"    utlbm(1,{idx})={fmt(u)},",
                    f"    utlbm(2,{idx})={fmt(u)},",
                    f"    utlbm(3,{idx})=0.0,",
                    f"    vtlbm(1,{idx})={fmt(v)},",
                    f"    vtlbm(2,{idx})={fmt(v)},",
                    f"    vtlbm(3,{idx})=0.0,",
                    f"    wtlbm(1,{idx})=0.0,",
                    f"    wtlbm(2,{idx})=0.0,",
                    f"    wtlbm(3,{idx})=0.0,",
                ]
            )
        else:
            power_lines.append(f"    powlbm(1,{idx})={fmt(p)},")
            velocity_lines.extend([f"    utlbm(1,{idx})={fmt(u)},", f"    vtlbm(1,{idx})={fmt(v)},", f"    wtlbm(1,{idx})=0.0,"])
    time_lines = ["    timlbm(1)=0.0,"]
    if turn_off:
        time_lines.extend([f"    timlbm(2)={fmt(scan_time)},", f"    timlbm(3)={fmt(laser_off_time)},"])
    return "\n".join(power_lines), "\n".join(velocity_lines), "\n".join(time_lines)


def laser_beam_block(
    beamlets: list[dict[str, float]],
    *,
    start: tuple[float, float, float],
    focal_distance: float,
    lens_radius: float,
    heat_distribution_code: int,
    width_lines: str,
    hflbm_line: str,
    spin_lines: str,
    power_table: str,
    velocity_table: str,
    time_table: str,
) -> str:
    def expand_beam_indexed(block: str, beam_count: int) -> str:
        if beam_count == 1:
            return block
        expanded: list[str] = []
        for idx in range(1, beam_count + 1):
            for line in block.splitlines():
                expanded.append(line.replace("(1,1)", f"(1,{idx})").replace("(1)", f"({idx})"))
        return "\n".join(expanded)

    lines = [f"    nlbeam={len(beamlets)},"]
    for idx, beamlet in enumerate(beamlets, start=1):
        beam_name = "Linear moving laser" if len(beamlets) == 1 else f"Linear moving laser {idx}"
        lines.extend(
            [
                f"    beam_name({idx})='{beam_name}',",
                f"    if_beam_enabled({idx})=1,",
                f"    dlflbm({idx})={fmt(focal_distance)},",
                f"    rflbm({idx})={fmt(beamlet['radius'])},",
                f"    r0lbm({idx})={fmt(lens_radius)},",
            ]
        )
    lines.append(expand_beam_indexed(width_lines, len(beamlets)))
    for idx in range(1, len(beamlets) + 1):
        lines.append(f"    ilbmhtb({idx})={heat_distribution_code},")
    lines.append(expand_beam_indexed(hflbm_line, len(beamlets)))
    lines.append(power_table)
    for idx, beamlet in enumerate(beamlets, start=1):
        lines.extend(
            [
                f"    xb0lbm({idx})={fmt(start[0] + beamlet['x'])},",
                f"    yb0lbm({idx})={fmt(start[1] + beamlet['y'])},",
                f"    zb0lbm({idx})={fmt(focal_distance)},",
                f"    rxlbmt(1,{idx})=0.0,",
                f"    rylbmt(1,{idx})=0.0,",
                f"    rzlbmt(1,{idx})=-1.0,",
                f"    iprstyp({idx})=0,",
                f"    iprstime({idx})=0,",
                f"    irbtime({idx})=0,",
            ]
        )
    lines.append(velocity_table)
    lines.append(expand_beam_indexed(spin_lines, len(beamlets)))
    lines.append(time_table)
    return "\n".join(line for line in lines if line)


def load_material(config_path: Path, spec: Any) -> tuple[dict[str, str], dict[str, Any]]:
    material_path = material_path_from_spec(spec, config_path)
    try:
        raw = json.loads(material_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Material file does not exist: {material_path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in material file {material_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"Material file must contain an object: {material_path}")
    if raw.get("schema_version") != 1:
        raise ConfigError(f"material schema_version must be exactly 1: {material_path}")
    material_name = raw.get("material_name")
    if not isinstance(material_name, str) or not material_name.strip():
        raise ConfigError("material_name must be a non-empty string")
    label = raw.get("flow3d_label", material_name)
    if not isinstance(label, str) or not label.strip():
        raise ConfigError("flow3d_label must be a non-empty string when provided")
    properties = raw.get("properties")
    if not isinstance(properties, dict):
        raise ConfigError("material properties must be an object")
    material_dir = material_path.parent

    required_properties = {
        "density_kg_m3": "rhof",
        "solid_density_kg_m3": "rhofs",
        "liquidus_temperature_k": "tl1",
        "solidus_temperature_k": "ts1",
        "specific_heat_j_kg_k": "cv1",
        "solid_specific_heat_j_kg_k": "cvs1",
        "thermal_conductivity_w_m_k": "thc1",
        "solid_thermal_conductivity_w_m_k": "thcs1",
        "dynamic_viscosity_pa_s": "mu1",
        "latent_heat_j_kg": "clht1",
        "surface_tension_n_m": "sigma",
        "vapor_pressure_pa": "pv1",
        "vaporization_temperature_k": "tv1",
        "vaporization_latent_heat_j_kg": "clhv1",
    }
    values: dict[str, float] = {}
    for key in required_properties:
        values[key] = number(properties.get(key), f"material.properties.{key}", positive=True)
    if values["solidus_temperature_k"] > values["liquidus_temperature_k"]:
        raise ConfigError("material solidus_temperature_k must not exceed liquidus_temperature_k")
    # Optional vapor molar mass (kg/mol). When present, render a per-material
    # Clausius-Clapeyron exponent tvexp = R/(M*clhv1) -- the solver's own recipe
    # for its saturation curve p_sat = pv1*exp[(1/tv1 - 1/T)/tvexp]. Absent, the
    # template's legacy constant stays, so existing Ti64 campaigns are unchanged
    # (assumptions.md A30).
    if properties.get("vaporization_molar_mass_kg_mol") is not None:
        values["vaporization_molar_mass_kg_mol"] = number(
            properties.get("vaporization_molar_mass_kg_mol"),
            "material.properties.vaporization_molar_mass_kg_mol", positive=True)

    required_tables = {
        "dynamic_viscosity_pa_s": "mu1(1)",
        "surface_tension_n_m": "sigma(1)",
        "density_kg_m3": "rhof(1)",
        "thermal_conductivity_w_m_k": "thc1",
        "specific_heat_j_kg_k": "cv1",
    }
    rendered_tables = []
    table_summary: dict[str, int] = {}
    for key, flow3d_name in required_tables.items():
        rows = table_rows(read_csv_table(material_dir / f"{key}.csv", key), f"material table {key}")
        rendered_tables.append(render_table(flow3d_name, rows))
        table_summary[key] = len(rows)

    replacements = {
        "__MATERIAL_LABEL__": label.replace("'", ""),
        "__MATERIAL_MU1__": fmt(values["dynamic_viscosity_pa_s"]),
        "__MATERIAL_MUCTST__": fmt(values["dynamic_viscosity_pa_s"]),
        "__MATERIAL_CV1__": fmt(values["specific_heat_j_kg_k"]),
        "__MATERIAL_TL1__": fmt(values["liquidus_temperature_k"]),
        "__MATERIAL_TS1__": fmt(values["solidus_temperature_k"]),
        "__MATERIAL_CLHT1__": fmt(values["latent_heat_j_kg"]),
        "__MATERIAL_CVS1__": fmt(values["solid_specific_heat_j_kg_k"]),
        "__MATERIAL_THCS1__": fmt(values["solid_thermal_conductivity_w_m_k"]),
        "__MATERIAL_PV1__": fmt(values["vapor_pressure_pa"]),
        "__MATERIAL_TV1__": fmt(values["vaporization_temperature_k"]),
        "__MATERIAL_CLHV1__": fmt(values["vaporization_latent_heat_j_kg"]),
        "__MATERIAL_RHOF__": fmt(values["density_kg_m3"]),
        "__MATERIAL_SIGMA__": fmt(values["surface_tension_n_m"]),
        "__MATERIAL_THC1__": fmt(values["thermal_conductivity_w_m_k"]),
        "__MATERIAL_TABLES__": "\n".join(rendered_tables),
    }
    if "vaporization_molar_mass_kg_mol" in values:
        replacements["__TVEXP__"] = fmt(
            8.314462618
            / (values["vaporization_molar_mass_kg_mol"] * values["vaporization_latent_heat_j_kg"])
        )
    optical_manifest: dict[str, Any] = {}
    optical = raw.get("optical")
    if optical is not None:
        if not isinstance(optical, dict):
            raise ConfigError("material optical must be an object when provided")
        if optical.get("normal_absorptivity") is not None:
            optical_manifest["normal_absorptivity"] = number(
                optical.get("normal_absorptivity"),
                "material.optical.normal_absorptivity",
                positive=True,
            )
        # Which Fresnel route is more accurate for this material: resistivity/Drude
        # (default true, good for transition metals) vs measured absorptivity
        # (set false for Cu/Al/noble metals where the DC-Drude form overestimates).
        prefer = optical.get("prefer_resistivity", True)
        if not isinstance(prefer, bool):
            raise ConfigError("material optical.prefer_resistivity must be true or false")
        optical_manifest["prefer_resistivity"] = prefer
    # Temperature-dependent electrical resistivity lives in a CSV alongside the JSON.
    # The highest-temperature (melt-relevant) value drives the Fresnel epsilon; the
    # full table is kept for the temperature-dependent ereflct(T) rendering.
    resistivity_csv = material_dir / "electrical_resistivity_ohm_m.csv"
    if resistivity_csv.is_file():
        resistivity_rows = table_rows(
            read_csv_table(resistivity_csv, "electrical_resistivity_ohm_m"),
            "material electrical_resistivity_ohm_m",
        )
        optical_manifest.setdefault("prefer_resistivity", True)
        optical_manifest["electrical_resistivity_ohm_m"] = resistivity_rows[-1][1]
        optical_manifest["electrical_resistivity_table"] = [list(row) for row in resistivity_rows]
    optical_manifest = optical_manifest or None
    manifest = {
        "name": material_name,
        "file": relative_to_root(material_path),
        "flow3d_label": label,
        "properties": values,
        "table_rows": table_summary,
        "optical": optical_manifest,
    }
    return replacements, manifest


def parallel_settings(raw: dict[str, Any], cores_override: int | None, threads_override: int | None) -> dict[str, int]:
    parallel = raw.get("parallel", {})
    if parallel is None:
        parallel = {}
    if not isinstance(parallel, dict):
        raise ConfigError("parallel must be an object when provided")
    cores = cores_override if cores_override is not None else integer(parallel.get("cores", 1), "parallel.cores")
    threads = (
        threads_override
        if threads_override is not None
        else integer(parallel.get("threads_per_rank", 1), "parallel.threads_per_rank")
    )
    if cores <= 0:
        raise ConfigError("parallel.cores must be a positive integer")
    if threads <= 0:
        raise ConfigError("parallel.threads_per_rank must be a positive integer")
    if cores % threads != 0:
        raise ConfigError("parallel.cores must be divisible by parallel.threads_per_rank")
    ranks = cores // threads
    return {"cores": cores, "ranks": ranks, "threads_per_rank": threads}


def nested_mesh_block(
    spec: Any,
    *,
    cell: float,
    bounds: tuple[float, float, float, float, float, float],
    track_xy: list[tuple[float, float]],
    radius: float,
    surface_z: float,
    ranks: int,
) -> tuple[str, str, dict | None]:
    """Render a finer `&mesh` block nested around the laser track.

    Uniform refinement from 10 to 5 um costs about 12x CPU time (measured on a
    matched pair of runs), which is what puts a 5 um dataset out of reach.  A nested block pays that only
    over the melt region.  `spec` is `mesh.refinement`:

        {"factor": 2, "margin_m": 3.0e-4}

    The fine block spans the beam footprint (every track position, plus one beam
    radius) grown by `margin_m`, snapped **outward onto parent cell faces** so the
    nesting is legal, then clipped to the parent.  In z it runs from
    `margin_m + radius` below the free surface to `margin_m` above it, so both the
    fusion boundary and the vapour depression sit inside it.

    Returns `(mesh_text, mesh_mpi_text, manifest_entry)`; `("", "", None)` when no
    refinement is requested.

    The margin is the user's responsibility: a keyhole deeper than
    `margin_m + radius` would run out of the fine block, and the metrics -- which
    read the fine block alone -- would then measure a truncated pool.  Choose it
    from the expected depth, not from the beam size."""
    if spec is None:
        return "", "", None
    if not isinstance(spec, dict):
        raise ConfigError("mesh.refinement must be an object")
    unknown = set(spec) - {"factor", "margin_m"}
    if unknown:
        raise ConfigError(f"mesh.refinement has unknown keys: {', '.join(sorted(unknown))}")

    factor = spec.get("factor")
    # bool is an int subclass; True would otherwise pass as factor 1.
    if isinstance(factor, bool) or not isinstance(factor, int) or factor not in (2, 3):
        raise ConfigError(
            "mesh.refinement.factor must be the integer 2 or 3 "
            "(FLOW-3D interpolates poorly across a larger jump between adjacent blocks)"
        )
    margin = number(spec.get("margin_m"), "mesh.refinement.margin_m", positive=True)
    if ranks != 1:
        raise ConfigError(
            "mesh.refinement requires a single rank: each mesh block is decomposed "
            "per rank, and the exporter cannot tell a decomposition brick from a "
            "nested block. Set parallel.threads_per_rank equal to parallel.cores "
            "(or pass --cores N --threads-per-rank N)."
        )

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    reach = radius + margin
    lo_hi = {
        "x": (min(p[0] for p in track_xy) - reach, max(p[0] for p in track_xy) + reach),
        "y": (min(p[1] for p in track_xy) - reach, max(p[1] for p in track_xy) + reach),
        "z": (surface_z - reach, surface_z + margin),
    }
    parent = {"x": (x_min, x_max), "y": (y_min, y_max), "z": (z_min, z_max)}

    box: dict[str, tuple[float, float]] = {}
    for axis, (lo, hi) in lo_hi.items():
        p_lo, p_hi = parent[axis]
        # Snap outward onto parent cell faces: an interface that cuts through a
        # parent cell is not a legal nesting and degrades quietly.
        snapped_lo = p_lo + math.floor((lo - p_lo) / cell) * cell
        snapped_hi = p_lo + math.ceil((hi - p_lo) / cell) * cell
        box[axis] = (max(p_lo, snapped_lo), min(p_hi, snapped_hi))

    if all(box[a] == parent[a] for a in ("x", "y", "z")):
        raise ConfigError(
            "mesh.refinement covers the whole domain, which is uniform refinement "
            "with extra steps: reduce mesh.refinement.margin_m, or set a smaller "
            "mesh.cell_size_m and drop the refinement block"
        )

    fine_cell = cell / factor
    name = "Refined block around the laser track"
    lines = [
        f"    MeshName='{name}',",
        f"    size={fmt(fine_cell)},",
        f"    px(1)={fmt(box['x'][0])},",
        f"    px(2)={fmt(box['x'][1])},",
        f"    py(1)={fmt(box['y'][0])},",
        f"    py(2)={fmt(box['y'][1])},",
        f"    pz(1)={fmt(box['z'][0])},",
        f"    pz(2)={fmt(box['z'][1])},",
        "    if_mesh_enabled=1,",
    ]
    body = "\n".join(lines)
    mesh_text = "\n &mesh\n" + body + "\n /\n"
    mesh_mpi_text = "\n &MESH_MPI\n" + body.replace(
        f"MeshName='{name}'", f"MeshName='{name} - MPI domain 1'"
    ) + "\n /\n"
    counts = {a: int(round((box[a][1] - box[a][0]) / fine_cell)) for a in ("x", "y", "z")}
    entry = {
        "factor": factor,
        "margin_m": margin,
        "cell_size_m": fine_cell,
        "bounds_m": {a: [box[a][0], box[a][1]] for a in ("x", "y", "z")},
        "cells": counts,
        "total_cells": counts["x"] * counts["y"] * counts["z"],
    }
    return mesh_text, mesh_mpi_text, entry


MAX_INDEPENDENT_BEAMS = 4
# FLOW-3D's beam tables (powlbm/utlbm/vtlbm/wtlbm/timlbm) hold at most 500 rows.
# Row 501 is rejected outright at preprocessing -- `error while reading name list
# "weld" ... unknown variable ... "powlbm(501,1)=0,"` -- which is how mb40_007 died
# on 2026-08-03 after rendering 529 rows. Independent beams share one time column,
# so the count grows with beam number: 107 (2 beams), 232 (3), 376 (4), 529 (4, over).
MAX_BEAM_TABLE_ROWS = 500


def multi_beam_tables(
    specs: list[dict[str, Any]],
    *,
    radius: float,
    bounds: tuple[float, float, float, float],
    default_power: float,
    default_speed: float,
    sample_spacing: float | None,
) -> tuple[str, str, str, list[dict[str, Any]]]:
    """Per-beam power/velocity tables on one shared time grid.

    Each entry of ``laser.beams`` is an independent beam: its own path, power and
    optional ``turn_off_time_s``. Paths may cross -- nothing keeps them apart.

    The shared grid is forced by FLOW-3D, not chosen: ``powlbm``/``utlbm``/``vtlbm``
    are indexed ``(row, beam)`` but ``timlbm`` is indexed by row alone, so there is
    exactly one time column. Each beam's own transition times are unioned into that
    grid and every beam is then held piecewise-constant across it, which reproduces
    each beam's plateaus exactly: between two union rows a beam's value is unchanged,
    and FLOW-3D's linear interpolation of two equal values is that same constant.
    """
    # Budget the shared grid across beams up front. Each leg contributes a two-row
    # plateau, so a beam's rows are ~2*(samples-1)+1; thinning samples is what keeps
    # the union under the solver's cap. Thinning coarsens the polyline the beam
    # traces, so it is recorded in the manifest rather than done silently.
    per_beam_rows = max(8, MAX_BEAM_TABLE_ROWS // max(1, len(specs)))
    max_samples = max(4, (per_beam_rows - 1) // 2)
    thinned = False

    series, meta = [], []
    for index, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            raise ConfigError(f"laser.beams[{index - 1}] must be an object")
        unknown = set(spec) - {"power_w", "path", "turn_off_time_s", "scan_speed_m_per_s"}
        if unknown:
            raise ConfigError(f"laser.beams[{index - 1}] has unknown keys: {', '.join(sorted(unknown))}")
        power = number(spec.get("power_w", default_power), f"laser.beams[{index - 1}].power_w", nonnegative=True)
        speed = number(spec.get("scan_speed_m_per_s", default_speed),
                       f"laser.beams[{index - 1}].scan_speed_m_per_s", positive=True)
        path_spec = spec.get("path")
        if not isinstance(path_spec, dict):
            raise ConfigError(f"laser.beams[{index - 1}].path is required and must be an object")
        try:
            built = laser_path.build_path(path_spec, radius=radius, bounds=bounds)
        except laser_path.PathError as error:
            raise ConfigError(f"laser.beams[{index - 1}].path: {error}") from error
        samples = built.samples
        if len(samples) > max_samples:
            # Keep the endpoints; drop interior points at a uniform stride.
            step = (len(samples) - 1) / (max_samples - 1)
            picks = sorted({int(round(i * step)) for i in range(max_samples)} | {0, len(samples) - 1})
            samples = [samples[i] for i in picks]
            thinned = True
        times, us, vs, powers = laser_path.velocity_time_series(power, samples, speed, turn_off=True)
        off = spec.get("turn_off_time_s")
        if off is not None:
            off = number(off, f"laser.beams[{index - 1}].turn_off_time_s", positive=True)
        duration = off if off is not None else times[-1]
        series.append({"t": times, "u": us, "v": vs, "p": powers, "off": off})
        meta.append({
            "index": index, "start_m": list(built.samples[0]),
            "power_w": power, "scan_speed_m_per_s": speed,
            "path_type": built.type, "turn_off_time_s": off, "duration_s": duration,
            "samples_used": len(samples),
            "path_length_m": built.length_m, "num_samples": len(built.samples),
            # The PATH ITSELF, as the single-beam manifest already records it
            # (`laser_path_xy_m`). Recording only `start_m` + speed + duration described
            # the beam parametrically, which is enough for FLOW-3D (it gets the velocity
            # table) but not for a consumer: pino's multibeam channels need per-beam
            # paths, and a `random` path cannot be inferred from a start point and a
            # speed. 31 of 61 members of the mb80 campaign were unusable for that reason
            # (RIGNO session, 2026-08-28). aiida-pino can now rebuild it from the seed,
            # but only because the seeded spec happens to be in the manifest -- recording
            # the samples here is the direct fix, and it keeps the two beam families
            # (tabular mbc40, parametric mbs40) describing themselves the same way.
            "path_xy_m": [[float(x), float(y)] for x, y in built.samples],
        })

    grid = sorted({t for s in series for t in s["t"]}
                  | {s["off"] for s in series if s["off"] is not None})
    if not grid:
        raise ConfigError("laser.beams produced no time rows")

    def hold(s: dict, t: float, key: str) -> float:
        """Value of this beam at time t: last row at or before t, zero once off."""
        if s["off"] is not None and t > s["off"]:
            return 0.0
        ts = s["t"]
        if t <= ts[0]:
            return s[key][0]
        lo, hi = 0, len(ts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ts[mid] <= t:
                lo = mid
            else:
                hi = mid - 1
        return s[key][lo]

    power_lines, vel_lines, time_lines = [], [], []
    for row, t in enumerate(grid, start=1):
        time_lines.append(f"    timlbm({row})={fmt(t)},")
    for beam, s in enumerate(series, start=1):
        for row, t in enumerate(grid, start=1):
            power_lines.append(f"    powlbm({row},{beam})={fmt(hold(s, t, 'p'))},")
        for row, t in enumerate(grid, start=1):
            vel_lines.append(f"    utlbm({row},{beam})={fmt(hold(s, t, 'u'))},")
        for row, t in enumerate(grid, start=1):
            vel_lines.append(f"    vtlbm({row},{beam})={fmt(hold(s, t, 'v'))},")
        for row in range(1, len(grid) + 1):
            vel_lines.append(f"    wtlbm({row},{beam})=0.0,")
    if len(grid) > MAX_BEAM_TABLE_ROWS:
        raise ConfigError(
            f"laser.beams needs {len(grid)} table rows but FLOW-3D accepts at most "
            f"{MAX_BEAM_TABLE_ROWS}; raise laser.path.sample_spacing_m or use fewer beams")
    for m in meta:
        m["beam_table_thinned"] = thinned
    return "\n".join(power_lines), "\n".join(vel_lines), "\n".join(time_lines), meta


def load_and_validate(
    path: Path,
    case_name_override: str | None,
    cores_override: int | None = None,
    threads_override: int | None = None,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError("Top-level JSON value must be an object")
    if raw.get("schema_version") != 1:
        raise ConfigError("schema_version must be exactly 1")

    case_name = case_name_override or raw.get("case_name")
    if not isinstance(case_name, str) or not CASE_NAME.fullmatch(case_name):
        raise ConfigError("case_name must use letters, digits, _ and - only")

    geometry = section(raw, "geometry")
    mesh = section(raw, "mesh")
    laser = section(raw, "laser")
    simulation = section(raw, "simulation")
    parallel = parallel_settings(raw, cores_override, threads_override)
    material_replacements, material_manifest = load_material(path, raw.get("material"))
    cell = number(mesh.get("cell_size_m"), "mesh.cell_size_m", positive=True)
    # With laser.beams each beam carries its own power and speed, so the top-level
    # values are only defaults for beams that omit them.
    _multi = isinstance(laser.get("beams"), list) and bool(laser.get("beams"))
    power = number(laser.get("power_w", 0.0) if _multi else laser.get("power_w"),
                   "laser.power_w", nonnegative=True)
    radius = number(laser.get("beam_radius_m"), "laser.beam_radius_m", positive=True)
    focal_distance = number(laser.get("focal_distance_m"), "laser.focal_distance_m", positive=True)
    lens_radius = number(laser.get("lens_radius_m"), "laser.lens_radius_m", positive=True)
    # A stationary (dwell) spot holds one position for `laser.dwell_time_s`; the
    # scan speed is then irrelevant and optional. This is the standard
    # experimental keyhole-onset geometry (Simonds et al. 2021).
    dwell_time = laser.get("dwell_time_s")
    if dwell_time is not None:
        dwell_time = number(dwell_time, "laser.dwell_time_s", positive=True)
        speed = number(laser.get("scan_speed_m_per_s"), "laser.scan_speed_m_per_s", positive=True) \
            if laser.get("scan_speed_m_per_s") is not None else 0.0
    else:
        speed = (number(laser.get("scan_speed_m_per_s", 0.1), "laser.scan_speed_m_per_s", positive=True)
                 if _multi else
                 number(laser.get("scan_speed_m_per_s"), "laser.scan_speed_m_per_s", positive=True))
    # A generalized in-plane path (laser.path) overrides the straight start/end
    # scan.  Its start seeds the domain bounds; the full path is built once the
    # bounds and beam radius are known (see the path branch below).
    # Independent beams: each with its own path, power and turn-off. Distinct from
    # laser.beam_profile, which splits ONE beam into co-moving beamlets.
    beam_specs = laser.get("beams")
    if beam_specs is not None:
        if not isinstance(beam_specs, list) or not beam_specs:
            raise ConfigError("laser.beams must be a non-empty list of beam objects")
        if len(beam_specs) > MAX_INDEPENDENT_BEAMS:
            raise ConfigError(
                f"laser.beams supports at most {MAX_INDEPENDENT_BEAMS} independent beams; "
                f"got {len(beam_specs)}")
        if laser.get("beam_profile") is not None:
            raise ConfigError(
                "laser.beams and laser.beam_profile cannot be combined: beams are "
                "independent sources, beam_profile splits one source into co-moving "
                "beamlets, and together the beam count is ambiguous")
        if laser.get("path") is not None:
            raise ConfigError("laser.beams replaces laser.path; give each beam its own path")
    manifest_beams = None
    path_spec = laser.get("path")
    if beam_specs is not None:
        # Each beam carries its own path; the domain-level start/end are unused and
        # the real per-beam origins are resolved in multi_beam_tables.
        start = end = (0.0, 0.0, 0.0)
    elif path_spec is not None:
        if not isinstance(path_spec, dict):
            raise ConfigError("laser.path must be an object")
        try:
            start = laser_path.path_start_point(path_spec)
        except laser_path.PathError as error:
            raise ConfigError(str(error)) from error
        # `random` may omit its start; it is defaulted to the domain centre later.
        end = start if start[0] is not None else (0.0, 0.0, 0.0)
    else:
        start = point(laser.get("start_point_m"), "laser.start_point_m", [0.0, 0.0, 0.0])
        # A stationary spot has no end point: it defaults to the start, and an
        # explicitly different one is rejected below as a contradiction.
        end = point(
            laser.get("end_point_m"),
            "laser.end_point_m",
            list(start) if dwell_time is not None else None,
        )
    temperature = number(simulation.get("initial_temperature_k"), "simulation.initial_temperature_k", positive=True)
    boundary_temperature = number(
        simulation.get("boundary_temperature_k", temperature),
        "simulation.boundary_temperature_k",
        positive=True,
    )
    ambient_pressure = simulation.get("ambient_pressure_pa")
    if ambient_pressure is not None:
        ambient_pressure = number(ambient_pressure, "simulation.ambient_pressure_pa", positive=True)
    finish_time = number(simulation.get("finish_time_s"), "simulation.finish_time_s", positive=True)
    max_iterations = integer(simulation.get("max_iterations"), "simulation.max_iterations")
    # Selected-data (spatial) output frames the exporter reads, set via the
    # FLOW-3D &xput selected-data output interval tapltd(1) (default TWFIN/100 ->
    # ~100 frames).  Omitted -> no line, so the default holds and the prepin is
    # unchanged; set -> tapltd(1) = finish_time / output_frames.
    output_frames_raw = simulation.get("output_frames")
    if output_frames_raw is None:
        output_frames = None
        selected_data_interval = None
        selected_data_interval_line = ""
    else:
        output_frames = integer(output_frames_raw, "simulation.output_frames")
        selected_data_interval = finish_time / output_frames
        selected_data_interval_line = f"    tapltd(1)={fmt(selected_data_interval)},"
    subprocess_id = int(simulation.get("subprocess_id", 0))
    if subprocess_id < 0:
        raise ConfigError("simulation.subprocess_id must not be negative")
    solver_profile = simulation.get("solver_profile", "keyhole")
    if not isinstance(solver_profile, str):
        raise ConfigError("simulation.solver_profile must be a string")
    solver_profile = solver_profile.strip().lower()
    if solver_profile not in {"standard", "legacy_case_fine_1", "keyhole"}:
        raise ConfigError("simulation.solver_profile must be standard, legacy_case_fine_1, or keyhole")
    # RSIZE is FLOW-3D's evaporation accommodation coefficient: it scales the
    # phase-change rate and, through the documented recoil recipe A = RSIZE*PV1,
    # the drilling pressure. The profiles hardcode it (1.0 for legacy/keyhole),
    # which is the physical maximum; literature values are 0.1-0.5 and the
    # 2026-07-28 experimental validation found the keyhole regime over-driven, so
    # it has to be tunable per case to calibrate against measurements.
    accommodation = simulation.get("accommodation_coefficient")
    if accommodation is not None:
        accommodation = number(
            accommodation, "simulation.accommodation_coefficient", positive=True
        )
        if accommodation > 1.0:
            raise ConfigError(
                "simulation.accommodation_coefficient must be <= 1 "
                "(it is a probability: the fraction of vapour molecules that stick)"
            )
    # output_frames drives tapltd(1); the legacy_case_fine_1/keyhole profiles
    # fix tapltd(1)=5e-06 in their &xput block, so when output_frames is set the
    # profile's line is dropped below (a duplicate aborts preprocessing).
    initialization = geometry.get("initialization", "legacy_halfspace")
    if not isinstance(initialization, str):
        raise ConfigError("geometry.initialization must be a string")
    initialization = initialization.strip().lower()
    if initialization not in {"substrate_with_gas", "filled_mesh", "legacy_halfspace"}:
        raise ConfigError("geometry.initialization must be substrate_with_gas, filled_mesh, or legacy_halfspace")

    bounds = geometry.get("bounds_m")
    placement = "explicit_bounds"
    if bounds is not None:
        if not isinstance(bounds, dict):
            raise ConfigError("geometry.bounds_m must be an object with x, y, and z bounds")
        x_min, x_max = bounds_pair(bounds.get("x"), "geometry.bounds_m.x")
        y_min, y_max = bounds_pair(bounds.get("y"), "geometry.bounds_m.y")
        z_min, z_max = bounds_pair(bounds.get("z"), "geometry.bounds_m.z")
        length = x_max - x_min
        width = y_max - y_min
        if "thickness_m" in geometry:
            thickness = number(geometry["thickness_m"], "geometry.thickness_m", positive=True)
        elif z_min < 0.0:
            thickness = abs(z_min)
        else:
            raise ConfigError(
                "geometry.bounds_m.z must extend below z=0 (the surface plane) so the substrate "
                f"has thickness; got z=[{z_min:g}, {z_max:g}]. Set z min < 0 or provide geometry.thickness_m."
            )
        clearance = number(geometry.get("gas_clearance_m", max(0.0, z_max)), "geometry.gas_clearance_m", nonnegative=True)
    else:
        d = config_schema.GEOMETRY_DEFAULTS
        length = number(geometry.get("length_m", d["length_m"]), "geometry.length_m", positive=True)
        width = number(geometry.get("width_m", d["width_m"]), "geometry.width_m", positive=True)
        thickness = number(geometry.get("thickness_m", d["thickness_m"]), "geometry.thickness_m", positive=True)
        clearance = number(geometry.get("gas_clearance_m", d["gas_clearance_m"]), "geometry.gas_clearance_m", positive=True)
        # laser_leading_margin frames a single forward scan around one start
        # point; a general path (which can head any direction) is centred instead.
        placement_value = geometry.get("placement", "centered" if path_spec is not None else "laser_leading_margin")
        if not isinstance(placement_value, str):
            raise ConfigError("geometry.placement must be a string")
        placement = placement_value.strip().lower()
        if placement == "laser_leading_margin" and path_spec is not None:
            raise ConfigError(
                "geometry.placement 'laser_leading_margin' is for a single straight scan; "
                "use 'centered' or geometry.bounds_m with laser.path"
            )
        if placement == "laser_leading_margin":
            x_min = start[0] - 4.0 * radius
            x_max = x_min + length
            y_min = start[1] - width / 2.0
            y_max = start[1] + width / 2.0
        elif placement == "centered":
            x_min = -length / 2.0
            x_max = length / 2.0
            y_min = -width / 2.0
            y_max = width / 2.0
        else:
            raise ConfigError("geometry.placement must be laser_leading_margin or centered")
        z_min = -thickness
        z_max = clearance
    surface_z = number(geometry.get("surface_z_m", 0.0), "geometry.surface_z_m")
    if not (z_min <= surface_z <= z_max):
        raise ConfigError("geometry.surface_z_m must lie within the mesh z bounds")
    # Defaults reproduce the validated case_fine_1 keyhole model (gaussian beam,
    # ray-traced with 20 reflections, full absorption); see the keyhole profile.
    heat_distribution = laser.get("heat_distribution", "gaussian")
    if not isinstance(heat_distribution, str):
        raise ConfigError("laser.heat_distribution must be a string")
    heat_distribution_code = {"uniform": 0, "gaussian": 2}.get(heat_distribution.strip().lower())
    if heat_distribution_code is None:
        raise ConfigError("laser.heat_distribution must be uniform or gaussian")
    beam_profile_type, beamlets = beam_profile_beamlets(laser, radius)
    radiation_flag = int(laser.get("radiation_distribution_flag", 2))
    reflection_flag = int(laser.get("reflection_flag", 3))
    max_reflections = int(laser.get("max_reflections", 20))
    # APLUME: the fraction of beam power the plume model removes (docs call it
    # "Attenuation factor of the heat source power"). Retired from all validated
    # recipes on 2026-07-29: the tv1 re-anchor replaced it, and matched-pair
    # energy audits showed it acts as a beam-power sink (see the RFPLUME note
    # below). Knob kept for future use.
    plume_attenuation = laser.get("plume_attenuation")
    if plume_attenuation is not None:
        plume_attenuation = number(plume_attenuation, "laser.plume_attenuation", nonnegative=True)
        if plume_attenuation >= 1.0:
            raise ConfigError("laser.plume_attenuation must be < 1 (a fraction of beam power)")
    # RFPLUME: the plume cylinder radius. With rfplume = 0 (the profile default)
    # the plume model never engages and APLUME is inert (observed 2026-07-28:
    # aplume 0.15 and 0.30 gave bit-identical results). With rfplume > 0 the
    # diverted fraction is REMOVED from the delivered beam power: matched-pair
    # energy audits (2026-07-29, four pairs) show the melt receives ~aplume less
    # energy and ACONDENS returns none of it. Despite the vendor docs calling it
    # "an additional heat source", treat aplume as a calibrated beam-power sink.
    # IF_PRSRECOIL: 1 = analytic A*exp(B(1-Tv/T)) overpressure (applies the full
    # saturation overpressure at Tv with no ambient netting -- suspected onset
    # cliff-maker); 2 = recoil computed from the actual evaporation mass flow
    # (nets against ambient through the phase-change rate, eq 41).
    recoil_model = simulation.get("recoil_model")
    if recoil_model is not None:
        if recoil_model not in (1, 2):
            raise ConfigError("simulation.recoil_model must be 1 (coefficients) or 2 (mass flow)")
        recoil_model = int(recoil_model)
    # PV0: partial pressure of vapour in void regions. The profiles hardcode
    # 50650 = 0.5*PV1, a GUI relic in the same family as the old frozen
    # avprslbm; it moves the net-evaporation zero to ~3008 K for Ti64.
    void_pv = simulation.get("void_vapor_pressure_pa")
    if void_pv is not None:
        void_pv = number(void_pv, "simulation.void_vapor_pressure_pa", nonnegative=True)
    plume_radius = laser.get("plume_radius_m")
    if plume_radius is not None:
        plume_radius = number(plume_radius, "laser.plume_radius_m", positive=True)
        if plume_attenuation in (None, 0.0):
            raise ConfigError(
                "laser.plume_radius_m has no effect without a non-zero "
                "laser.plume_attenuation (the plume would carry no energy)"
            )
    # ACONDENS: documented by the vendor as recondensing the plume-diverted
    # power onto the fluid inside the RFPLUME cylinder. Measured ablations
    # (2026-07-29, p2c/p4) return NOTHING at either setting -- APLUME is a
    # loss regardless of this knob. Profile hardcodes 1.0.
    plume_condensation = laser.get("plume_condensation")
    if plume_condensation is not None:
        plume_condensation = number(plume_condensation, "laser.plume_condensation", nonnegative=True)
        if plume_condensation > 1.0:
            raise ConfigError("laser.plume_condensation must be <= 1 (a fraction of plume energy)")
        if plume_attenuation in (None, 0.0):
            raise ConfigError(
                "laser.plume_condensation has no effect without a non-zero "
                "laser.plume_attenuation (there is no plume energy to recondense)"
            )
    reflection_limit = number(laser.get("reflection_limit", 0.01), "laser.reflection_limit", positive=True)
    absorptivity_flag = int(laser.get("absorptivity_flag", 0))
    absorptivity = number(laser.get("absorptivity", 1.0), "laser.absorptivity", nonnegative=True)
    reflectivity = number(laser.get("reflectivity", 0.35), "laser.reflectivity", nonnegative=True)
    wavelength = number(laser.get("wavelength_m", 1.064e-6), "laser.wavelength_m", positive=True)
    # Fresnel-per-material: derive the Fresnel coefficient EREFLCT (epsilon) from the
    # material. Each route's value is computed when its input exists, so the chosen
    # one and the alternative can both be logged; the material's prefer_resistivity
    # flag picks which is more accurate (resistivity/Drude default; false for
    # Cu/Al where the DC-Drude form overestimates). Only meaningful with the Fresnel
    # reflection model (reflection_flag/irefopt=3).
    fresnel_method = "fixed"
    fresnel_alternatives: dict[str, dict[str, float]] = {}
    fresnel_eps_table: list[tuple[float, float]] = []
    optical = material_manifest.get("optical")
    # Default the material-derived Fresnel coupling ON whenever the material carries
    # the optical data to drive it (the resistivity CSV or a measured absorptivity),
    # so temperature-dependent absorption is used automatically. A config can still
    # set laser.fresnel_from_material=false to force the fixed scalar reflectivity.
    default_fresnel = bool(optical) and (
        optical.get("electrical_resistivity_ohm_m") is not None
        or optical.get("normal_absorptivity") is not None
    )
    fresnel_from_material = laser.get("fresnel_from_material", default_fresnel)
    if not isinstance(fresnel_from_material, bool):
        raise ConfigError("laser.fresnel_from_material must be true or false")
    if fresnel_from_material:
        if not optical:
            raise ConfigError(
                "laser.fresnel_from_material is true but the material has no optical block; "
                "add optical.electrical_resistivity_ohm_m (preferred) or optical.normal_absorptivity"
            )
        routes: dict[str, float] = {}
        if optical.get("electrical_resistivity_ohm_m"):
            routes["resistivity_drude"] = fresnel_epsilon_from_resistivity(
                optical["electrical_resistivity_ohm_m"], wavelength
            )
        if optical.get("normal_absorptivity") is not None:
            routes["absorptivity_inversion"] = fresnel_epsilon_from_absorptivity(optical["normal_absorptivity"])
        if not routes:
            raise ConfigError(
                "laser.fresnel_from_material is true but the material optical block has neither "
                "electrical_resistivity_ohm_m nor normal_absorptivity"
            )
        prefer = optical.get("prefer_resistivity", True)
        order = (["resistivity_drude", "absorptivity_inversion"] if prefer
                 else ["absorptivity_inversion", "resistivity_drude"])
        fresnel_method = next(name for name in order if name in routes)
        reflectivity = routes[fresnel_method]
        fresnel_alternatives = {
            name: {"ereflct": eps, "normal_absorptivity": 1.0 - (1.0 + (1.0 - eps) ** 2) / (1.0 + (1.0 + eps) ** 2)}
            for name, eps in routes.items()
        }
        # Temperature-dependent epsilon: if the resistivity route is chosen and the
        # material has a multi-point resistivity(T) table, render a tabular ereflct(T)
        # (iereflct=1), analogous to the sigma(T) table. Otherwise ereflct stays a
        # single value (the scalar EREFLCT, iereflct=0).
        resistivity_table = optical.get("electrical_resistivity_table")
        if fresnel_method == "resistivity_drude" and resistivity_table and len(resistivity_table) > 1:
            fresnel_eps_table = [
                (temperature, fresnel_epsilon_from_resistivity(rho, wavelength))
                for temperature, rho in resistivity_table
            ]
    vapor_a = number(laser.get("vapor_pressure_a", 50650.0), "laser.vapor_pressure_a", positive=True)
    vapor_b = number(laser.get("vapor_pressure_b", 10.95913), "laser.vapor_pressure_b", positive=True)
    turn_off = laser.get("turn_off_at_end", True)
    if not isinstance(turn_off, bool):
        raise ConfigError("laser.turn_off_at_end must be true or false")

    if beam_specs is not None:
        # ---- Independent beams: geometry is validated per beam by build_path,
        # which enforces the 4-radius keep-out on every sampled point. ----
        laser_path_obj = None
        path_length = 0.0
        u = v = 0.0
        beam_power_table, beam_velocity_table, beam_time_table, manifest_beams = multi_beam_tables(
            beam_specs, radius=radius, bounds=(x_min, x_max, y_min, y_max),
            default_power=power, default_speed=speed if speed else 0.1, sample_spacing=None,
        )
        # The run is over when the last beam stops, not when the first does.
        scan_time = max(m["duration_s"] for m in manifest_beams)
    elif path_spec is None:
        # ---- Straight single-segment scan (the original path). ----
        # The coordinate convention is intentionally fixed: start/end points are
        # on the top workpiece surface.  This makes [0, 0, 0] unambiguous.
        if abs(start[2]) > 1.0e-15 or abs(end[2]) > 1.0e-15:
            raise ConfigError("laser start_point_m and end_point_m must have z=0 (the beam focal plane)")
        if placement == "laser_leading_margin" and length <= 5.0 * radius:
            raise ConfigError("geometry.length_m must exceed 5 beam radii (four before start and one after)")
        if width <= 2.0 * radius:
            raise ConfigError("geometry.width_m must exceed two beam radii")
        for label, laser_point in (("start_point_m", start), ("end_point_m", end)):
            if not (
                x_min + radius <= laser_point[0] <= x_max - radius
                and y_min + radius <= laser_point[1] <= y_max - radius
            ):
                raise ConfigError(
                    f"laser.{label} must remain at least one beam radius inside the rectangular substrate"
                )
        path_length = math.hypot(end[0] - start[0], end[1] - start[1])
        laser_path_obj = None
        if dwell_time is not None:
            # Stationary spot: the beam holds `start_point_m` for dwell_time_s.
            # FLOW-3D gets zero in-plane velocity and the dwell as its on-time.
            if path_length > 0:
                raise ConfigError(
                    "laser.dwell_time_s is a stationary spot: omit laser.end_point_m "
                    "or make it equal to laser.start_point_m"
                )
            scan_time = dwell_time
            u = v = 0.0
        else:
            if path_length <= 0:
                raise ConfigError("laser start_point_m and end_point_m must differ")
            scan_time = path_length / speed
            u = (end[0] - start[0]) / scan_time
            v = (end[1] - start[1]) / scan_time
    else:
        # ---- Generalized constant-speed path (waypoints / sinusoidal / random). ----
        # build_path resamples the path at equal arc length (constant speed) and
        # enforces the hard 4-beam-radius keep-out from every domain edge.
        try:
            laser_path_obj = laser_path.build_path(
                path_spec,
                radius=radius,
                bounds=(x_min, x_max, y_min, y_max),
            )
        except laser_path.PathError as error:
            raise ConfigError(str(error)) from error
        samples = laser_path_obj.samples
        start = (samples[0][0], samples[0][1], 0.0)
        end = (samples[-1][0], samples[-1][1], 0.0)
        path_length = laser_path_obj.length_m
        if laser_path_obj.times_s is not None:
            # variable_speed: the knots carry their own timing; the total scan
            # time is the last knot time and the first-leg velocity is chord/dt0.
            times_s = laser_path_obj.times_s
            scan_time = times_s[-1]
            first_dt = times_s[1] - times_s[0]
        else:
            # Constant-speed path: uniform scan time = length/speed.
            scan_time = path_length / speed
            first_dt = math.dist(samples[0], samples[1]) / speed
        # First-leg velocity for the manifest; the full per-leg table is built by
        # laser_path.(velocity|variable_speed)_time_tables below.
        u = (samples[1][0] - samples[0][0]) / first_dt
        v = (samples[1][1] - samples[0][1]) / first_dt
    tolerance = max(1.0e-12, scan_time * 1.0e-9)
    laser_off_time = scan_time + tolerance
    if len(beamlets) > 1 and laser_path_obj is not None:
        raise ConfigError("laser.beam_profile with multiple beamlets is currently supported only for straight or dwell cases")
    if turn_off and finish_time < laser_off_time:
        # State the DEFICIT, not just the two numbers. At %g both sides print as
        # near-identical 6-significant-digit values, so a generator that rounded
        # finish_time down by a picosecond reads exactly like a real design error.
        # That cost a diagnosis on 2026-08-20: 8 cases of a 150-case sweep were
        # short by 7e-11 to 1e-9 s because the generator derived finish_time from
        # full-precision draws while writing rounded length/speed.
        deficit = laser_off_time - finish_time
        raise ConfigError(
            f"simulation.finish_time_s ({finish_time!r}) is shorter than the laser "
            f"travel time plus stop event ({laser_off_time!r} s) by {deficit:.3e} s. "
            f"If the deficit is far below one solver timestep, the config was most "
            f"likely rounded down: derive finish_time_s from the length and speed "
            f"AS WRITTEN and round it up. It must be at least {laser_off_time!r} s, "
            f"so round up on a grid COARSER than the {tolerance:.1e} s margin -- "
            f"rounding up to a grid the same size as the margin lands short about "
            f"half the time."
        )
    # `turn_off_at_end: false` renders ONE laser time row and FLOW-3D holds it,
    # so the beam fires until finish_time -- past the declared travel or dwell,
    # and for a scan it keeps moving past end_point_m. That may be deliberate
    # (run the track to the domain edge) or a copy-paste from a config where
    # finish_time == scan_time. It is never visible in the config, so say it out
    # loud and record it in the manifest rather than erroring: 67 tracked
    # validation configs do this at 15-26 %, and silently changing their physics
    # is not this function's call.
    exposure_overrun_s = 0.0 if turn_off else max(0.0, finish_time - scan_time)
    if exposure_overrun_s > tolerance:
        kind = "dwell" if dwell_time is not None else "scan"
        print(
            f"WARNING: laser.turn_off_at_end is false, so the laser stays ON for the "
            f"whole run: {finish_time * 1e6:.1f} us against a declared {kind} time of "
            f"{scan_time * 1e6:.1f} us (+{exposure_overrun_s / scan_time * 100:.0f}%)."
            + (f" The beam also travels {math.hypot(u, v) * exposure_overrun_s * 1e6:.0f} um "
               f"past laser.end_point_m." if dwell_time is None else "")
            + " Set laser.turn_off_at_end true if you meant a pulse followed by cooling."
        )
    if cell > radius / 2.0:
        raise ConfigError("mesh.cell_size_m must be no larger than half the beam radius (at least two cells per radius)")
    if min(length, width, thickness, clearance) < cell:
        raise ConfigError("each geometry dimension must be at least one mesh cell")
    track_xy = [(p[0], p[1]) for p in laser_path_obj.samples] if laser_path_obj else [
        (start[0], start[1]),
        (end[0], end[1]),
    ]
    refine_mesh, refine_mpi, refinement = nested_mesh_block(
        mesh.get("refinement"),
        cell=cell,
        bounds=(x_min, x_max, y_min, y_max, z_min, z_max),
        track_xy=track_xy,
        radius=radius,
        surface_z=surface_z,
        ranks=parallel["ranks"],
    )
    replacements = {
        "__FINAL_TIME_S__": fmt(finish_time),
        "__MAX_ITERATIONS__": str(max_iterations),
        "__SUBPROCESS_ID__": str(subprocess_id),
        "__CELL_SIZE_M__": fmt(cell),
        "__EXTRA_MESH_BLOCKS__": refine_mesh,
        "__EXTRA_MESH_MPI_BLOCKS__": refine_mpi,
        "__X_MIN_M__": fmt(x_min),
        "__X_MAX_M__": fmt(x_max),
        "__Y_MIN_M__": fmt(y_min),
        "__Y_MAX_M__": fmt(y_max),
        "__Z_MIN_M__": fmt(z_min),
        "__Z_MAX_M__": fmt(z_max),
        "__INITIAL_TEMPERATURE_K__": fmt(temperature),
        "__BOUNDARY_TEMPERATURE_K__": fmt(boundary_temperature),
        "__AMBIENT_PRESSURE_PA__": fmt(ambient_pressure) if ambient_pressure is not None else "101300.0",
        "__PROJECT_NAME__": case_name,
        "__FOCAL_DISTANCE_M__": fmt(focal_distance),
        "__BEAM_RADIUS_M__": fmt(radius),
        "__LENS_RADIUS_M__": fmt(lens_radius),
        "__LASER_POWER_W__": fmt(power),
        "__START_X_M__": fmt(start[0]),
        "__START_Y_M__": fmt(start[1]),
        "__LASER_U_MPS__": fmt(u),
        "__LASER_V_MPS__": fmt(v),
        "__SCAN_TIME_S__": fmt(scan_time),
        "__LASER_OFF_TIME_S__": fmt(laser_off_time),
        "__THREADS_PER_RANK__": str(parallel["threads_per_rank"]),
        "__FLUID_INITIALIZATION__": fluid_initialization_block(
            mode=initialization,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            surface_z=surface_z,
            cell=cell,
            temperature=temperature,
            pressure=ambient_pressure,
        ),
        "__LASER_RADIATION_FLAG__": str(radiation_flag),
        "__LASER_REFLECTION_FLAG__": str(reflection_flag),
        "__LASER_MAX_REFLECTIONS__": str(max_reflections),
        "__LASER_REFLECTION_LIMIT__": fmt(reflection_limit),
        "__LASER_ABSORPTIVITY_FLAG__": str(absorptivity_flag),
        "__LASER_ABSORPTIVITY__": fmt(absorptivity),
        "__LASER_REFLECTIVITY__": fmt(reflectivity),
        "__LASER_IEREFLCT_LINE__": "    iereflct=1," if fresnel_eps_table else "",
        "__FRESNEL_EREFLCT_TABLE__": render_table("ereflct(1)", fresnel_eps_table) if fresnel_eps_table else "",
        "__LASER_VAPOR_A__": fmt(vapor_a),
        "__LASER_VAPOR_B__": fmt(vapor_b),
        "__LASER_HEAT_DISTRIBUTION_FLAG__": str(heat_distribution_code),
        "__SELECTED_DATA_INTERVAL_LINE__": selected_data_interval_line,
    }
    replacements.update(solver_profile_replacements(solver_profile))
    if ambient_pressure is not None:
        replacements["__P_REFERENCE__"] = fmt(ambient_pressure)
        zmax_lines = replacements.get("__BC_ZMAX_LINES__", "")
        if "pbct(1,6)=" in zmax_lines:
            replacements["__BC_ZMAX_LINES__"] = "\n".join(
                f"    pbct(1,6)={fmt(ambient_pressure)}," if line.strip().startswith("pbct(1,6)=") else line
                for line in zmax_lines.splitlines()
            )
        else:
            replacements["__BC_ZMAX_LINES__"] = "\n".join(
                [
                    f"    pbct(1,6)={fmt(ambient_pressure)},",
                    f"    tbct(1,6)={fmt(boundary_temperature)},",
                    zmax_lines,
                ]
            )
    replacements["__IF_PRSRECOIL__"] = str(recoil_model if recoil_model is not None else 1)
    if void_pv is not None:
        replacements["__PV0__"] = fmt(void_pv)
    if plume_attenuation:
        # IPLUMEOPT is the master switch for the plume energy-scattering model.
        # APLUME and RFPLUME are only its parameters: with the flag at its default
        # 0 the solver never runs the model and every value gives bit-identical
        # results (cost 4 solves on 2026-07-28 before the preprocessor echo was
        # read). Asking for attenuation therefore enables the model, rather than
        # leaving a second switch the caller has to remember.
        block = replacements.get("__WELD_LEGACY_AFTER_REFLECTION__", "")
        if "iplumeopt=" not in block:
            replacements["__WELD_LEGACY_AFTER_REFLECTION__"] = "    iplumeopt=1,\n" + block
    if plume_radius is not None:
        block = replacements.get("__WELD_LEGACY_AFTER_REFLECTION__", "")
        if "rfplume=" in block:
            replacements["__WELD_LEGACY_AFTER_REFLECTION__"] = "\n".join(
                f"    rfplume={fmt(plume_radius)}," if line.strip().startswith("rfplume=") else line
                for line in block.splitlines()
            )
    if plume_attenuation is not None:
        block = replacements.get("__WELD_LEGACY_AFTER_REFLECTION__", "")
        if "aplume=" in block:
            replacements["__WELD_LEGACY_AFTER_REFLECTION__"] = "\n".join(
                f"    aplume={fmt(plume_attenuation)}," if line.strip().startswith("aplume=") else line
                for line in block.splitlines()
            )
    if plume_condensation is not None:
        block = replacements.get("__WELD_LEGACY_AFTER_REFLECTION__", "")
        if "acondens=" in block:
            replacements["__WELD_LEGACY_AFTER_REFLECTION__"] = "\n".join(
                f"    acondens={fmt(plume_condensation)}," if line.strip().startswith("acondens=") else line
                for line in block.splitlines()
            )
    if accommodation is not None:
        replacements["__RSIZE__"] = fmt(accommodation)
        # The legacy/keyhole profiles also emit rsize_cond (the condensation
        # side); keep the two consistent unless the profile omits the line.
        if replacements.get("__RSIZE_COND_LINE__", "").strip():
            replacements["__RSIZE_COND_LINE__"] = f"    rsize_cond={fmt(accommodation)},"
    if output_frames is not None and solver_profile != "standard":
        replacements["__XPUT_LEGACY_LINES__"] = "\n".join(
            line for line in replacements["__XPUT_LEGACY_LINES__"].split("\n")
            if "tapltd(1)=" not in line
        )
    replacements.update(boundary_replacements(raw))
    if beam_specs is not None:
        power_table, velocity_table, time_table = beam_power_table, beam_velocity_table, beam_time_table
        beam_meta = manifest_beams
        # Each independent beam is one entry in the beamlet list, offset from the
        # first beam's start (laser_beam_block places it at start + offset).
        origin = beam_meta[0]["start_m"]
        start = (origin[0], origin[1], 0.0)
        beamlets = [
            {"x": m["start_m"][0] - origin[0], "y": m["start_m"][1] - origin[1],
             "radius": radius, "power_fraction": 1.0}
            for m in beam_meta
        ]
    elif laser_path_obj is None:
        power_table, velocity_table, time_table = indexed_time_tables(
            beamlets, power, u, v, scan_time, laser_off_time, turn_off
        )
    elif laser_path_obj.times_s is not None:
        table_replacements = laser_path.variable_speed_time_tables(
            power, laser_path_obj.samples, laser_path_obj.times_s, turn_off, laser_path_obj.powers_w
        )
        power_table = table_replacements["__LASER_POWER_TABLE__"]
        velocity_table = table_replacements["__LASER_VELOCITY_TABLE__"]
        time_table = table_replacements["__LASER_TIME_TABLE__"]
    else:
        table_replacements = laser_path.velocity_time_tables(power, laser_path_obj.samples, speed, turn_off)
        power_table = table_replacements["__LASER_POWER_TABLE__"]
        velocity_table = table_replacements["__LASER_VELOCITY_TABLE__"]
        time_table = table_replacements["__LASER_TIME_TABLE__"]
    replacements["__LASER_BEAM_BLOCK__"] = laser_beam_block(
        beamlets,
        start=start,
        focal_distance=focal_distance,
        lens_radius=lens_radius,
        heat_distribution_code=heat_distribution_code,
        width_lines=replacements["__WELD_WIDTH_LINES__"],
        hflbm_line=replacements["__WELD_HFLBM_LINE__"],
        spin_lines=replacements["__WELD_SPIN_LINES__"],
        power_table=power_table,
        velocity_table=velocity_table,
        time_table=time_table,
    )
    replacements.update(material_replacements)
    manifest = {
        "schema_version": 1,
        "source_config": str(path.resolve()),
        "case_name": case_name,
        "units": {"length": "m", "mass": "kg", "time": "s", "temperature": "K", "power": "W", "pressure": "Pa"},
        "input": raw,
        "material": material_manifest,
        "parallel": parallel,
        "derived": {
            "substrate_bounds_m": {"x": [x_min, x_max], "y": [y_min, y_max], "z": [z_min, 0.0]},
            "mesh_bounds_m": {"x": [x_min, x_max], "y": [y_min, y_max], "z": [z_min, z_max]},
            # None unless mesh.refinement is set; the exported NPZ is then the fine
            # block alone (npz_block_merge.select_finest), so these are the bounds
            # and cell size the metrics actually see.
            "mesh_refinement": refinement,
            # Only present for laser.beams runs; absent for a single-beam config.
            **({"beam_count": len(manifest_beams), "beams": manifest_beams,
                "beam_table_thinned": bool(manifest_beams[0].get("beam_table_thinned"))}
               if manifest_beams else {}),
            "geometry_dimensions_m": {
                "length": length,
                "width": width,
                "thickness": thickness,
                "gas_clearance": clearance,
            },
            "geometry_placement": placement,
            "initialization": initialization,
            "solver_profile": solver_profile,
            "surface_z_m": surface_z,
            "beam_focus_start_m": list(start),
            "beam_focus_end_m": list(end),
            "beam_optical_origin_at_t0_m": [start[0], start[1], focal_distance],
            "beam_velocity_m_per_s": [u, v, 0.0],
            "scan_path_length_m": path_length,
            "scan_time_s": scan_time,
            # None when the laser is never turned off. It used to be scan_time
            # regardless, which described a turn-off event that the deck does
            # not contain -- see laser_exposure_time_s below.
            "laser_off_time_s": laser_off_time if turn_off else None,
            "laser_turn_off_at_end": turn_off,
            # What the SOLVER actually does, as opposed to what the config
            # declares. `turn_off_at_end: false` renders one laser time row and
            # FLOW-3D holds it, so the beam fires at full power (and, for a
            # scan, keeps moving) until finish_time. These three fields are the
            # deck's truth; scan_time_s remains the config's intent.
            "laser_exposure_time_s": scan_time if turn_off else finish_time,
            "laser_on_past_declared_time": bool(exposure_overrun_s > 0.0),
            "laser_overrun_distance_m": math.hypot(u, v) * exposure_overrun_s,
            "laser_path_type": laser_path_obj.type if laser_path_obj else ("stationary" if dwell_time is not None else "linear"),
            # True for a dwell (laser.dwell_time_s): one fixed beam position,
            # zero in-plane velocity, scan_path_length_m == 0.
            "laser_stationary": dwell_time is not None,
            # None = the solver profile's own RSIZE was used unchanged.
            "accommodation_coefficient": accommodation,
            # None = the profile's own APLUME (0.0, transparent plume).
            "plume_attenuation": plume_attenuation,
            "plume_radius_m": plume_radius,
            "plume_condensation": plume_condensation,
            "ambient_pressure_pa": ambient_pressure,
            "recoil_model": recoil_model,
            "void_vapor_pressure_pa": void_pv,
            "laser_path_interpolation": laser_path_obj.interpolation if laser_path_obj else None,
            "laser_path_num_samples": len(laser_path_obj.samples) if laser_path_obj else None,
            "laser_path_sample_spacing_m": laser_path_obj.spacing_m if laser_path_obj else None,
            # variable_speed paths carry per-knot timing (else None): the arrival
            # times, per-leg speeds, and any dwells (zero-speed legs) for
            # inspection and stall/divergence monitoring.
            "laser_path_variable_speed": bool(laser_path_obj and laser_path_obj.times_s is not None),
            "laser_path_times_s": (
                list(laser_path_obj.times_s)
                if laser_path_obj and laser_path_obj.times_s is not None
                else None
            ),
            "laser_path_leg_speeds_m_per_s": (
                [
                    math.dist(a, b) / (laser_path_obj.times_s[i + 1] - laser_path_obj.times_s[i])
                    for i, (a, b) in enumerate(zip(laser_path_obj.samples, laser_path_obj.samples[1:]))
                ]
                if laser_path_obj and laser_path_obj.times_s is not None
                else None
            ),
            # In-plane path points (metres) for inspection / the path.png plot;
            # the full sample list for a path, or start->end for a straight scan.
            "laser_path_xy_m": (
                [[x, y] for x, y in laser_path_obj.samples]
                if laser_path_obj
                else [[start[0], start[1]], [end[0], end[1]]]
            ),
            "laser_heat_distribution": heat_distribution.strip().lower(),
            "laser_beam_profile_type": beam_profile_type,
            "laser_beam_count": len(beamlets),
            "laser_beamlets": [
                {
                    "offset_m": [beamlet["x"], beamlet["y"]],
                    "radius_m": beamlet["radius"],
                    "power_fraction": beamlet["power_fraction"],
                }
                for beamlet in beamlets
            ],
            "output_frames": output_frames,
            "selected_data_interval_s": selected_data_interval,
            "laser_reflection_model_irefopt": reflection_flag,
            "laser_max_reflections": max_reflections,
            "laser_absorptivity_cbmengy": absorptivity,
            "laser_fresnel_from_material": fresnel_from_material,
            "laser_fresnel_method": fresnel_method,
            "laser_wavelength_m": wavelength,
            "laser_fresnel_coefficient_ereflct": reflectivity,
            "laser_normal_absorptivity_from_ereflct": 1.0
            - (1.0 + (1.0 - reflectivity) ** 2) / (1.0 + (1.0 + reflectivity) ** 2),
            "laser_fresnel_alternatives": fresnel_alternatives,
            "laser_ereflct_temperature_dependent": bool(fresnel_eps_table),
            "nominal_mesh_cells": {
                "x": (x_max - x_min) / cell,
                "y": (y_max - y_min) / cell,
                "z": (z_max - z_min) / cell,
            },
        },
    }
    return case_name, replacements, manifest


def machine_file_text(parallel: dict[str, int]) -> str:
    hostname = socket.gethostname()
    return "".join(f"{hostname}\n" for _ in range(parallel["ranks"]))


def existing_run_artifacts(case_dir: Path, case_name: str) -> list[Path]:
    """Solver output / exports a rerun under the same case_name would clobber.
    Inputs alone (prepin/manifest/machines from prepare-only or a pre-solve
    failure) are not counted, so those re-render freely; only results are
    guarded, so an expensive completed run is never silently overwritten."""
    if not case_dir.is_dir():
        return []
    found = list(case_dir.glob(f"flsgrf*.{case_name}*"))
    found += [case_dir / sub for sub in ("npz", "paraview") if (case_dir / sub).is_dir()]
    return found


def render(case_dir: Path, case_name: str, replacements: dict[str, str], manifest: dict[str, Any]) -> Path:
    if not TEMPLATE.is_file():
        raise RuntimeError(f"Template is missing: {TEMPLATE}")
    text = TEMPLATE.read_text(encoding="utf-8")
    for _ in range(3):
        previous = text
        for key, value in replacements.items():
            text = text.replace(key, value)
        if text == previous:
            break
    unresolved = sorted(set(TOKEN.findall(text)))
    if unresolved:
        raise RuntimeError(f"Unresolved template values: {', '.join(unresolved)}")
    case_dir.mkdir(parents=True, exist_ok=True)
    input_file = case_dir / f"prepin.{case_name}"
    manifest_file = case_dir / "case_manifest.json"
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    # Solver output / exports block a rerun (see existing_run_artifacts), so
    # inputs from an earlier prepare-only or pre-solve failure are regenerated
    # here.  Same parameters is a harmless
    # re-render (continuation); different parameters silently replacing the old
    # inputs is the trap, so warn before overwriting them.
    # The comparison ignores the post-solve "run" block (write_run_summary merges the
    # solver's wall-clock in after the fact): it is an output, not a parameter, so a
    # continuation re-render of the same case must not report a parameter change.
    previous_text = ""
    if manifest_file.is_file():
        previous_text = manifest_file.read_text(encoding="utf-8")
        try:
            previous = json.loads(previous_text)
            if previous.pop("run", None) is not None:
                previous_text = json.dumps(previous, indent=2, sort_keys=True) + "\n"
        except json.JSONDecodeError:
            pass
    if previous_text and previous_text != manifest_text:
        print(
            f"WARNING: overwriting inputs in {case_dir} that had different parameters "
            "(from an earlier prepare-only or failed run for this case_name)",
            file=sys.stderr,
        )
    verify_rendered_semantics(text, manifest)
    input_file.write_text(text, encoding="utf-8")
    manifest_file.write_text(manifest_text, encoding="utf-8")
    (case_dir / "machines").write_text(machine_file_text(manifest["parallel"]), encoding="utf-8")
    write_path_png(case_dir, case_name, manifest)
    return input_file


def verify_rendered_semantics(text: str, manifest: dict[str, Any]) -> None:
    """Assert configured values reached the deck under the RIGHT solver keys.

    The 2026-08-10 preheat bug: a 230-case sweep's initial_temperature_k appeared
    in every prepin 11 times (tvoid, tbct) but never as tempi, so the metal
    started at FLOW-3D's default and the sweep's whole axis was dead. The
    manifest recorded the intended value, so every downstream view agreed with
    the config while the solver disagreed with both. Presence of a NUMBER proves
    nothing; only key=value does — which is what this checks, at render time,
    so the failure costs a prepare-only instead of a week of solver time.

    Extend this check whenever a config field maps to a specific namelist key;
    it is the render-time complement of tests/test_initial_temperature.py.
    """
    t0 = (manifest.get("input", {}).get("simulation", {})
          or {}).get("initial_temperature_k")
    if t0 is None:
        return
    m = re.search(r"^\s*tempi\s*=\s*([0-9.eE+-]+)", text, re.M)
    if m is None:
        raise ConfigError(
            "rendered prepin has no `tempi=` in &fl: simulation."
            "initial_temperature_k would be silently ignored by the solver "
            "(the metal starts at FLOW-3D's default). This is the "
            "legacy_halfspace preheat bug; the initialization branch that "
            "rendered this deck dropped the key.")
    if abs(float(m.group(1)) - float(t0)) > 0.05:
        raise ConfigError(
            f"rendered tempi={m.group(1)} does not match "
            f"simulation.initial_temperature_k={t0}")


def write_path_png(case_dir: Path, case_name: str, manifest: dict[str, Any]) -> None:
    """Best-effort ``path.png`` of the scan path in the run folder.

    matplotlib lives in a separate module imported here so the runner stays
    stdlib-only; a missing library or plot error warns but never fails the run.
    """
    try:
        import plot_laser_path

        derived = manifest["derived"]
        path_xy = derived.get("laser_path_xy_m")
        if not path_xy:
            return
        mesh = derived["mesh_bounds_m"]
        radius = float(manifest["input"]["laser"]["beam_radius_m"])
        plot_laser_path.plot_path(
            path_xy,
            mesh["x"],
            mesh["y"],
            radius,
            case_dir / "path.png",
            case_name=case_name,
            path_type=str(derived.get("laser_path_type", "linear")),
            length_m=derived.get("scan_path_length_m"),
        )
    except Exception as error:  # noqa: BLE001 - plotting must never break a run
        print(f"WARNING: could not write path.png ({error})", file=sys.stderr)


def run_logged(command: str, log_file: Path) -> None:
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(["bash", "-lc", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        status = process.wait()
    if status != 0:
        raise RuntimeError(f"FLOW-3D command failed with exit code {status}; read {log_file}")


@contextlib.contextmanager
def seat_lock():
    """Hold the shared solver lock for the duration of an actual solve.

    Solver runs that share a licence are serialised by this lock.  Both a direct run and a spooled run
    reach the solver through ``solve()``, so acquiring the lock here serializes
    every solve -- direct or queued, in this repo or a sibling -- that shares the
    same ``F3D_SPOOL_DIR`` (default ``~/.flow3d/spool``).  The lock is a distinct
    file from the spooler's ``.worker.lock`` (which only marks a live worker), so
    an idle worker does not block a direct run; only an in-flight solve does.
    Blocks until the lock frees; Ctrl-C to stop waiting.

    A caller that already holds the lock sets ``F3D_SEAT_HELD=1`` and we skip the
    acquire, because ``flock`` is per open file description: re-opening the same
    path here would block forever against our own parent.  This lets a launcher
    take the lock *outside* a wall-clock timeout and hand it down, so queueing
    time is not charged to the solve budget (see the aiida-pino run adapter).
    """
    if os.environ.get("F3D_SEAT_HELD") == "1":
        print("[seat] seat already held by the launcher; not re-acquiring", flush=True)
        yield
        return
    lock_dir = default_spool_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_dir / ".seat.lock", os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "[seat] another FLOW-3D run holds the solver lock; waiting for it "
                f"to finish (lock: {lock_dir / '.seat.lock'}) ...",
                flush=True,
            )
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def solve(case_dir: Path, case_name: str, license_endpoint: str, flow3d_env: Path, parallel: dict[str, int]) -> None:
    if not flow3d_env.is_file():
        raise RuntimeError(f"FLOW-3D environment file not found: {flow3d_env}")
    if not license_endpoint:
        raise RuntimeError("Set F3D_LICENSE to a PORT@HOST endpoint or use --license PORT@HOST")
    quoted_env = shlex.quote(str(flow3d_env))
    quoted_license = shlex.quote(license_endpoint)
    quoted_case_dir = shlex.quote(str(case_dir))
    quoted_case_name = shlex.quote(case_name)
    prefix = (
        "set -e; : \"${DONT_REDEFINE_F3D_HOME:=}\"; "
        f". {quoted_env}; export F3DTKNUX_LICENSE_FILE={quoted_license}; "
        f"export LM_LICENSE_FILE={quoted_license}; cd {quoted_case_dir}; "
    )
    # Hold the shared lock only while the solver is actually running; rendering,
    # prepare-only, and export do not start the solver and stay outside it.
    with seat_lock():
        if parallel["ranks"] > 1:
            run_logged(
                prefix
                + f'"$F3D_HOME/local/runpre_par" -9 {parallel["ranks"]} {parallel["threads_per_rank"]} {quoted_case_name}',
                case_dir / "runpre_decompose.log",
            )
        # With one rank, runhyd_par uses the pre-split one-block input rendered by
        # this repo.  With more ranks, runpre_par -9 asks FLOW-3D's ADT to overwrite
        # the seed decomposition before the solver starts.
        run_logged(prefix + f'"$F3D_HOME/local/runhyd_par" {quoted_case_name}', case_dir / "runhyd.log")
    # flsgrf1000 is the preprocessor's initial restart snapshot; it is not a
    # completed solver result.  The normal result series starts at flsgrf1.
    outputs = list(case_dir.glob(f"flsgrf1.{case_name}*"))
    if not any(path.is_file() and path.stat().st_size > 0 for path in outputs):
        raise RuntimeError(f"Solver completed but produced no non-empty flsgrf result in {case_dir}")


def free_space_gb(path: Path) -> float:
    """Free space in GB on the filesystem holding `path`."""
    return shutil.disk_usage(path).free / 1024**3


def require_free_space(path: Path, min_gb: float, wait_s: float = 0.0, poll_s: float = 60.0) -> None:
    """Hold off a solve until the filesystem has room, then give up if it never does.

    A solve that fills the disk does not fail cleanly: FLOW-3D blocks mid-write,
    sim_time stops advancing, and the stall monitor kills the case hours later.

    It waits rather than refusing outright because disk pressure here is usually
    transient -- another case is mid-solve and will drop its flsgrf when it exports.
    Refusing instantly is what cost the whole material_ck300 sweep on 2026-07-21:
    the spooler fed 300 queued cases through a refusing gate in five minutes and
    marked every one FAILED, for a dip that cleared on its own within the hour.
    A `min_gb` of 0 disables the floor; `wait_s` of 0 fails fast. Under
    `F3D_RESOURCE_GUARDS=warn` (the public copy's default, see resource_guards.py)
    a low reading is one warning line, with no wait and no refusal.
    """
    if min_gb <= 0:
        return
    if resource_guards.mode() == "warn":
        wait_s = 0.0
    deadline = time.monotonic() + max(wait_s, 0.0)
    announced = False
    while True:
        free = free_space_gb(path)
        if free >= min_gb:
            return
        if time.monotonic() >= deadline:
            resource_guards.resource_guard(
                f"Refusing to start: {free:.1f} GB free disk under {path}, floor is {min_gb:g} GB "
                f"and it did not clear in {wait_s:g}s (lower with --min-free-gb, or --min-free-gb 0 to disable)"
            )
            return
        if not announced:
            print(f"Waiting for free disk: {free:.1f} GB under {path}, need {min_gb:g} GB", flush=True)
            announced = True
        time.sleep(min(poll_s, max(deadline - time.monotonic(), 0.0)) or poll_s)


def available_memory_gb() -> float:
    """MemAvailable from /proc/meminfo, in GB."""
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024**2
    return float("inf")  # not Linux: nothing to gate on


def load_average_1min() -> float:
    """The kernel's 1-minute load average."""
    return os.getloadavg()[0]


def nice_target(requested: int, current: int) -> int | None:
    """Absolute niceness to set before solving, or None to leave the process alone.

    Wrapping a solve in `systemd-run --nice=10` makes the solver yield CPU to
    sshd and to other jobs sharing the machine. That only applies when a human
    types the wrapper: neither the spool worker nor the AiiDA path uses it, so
    queued solves ran at whatever priority they inherited (observed: nice 5 while
    the 1-minute load sat at 32). Putting it here covers every caller, the same reason the memory and pause gates live here.

    Only ever RAISES niceness. Lowering it needs privilege and would take CPU away
    from sshd, which is the failure this exists to prevent. `requested <= 0`
    disables the adjustment.
    """
    if requested <= 0 or current >= requested:
        return None
    return requested


def require_machine_headroom(
    min_mem_gb: float, max_load: float, wait_s: float = 1800.0, poll_s: float = 30.0
) -> None:
    """Hold off a solve until the box has room for the OS and an interactive login.

    On 2026-07-09 a solve stacked on two ML trainings drove the machine into swap
    thrashing and new SSH logins timed out faulting sshd's pages back in. The batch
    wrappers have gated on this ever since; runs submitted through AiiDA had no gate
    at all, so it belongs here, where every caller passes through.

    Gates on `MemAvailable`, because the lockout risk is active thrashing and that is
    what measures it. The load check is **opt-in** (`max_load=0` disables it): load1
    is a lagging one-minute average, and a busy machine legitimately sits above the
    documented threshold of 16 while a 10-core sweep runs, so gating on it by default
    would stall every back-to-back solve waiting for an average that never decays.
    Under `F3D_RESOURCE_GUARDS=warn` (the public copy's default) a short reading is
    one warning line, with no wait and no refusal.
    """
    if resource_guards.mode() == "warn":
        wait_s = 0.0
    deadline = time.monotonic() + max(wait_s, 0.0)
    announced = False
    while True:
        memory = available_memory_gb()
        load = load_average_1min() if max_load > 0 else 0.0
        short_on_memory = min_mem_gb > 0 and memory < min_mem_gb
        overloaded = max_load > 0 and load > max_load
        if not short_on_memory and not overloaded:
            return
        reason = f"{memory:.0f} GB available RAM (need {min_mem_gb:g})" if short_on_memory else ""
        if overloaded:
            reason = (reason + "; " if reason else "") + f"load {load:.1f} (max {max_load:g})"
        if time.monotonic() >= deadline:
            resource_guards.resource_guard(
                f"Refusing to start: {reason}, and it did not clear in {wait_s:g}s "
                "(tune with --min-free-mem-gb / --max-load, or 0 to disable)"
            )
            return
        if not announced:
            print(f"Waiting for machine headroom: {reason}", flush=True)
            announced = True
        time.sleep(poll_s)


def wait_while_paused(spool_dir: Path, poll_s: float = 30.0, _on_poll=None) -> None:
    """Block while the shared pause sentinel exists, so one switch stops every path.

    One solver lock is fed by several independent queues -- the spool worker, AiiDA
    workchains, and direct runs. Stopping the spool worker did not stop
    AiiDA-submitted solves, which took the lock anyway. `<spool_dir>/PAUSED` is the
    one switch they all see, because they all shell out to this runner.

    The wait is deliberately unbounded: a pause means "hold", not "hold briefly".
    A long pause can therefore outlast a CalcJob's own timeout, which is visible and
    intended rather than silent.
    """
    sentinel = spool_dir / "PAUSED"
    announced = False
    while sentinel.exists():
        if not announced:
            note = sentinel.read_text(encoding="utf-8").strip()
            print(f"Paused by {sentinel}{': ' + note if note else ''}", flush=True)
            announced = True
        if _on_poll is not None:
            _on_poll()
        time.sleep(poll_s)


def npz_export_is_complete(npz: Path) -> bool:
    """True when `npz` is a readable archive holding at least one array.

    An .npz is a zip, so a truncated or still-being-written export cannot open
    its central directory -- enough to tell a finished export from a dead one.
    """
    try:
        with zipfile.ZipFile(npz) as archive:
            return bool(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def discard_solver_result(case_dir: Path, case_name: str) -> int:
    """Delete the flsgrf result database once the NPZ export is complete.

    flsgrf is the bulk of a run -- 0.7-3.6 GB per case on the batches run here --
    and nothing downstream reads it: summary.md works off runhyd.log, and the ML
    datasets ship the NPZ plus case_manifest.json.  Keeping it is what filled the
    root filesystem on 2026-07-20 and refused a whole 300-case sweep.

    Returns the bytes reclaimed, or 0 when the export is missing or partial --
    then flsgrf is the only copy of an expensive solve, so it is kept.
    """
    if not npz_export_is_complete(case_dir / "npz" / f"{case_name}.npz"):
        return 0
    freed = 0
    for result in sorted(case_dir.glob(f"flsgrf*.{case_name}*")):
        if result.is_file():
            freed += result.stat().st_size
            result.unlink()
    return freed


def run_vtk_exporter(
    case_dir: Path,
    case_name: str,
    post_home: Path,
    exporter_name: str,
    output_dir: Path,
    log_name: str,
    extra_args: str = "",
) -> None:
    """Run a FLOW-3D POST pvpython exporter for one case, logging its output."""
    pvpython = post_home / "bin" / "pvpython"
    exporter = ROOT_DIR / "python" / exporter_name
    result = case_dir / f"flsgrf1.{case_name}"
    if not pvpython.is_file():
        raise RuntimeError(f"FLOW-3D POST pvpython was not found: {pvpython}; rerun with --export none to skip export")
    if not exporter.is_file():
        raise RuntimeError(f"VTK exporter is missing: {exporter}")
    if not result.is_file():
        raise RuntimeError(f"Cannot export VTK: solver result is missing: {result}")
    command = (
        f"export F3D_POST_HOME={shlex.quote(str(post_home))}; "
        f"{shlex.quote(str(pvpython))} {shlex.quote(str(exporter))} "
        f"{shlex.quote(str(result))} {shlex.quote(str(output_dir))} "
        f"--prefix {shlex.quote(case_name)}{extra_args}"
    )
    run_logged(command, case_dir / log_name)


def export_provenance_scalars(laser: dict[str, Any]) -> tuple[float, float]:
    """(power_w, scan_speed_m_per_s) recorded in the exported NPZ.

    A multi-beam config has no top-level power or speed -- each beam carries its own --
    so reading ``laser["power_w"]`` raises KeyError *at export time*, after the solve is
    already paid for. One scalar cannot describe 2-4 independent beams; this records the
    **total delivered power** and the **mean scan speed**, which is what a surrogate
    conditioned on (P, V) can use. The per-beam breakdown stays in
    ``case_manifest.json`` under ``derived.beams``.
    """
    beams = laser.get("beams")
    if isinstance(beams, list) and beams:
        power = sum(float(b.get("power_w", 0.0)) for b in beams)
        # A beam without its own speed falls back to the top-level value, and to the
        # runner's 0.1 m/s default if that is absent too (load_and_validate uses
        # `default_speed = speed if speed else 0.1`). Recording 0.0 here would state
        # that a moving beam was stationary.
        fallback = float(laser.get("scan_speed_m_per_s") or 0.0) or 0.1
        speeds = [float(b.get("scan_speed_m_per_s") or fallback) for b in beams]
        return power, sum(speeds) / len(speeds)
    return float(laser["power_w"]), float(laser.get("scan_speed_m_per_s", 0.0) or 0.0)


def export_all_timesteps_npz(
    case_dir: Path, case_name: str, post_home: Path, power_w: float, scan_speed_m_per_s: float
) -> Path:
    """Write the complete stored time series to one structured-array NPZ."""
    output_dir = case_dir / "npz"
    extra_args = f" --power-w {power_w} --scan-speed-m-per-s {scan_speed_m_per_s}"
    run_vtk_exporter(
        case_dir, case_name, post_home, "export_all_timesteps_to_npz.py", output_dir, "export_npz.log", extra_args
    )
    npz = output_dir / f"{case_name}.npz"
    if not npz.is_file():
        raise RuntimeError(f"NPZ exporter did not produce its file: {npz}")
    return npz


def export_all_timesteps(case_dir: Path, case_name: str, post_home: Path, representation: str = "volume") -> Path:
    """Write the complete stored time series to VTK/PVD using FLOW-3D POST.

    representation="volume" writes the 3D Fluid isovolume (default, for training);
    "surface" writes the Solidified Liquid 0.5 isosurface for geometry inspection.
    """
    if representation == "volume":
        output_dir = case_dir / "paraview" / "all_steps"
        log_name, extra_args = "export_vtk.log", ""
    else:
        output_dir = case_dir / "paraview" / "all_steps_surface"
        log_name, extra_args = "export_vtk_surface.log", " --representation surface"
    run_vtk_exporter(case_dir, case_name, post_home, "export_all_timesteps_to_vtk.py", output_dir, log_name, extra_args)
    collection = output_dir / f"{case_name}.pvd"
    if not collection.is_file():
        raise RuntimeError(f"VTK exporter did not produce its PVD collection: {collection}")
    return collection


def export_legacy_even_timesteps(case_dir: Path, case_name: str, post_home: Path, frames: int = 60) -> Path:
    """Write the old PINO training-data style VTK sample set."""
    output_dir = case_dir / "paraview" / "legacy_even_steps"
    run_vtk_exporter(
        case_dir, case_name, post_home, "export_legacy_even_timesteps_to_vtk.py",
        output_dir, "export_vtk_legacy_even_steps.log", f" --frames {frames}",
    )
    if not output_dir.is_dir() or not any(output_dir.glob(f"{case_name}_*.vtk")):
        raise RuntimeError(f"Legacy VTK exporter did not produce frames in: {output_dir}")
    return output_dir


def parse_solver_log(log_path: Path) -> dict[str, str]:
    """Pull completion status and timing from the runhyd.log tail (stdlib only)."""
    info: dict[str, str] = {}
    if not log_path.is_file():
        return info
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    end = re.search(r"end of calculation at\s+t\s*=\s*([0-9.eE+-]+),\s*cycle\s*=\s*(\d+)", text)
    if end:
        info["final_time_s"] = end.group(1)
        info["cycles"] = end.group(2)
    status = re.search(r"^\s*(normal completion[^\n]*)", text, re.MULTILINE | re.IGNORECASE)
    if not status:
        status = re.search(r"^\s*([^\n]*(?:abnormal|termination|fatal error)[^\n]*)", text, re.MULTILINE | re.IGNORECASE)
    if status:
        info["completion"] = status.group(1).strip()
    elapsed = re.search(r"elapsed time\s*=\s*([0-9.eE+-]+)\s*seconds,\s*or\s*\n([^\n]+)", text)
    if elapsed:
        info["elapsed_seconds"] = elapsed.group(1)
        info["elapsed_human"] = elapsed.group(2).strip()
    cpu = re.search(r"cpu time\s*=\s*([0-9.eE+-]+)\s*seconds", text)
    if cpu:
        info["cpu_seconds"] = cpu.group(1)
    when = re.search(r"date of completion\s*=\s*([^\n]+)\s*\n\s*time\s*=\s*([^\n]+)", text)
    if when:
        info["completion_datetime"] = f"{when.group(1).strip()} {when.group(2).strip()}"
    size = re.search(r"flsgrf[^\n]*file size:\s*([^\n]+)", text)
    if size:
        info["result_size"] = size.group(1).strip()
    return info


def npz_stats_text(npz_path: Path, initial_temperature_k: float | None = None) -> str:
    """Run the NPZ validator and capture its one-line summary (keeps the runner
    stdlib-only by leaving numpy in the validator's own process).

    When the case's configured initial temperature is known, the validator also
    checks that frame 0 of the export actually STARTS there (--check-initial).
    That is the guard the 230-case preheat sweep lacked: its configs asked for
    293-773 K, the solver ran every metal block at 273.1 K, and the first
    export already contained the proof.  A failed check is put in the summary
    where a human reads it; it does not delete the export.
    """
    validator = SCRIPT_DIR / "validate_npz_series.py"
    if not npz_path.is_file() or not validator.is_file():
        return ""
    cmd = [sys.executable, str(validator), str(npz_path)]
    if initial_temperature_k is not None:
        cmd += ["--check-initial", "--initial-temperature-k", str(initial_temperature_k)]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    line = (result.stdout.strip() or result.stderr.strip()).splitlines()
    return line[-1].strip() if line else ""


def _md_table(rows: list[tuple[str, str]]) -> str:
    keep = [(k, v) for k, v in rows if v not in ("", None)]
    if not keep:
        return "_none_"
    out = ["| Field | Value |", "| --- | --- |"]
    out += [f"| {k} | {v} |" for k, v in keep]
    return "\n".join(out)


def _ray_tracing_label(irefopt: Any) -> str:
    """Human-readable FLOW-3D laser absorption mode (irefopt)."""
    return {
        0: "off (no ray tracing)",
        1: "on, constant per-hit absorption (irefopt=1)",
        3: "on, angle-dependent Fresnel (irefopt=3)",
    }.get(irefopt, "" if irefopt is None else f"irefopt={irefopt}")


def build_summary_markdown(
    case_name: str,
    manifest: dict[str, Any],
    log_info: dict[str, str],
    npz_stats: str,
    generated_at: str,
    artifacts: list[tuple[str, str]],
) -> str:
    """Render a run summary as organized Markdown.  Pure: same inputs -> same text."""

    def g(value: Any) -> str:
        return format(value, ".6g") if isinstance(value, (int, float)) else str(value)

    d = manifest.get("derived", {})
    inp = manifest.get("input", {})
    geom = d.get("geometry_dimensions_m", {})
    cells = d.get("nominal_mesh_cells", {})
    laser = inp.get("laser", {})
    sim = inp.get("simulation", {})
    mesh = inp.get("mesh", {})
    par = manifest.get("parallel", {})
    mat = manifest.get("material", {})

    cell_size = mesh.get("cell_size_m")
    beam_radius = laser.get("beam_radius_m")
    total_cells = None
    if cells:
        total_cells = 1
        for value in cells.values():
            total_cells *= value
    cells_per_radius = (beam_radius / cell_size) if (beam_radius and cell_size) else None

    status_rows = [
        ("Completion", log_info.get("completion", "unknown (log not parsed)")),
        ("Final simulation time (s)", log_info.get("final_time_s", "")),
        ("Solver cycles", log_info.get("cycles", "")),
        ("Wall-clock elapsed", log_info.get("elapsed_human", "")),
        ("Wall-clock elapsed (s)", log_info.get("elapsed_seconds", "")),
        ("CPU time (s)", log_info.get("cpu_seconds", "")),
        ("Solver finished at", log_info.get("completion_datetime", "")),
        ("Result file size", log_info.get("result_size", "")),
    ]
    geom_rows = [
        ("Length x Width x Thickness (m)", f"{g(geom.get('length'))} x {g(geom.get('width'))} x {g(geom.get('thickness'))}"),
        ("Gas clearance (m)", g(geom.get("gas_clearance")) if "gas_clearance" in geom else ""),
        ("Bottom clearance (m)", g(inp.get("geometry", {}).get("bottom_clearance_m")) if inp.get("geometry", {}).get("bottom_clearance_m") is not None else ""),
        ("Placement", d.get("geometry_placement", "")),
        ("Initialization", d.get("initialization", "")),
        ("Cell size (m)", g(cell_size) if cell_size is not None else ""),
        ("Nominal cells (x, y, z)", f"{g(cells.get('x'))}, {g(cells.get('y'))}, {g(cells.get('z'))}" if cells else ""),
        ("Total cells (nominal)", g(total_cells) if total_cells is not None else ""),
        ("Cells per beam radius", g(cells_per_radius) if cells_per_radius is not None else ""),
    ]
    laser_rows = [
        ("Power (W)", g(laser.get("power_w"))),
        ("Beam radius (m)", g(beam_radius) if beam_radius is not None else ""),
        ("Scan speed (m/s)", g(laser.get("scan_speed_m_per_s"))),
        ("Scan path length (m)", g(d.get("scan_path_length_m")) if d.get("scan_path_length_m") is not None else ""),
        ("Scan time (s)", g(d.get("scan_time_s")) if d.get("scan_time_s") is not None else ""),
        ("Heat distribution", d.get("laser_heat_distribution", "")),
        ("Ray tracing (irefopt)", _ray_tracing_label(d.get("laser_reflection_model_irefopt"))),
        ("Max reflections", g(d.get("laser_max_reflections")) if d.get("laser_max_reflections") is not None else ""),
        ("Absorptivity (cbmengy)", g(d.get("laser_absorptivity_cbmengy")) if d.get("laser_absorptivity_cbmengy") is not None else ""),
        ("Fresnel coefficient (ereflct)", g(d.get("laser_fresnel_coefficient_ereflct")) if d.get("laser_fresnel_coefficient_ereflct") is not None else ""),
        ("Turn off at end", str(d.get("laser_turn_off_at_end")) if "laser_turn_off_at_end" in d else ""),
        ("Laser off time (s)", g(d.get("laser_off_time_s")) if d.get("laser_off_time_s") is not None else ""),
        ("Solver profile", d.get("solver_profile", "")),
    ]
    solver_rows = [
        ("Cores", g(par.get("cores"))),
        ("Ranks", g(par.get("ranks"))),
        ("Threads per rank", g(par.get("threads_per_rank"))),
        ("Finish time (s)", g(sim.get("finish_time_s")) if sim.get("finish_time_s") is not None else ""),
        ("Max iterations", g(sim.get("max_iterations")) if sim.get("max_iterations") is not None else ""),
        ("Initial temperature (K)", g(sim.get("initial_temperature_k")) if sim.get("initial_temperature_k") is not None else ""),
    ]
    material_rows = [("Name", mat.get("name", "")), ("Label", mat.get("flow3d_label", "")), ("File", mat.get("file", ""))]
    material_rows += [(str(k), g(v)) for k, v in sorted(mat.get("properties", {}).items())]

    sections = [
        f"# Run summary: {case_name}",
        f"_Generated {generated_at}_",
        "## Status\n" + _md_table(status_rows),
        "## Geometry and mesh\n" + _md_table(geom_rows),
        "## Laser\n" + _md_table(laser_rows),
        "## Solver and parallelism\n" + _md_table(solver_rows),
        "## Material\n" + _md_table(material_rows),
    ]

    gas = d.get("assist_gas_shield")
    assist = d.get("assist_gas_pressure_pa")
    if assist is not None or gas:
        gas_rows = [("Assist gas pressure (Pa)", g(assist) if assist is not None else "")]
        if gas:
            gas_rows.append(("Shield gas density (kg/m^3)", g(gas.get("density_kg_m3"))))
            vel = gas.get("velocity_m_per_s")
            if vel:
                gas_rows.append(("Shield gas velocity (m/s)", ", ".join(g(c) for c in vel)))
        sections.append("## Assist / shield gas\n" + _md_table(gas_rows))

    sections.append("## Results\n" + (npz_stats if npz_stats else "_NPZ not exported; run with --export npz or both for result statistics._"))
    sections.append("## Artifacts\n" + _md_table(artifacts))
    return "\n\n".join(sections) + "\n"


def runtime_block(log_info: dict[str, str]) -> dict[str, Any]:
    """The manifest "run" block for a parsed solver log ({} if there is nothing to record)."""
    run: dict[str, Any] = {}
    for key in ("completion", "completion_datetime", "cycles", "elapsed_human", "result_size"):
        if log_info.get(key):
            run[key] = log_info[key]
    for key in ("elapsed_seconds", "cpu_seconds", "final_time_s"):
        try:
            run[key] = float(log_info[key])
        except (KeyError, TypeError, ValueError):
            pass
    return run


def _merge_runtime_into_manifest(case_dir: Path, log_info: dict[str, str]) -> None:
    """Write the solve's timing into case_manifest.json under a "run" block.

    The manifest is rendered at prepare time, before the solver starts, so it cannot
    carry a wall-clock of its own; and it is the file copied next to the NPZ into the
    ML datasets (lpno/data/flow3D/<batch>/<case>.manifest.json), where
    summary.md and the batch status.tsv do not follow.  Merging here covers every
    caller, since write_run_summary is the single post-solve reporting point.
    """
    manifest_file = case_dir / "case_manifest.json"
    if not log_info or not manifest_file.is_file():
        return
    run = runtime_block(log_info)
    if not run:
        return
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    manifest["run"] = run
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run_summary(case_dir: Path, case_name: str, manifest: dict[str, Any]) -> Path:
    """Collect log/result stats and write a Markdown run summary into case_dir."""
    log_info = parse_solver_log(case_dir / "runhyd.log")
    _merge_runtime_into_manifest(case_dir, log_info)
    npz = case_dir / "npz" / f"{case_name}.npz"
    t0 = (manifest.get("input", {}).get("simulation", {}) or {}).get("initial_temperature_k")
    npz_stats = npz_stats_text(npz, initial_temperature_k=t0)
    artifacts: list[tuple[str, str]] = []
    candidates = [
        ("FLOW-3D input", case_dir / f"prepin.{case_name}"),
        ("Manifest", case_dir / "case_manifest.json"),
        ("Solver result", case_dir / f"flsgrf1.{case_name}"),
        ("Solver log", case_dir / "runhyd.log"),
        ("NPZ export", npz),
        ("ParaView collection", case_dir / "paraview" / "all_steps" / f"{case_name}.pvd"),
    ]
    for label, path in candidates:
        if path.exists():
            artifacts.append((label, relative_to_root(path)))
    generated_at = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    text = build_summary_markdown(case_name, manifest, log_info, npz_stats, generated_at, artifacts)
    summary = case_dir / "summary.md"
    summary.write_text(text, encoding="utf-8")
    return summary


def main() -> int:
    # Handled before argparse so it works without the required config argument.
    if "--print-config-schema" in sys.argv[1:]:
        print(config_schema.to_markdown())
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-config-schema", action="store_true", help="Print the config schema table (Markdown) and exit")
    parser.add_argument("config", type=Path, help="JSON configuration with SI values")
    parser.add_argument("--case-name", help="Override config case_name; letters, digits, _ and - only")
    parser.add_argument("--out-root", type=Path, default=RUNS_DIR, help="Parent directory for the run's <case_name> folder (default: runs/). Batches set this so all of a batch's cases live in one folder.")
    parser.add_argument("--prepare-only", action="store_true", help="Render prepin and manifest but do not consume a solver license")
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing a case_name whose run folder already holds solver output or exports (otherwise refused, so an expensive run is not silently overwritten)")
    parser.add_argument("--cores", type=int, help="Override parallel.cores for this run")
    parser.add_argument("--threads-per-rank", type=int, help="Override parallel.threads_per_rank for this run")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=float(os.environ.get("F3D_MIN_FREE_GB", 30.0)),
        help="Hold off solving with less than this much free disk (default 30, 0 disables). A full disk stalls FLOW-3D mid-write instead of failing cleanly.",
    )
    parser.add_argument(
        "--min-free-wait-s",
        type=float,
        default=float(os.environ.get("F3D_MIN_FREE_WAIT_S", 1800.0)),
        help="How long to wait for --min-free-gb to clear before failing (default 1800; 0 fails immediately). Disk pressure is usually another case about to drop its flsgrf.",
    )
    parser.add_argument(
        "--min-free-mem-gb",
        type=float,
        default=float(os.environ.get("F3D_MIN_FREE_MEM_GB", 25.0)),
        help="Wait for at least this much MemAvailable before solving (default 25, 0 disables). Low available RAM is what locks out interactive SSH.",
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=int(os.environ.get("F3D_NICE", 10)),
        help=(
            "raise this process's niceness before solving so the solver yields CPU to "
            "sshd and other jobs (default 10; 0 disables). Never lowers "
            "niceness."
        ),
    )
    parser.add_argument(
        "--max-load",
        type=float,
        default=float(os.environ.get("F3D_MAX_LOAD", 0.0)),
        help="Wait until the 1-minute load average is at or below this before solving (default 0 = disabled). Off by default because load1 lags and a running sweep keeps it high.",
    )
    parser.add_argument(
        "--keep-flsgrf",
        action="store_true",
        help="Keep the multi-GB flsgrf result database after a successful NPZ export (by default it is deleted, since the NPZ and manifest are what downstream tools read)",
    )
    parser.add_argument("--license", default=local_license_endpoint(), help="FlexNet endpoint PORT@HOST (or set F3D_LICENSE or local license.env)")
    parser.add_argument("--flow3d-env", type=Path, default=Path(os.environ.get("FLOW3D_ENV", DEFAULT_ENV)), help="Path to flow3dvars.sh")
    parser.add_argument(
        "--export",
        dest="export",
        choices=["npz", "vtk", "vtk-surface", "vtk-legacy60", "both", "none"],
        default="npz",
        help=(
            "Export mode after solving. npz (default) writes one compact "
            "structured-array .npz for the whole run; vtk writes full 3D volume "
            "VTK/PVD for all stored steps; vtk-surface writes the Solidified "
            "Liquid isosurface VTK/PVD for all steps; vtk-legacy60 writes the old "
            "60-frame Fluid isovolume sequence; both writes npz and vtk. All VTK "
            "modes carry cell and interpolated point data so ParaView Contour works."
        ),
    )
    parser.add_argument("--post-home", type=Path, default=Path(os.environ.get("F3D_POST_HOME", DEFAULT_POST_HOME)), help="FLOW-3D POST installation directory")
    args = parser.parse_args()
    try:
        case_name, replacements, manifest = load_and_validate(
            args.config,
            args.case_name,
            args.cores,
            args.threads_per_rank,
        )
        case_dir = args.out_root / case_name
        existing = existing_run_artifacts(case_dir, case_name)
        if existing and not args.overwrite:
            names = ", ".join(sorted(p.name for p in existing)[:5])
            raise RuntimeError(
                f"Refusing to overwrite existing run artifacts in {case_dir} ({names}); "
                f"choose a new case_name or pass --overwrite"
            )
        input_file = render(case_dir, case_name, replacements, manifest)
        print(f"Prepared {case_dir}")
        print(f"Input: {input_file}")
        print(
            "Parallel allocation: "
            f"{manifest['parallel']['cores']} cores, "
            f"{manifest['parallel']['ranks']} ranks, "
            f"{manifest['parallel']['threads_per_rank']} threads/rank"
        )
        print(f"Material: {manifest['material']['name']} ({manifest['material']['file']})")
        print(f"Laser travel: {manifest['derived']['scan_path_length_m']:.6g} m in {manifest['derived']['scan_time_s']:.6g} s")
        derived = manifest["derived"]
        if derived["laser_fresnel_from_material"]:
            chosen = derived["laser_fresnel_method"]
            alts = "; ".join(
                f"{'*' if name == chosen else ''}{name}: ereflct={vals['ereflct']:.4g}, A0={vals['normal_absorptivity']:.3f}"
                for name, vals in derived["laser_fresnel_alternatives"].items()
            )
            print(f"Fresnel coupling (* = used, {chosen}): {alts}")
        if not args.prepare_only:
            # Pause first: a paused machine should not even be measured for headroom.
            wait_while_paused(default_spool_dir())
            require_free_space(case_dir, args.min_free_gb, args.min_free_wait_s)
            require_machine_headroom(args.min_free_mem_gb, args.max_load, args.min_free_wait_s)
            # Children inherit this, so the solver and its MPI ranks all yield.
            target = nice_target(args.nice, os.getpriority(os.PRIO_PROCESS, 0))
            if target is not None:
                try:
                    os.setpriority(os.PRIO_PROCESS, 0, target)
                    print(f"Niceness raised to {target} so the solver yields CPU to sshd")
                except OSError as exc:      # never block a solve on a scheduling hint
                    print(f"WARNING: could not set niceness to {target}: {exc}")
            solve(case_dir, case_name, args.license, args.flow3d_env, manifest["parallel"])
            print(f"SUCCESS: solver output written to {case_dir}")
            export = args.export
            laser = manifest["input"]["laser"]
            if export in {"npz", "both"}:
                # A stationary spot (laser.dwell_time_s) has no scan speed, so the
                # key is absent; record 0 m/s rather than crashing after a solve
                # that already cost solver time.
                _p_prov, _v_prov = export_provenance_scalars(laser)
                npz = export_all_timesteps_npz(
                    case_dir,
                    case_name,
                    args.post_home,
                    _p_prov,
                    _v_prov,
                )
                print(f"SUCCESS: all stored time steps exported to {npz}")
            if export in {"vtk", "both"}:
                collection = export_all_timesteps(case_dir, case_name, args.post_home)
                print(f"SUCCESS: all stored time steps exported to {collection}")
            if export == "vtk-surface":
                collection = export_all_timesteps(case_dir, case_name, args.post_home, representation="surface")
                print(f"SUCCESS: all stored surface time steps exported to {collection}")
            if export == "vtk-legacy60":
                output_dir = export_legacy_even_timesteps(case_dir, case_name, args.post_home, frames=60)
                print(f"SUCCESS: legacy 60-frame VTK export written to {output_dir}")
            # Before the summary, so its artifact list only names files that exist.
            if not args.keep_flsgrf:
                freed = discard_solver_result(case_dir, case_name)
                if freed:
                    print(f"SUCCESS: solver result discarded after NPZ export, {freed / 1024**3:.2f} GB reclaimed")
            summary = write_run_summary(case_dir, case_name, manifest)
            print(f"SUCCESS: run summary written to {summary}")
    except (ConfigError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
