# Active node stack: masking / liquid-fraction / spatial-align post-processing.
# The physics-loss PDE residuals (energy, momentum, VOF, FVM, boundary
# conditions) were deprecated and moved to `deprecated.equations`; the
# `pde_loss` branches in `models/pipeline/model_domain.py` import them lazily
# from there.
from .FNO_postprocess import *
