"""A `+`/`++` arg naming a real config section must not vanish into `custom.`.

Reported 2026-08-25. `remap_plus_args_to_custom` prefixes every `+`/`++` argument
with `custom.`, so `++arch.rigno.processor_steps=8` becomes
`++custom.arch.rigno.processor_steps=8`: a key nothing reads, while the real
`arch.rigno.processor_steps` keeps its config default. Nothing warned.

Cost four RIGNO attempts (66684, 66690 and two before them). They trained at
depth 18 instead of the intended 8 and then died with CUDA OOM, because the VRAM
booking had been measured at depth 8 -- so the failure reads as a resource
problem and is a configuration one. The trap was documented and it happened anyway, which is the argument for a guard rather
than more prose.

The remap's behaviour is deliberately unchanged (someone may rely on it); what
changes is that it says so.
"""
import pytest

from lpno.models.argv_custom_remap import remap_plus_args_to_custom


def test_top_level_config_key_is_warned_about(capsys):
    out = remap_plus_args_to_custom(['++arch.rigno.processor_steps=8'])
    warning = capsys.readouterr().out
    assert 'arch.rigno.processor_steps' in warning
    assert 'custom.arch.rigno.processor_steps' in warning, 'name what it became'
    # behaviour unchanged: still remapped
    assert out == ['++custom.arch.rigno.processor_steps=8']


@pytest.mark.parametrize('section', [
    'arch', 'training', 'optimizer', 'scheduler', 'loss', 'stop_criterion',
])
def test_every_known_section_is_covered(section, capsys):
    remap_plus_args_to_custom([f'++{section}.some_key=1'])
    assert section in capsys.readouterr().out, section


def test_genuine_custom_keys_are_silent(capsys):
    remap_plus_args_to_custom(['+melt_dice_weight=50.0', '++custom.foo=1', '+data_dir=/x'])
    assert capsys.readouterr().out == '', 'must not cry wolf on ordinary custom keys'


def test_non_override_args_are_untouched(capsys):
    argv = ['--config-name=config_flow3D_RIGNO', 'training.max_steps=10']
    assert remap_plus_args_to_custom(list(argv)) == argv
    assert capsys.readouterr().out == ''
