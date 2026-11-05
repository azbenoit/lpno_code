"""How big, how fast, how much memory — answerable from the database.

AiiDA holds provenance well and performance not at all: every such question
needed a log in the calcjob's scratch workdir, and for finished runs those are
gone. Reported 2026-08-06, with one part of the report corrected by measurement:
the GAOT graph line was believed to live only in the live-tail log, but it is
printed to stdout, which AiiDA retrieves and keeps — present in **23 of 151**
retained training stdouts, so it was recoverable for every GAOT run ever done.
"""

from __future__ import annotations

from aiida_pino.utils.run_metrics import (
    parse_graph_size,
    parse_throughput,
    performance_extras,
)

GRAPH_LINE = (
    "GAOT graph: grid (150, 100, 40) (600000 nodes) <-> latent (50, 34, 40) "
    "(68000 tokens); encoder edges 2400000, decoder edges 4800000\n"
)


def test_the_graph_line_is_parsed_whole():
    parsed = parse_graph_size(GRAPH_LINE)
    assert parsed["graph_grid"] == [150, 100, 40]
    assert parsed["graph_nodes"] == 600000
    assert parsed["graph_latent"] == [50, 34, 40]
    assert parsed["graph_latent_tokens"] == 68000
    assert parsed["graph_encoder_edges"] == 2400000
    assert parsed["graph_decoder_edges"] == 4800000


def test_a_line_that_does_not_match_yields_nothing():
    """A wrong performance number is worse than a missing one."""
    assert parse_graph_size("GraphManager: {'_func_arch': False}") == {}
    assert parse_graph_size("") == {}
    assert parse_graph_size(None) == {}


def test_throughput_comes_from_the_timestamps_already_in_stdout():
    stdout = (
        "[10:00:00] - [step:  1000] loss: 1\n"
        "[10:00:30] - [step:  2000] loss: 0.5\n"
        "[10:01:40] - [step:  3000] loss: 0.4\n"
    )
    parsed = parse_throughput(stdout)
    # Widest pair: 100 s over 2000 steps = 50 ms/iter, so a slow first step is
    # diluted rather than reported as the steady rate.
    assert parsed["ms_per_iter"] == 50.0
    assert parsed["throughput_sampled_steps"] == 2000


def test_a_run_crossing_midnight_is_not_negative():
    stdout = "[23:59:00] - [step: 1000] x\n[00:01:00] - [step: 2000] x\n"
    assert parse_throughput(stdout)["ms_per_iter"] == 120.0


def test_one_step_line_is_not_a_rate():
    assert parse_throughput("[10:00:00] - [step: 1000] loss: 1\n") == {}
    assert parse_throughput("no steps at all") == {}


def test_the_summary_alone_answers_cost():
    """`train_summary` is stored on the node, so this survives everything."""
    extras = performance_extras({
        "steps_trained": 20000,
        "gpu_vram_peak_mb": 18268,
        "host_ram_peak_mb": 41000,
        "runtime": {"wall_seconds": 32300.0, "cpu_user_seconds": 91000.0},
    })
    assert extras["wall_seconds"] == 32300.0
    assert extras["gpu_vram_peak_mb"] == 18268
    assert extras["host_ram_peak_mb"] == 41000
    assert extras["ms_per_iter"] == 1615.0


def test_stdout_measured_throughput_overrides_the_summary_estimate():
    stdout = "[10:00:00] - [step: 0] x\n[10:00:10] - [step: 1000] x\n"
    extras = performance_extras(
        {"steps_trained": 1000, "runtime": {"wall_seconds": 600.0}}, stdout
    )
    assert extras["ms_per_iter"] == 10.0, "the measured rate must win over wall/steps"


def test_nothing_measured_means_nothing_claimed():
    assert performance_extras({}) == {}
    assert performance_extras(None) == {}
    assert performance_extras({"runtime": {"wall_seconds": 0}}) == {}
