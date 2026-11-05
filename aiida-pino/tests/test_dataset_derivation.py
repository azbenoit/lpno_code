

def test_the_time_avg_error_names_the_other_half_of_the_fix():
    """Alone, this message points at relaxing a correct check.

    It cost three submissions and nearly an request to weaken the guard: each
    validator is individually right, neither knows the other exists, and the
    reader only sees one. The message must say what a steady run actually
    requires — both profiles stating it — not just what is rejected.
    """
    import pytest

    from aiida_pino.utils.dataset_derivation import validate_derivation_profile

    with pytest.raises(ValueError) as excinfo:
        validate_derivation_profile(
            {
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": False,
                "time_avg_window": 30,
                "moving_ref_frame": False,
            }
        )
    message = str(excinfo.value)
    assert "derivation profile" in message.lower()
    assert "training profile" in message.lower()
    assert "explicitly" in message.lower()
