import lpno.models.ns_energy_pipeline as pipeline


def test_facade_exports_core_symbols() -> None:
    expected = [
        "KeyConfig",
        "RunConfig",
        "SplitSpec",
        "DatasetBundle",
        "ModelBundle",
        "InferenceMetrics",
        "configure_device",
        "seed_all",
        "load_cfg_and_paths",
        "setup_wandb_run",
        "_collect_sim_param_overrides",
        "load_and_preprocess_data",
        "print_dict_shapes",
        "get_splits",
        "build_datasets",
        "build_model_and_nodes",
        "build_domain",
        "solve_domain",
        "run_inference_and_metrics",
        "write_val_error_csv",
        "run_training_pipeline",
    ]

    for name in expected:
        assert hasattr(pipeline, name), f"Missing symbol in ns_energy_pipeline facade: {name}"


def test_facade_symbols_are_callable_where_expected() -> None:
    callable_names = [
        "configure_device",
        "seed_all",
        "load_cfg_and_paths",
        "setup_wandb_run",
        "_collect_sim_param_overrides",
        "load_and_preprocess_data",
        "print_dict_shapes",
        "get_splits",
        "build_datasets",
        "build_model_and_nodes",
        "build_domain",
        "solve_domain",
        "run_inference_and_metrics",
        "write_val_error_csv",
        "run_training_pipeline",
    ]
    for name in callable_names:
        assert callable(getattr(pipeline, name))
