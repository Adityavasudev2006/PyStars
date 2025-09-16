# PyAIStatus/PyAIStatus/evaluation.py

from . import (
    environment,
    data,
    model,
    preprocessing,
    metrics,
    plots,
    robustness,
    explainability,
    efficiency,
    baseline,
    statistical_tests,
    reporting,
)

def evaluate(model_path: str, dataset_dir: str, output_dir: str):
    
    # 1. Environment snapshot
    env_info = environment.get_environment_snapshot()

    # 2. Data list and counts
    class_names, image_counts = data.get_dataset_summary(dataset_dir)
    if not class_names:
        reporting.create_error_report("Failed to read dataset summary.", output_dir)
        return
    dataset_summary = dict(zip(class_names, image_counts))

    # 3. Data split
    TEST_SPLIT_RATIO = 0.2
    RANDOM_SEED = 42
    train_df, test_df = data.split_data(dataset_dir, test_size=TEST_SPLIT_RATIO, seed=RANDOM_SEED)
    if train_df is None:
        reporting.create_error_report("Failed to split data.", output_dir)
        return

    # 4. Load model
    keras_model, model_summary_str = model.load_keras_model(model_path)
    if not keras_model:
        reporting.create_error_report("Failed to load Keras model.", output_dir)
        return
    
    # Create the preprocessing description string
    model_input_shape = keras_model.input_shape[1:3]
    preprocessing_desc = (
        f"The model expects input images of size {model_input_shape}. "
        "The preprocessing pipeline resizes all test images to this target size "
        "and normalizes pixel values to the [0, 1] range by dividing by 255."
    )

    # 5. Preprocessing
    test_generator = preprocessing.create_data_generator(test_df, image_size=model_input_shape)

    # 6. Model inference
    predictions = model.run_inference(keras_model, test_generator)
    if predictions is None:
        reporting.create_error_report("Model inference failed.", output_dir)
        return

    # 7. Primary metrics and per-class metrics
    all_metrics = metrics.compute_all_metrics(test_generator.classes, predictions, class_names)

    # 8. Confusion matrix and ROC/PR curves
    # This now returns a dictionary of base64 encoded plot images
    plot_data = plots.generate_all_plots(test_generator.classes, predictions, class_names, output_dir)

    # 9. Bootstrapped confidence intervals
    primary_metric_ci = metrics.get_bootstrapped_ci(
    test_generator.classes, predictions
)

    # 10. Baseline comparison
    # Note: baseline.train_simple_cnn expects train_df and a val_df. For simplicity here, we can pass train_df twice.
    # In a real scenario, you'd create a validation split from the training data.
    baseline_model = baseline.train_simple_cnn(train_df, train_df) 
    baseline_predictions = baseline.evaluate_baseline(baseline_model, test_df)
    baseline_metrics = metrics.compute_all_metrics(test_generator.classes, baseline_predictions, class_names)


    # 11. Statistical test
    stat_test_results = statistical_tests.compare_models(
        test_generator.classes, predictions, baseline_predictions
    )

    # 12. Calibration and uncertainty
    calibration_metrics = metrics.compute_calibration_metrics(
        test_generator.classes, predictions
    )
    
    #13
    robustness_results = robustness.run_robustness_tests(
        keras_model, 
        test_df, 
        all_metrics['overall']['accuracy'], # Pass clean accuracy for delta calculation
        class_names
    )

    # 14. Explainability
    explainability_results = explainability.run_explainability_tasks(
        keras_model, test_generator, class_names
    )

    # 15. Efficiency Metrics
    efficiency_metrics = efficiency.get_efficiency_metrics(model_path)
    
    # 16. Generate Summary Text (moved so it can be included in the HTML report)
    summary_text = reporting.generate_summary_txt({
        "metrics": all_metrics,
        "baseline_metrics": baseline_metrics,
        "stat_test_results": stat_test_results,
    }, output_dir)


    # 17. Assemble final data package for the report
    report_data = {
        "env_info": env_info,
        "dataset_summary": dataset_summary,
        "preprocessing_and_model_info": {
            "preprocessing_desc": preprocessing_desc,
            "model_summary": model_summary_str
        },
        "reproducibility_info": {
            "test_split_ratio": TEST_SPLIT_RATIO,
            "random_seed": RANDOM_SEED
        },
        "metrics": all_metrics,
        "primary_metric_ci": primary_metric_ci,
        "baseline_metrics": baseline_metrics,
        "stat_test_results": stat_test_results,
        "calibration_metrics": calibration_metrics,
        "robustness_results": robustness_results,
        "explainability_results": explainability_results,
        "efficiency_metrics": efficiency_metrics,
        "plot_data": plot_data,  # Contains all base64 plot strings
        "summary_text": summary_text, # Add the generated summary
    }
    
    # Generate the final HTML report
    reporting.generate_html_report(report_data, output_dir)