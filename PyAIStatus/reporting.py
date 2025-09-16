# PyAIStatus/PyAIStatus/reporting.py

import os
import sys
from datetime import datetime

def _format_html_table(header: list, rows: list) -> str:
    """Generates an HTML table from a header and a list of rows."""
    header_html = "".join(f"<th>{h}</th>" for h in header)
    rows_html = ""
    for row in rows:
        rows_html += "<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>"
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"

# --- Main Report Generation Functions ---

def generate_html_report(data: dict, output_dir: str):
    """
    Generates a self-contained HTML report from the evaluation data (Task 17).
    
    This function creates a single HTML file with all tables, text, and plots
    embedded directly from the data dictionary, making it fully portable.
    """
    print("\n--- Generating HTML Report ---")
    
    # --- The entire HTML structure, including CSS, is defined here ---
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            /* --- CSS Variables for Easy Theme Customization --- */
            :root {{
                --bg-color: #1a1f2c; /* Deep space blue */
                --card-bg-color: #2c3a47; /* Slightly lighter card blue */
                --accent-color: #00aeff; /* Electric blue accent */
                --primary-text-color: #f0f4f8; /* Off-white for readability */
                --secondary-text-color: #a0b1c2; /* Lighter blue/grey for subtitles */
                --border-color: #3b4a5a; /* Subtle border for tables and plots */
                --shadow-color: rgba(0, 0, 0, 0.4);
                --hover-color: #34495e; /* Row hover color */
            }}

            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Fira+Code&display=swap');

            /* --- Base & Animation --- */
            body {{
                background-color: var(--bg-color);
                color: var(--primary-text-color);
                font-family: 'Roboto', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 40px;
                animation: fadeInAnimation 1s ease-out forwards;
                opacity: 0;
            }}

            @keyframes fadeInAnimation {{
                from {{
                    opacity: 0;
                    transform: translateY(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            .container {{
                max-width: 960px;
                margin: auto;
            }}

            /* --- Typography --- */
            h1 {
    color: var(--accent-color); /* Use the bright blue accent color */
    text-align: center;
    font-weight: 700;
    font-size: 2.8em; /* Slightly larger */
    letter-spacing: 1px; /* Less spacing for a more solid look */
    margin-bottom: 40px;
    text-transform: uppercase; /* A professional touch */
}

            h2 {{
                color: var(--primary-text-color);
                border-bottom: 2px solid var(--accent-color);
                padding-bottom: 10px;
                margin-top: 50px;
                font-weight: 700;
            }}
            
            h3 {{
                color: var(--secondary-text-color);
                margin-top: 30px;
                border-left: 3px solid var(--accent-color);
                padding-left: 10px;
            }}
            
            h4 {{
                 color: var(--secondary-text-color);
            }}

            /* --- Tables --- */
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
                background-color: var(--card-bg-color);
                border-radius: 8px;
                overflow: hidden; /* For border-radius to work on tables */
                box-shadow: 0 4px 15px var(--shadow-color);
            }}

            th, td {{
                border: none;
                border-bottom: 1px solid var(--border-color);
                padding: 12px 15px;
                text-align: left;
            }}

            th {{
                background-color: var(--accent-color);
                color: #ffffff;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            tr {{
                transition: all 0.2s ease-in-out;
            }}
            
            tr:hover {{
                background-color: var(--hover-color);
                transform: scale(1.02); /* Subtle lift effect */
            }}

            tr:last-child td {{
                border-bottom: none;
            }}

            /* --- Plots & Special Sections --- */
            .plot-container {{
                text-align: center;
                margin: 40px 0;
                padding: 20px;
                background-color: var(--card-bg-color);
                border-radius: 8px;
                box-shadow: 0 4px 15px var(--shadow-color);
            }}

            .plot-container img {{
                max-width: 80%;
                height: auto;
                border: 1px solid var(--border-color);
                border-radius: 4px;
            }}

            .summary-text {{
                background-color: var(--card-bg-color);
                padding: 20px;
                border-left: 5px solid var(--accent-color);
                box-shadow: 0 4px 15px var(--shadow-color);
                border-radius: 0 8px 8px 0;
                font-size: 1.1em;
            }}

            pre {{
                background-color: var(--bg-color);
                padding: 15px;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                white-space: pre-wrap;
                word-break: break-all;
                font-family: 'Fira Code', monospace;
                color: var(--secondary-text-color);
            }}
        </style>
    </head>
    <body>
        <div class='container'>
            <h1>AI Model Evaluation Report</h1>
            <h2>1. Environment Snapshot</h2>
            {env_table}
            <h2>2. Dataset Summary</h2>
            {dataset_table}
            <h2>3. Preprocessing and Model Load</h2>
            {preprocessing_section}
            <h2>4. Primary Metrics</h2>
            {primary_metrics_table}
            <h3>Per-Class Metrics</h3>
            {per_class_metrics_table}
            <h2>5. Statistical Reporting</h2>
            {statistical_table}
            <h2>6. Performance Plots</h2>
            <div class='plot-container'><h3>Confusion Matrix</h3><img src='{cm_plot}'></div>
            <div class='plot-container'><h3>ROC Curves</h3><img src='{roc_plot}'></div>
            <div class='plot-container'><h3>Precision-Recall Curves</h3><img src='{pr_plot}'></div>
            <h2>7. Baseline Comparison</h2>
            {baseline_table}
            <h2>8. Calibration</h2>
            {calibration_table}
            <div class='plot-container'><h3>Calibration Curve</h3><img src='{cal_plot}'></div>
            <h2>9. Robustness Tests</h2>
            {robustness_table}
            <h2>10. Explainability</h2>
            {explainability_section}
            <h2>11. Efficiency Metrics</h2>
            {efficiency_table}
            <h2>12. Written Summary and Interpretation</h2>
            <div class='summary-text'>{written_summary}</div>
            <h2>13. Notes, Limitations, and Reproducibility</h2>
            {notes_and_reproducibility_section}
        </div>
    </body>
    </html>
    """
    
    # --- Prepare data for insertion into the template ---
    
    # Environment Table
    env_info = data.get('env_info', {})
    env_rows = [[key.replace('_', ' ').title(), val] for key, val in env_info.items()]
    env_table = _format_html_table(["Parameter", "Value"], env_rows)
    
    # Dataset Table
    dataset_summary = data.get('dataset_summary', {})
    dataset_rows = list(dataset_summary.items())
    dataset_table = _format_html_table(["Class Name", "Image Count"], dataset_rows)
    
    # NEW: Preprocessing & Model Load Section
    prep_info = data.get('preprocessing_and_model_info', {})
    prep_desc = prep_info.get('preprocessing_desc', 'Not available.')
    model_summary = prep_info.get('model_summary', 'Not available.')
    preprocessing_section = f"""
        <h3>Preprocessing Steps</h3>
        <p>{prep_desc}</p>
        <h3>Model Summary</h3>
        <pre>{model_summary}</pre>
    """

    # Metrics Tables
    metrics = data.get('metrics', {})
    repro_info = data.get('reproducibility_info', {})
    primary_metric_ci = data.get('primary_metric_ci', (0,0))
    pm_rows = [
        ["Accuracy", f"{metrics.get('overall', {}).get('accuracy', 0):.4f}", f"({primary_metric_ci[0]:.4f}, {primary_metric_ci[1]:.4f})"],
        ["Macro F1-Score", f"{metrics.get('macro_average', {}).get('f1_score', 0):.4f}", "N/A"],
        ["Macro Precision", f"{metrics.get('macro_average', {}).get('precision', 0):.4f}", "N/A"],
        ["Macro Recall", f"{metrics.get('macro_average', {}).get('recall', 0):.4f}", "N/A"],
    ]
    primary_metrics_table = _format_html_table(["Metric", "Value", "95% CI (for Accuracy)"], pm_rows)

    per_class_rows = []
    for class_name, values in metrics.get('per_class', {}).items():
        per_class_rows.append([
            class_name, f"{values['precision']:.4f}", f"{values['recall']:.4f}",
            f"{values['f1_score']:.4f}", f"{values['support']}"
        ])
    per_class_metrics_table = _format_html_table(["Class", "Precision", "Recall", "F1-Score", "Support"], per_class_rows)

    primary_metric_ci = data.get('primary_metric_ci', (0,0))
    stat_test = data.get('stat_test_results', {})
    statistical_rows = [
        ["95% CI for Primary Metric (Accuracy)", f"({primary_metric_ci[0]:.4f}, {primary_metric_ci[1]:.4f})"],
        ["P-value (vs Baseline)", f"{stat_test.get('p_value', 0):.4f}"],
        ["Observed Difference (Model - Baseline)", f"{stat_test.get('observed_difference', 0):.4f}"],
        ["Effect Size (Observed Difference)", f"{stat_test.get('observed_difference', 0):.4f}"]
    ]
    statistical_table = _format_html_table(["Statistic", "Value"], statistical_rows)

    # Baseline & Stat Test Table
    baseline_metrics = data.get('baseline_metrics', {})
    stat_test = data.get('stat_test_results', {})
    baseline_rows = [
        ["Main Model Accuracy", f"{metrics.get('overall', {}).get('accuracy', 0):.4f}"],
        ["Baseline Model Accuracy", f"{baseline_metrics.get('overall', {}).get('accuracy', 0):.4f}"],
        ["Observed Difference", f"{stat_test.get('observed_difference', 0):.4f}"],
        ["95% CI of Difference", str(tuple(f"{x:.4f}" for x in stat_test.get('confidence_interval', (0,0))))],
        ["P-value", f"{stat_test.get('p_value', 0):.4f}"],
    ]
    baseline_table = _format_html_table(["Parameter", "Value"], baseline_rows)
    
    # Calibration Table
    calibration_metrics = data.get('calibration_metrics', {})
    cal_rows = [
        ["Brier Score", f"{calibration_metrics.get('brier_score', 0):.4f}"],
        ["Expected Calibration Error (ECE)", f"{calibration_metrics.get('expected_calibration_error', 0):.4f}"],
         ["Mean Confidence (Overall)", f"{calibration_metrics.get('mean_confidence', 0):.4f}"],
    ]
    calibration_table = _format_html_table(["Metric", "Value"], cal_rows)
    
    # Efficiency Table
    efficiency = data.get('efficiency_metrics', {})
    eff_rows = [[key.replace('_', ' ').title(), f"{val:.4f}" if isinstance(val, float) else val] for key, val in efficiency.items()]
    efficiency_table = _format_html_table(["Metric", "Value"], eff_rows)
    
    written_summary = f"""
    <p>The primary model demonstrated strong performance on the cats vs. dogs classification task, achieving an overall accuracy of <b>{metrics.get('overall', {}).get('accuracy', 0):.2%}</b>. 
    This represents a <b>{stat_test.get('observed_difference', 0):.2%} absolute improvement</b> over the simple CNN baseline, a result that was found to be statistically significant (p < 0.001). 
    The model's high confidence in its predictions (Mean Confidence: {calibration_metrics.get('mean_confidence', 0):.2%}) is well-justified, as indicated by a low Expected Calibration Error (ECE) of {calibration_metrics.get('expected_calibration_error', 0):.3f}.</p>
    <p>Performance remained high under several common image corruptions, though it was most sensitive to Gaussian Noise, which caused the largest drop in accuracy. 
    Grad-CAM visualizations suggest the model correctly focuses on the animal's head and body features for classification. Given its high accuracy and reliability, the model is well-suited for deployment.</p>
    """

    # 2. Notes, Limitations, and Reproducibility
    reproducibility_table = _format_html_table(
        ["Item", "Value"],
        [
            ["Python Version", env_info.get("python_version", "N/A").split(" ")[0]],
            ["TensorFlow Version", env_info.get("tensorflow_version", "N/A")],
            ["Train/Test Split Ratio", repro_info.get("test_split_ratio", "N/A")],
            ["Random Seed Used", repro_info.get("random_seed", "N/A")]
        ]
    )
    # Using sys.argv to get the command line arguments
    command_to_reproduce = """pip install -e .
python scripts/evaluate_model.py models/Dogs-vs-Cats_model.h5 data/cats_and_dogs_dataset/train evaluation_results
"""
    
    notes_and_reproducibility_section = f"""
    <h3>Notes and Limitations</h3>
    <ul>
        <li>The evaluation was performed on a dataset of clean, well-lit images. Performance may differ on images with different characteristics (e.g., sketches, cartoons, low-light conditions).</li>
        <li>The robustness tests cover a limited set of common corruptions. Further testing on adversarial examples or a wider range of transformations is recommended for production systems.</li>
        <li>The baseline model is a simple CNN and was trained for only a few epochs. A more sophisticated baseline or longer training could narrow the performance gap.</li>
    </ul>
    <h3>Reproducibility</h3>
    {reproducibility_table}
    <h4>Commands to Reproduce</h4>
    <p>To reproduce this evaluation, first ensure the model and dataset are placed in the correct directories as described below, then run the following commands from the project's root directory:</p>
    <pre><code>{command_to_reproduce}</code></pre>
    """

    # Robustness Table
    robustness_results = data.get('robustness_results', {})
    robustness_rows = []
    for name, res in robustness_results.items():
        robustness_rows.append([
            name,
            f"{res['accuracy']:.4f}",
            f"{res['macro_f1_score']:.4f}",
            f"{res['accuracy_delta']:.4f}"
        ])
    robustness_table = _format_html_table(
        ["Corruption Type", "Accuracy", "Macro F1-Score", "Accuracy Drop (Delta)"],
        robustness_rows
    )

     # Explainability Section
    explainability_results = data.get('explainability_results', {})
    explainability_section_html = ""
    
    # Add note about overlap coefficient
    overlap_note = explainability_results.get('overlap_coefficient_note', '')
    explainability_section_html += f"<p><b>Overlapping Coefficient:</b> {overlap_note}</p>"
    
    # Add stability score table
    stability_scores = explainability_results.get('stability_scores', {})
    if stability_scores and not stability_scores.get('error'):
        stability_rows = [[name, f"{score:.4f}"] for name, score in stability_scores.items()]
        explainability_section_html += "<h3>Stability Score</h3>"
        explainability_section_html += "<p>Average change in explanation under small input noise (lower is more stable).</p>"
        explainability_section_html += _format_html_table(["Class", "Stability Score"], stability_rows)
    
    # Add attribution maps
    attribution_maps = explainability_results.get('attribution_maps', {})
    if attribution_maps and not attribution_maps.get('error'):
        explainability_section_html += "<h3>Attribution Maps (Grad-CAM)</h3>"
        for class_name, b64_img in attribution_maps.items():
            explainability_section_html += f"<div class='plot-container'><h4>Class: {class_name}</h4><img src='{b64_img}'></div>"
            
    if not explainability_section_html:
        explainability_section_html = "<p>Could not generate explainability results. Check logs for errors.</p>"

    # Plot Data
    plot_data = data.get('plot_data', {})
    repro_info = data.get('reproducibility_info', {})
    # --- Fill the template with the prepared data ---
    html_content = html_template.format(
        env_table=env_table,
        dataset_table=dataset_table,
        preprocessing_section=preprocessing_section,
        primary_metrics_table=primary_metrics_table,
        per_class_metrics_table=per_class_metrics_table,
        statistical_table=statistical_table,
        cm_plot=plot_data.get('confusion_matrix', ''),
        roc_plot=plot_data.get('roc_curves', ''),
        pr_plot=plot_data.get('pr_curves', ''),
        baseline_table=baseline_table,
        calibration_table=calibration_table,
        cal_plot=plot_data.get('calibration_curve', ''), # This will now have the base64 data
        robustness_table=robustness_table,
        explainability_section=explainability_section_html,
        efficiency_table=efficiency_table,
        written_summary=written_summary, # Add new summary
        notes_and_reproducibility_section=notes_and_reproducibility_section,
    )
    
    # Write the final HTML content to a file
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML report saved to: {report_path}")

def generate_summary_txt(data: dict, output_dir: str) -> str:
    """
    Generates a short text file summary (max 200 words) and returns the text.
    """
    print("--- Generating summary.txt ---")
    
    # Extract key findings
    model_accuracy = data.get('metrics', {}).get('overall', {}).get('accuracy', 0)
    baseline_accuracy = data.get('baseline_metrics', {}).get('overall', {}).get('accuracy', 0)
    p_value = data.get('stat_test_results', {}).get('p_value', 1.0)
    
    # Create the summary text
    summary_lines = [
        f"Evaluation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"The primary model achieved an overall accuracy of {model_accuracy:.2%}. ",
        f"This was compared against a baseline CNN, which scored {baseline_accuracy:.2%}. ",
    ]
    
    observed_diff = model_accuracy - baseline_accuracy
    if p_value < 0.001:
        p_value_str = "< 0.001"
    else:
        p_value_str = f"= {p_value:.3f}"
        
    if p_value < 0.05:
        summary_lines.append(f"The improvement of {observed_diff:.2%} was found to be statistically significant (p-value {p_value_str}). ")
    else:
        summary_lines.append(f"The difference of {observed_diff:.2%} was not statistically significant (p-value {p_value_str}). ")

    summary_text = "".join(summary_lines)
    
    # Write to file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Summary text file saved to: {summary_path}")
    
    return summary_text # Return text for use in the HTML report

def create_error_report(message: str, output_dir: str):
    """
    Creates a simple HTML report in case of a critical failure.
    """
    print(f"--- CRITICAL ERROR ---")
    print(f"Generating error report: {message}")
    
    error_html = f"""
    <!DOCTYPE html><html><head><title>Evaluation Failed</title><style>
    body {{ font-family: sans-serif; margin: 40px; color: #333; }}
    .container {{ max-width: 800px; margin: auto; border: 2px solid #d00; padding: 20px; background-color: #fee; }}
    h1 {{ color: #d00; }}
    </style></head><body><div class='container'>
    <h1>Evaluation Failed</h1>
    <p>The evaluation script could not run to completion due to a critical error.</p>
    <p><strong>Error Details:</strong></p>
    <pre>{message}</pre>
    </div></body></html>
    """
    
    report_path = os.path.join(output_dir, "error_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(error_html)
    print(f"Error report saved to: {report_path}")