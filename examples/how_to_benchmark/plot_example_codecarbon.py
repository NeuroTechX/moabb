"""
=================================================
Benchmarking with MOABB showing the CO2 footprint
=================================================

This example shows how to use MOABB to track the CO2 footprint
using `CodeCarbon library <https://codecarbon.io/>`__.
For this example, we will use only one
dataset to keep the computation time low, but this benchmark is designed
to easily scale to many datasets. Due to limitation of online documentation
generation, the results is computed on a local cluster but could be easily
replicated on your infrastructure.
"""

# Authors: Igor Carrara <igor.carrara@inria.fr>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Ethan Davis <davisethan@gmail.com>
#
# License: BSD (3-clause)

###############################################################################
from moabb import benchmark, set_log_level
from moabb.analysis.plotting import codecarbon_plot, emissions_summary
from moabb.datasets import BNCI2014_001, Zhou2016
from moabb.paradigms import LeftRightImagery


set_log_level("info")

###############################################################################
# Loading the pipelines
# ---------------------
#
# To run this example we use several pipelines, ML and DL (Keras) and also
# pipelines that need an optimization of the hyperparameter.
# All this different pipelines are stored in ``pipelines_codecarbon``

###############################################################################
# Selecting the datasets (optional)
# ---------------------------------
#
# If you want to limit your benchmark on a subset of datasets, you can use the
# ``include_datasets`` and ``exclude_datasets`` arguments. You will need either
# to provide the dataset's object, or a dataset's code. To get the list of
# available dataset's code for a given paradigm, you can use the following
# command:

paradigm = LeftRightImagery()
for d in paradigm.datasets:
    print(d.code)

###############################################################################
# In this example, we will use only the last dataset, 'Zhou 2016', considering
# only the first subject.
#
# Running the benchmark
# ---------------------
#
# The benchmark is run using the ``benchmark`` function. You need to specify the
# folder containing the pipelines to use, the kind of evaluation and the paradigm
# to use. By default, the benchmark will use all available datasets for all
# paradigms listed in the pipelines. You could restrict to specific evaluation
# and paradigm using the ``evaluations`` and ``paradigms`` arguments.
#
# To save computation time, the results are cached. If you want to re-run the
# benchmark, you can set the ``overwrite`` argument to ``True``.
#
# It is possible to indicate the folder to cache the results and the one to
# save the analysis & figures. By default, the results are saved in the
# ``results`` folder, and the analysis & figures are saved in the ``benchmark``
# folder.

dataset = Zhou2016()
dataset2 = BNCI2014_001()
dataset.subject_list = dataset.subject_list[:1]
dataset2.subject_list = dataset2.subject_list[:1]
datasets = [dataset, dataset2]

###############################################################################
# Configuring CodeCarbon Tracking
# --------------------------------
#
# The ``benchmark`` function supports CodeCarbon configuration through the
# ``codecarbon_config`` parameter. This allows fine-grained control over how
# emissions are tracked and reported.
#
# CodeCarbon provides many configuration options:
#
# **Output Options:**
#  - ``save_to_file`` (bool): Save results to CSV file (default: False)
#  - ``log_level`` (str): Logging verbosity (default: 'error')
#  - ``output_dir`` (str): Directory for output files (default: '.')
#  - ``output_file`` (str): CSV filename (default: 'emissions.csv')
#
# **Tracking Options:**
#  - ``tracking_mode`` (str): 'machine' for system-wide, 'process' for isolated
#  - ``measure_power_secs`` (int): Power measurement interval in seconds
#  - ``experiment_name`` (str): Label for the experiment
#  - ``project_name`` (str): Project identifier
#
# **Hardware Options:**
#  - ``gpu_ids`` (str): Comma-separated GPU IDs to track
#  - ``force_cpu_power`` (float): Manual CPU power in watts
#  - ``force_ram_power`` (float): Manual RAM power in watts
#
# **API & Output Backends:**
#  - ``save_to_api`` (bool): Send data to CodeCarbon API
#  - ``api_endpoint`` (str): Custom API endpoint
#  - ``save_to_prometheus`` (bool): Push to Prometheus
#  - ``prometheus_url`` (str): Prometheus server address
#
# **Location & Electricity:**
#  - ``country_2letter_iso_code`` (str): Country code for carbon intensity
#  - ``electricitymaps_api_token`` (str): API token for real-time data
#  - ``pue`` (float): Power Usage Effectiveness of data center
#
# Example 1: Basic configuration with CSV output and verbose logging
codecarbon_config = {
    "save_to_file": True,
    "log_level": "info",
    "output_file": "emissions_results.csv",
    "experiment_name": "MOABB_Benchmark_Zhou2016",
}

results = benchmark(
    pipelines="./pipelines_codecarbon/",
    evaluations=["WithinSession"],
    paradigms=["LeftRightImagery"],
    include_datasets=datasets,
    results="./results/",
    overwrite=False,
    plot=False,
    output="./benchmark/",
    codecarbon_config=codecarbon_config,
)

###############################################################################
# Benchmark prints a summary of the results. Detailed results are saved in a
# pandas dataframe, and can be used to generate figures. The analysis & figures
# are saved in the ``benchmark`` folder.
results.head()

order_list = [
    "CSP + SVM",
    "Tangent Space LR",
    "EN Grid",
    "CSP + LDA Grid",
]

###############################################################################
# Comprehensive CodeCarbon Visualization Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``codecarbon_plot`` function provides multiple visualization modes to
# analyze emissions data from different perspectives. Each mode answers specific
# questions about the sustainability and efficiency of your pipelines.

###############################################################################
# Visualization Mode 1: Basic CO2 Emissions (Default)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This shows the raw CO2 emissions per dataset and algorithm. It helps you
# understand which combinations of dataset and pipeline produce the most
# emissions.
#
# **What it shows:**
#  - X-axis: Different datasets used in benchmarking
#  - Y-axis: CO2 emissions in kg (log scale)
#  - Colors: Different pipeline algorithms
#
# **Best for:** Understanding overall emissions impact

fig1 = codecarbon_plot(results, order_list, country="(France)")
print("Mode 1 created: Basic CO2 emissions visualization")

###############################################################################
# Visualization Mode 2: Energy Efficiency Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This mode adds a subplot showing energy efficiency, calculated as:
# **Efficiency = Accuracy Score / CO2 Emissions (kg)**
#
# Higher efficiency means the pipeline achieves better accuracy with less
# carbon cost. This is the key metric for sustainable machine learning.
#
# **What it shows:**
#  - Bar chart: Pipelines ranked by energy efficiency
#  - Values: Efficiency score (higher is better)
#  - Colors: Pipeline identification
#
# **Best for:** Identifying which pipelines are most sustainable
# **Use case:** When you care about accuracy-to-emissions ratio

fig2 = codecarbon_plot(
    results,
    order_list,
    country="(France)",
    include_efficiency=True,
)
print("Mode 2 created: Added energy efficiency analysis")

###############################################################################
# Visualization Mode 3: Complete Analysis with Pareto Frontier
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This comprehensive mode shows ALL three visualizations:
#  1. CO2 emissions per dataset (shows raw environmental impact)
#  2. Energy efficiency ranking (shows best accuracy/emissions ratio)
#  3. Accuracy vs emissions scatter (shows performance-sustainability trade-off)
#
# The third plot shows the **Pareto frontier**: pipelines in the upper-right
# are Pareto-optimal (you cannot improve accuracy without increasing emissions
# or vice versa).
#
# **What each plot shows:**
#  - Plot 1: Raw emissions across datasets and pipelines
#  - Plot 2: Which pipelines are most efficient (sorted ranking)
#  - Plot 3: Accuracy vs emissions scatter (find the best balance)
#
# **Best for:** Complete sustainability analysis and informed decision-making
# **Use case:** Selecting the best pipeline considering both performance
#              and environmental impact

fig3 = codecarbon_plot(
    results,
    order_list,
    country="(France)",
    include_efficiency=True,
    include_power_vs_score=True,
)
print("Mode 3 created: Complete analysis with Pareto frontier visualization")

###############################################################################
# CodeCarbon Configuration Examples
# ----------------------------------
#
# Below are additional configuration examples for different use cases:
#
# **Example 2: Process-level tracking with custom tracking interval**
# .. code-block:: python
#
#     codecarbon_config = {
#         'tracking_mode': 'process',
#         'measure_power_secs': 30,
#         'save_to_file': True,
#         'log_level': 'debug'
#     }
#
# **Example 3: GPU tracking with specific IDs**
# .. code-block:: python
#
#     codecarbon_config = {
#         'gpu_ids': '0,1,2',  # Track GPUs 0, 1, 2
#         'save_to_file': True,
#         'experiment_name': 'multi_gpu_benchmark'
#     }
#
# **Example 4: Real-time carbon intensity data with Electricity Maps API**
# .. code-block:: python
#
#     codecarbon_config = {
#         'electricitymaps_api_token': 'your-token-here',
#         'country_2letter_iso_code': 'FR',
#         'save_to_file': True,
#         'output_file': 'emissions_real_time.csv'
#     }
#
# **Example 5: API-based tracking and reporting**
# .. code-block:: python
#
#     codecarbon_config = {
#         'save_to_api': True,
#         'api_endpoint': 'https://api.codecarbon.io',
#         'api_key': 'your-api-key',
#         'project_name': 'MOABB_Project'
#     }
#
# **Example 6: Prometheus metrics export**
# .. code-block:: python
#
#     codecarbon_config = {
#         'save_to_prometheus': True,
#         'prometheus_url': 'http://localhost:9091',
#         'experiment_name': 'moabb_metrics'
#     }
#
# **Example 7: Custom data center with manual power specifications**
# .. code-block:: python
#
#     codecarbon_config = {
#         'force_cpu_power': 150.0,  # Watts
#         'force_ram_power': 20.0,   # Watts
#         'pue': 1.2,                # Data center PUE
#         'save_to_file': True
#     }

###############################################################################
# Emissions Summary Report and Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Beyond visualizations, you can generate a detailed summary report using
# the ``emissions_summary`` function. This provides comprehensive metrics
# for data-driven decision making.

summary = emissions_summary(results, order_list=order_list)
print("\n" + "=" * 80)
print("EMISSIONS SUMMARY REPORT")
print("=" * 80)
print("\nDetailed Metrics Table:")
print(summary.to_string())

###############################################################################
# Understanding the Metrics
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The summary report includes the following columns:
#
# **Performance Metrics:**
#  - avg_score: Average accuracy/performance across all evaluations
#  - std_score: Standard deviation (variability) of accuracy
#
# **Emissions Metrics:**
#  - avg_emissions: Average CO2 per evaluation in kg
#  - total_emissions: Total CO2 for all evaluations in kg
#  - emissions_per_eval: Average emissions per fold
#
# **Efficiency Metrics:**
#  - efficiency: **Score / kg CO2** (MOST IMPORTANT - higher is better)
#  - n_evaluations: Number of evaluations performed

print("\n" + "=" * 80)
print("METRIC EXPLANATIONS")
print("=" * 80)
metrics_info = {
    "avg_score": "Higher = Better accuracy",
    "std_score": "Lower = More consistent accuracy",
    "avg_emissions": "Lower = Less carbon per evaluation",
    "total_emissions": "Lower = Less total carbon footprint",
    "efficiency": "HIGHER = Better (accuracy with less carbon)",
    "n_evaluations": "Number of CV folds evaluated",
}
for metric, explanation in metrics_info.items():
    print(f"  {metric:20s}: {explanation}")

###############################################################################
# Sustainability Analysis
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Identify the most sustainable and efficient pipelines.

print("\n" + "=" * 80)
print("SUSTAINABILITY RANKINGS")
print("=" * 80)

# Find best efficiency
best_efficiency = summary["efficiency"].idxmax()
worst_efficiency = summary["efficiency"].idxmin()
print("\n1. Most Efficient Pipeline (Best accuracy-to-emissions ratio):")
print(f"   Pipeline: {best_efficiency}")
print(f"   - Accuracy: {summary.loc[best_efficiency, 'avg_score']:.4f}")
print(f"   - Efficiency: {summary.loc[best_efficiency, 'efficiency']:.4f} score/kg CO2")
print(
    f"   - Total emissions: {summary.loc[best_efficiency, 'total_emissions']:.6f} kg CO2"
)

print("\n2. Lowest Emissions Pipeline:")
lowest_emissions = summary["avg_emissions"].idxmin()
print(f"   Pipeline: {lowest_emissions}")
print(
    f"   - Avg emissions: {summary.loc[lowest_emissions, 'avg_emissions']:.6f} kg CO2/eval"
)
print(f"   - Accuracy: {summary.loc[lowest_emissions, 'avg_score']:.4f}")

print("\n3. Highest Accuracy Pipeline:")
best_accuracy = summary["avg_score"].idxmax()
print(f"   Pipeline: {best_accuracy}")
print(f"   - Accuracy: {summary.loc[best_accuracy, 'avg_score']:.4f}")
print(f"   - Efficiency: {summary.loc[best_accuracy, 'efficiency']:.4f} score/kg CO2")

print("\n4. Efficiency Comparison:")
for pipeline in summary.index:
    efficiency = summary.loc[pipeline, "efficiency"]
    accuracy = summary.loc[pipeline, "avg_score"]
    emissions = summary.loc[pipeline, "avg_emissions"]
    print(
        f"   {pipeline:25s}: {efficiency:6.4f} score/kg | {accuracy:.4f} acc | {emissions:.6f} kg CO2/eval"
    )

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print(f"\nMost Sustainable Choice: {best_efficiency}")
print("  → Best balance of accuracy and environmental impact")
print(
    f"  → Efficiency score: {summary.loc[best_efficiency, 'efficiency']:.4f} score/kg CO2"
)

if best_accuracy != best_efficiency:
    efficiency_loss = (
        (
            summary.loc[best_accuracy, "avg_score"]
            - summary.loc[best_efficiency, "avg_score"]
        )
        / summary.loc[best_accuracy, "avg_score"]
        * 100
    )
    emissions_saving = (
        (
            summary.loc[best_accuracy, "avg_emissions"]
            - summary.loc[best_efficiency, "avg_emissions"]
        )
        / summary.loc[best_accuracy, "avg_emissions"]
        * 100
    )
    print(f"\nSwitch from {best_accuracy} to {best_efficiency}:")
    print(f"  → Accuracy reduction: {efficiency_loss:.1f}%")
    print(f"  → Carbon savings: {emissions_saving:.1f}%")
    print(
        f"  → Better efficiency: {summary.loc[best_efficiency, 'efficiency'] / summary.loc[best_accuracy, 'efficiency']:.2f}x more sustainable"
    )

###############################################################################
# The result expected will be the following image, but varying depending on the
# machine and the country used to run the example.
#
# .. image:: ../../images/example_codecarbon.png
#    :align: center
#    :alt: carbon_example
#
###############################################################################
