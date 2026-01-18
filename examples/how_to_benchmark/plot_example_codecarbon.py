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
# Plotting the results
# --------------------
# We can plot the results using the ``codecarbon_plot`` function, generated
# below. This function takes the dataframe returned by the ``benchmark``
# function as input, and returns a pyplot figure with comprehensive emissions
# analysis.
#
# The function provides multiple visualization options:
#
# **Basic usage (emissions only):**
# Shows CO2 emissions per dataset and algorithm with logarithmic scale.
#
# **With efficiency metrics:**
# Adds a subplot showing energy efficiency (accuracy score per kg CO2).
# Higher bars indicate better efficiency.
#
# **With power vs score analysis:**
# Adds a scatter plot showing the trade-off between accuracy and emissions.
# Pipelines in the upper-right are better (higher accuracy, lower emissions).

# Example 1: Basic emissions visualization
fig1 = codecarbon_plot(results, order_list, country="(France)")

# Example 2: Include efficiency analysis
# This shows which pipelines provide the best accuracy-to-emissions ratio
fig2 = codecarbon_plot(
    results,
    order_list,
    country="(France)",
    include_efficiency=True,
)

# Example 3: Full analysis with accuracy vs emissions trade-off
# This comprehensive view shows three plots:
# 1. CO2 emissions per dataset (log scale)
# 2. Energy efficiency ranking (accuracy / kg CO2)
# 3. Accuracy vs emissions scatter (Pareto frontier)
fig3 = codecarbon_plot(
    results,
    order_list,
    country="(France)",
    include_efficiency=True,
    include_power_vs_score=True,
)

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
# Emissions Summary Report
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Beyond visualizations, you can generate a detailed summary report of
# emissions metrics using the ``emissions_summary`` function. This provides
# a table with comprehensive efficiency metrics for each pipeline.

summary = emissions_summary(results, order_list=order_list)
print("Emissions Summary Report:")
print("=" * 80)
print(summary.to_string())
print("\nKey Metrics:")
print("  - avg_score: Average accuracy across all evaluations")
print("  - avg_emissions: Average CO2 emissions per evaluation (kg)")
print("  - total_emissions: Total CO2 emissions for this pipeline (kg)")
print("  - efficiency: Score per kg CO2 (higher is better)")
print("  - emissions_per_eval: Average emissions per individual evaluation")

# Identify the most efficient pipeline
best_efficiency = summary["efficiency"].idxmax()
print(f"\nMost efficient pipeline: {best_efficiency}")
print(f"  - Accuracy: {summary.loc[best_efficiency, 'avg_score']:.3f}")
print(f"  - Efficiency: {summary.loc[best_efficiency, 'efficiency']:.3f} score/kg CO2")

###############################################################################
# The result expected will be the following image, but varying depending on the
# machine and the country used to run the example.
#
# .. image:: ../../images/example_codecarbon.png
#    :align: center
#    :alt: carbon_example
#
###############################################################################
