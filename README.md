# Performance analysis of the Survival-SVM classifier applied to gene expression databases

Source code of the paper Camele & Hasperué, _Performance analysis of the Survival-SVM classifier applied to gene expression databases_, CACIC (2023).


## Installation

To run the code you need to install the dependencies:

1. Create a virtual env: `python3 -m venv venv` (only once).
2. Activate virtual env: `source venv/bin/activate` (only when used)
    1. To deactivate the virtual env just run: `deactivate`
3. Install all the dependencies: `pip install -r requirements.txt`


## Datasets

The dataset used is [Breast Invasive Carcinoma (TCGA, PanCancer Atlas)][survival-dataset] (which is liste in [cBioPortal datasets page][cbioportal-datasets]). The files `data_clinical_patient.txt` and `data_RNA_Seq_v2_mRNA_median_Zscores.txt` must be placed in the `Datasets` folder.


## Code structure

The main code is in the `core.py` file where the metaheuristic is executed and the results are reported and saved in a CSV file that will go to the `Results` folder.

The `utils.py` file contains import functions, dataset label column binarization, preprocessing, among other useful functions.

The metaheuristics algorithm and its variants can be found in the `metaheuristics.py` file.

The `times.py` file contains the code to evaluate how long the execution takes using different amounts of features. In `plot_times.py` is the code to plot these times using the JSON file generated with the first `plot_times.py` file. script.

The `get_logs_metrics.py` is useful to create a JSON file from logs files (see command to run scripts below) and get the metrics to be plotted in the `plot_times.py` script.

To learn more about SVM Survival or Random Survival Forest models read the [Scikit-survival][scikit-survival-blog] blog.


## Usage

Spark has problems with importing user-defined modules, so we need to leave a file called `scripts.zip` that contains all the necessary modules to be distributed among Spark's Workers. Run the following commands to get everything working:

1. Configure all the experiment's parameters in the `times.py` file.
2. Compress all the needed scripts inside `scripts.zip` running: `./zip_modules.sh`
3. Inside the Spark Cluster's master container run: `spark-submit --py-files scripts.zip times.py &> logs.txt &`. When the execution is finished, a `.csv` with the exact datetime at which the script was run will remain in the `Results` folder with all the results obtained. The file `logs.txt` can be processed by the script `get_logs_metrics.py` to generate a JSON and plot that data, we recommend to store it in the `Logs` folder as the script is pointing to that directory.


<!-- ## Considerations

If you use any part of our code is useful for your research, please consider citing:

```

```
-->


## License

This code is distributed under the MIT license.


[scikit-survival-blog]: https://scikit-survival.readthedocs.io/en/stable/user_guide/understanding_predictions.html
[survival-dataset]: https://cbioportal-datahub.s3.amazonaws.com/brca_tcga_pan_can_atlas_2018.tar.gz
[cbioportal-datasets]: https://www.cbioportal.org/datasets
[so-memory-leak]: https://stackoverflow.com/questions/53105508/pyspark-numpy-memory-not-being-released-in-executor-map-partition-function-mem/71700592#71700592
[paper-link]: #
