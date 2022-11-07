import json
from math import ceil
from typing import List, Optional, Callable, Union, Tuple, Any
from metaheuristics import binary_black_hole, improved_binary_black_hole, binary_black_hole_spark, \
    CrossValidationSparkResult, parallelize_fitness_execution_by_partitions
import os
import pandas as pd
from pyspark import SparkContext, Broadcast
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import DataFrame as SparkDataFrame
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
from utils import get_columns_from_df, read_survival_data, WorkerTimeDict, store_times, report_exec_and_idle_times
import time
import logging
import matplotlib.pyplot as plt

logging.getLogger().setLevel(logging.INFO)

# Prevents 'A value is trying to be set on a copy of a slice from a DataFrame.' error
pd.options.mode.chained_assignment = None

# Number of groups resulting from the clustering algorithm
NUMBER_OF_GROUPS = 3

# Prevents 'A value is trying to be set on a copy of a slice from a DataFrame.' error
pd.options.mode.chained_assignment = None

# Algunos tipos utiles
ParameterFitnessFunctionSequential = Tuple[pd.DataFrame, np.ndarray]
ParsedDataCallable = Callable[[np.ndarray, Any, np.ndarray], Union[ParameterFitnessFunctionSequential, SparkDataFrame]]

# Fitness function result structure. It's a function that takes a Pandas DF/Spark Broadcast variable, the bool subset
# of features, the original data (X), the target vector (Y) and a bool flag indicating if it's a broadcast variable
CrossValidationCallback = Callable[[Union[pd.DataFrame, Broadcast], np.ndarray, np.ndarray, bool],
                                   CrossValidationSparkResult]


def fitness_function_with_checking(
        compute_cross_validation: CrossValidationCallback,
        index_array: np.ndarray,
        x: Union[pd.DataFrame, Broadcast],
        y: np.ndarray,
        is_broadcast: bool
) -> CrossValidationSparkResult:
    """
    Funcion de fitness de una estrella evaluada en el Binary Black hole, incluye chequeo de vector sin features

    :param compute_cross_validation: Funcion de Cross valitadion incluida la funcion de fitness
    :param index_array: Lista de booleanos indicando cual feature debe ser incluido en la evaluacion y cual no
    :param x: Data with features
    :param y: Classes
    :param is_broadcast: True if x is a Spark Broadcast to retrieve its values
    :return: Promedio de la metrica obtenida en cada fold del CrossValidation. -1 si no hay features a evaluar
    """
    if not np.count_nonzero(index_array):
        return -1.0, -1.0, -1, '', -1, '', -1.0, -1.0, -1.0

    return compute_cross_validation(x, index_array, y, is_broadcast)


def run_experiment(
        compute_cross_validation: CrossValidationCallback,
        metric_description: str,
        model_description: str,
        add_epsilon: bool,
        filter_top_50: bool = True,
        run_improved_bbha: Optional[bool] = None,
        run_in_spark: bool = False,
        sc: Optional[SparkContext] = None,
        number_of_independent_runs: int = 5,
        n_stars: int = 10,
        n_iterations: int = 25,
        coeff_1: float = 2.35,
        coeff_2: float = 0.2,
        binary_threshold: Optional[float] = None,
        use_broadcasts_in_spark: Optional[bool] = True
):
    """
    Hace lo mismo que run_experiment pero con datos de supervivencia en vez de los datasets con clases categoricas

    :param compute_cross_validation: Funcion de Cross valitadion incluida la funcion de fitness
    :param metric_description: Descripcion de la metrica que devuelve la funcion de CrossValidation para mostrar en el CSV
    :param model_description: Descripcion del modelo utilizado como funcion de fitness de CrossValidation para mostrar en el CSV
    :param add_epsilon: If True it adds an epsilon to 0s in Y data to prevent errors in SVM training
    :param filter_top_50: True si se quiere alimentar al BBHA con el top 50 de feature, False para usar todos
    :param run_improved_bbha: If None runs both algorithm versions. True for improved, False to run the original
    :param run_in_spark: True to run the stars of the BBHA in a distributed Apache Spark cluster
    :param sc: Spark Context
    :param number_of_independent_runs: Numero de corridas independientes a ejecutar
    :param n_stars: Numero de estrellas a utilizar en el BBHA
    :param n_iterations: Numero de iteraciones a utilizar en el BBHA
    :param coeff_1: Coeficiente 1 requerido por la version mejorada del BBHA
    :param coeff_2: Coeficiente 2 requerido por la version mejorada del BBHA
    :param binary_threshold: Threshold usado en BBHA, None para que se compute de manera aleatoria
    :param use_broadcasts_in_spark: If True, it generates a Broadcast value to pass to the fitness function instead of pd.DataFrame. Is ignored if run_in_spark = False
    """
    # CSV donde se van a guardar las cosas
    now = time.strftime('%Y-%m-%d_%H_%M_%S')
    dir_name = os.path.dirname(__file__)

    # Configures CSV file
    res_csv_file_path = os.path.join(dir_name, f'Results/{now}/result_{now}.csv')

    logging.info(f'Results will be saved in {res_csv_file_path}')

    # Creates a folder to save all the results and figures
    mode = 0o777
    dir_path = os.path.join(dir_name, f'Results/{now}')
    os.mkdir(dir_path, mode)

    res_csv = pd.DataFrame(columns=['dataset', 'Improved BBHA', 'Model',
                                    f'Best {metric_description} (in {number_of_independent_runs} runs)',
                                    f'Features with best {metric_description} (in {number_of_independent_runs} runs)',
                                    f'CPU execution time ({number_of_independent_runs} runs) in seconds'])

    # Obtiene los datos necesarios de supervivencia
    x, y = read_survival_data(add_epsilon)

    number_samples, number_features = x.shape

    logging.info(f'Dataset de supervivencia')
    logging.info(f'\tSamples (filas) -> {number_samples} | Features (columnas) -> {number_features}')
    logging.info(f'\tY shape -> {y.shape}')

    # Gets concordance index with all the features
    start = time.time()
    all_features_concordance_index = compute_cross_validation(x, np.ones(number_features), y, use_broadcasts_in_spark)

    # In Spark it's only the fitness value it's the first value
    if run_in_spark:
        all_features_concordance_index = all_features_concordance_index[0]

    logging.info(f'Cross validation con todos los features terminado en {time.time() - start} segundos')
    logging.info(f'Concordance index with all the features -> {all_features_concordance_index}')

    # Hace primero un Feature Selection filtrando el top por Chi Square Test
    if filter_top_50:
        # FIXME: hay que solucionar el filtrado teniendo en cuenta los datos de supervivencia
        top_features = get_most_related_pairs(x, n=50)
        x = x[top_features]

        # Imprimo para probar
        logging.info(f'Top 50 Features obtenidos por Chi Square -> ')
        logging.info(top_features)
        # print(x)
        # print(top_features)
        # print(x.columns.values)

        # Gets concordance index with the top features
        # TODO: uncomment both lines when N best features filtering is implemented
        # initial_concordance_index = compute_cross_validation(x, y)
        # logging.info(f'Concordance index with top features -> {initial_concordance_index}')

    # Needed parameter for the Binary Black Hole Algorithm
    n_features = x.shape[1]

    # Check which version of the algorithm want to run
    if run_improved_bbha is None:
        improved_options = [False, True]
    elif run_improved_bbha is True:
        improved_options = [True]
    else:
        improved_options = [False]

    # If it was set, generates a broadcast value
    using_broadcast = run_in_spark and use_broadcasts_in_spark
    if using_broadcast:
        logging.info('Using Broadcast')
        x = sc.broadcast(x)

    experiment_start = time.time()
    for run_improved in improved_options:
        improved_mode_str = 'improved' if run_improved else 'normal'
        spark_mode_str = '(in Spark)' if run_in_spark else ''
        logging.info(f'Running {improved_mode_str} algorithm {spark_mode_str}')
        independent_start_time = time.time()

        final_subset = None  # Final best subset
        best_metric = -1  # Final best metric

        for i in range(number_of_independent_runs):
            # Binary Black Hole
            bh_start = time.time()
            if run_improved:
                best_subset, current_metric = improved_binary_black_hole(
                    n_stars=n_stars,
                    n_features=n_features,
                    n_iterations=n_iterations,
                    fitness_function=lambda subset: fitness_function_with_checking(
                        compute_cross_validation,
                        subset,
                        x,
                        y,
                        is_broadcast=using_broadcast
                    ),
                    coeff_1=coeff_1,
                    coeff_2=coeff_2,
                    binary_threshold=binary_threshold,
                    debug=True
                )
            else:
                if run_in_spark:
                    best_subset, current_metric, _best_data = binary_black_hole_spark(
                        n_stars=n_stars,
                        n_features=n_features,
                        n_iterations=n_iterations,
                        fitness_function=lambda subset: fitness_function_with_checking(
                            compute_cross_validation,
                            subset,
                            x,
                            y,
                            is_broadcast=using_broadcast
                        ),
                        sc=sc,
                        binary_threshold=binary_threshold,
                        debug=True
                    )
                else:
                    best_subset, current_metric = binary_black_hole(
                        n_stars=n_stars,
                        n_features=n_features,
                        n_iterations=n_iterations,
                        fitness_function=lambda subset: fitness_function_with_checking(
                            compute_cross_validation,
                            subset,
                            x,
                            y,
                            is_broadcast=using_broadcast
                        ),
                        binary_threshold=binary_threshold,
                        debug=True
                    )

            logging.info(f'Independent run {i + 1} of {number_of_independent_runs} | '
                         f'Binary Black Hole with {n_iterations} iterations y {n_stars} '
                         f'stars, finished in {time.time() - bh_start} seconds')

            # Check if current is the best metric
            if current_metric > best_metric:
                best_metric = current_metric

                # Gets columns names
                x_df = x.value if run_in_spark and use_broadcasts_in_spark else x
                column_names = get_columns_from_df(best_subset, x_df).columns.values
                final_subset = column_names

        # Reports final result
        independent_run_time = round(time.time() - independent_start_time, 3)
        logging.info(f'{number_of_independent_runs} indenpendent runs finished in {independent_run_time} seconds')

        experiment_results_dict = {
            'dataset': '',
            'Improved BBHA': 1 if run_improved else 0,
            'Model': model_description,
            f'Best {metric_description} (in {number_of_independent_runs} runs)': round(best_metric, 4),
            f'Features with best {metric_description} (in {number_of_independent_runs} runs)': ' | '.join(final_subset),
            f'CPU execution time ({number_of_independent_runs} runs) in seconds': independent_run_time
        }

        # Some extra reporting
        algorithm = 'BBHA' + (' (improved)' if run_improved else '')
        logging.info(f'Features con {algorithm} ({metric_description} '
                     f'= {best_metric}) ->')
        logging.info(final_subset)

        # Saves new data to final CSV
        res_csv = res_csv.append(experiment_results_dict, ignore_index=True)
        res_csv.to_csv(res_csv_file_path)

    logging.info(f'Experiment completed in {time.time() - experiment_start} seconds')


def plot_charts(x: pd.DataFrame, y: np.ndarray, final_subset: np.ndarray, run_improved: bool, save: bool = False):
    """
    TODO: add docs
    """
    now = time.strftime('%Y-%m-%d_%H_%M_%S')

    # First plots Kaplan Meier and runs cross validation with all the features
    plot_kaplan_meier(f'Results/{now}/kaplan_meier_{now}_antes_BBHA', x, y, save)

    # Shows Kaplan Meier plot with bests parameters
    x_best_subset = get_columns_from_df(final_subset, x)
    improved_str = '_improved' if run_improved else ''
    plot_kaplan_meier(f'Results/{now}/kaplan_meier_{now}_despues_BBHA{improved_str}', x_best_subset, y, save)


def plot_kaplan_meier(fig_path: str, data_x: pd.DataFrame, data_y: np.ndarray, save_fig: bool):
    """
    TODO: add docs
    """
    # k_means_result = KMeans(n_clusters=NUMBER_OF_GROUPS).fit(data_x.values)
    # k_means_result = DBSCAN(eps=0.3, min_samples=10).fit(data_x.values)
    k_means_result = MiniBatchKMeans(
        init="k-means++",
        n_clusters=NUMBER_OF_GROUPS,
        batch_size=100,
        n_init=10,
        max_no_improvement=10,
        random_state=0
    ).fit(data_x.values)

    data_x['group'] = k_means_result.labels_
    for group in range(NUMBER_OF_GROUPS):
        mask_group = data_x["group"] == group
        group_y_data = data_y[mask_group]
        number_of_samples = group_y_data.shape[0]
        # if number_of_samples < 50:
        #     continue

        logging.info(f'Samples in group {group} -> {number_of_samples}')
        time_treatment, survival_prob_treatment = kaplan_meier_estimator(
            group_y_data['event'],
            group_y_data['time']
        )

        plt.step(time_treatment, survival_prob_treatment, where="post",
                 label="Group = %s" % group)

    # Removes 'group' column
    data_x.drop('group', axis=1, inplace=True)

    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")
    if save_fig:
        plt.savefig(fig_path)
    plt.show()
    plt.clf()


def get_most_related_pairs(df: pd.DataFrame, method='pearson', n: int = 5) -> pd.Series:
    """
    Computa los pares de genes mas correlacionados
    :param df: DataFrame para computa la correlacion entre todos los pares
    :param method: Metodo de correlacion a utilizar: 'pearson', 'kendall', 'spearman'
    :param n: Cantidad de pares a devolver (estan ordenados decrecientemente)
    :return: Series de Pandas con los N pares de genes mas correlacionados
    """
    corr_matrix = df.corr(method=method).abs()

    # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
           .stack()
           .sort_values(ascending=False))
    return sol[0: n + 1]


def filter_by_chi_squared(x: pd.DataFrame, y: np.ndarray) -> List[str]:
    """
    Según el paper, se realiza un filtrado previo dejando el top de 50 features mas relevantes
    segun el test estadístico Chi Square.
    Ver -> https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223
    :param x: DataFrame con los features
    :param y: Labels
    :return: Top 50 de features mas significativos
    """
    # TODO: chequear si esta bien. chi2 no acepta numero negativos asi que estoy normalizando [0-1]
    # Escala entre [0-1]
    x_scaled = MinMaxScaler().fit_transform(x)
    scaled_df = pd.DataFrame(x_scaled, columns=x.columns, index=x.index)

    # Computa el Chi-Square
    chi_scores = chi2(scaled_df, y)

    # Ordena por p-valor y devuelve los primeros 50
    p_values = pd.Series(chi_scores[1], index=x.columns)
    p_values.sort_values(ascending=True, inplace=True)
    return p_values.index.values[:50]


def assign_ids(stars_subsets: np.ndarray, number_of_workers: int):
    """
    Assigns partitions IDs equally among all the stars
    :param stars_subsets: Stars numpy array. First element is index, second is subset of features
    :param number_of_workers: Number of workers to compute the partitions equally
    :return:
    """
    current_n_stars = len(stars_subsets)
    rows_per_partition = ceil(current_n_stars / number_of_workers)
    partition_id = 0
    current_total_rows_per_part = 0
    last_row = number_of_workers - 1
    for i in range(current_n_stars):
        stars_subsets[i][0] = partition_id
        current_total_rows_per_part += 1
        if current_total_rows_per_part == rows_per_partition and partition_id != last_row:
            partition_id += 1
            current_total_rows_per_part = 0


def run_times_experiment(
        compute_cross_validation: CrossValidationCallback,
        n_iterations: int,
        metric_description: str,
        model_description: str,
        add_epsilon: bool,
        sc: Optional[SparkContext] = None,
        use_broadcasts_in_spark: Optional[bool] = True
):
    """
    Hace lo mismo que run_experiment pero con datos de supervivencia en vez de los datasets con clases categoricas

    :param n_iterations: Number of iterations
    :param compute_cross_validation: Fitness function
    :param metric_description: Metric description to report in results
    :param model_description: model description to report in results
    :param add_epsilon: If True it adds an epsilon to 0s in Y data to prevent errors in SVM training
    :param sc: Spark Context
    :param use_broadcasts_in_spark: If True, it generates a Broadcast value to pass to the fitness function instead of pd.DataFrame. Is ignored if run_in_spark = False
    """
    # Obtiene los datos necesarios de supervivencia
    x, y = read_survival_data(add_epsilon)

    number_samples, number_features = x.shape
    n_stars = 15  # Five stars per worker
    step = 100

    logging.info(f'Running times experiment with {n_iterations} iterations and {n_stars} stars ({n_stars // 3} stars '
                 f'per worker)')
    logging.info(f'Metric -> {metric_description} | Model -> {model_description}')
    logging.info(f'Survival dataset')
    logging.info(f'\tSamples (rows) -> {number_samples} | Features (columns) -> {number_features}')
    logging.info(f'\tY shape -> {y.shape}')

    # Needed parameter for the Binary Black Hole Algorithm
    total_n_features = x.shape[1]

    if use_broadcasts_in_spark:
        x = sc.broadcast(x)

        # Runs an initial experiment with 1000 features to broadcast the data and prevent issues with execution times
        # due to data distribution
        initial_n_features = 1000
        stars_subsets_initial = np.empty((n_stars, 2), dtype=object)  # 2 = (1, features)
        for i in range(n_stars):
            random_features_to_select_initial = np.zeros(total_n_features, dtype=int)
            random_features_to_select_initial[:initial_n_features] = 1
            np.random.shuffle(random_features_to_select_initial)
            stars_subsets_initial[i] = (i + 1, random_features_to_select_initial)

        _results_values, total_initial_time = parallelize_fitness_execution_by_partitions(
            sc,
            stars_subsets_initial,
            fitness_function=lambda subset: fitness_function_with_checking(
                compute_cross_validation,
                subset,
                x,
                y,
                is_broadcast=use_broadcasts_in_spark
            )
        )
        logging.info(f'Initial running finished in {total_initial_time}')
    else:
        logging.info(f'Broadcasting disabled. Initial run with all features discarded ')

    execution_times: WorkerTimeDict = {}
    idle_times: WorkerTimeDict = {}

    # Lists for reporting
    number_of_features: List[int] = []
    time_exec: List[float] = []
    idle_exec: List[float] = []
    times_by_iteration: List[float] = []
    time_test: List[float] = []
    num_of_iterations: List[float] = []
    train_scores: List[float] = []

    # Runs the iterations
    for i_iter in range(n_iterations):
        logging.info(f'Iteration {i_iter + 1}/{n_iterations}')

        stars_subsets = np.empty((n_stars, 2), dtype=object)  # 2 = (1, features)
        current_n_features = step

        while current_n_features <= total_n_features:
            for i in range(n_stars):
                # Initializes 'Population' with a key for partitionBy()
                random_features_to_select = np.zeros(total_n_features, dtype=int)
                random_features_to_select[:current_n_features] = 1
                np.random.shuffle(random_features_to_select)
                # stars_subsets[i] = (i + 1, random_features_to_select)
                stars_subsets[i] = (i, random_features_to_select)

                # Jumps by 'step' elements
                current_n_features += step

                # If it's arraised the maximum number of features, slices the stars array
                if current_n_features > total_n_features:
                    stars_subsets = stars_subsets[:i + 1]
                    break

            # Assigns partition IDs
            assign_ids(stars_subsets, number_of_workers=3)

            results_values, total_time = parallelize_fitness_execution_by_partitions(
                sc,
                stars_subsets,
                fitness_function=lambda subset: fitness_function_with_checking(
                    compute_cross_validation,
                    subset,
                    x,
                    y,
                    is_broadcast=use_broadcasts_in_spark
                )
            )

            for init_idx in range(len(stars_subsets)):
                current_data = results_values[init_idx]
                worker_time = current_data[1]
                evaluated_features = current_data[4]
                time_by_iteration = current_data[6]
                model_test_time = current_data[7]
                mean_num_of_iterations = current_data[8]
                train_score = current_data[9]
                idle_time = total_time - worker_time

                number_of_features.append(evaluated_features)
                time_exec.append(round(worker_time, 4))
                idle_exec.append(round(idle_time, 4))
                times_by_iteration.append(round(time_by_iteration, 4))
                time_test.append(round(model_test_time, 4))
                num_of_iterations.append(round(mean_num_of_iterations, 4))
                train_scores.append(round(train_score, 4))

                store_times(init_idx, current_data, total_time, execution_times, idle_times, debug=True)

    report_exec_and_idle_times(execution_times, idle_times)

    # Saves times in JSON for post-processing
    now = time.strftime('%Y-%m-%d_%H_%M_%S')
    json_file = f'{now}_times.json'
    json_dest = os.path.join('Times results', json_file)

    logging.info(f'Saving lists in JSON format in {json_dest}')
    result_dict = {
        'n_features': number_of_features,
        'execution_times': time_exec,
        'idle_times': idle_exec,
        'times_by_iteration': times_by_iteration,
        'test_times': time_test,
        'train_scores': train_scores,
        'num_of_iterations': num_of_iterations,
    }

    with open(json_dest, 'w+') as file:
        file.write(json.dumps(result_dict))

    logging.info('Saved.')


def fitness_function_with_checking_sequential(
        compute_cross_validation: Callable[[pd.DataFrame, np.ndarray], float],
        index_array: np.array,
        x: Union[pd.DataFrame, Broadcast],
        y: np.ndarray,
) -> float:
    """
    Fitness function of a star evaluated in the Binary Black hole, including featureless vector check for sequential
    experiment.

    :param compute_cross_validation: Funcion de Cross valitadion incluida la funcion de fitness
    :param index_array: Lista de booleanos indicando cual feature debe ser incluido en la evaluacion y cual no
    :param x: Data with features
    :param y: Classes
    :return: Promedio de la metrica obtenida en cada fold del CrossValidation. -1 si no hay features a evaluar
    """
    if not np.count_nonzero(index_array):
        return -1.0

    parsed_data = get_columns_from_df(index_array, x)
    return compute_cross_validation(parsed_data, y)


def run_times_experiment_sequential(
        compute_cross_validation: Callable[[pd.DataFrame, np.ndarray], float],
        n_iterations: int,
        metric_description: str,
        model_description: str,
        add_epsilon: bool
):
    """
    Hace lo mismo que run_experiment pero con datos de supervivencia en vez de los datasets con clases categoricas

    :param n_iterations: Number of iterations
    :param compute_cross_validation: Fitness function
    :param metric_description: Metric description to report in results
    :param model_description: model description to report in results
    :param add_epsilon: If True it adds an epsilon to 0s in Y data to prevent errors in SVM training
    """
    # Obtiene los datos necesarios de supervivencia
    x, y = read_survival_data(add_epsilon)

    number_samples, number_features = x.shape
    step = 100

    logging.info(f'Running times experiment with {n_iterations} iterations sequentially')
    logging.info(f'Metric -> {metric_description} | Model -> {model_description}')
    logging.info(f'Survival dataset')
    logging.info(f'\tSamples (rows) -> {number_samples} | Features (columns) -> {number_features}')
    logging.info(f'\tY shape -> {y.shape}')

    # Needed parameter for the Binary Black Hole Algorithm
    total_n_features = x.shape[1]

    # Lists for reporting
    number_of_features: List[int] = []
    exec_times: List[float] = []

    # Runs the iterations
    for i_iter in range(n_iterations):
        logging.info(f'Iteration {i_iter + 1}/{n_iterations}')

        current_n_features = step

        while current_n_features <= total_n_features:
            random_features_to_select = np.zeros(total_n_features, dtype=int)
            random_features_to_select[:current_n_features] = 1
            np.random.shuffle(random_features_to_select)

            start_worker_time = time.time()
            _star_result_values = fitness_function_with_checking_sequential(
                compute_cross_validation,
                random_features_to_select,
                x,
                y
            )
            number_of_features.append(current_n_features)

            cur_exec_time = time.time() - start_worker_time
            cur_exec_time = round(cur_exec_time, 4)
            exec_times.append(cur_exec_time)

            current_n_features += step

    # Saves times in JSON for post-processing
    now = time.strftime('%Y-%m-%d_%H_%M_%S')
    json_file = f'{now}_times.json'
    json_dest = os.path.join('Times results', json_file)

    logging.info(f'Saving lists in JSON format in {json_dest}')
    result_dict = {
        'n_features': number_of_features,
        'execution_times': exec_times,
    }

    with open(json_dest, 'w+') as file:
        file.write(json.dumps(result_dict))

    logging.info('Saved.')


def run_times_experiment_features_shuffle(
        compute_cross_validation: CrossValidationCallback,
        n_iterations: int,
        metric_description: str,
        model_description: str,
        add_epsilon: bool,
        sc: Optional[SparkContext] = None,
        use_broadcasts_in_spark: Optional[bool] = True
):
    """
    Hace lo mismo que run_experiment pero se hace un shuffle aleatorio del orden de  los numeros de features a probar

    :param n_iterations: Number of iterations
    :param compute_cross_validation: Fitness function
    :param metric_description: Metric description to report in results
    :param model_description: model description to report in results
    :param add_epsilon: If True it adds an epsilon to 0s in Y data to prevent errors in SVM training
    :param sc: Spark Context
    :param use_broadcasts_in_spark: If True, it generates a Broadcast value to pass to the fitness function instead of pd.DataFrame. Is ignored if run_in_spark = False
    """
    # Obtiene los datos necesarios de supervivencia
    x, y = read_survival_data(add_epsilon)

    number_samples, number_features = x.shape
    n_stars = 15  # Five stars per worker

    logging.info(f'Running times experiment with {n_iterations} iterations and {n_stars} stars ({n_stars // 3} stars '
                 f'per worker)')
    logging.info(f'Metric -> {metric_description} | Model -> {model_description}')
    logging.info(f'Survival dataset')
    logging.info(f'\tSamples (rows) -> {number_samples} | Features (columns) -> {number_features}')
    logging.info(f'\tY shape -> {y.shape}')

    # Needed parameter for the Binary Black Hole Algorithm
    total_n_features = x.shape[1]

    if use_broadcasts_in_spark:
        x = sc.broadcast(x)

        # Runs an initial experiment with 1000 features to broadcast the data and prevent issues with execution times
        # due to data distribution
        initial_n_features = 1000
        stars_subsets_initial = np.empty((n_stars, 2), dtype=object)  # 2 = (1, features)
        for i in range(n_stars):
            random_features_to_select_initial = np.zeros(total_n_features, dtype=int)
            random_features_to_select_initial[:initial_n_features] = 1
            np.random.shuffle(random_features_to_select_initial)
            stars_subsets_initial[i] = (i + 1, random_features_to_select_initial)

        _results_values, total_initial_time = parallelize_fitness_execution_by_partitions(
            sc,
            stars_subsets_initial,
            fitness_function=lambda subset: fitness_function_with_checking(
                compute_cross_validation,
                subset,
                x,
                y,
                is_broadcast=use_broadcasts_in_spark
            )
        )
        logging.info(f'Initial running finished in {total_initial_time}')
    else:
        logging.info(f'Broadcasting disabled. Initial run with all features discarded ')

    execution_times: WorkerTimeDict = {}
    idle_times: WorkerTimeDict = {}

    # Lists for reporting
    number_of_features: List[int] = []
    time_exec: List[float] = []
    idle_exec: List[float] = []

    # List with all the number of features to test.
    # This is useful to detect issues with Spark with peaks in execution times
    number_features_array = np.arange(100, 20100, 100)
    np.random.shuffle(number_features_array)  # TODO: comment this for experiment 2

    # Runs the iterations
    for i_iter in range(n_iterations):
        logging.info(f'Iteration {i_iter + 1}/{n_iterations}')

        stars_subsets = np.empty((n_stars, 2), dtype=object)  # 2 = (1, features)

        # Randomizes the list of features to test
        # np.random.shuffle(number_features_array)  # TODO: uncomment this for experiment 2

        idx_n_feature = 0
        len_number_features_array = len(number_features_array)
        while idx_n_feature < len_number_features_array:
            for i in range(n_stars):
                # Initializes 'Population' with a key for partitionBy()
                current_n_features = number_features_array[idx_n_feature]
                random_features_to_select = np.zeros(total_n_features, dtype=int)
                random_features_to_select[:current_n_features] = 1
                np.random.shuffle(random_features_to_select)
                stars_subsets[i] = (i, random_features_to_select)

                # Jumps by 'step' elements
                # current_n_features += step
                idx_n_feature += 1

                # If it's arraised the maximum number of features, slices the stars array
                # if current_n_features > total_n_features:
                if idx_n_feature == len_number_features_array:
                    stars_subsets = stars_subsets[:i + 1]
                    break

            # Assigns partition IDs
            assign_ids(stars_subsets, number_of_workers=3)

            results_values, total_time = parallelize_fitness_execution_by_partitions(
                sc,
                stars_subsets,
                fitness_function=lambda subset: fitness_function_with_checking(
                    compute_cross_validation,
                    subset,
                    x,
                    y,
                    is_broadcast=use_broadcasts_in_spark
                )
            )

            for init_idx in range(len(stars_subsets)):
                current_data = results_values[init_idx]
                worker_time = current_data[1]
                evaluated_features = current_data[4]
                idle_time = total_time - worker_time

                number_of_features.append(evaluated_features)
                time_exec.append(round(worker_time, 4))
                idle_exec.append(round(idle_time, 4))

                store_times(init_idx, current_data, total_time, execution_times, idle_times, debug=True)

    report_exec_and_idle_times(execution_times, idle_times)

    # Saves times in JSON for post-processing
    now = time.strftime('%Y-%m-%d_%H_%M_%S')
    json_file = f'{now}_times.json'
    json_dest = os.path.join('Times results', json_file)

    logging.info(f'Saving lists in JSON format in {json_dest}')
    result_dict = {
        'n_features': number_of_features,
        'execution_times': time_exec,
        'idle_times': idle_exec
    }

    with open(json_dest, 'w+') as file:
        file.write(json.dumps(result_dict))

    logging.info('Saved.')