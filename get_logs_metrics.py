import json
import re
from typing import Tuple, Optional


def get_n_features_and_metric(line: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Retrieves the number of features and the metric obtained
    :param line: Current line to parse
    :return: Tuple with both numbers. None in both data if it's not a valid line
    """
    important_str = 'Fitness: '
    if important_str not in line:
        return None, None

    splitted = line.split(important_str)

    # Gets metric
    metric = float(splitted[1])

    # Gets number of features
    suffix = ' features.'
    new_result = re.findall(r'\d+' + suffix, splitted[0])
    n_features = new_result[0].rstrip(suffix)
    return int(n_features), metric


def get_all_n_features_and_metrics(file_path: str):
    json_result = {
        'n_features': [],
        'fitness': []
    }
    with open(file_path, 'r') as f:
        for line in f.readlines():
            n_features, fitness_value = get_n_features_and_metric(line)
            if n_features is None or fitness_value is None:
                continue
            json_result['n_features'].append(n_features)
            json_result['fitness'].append(fitness_value)

    with open(f'{file_path}_result.json', 'w+') as outfile:
        json.dump(json_result, outfile)


def main():
    get_all_n_features_and_metrics('./Logs/logs_times_optimizer_rbtree_kernel_cosine_30_it.txt')
    get_all_n_features_and_metrics('./Logs/logs_times_optimizer_rbtree_kernel_sigmoid_30_it.txt')
    get_all_n_features_and_metrics('./Logs/logs_times_optimizer_rbtree_kernel_rbf_30_it.txt')
    get_all_n_features_and_metrics('./Logs/logs_times_optimizer_rbtree_kernel_poly_30_it.txt')
    get_all_n_features_and_metrics('./Logs/logs_times_optimizer_rbtree_kernel_linear_30_it.txt')
    get_all_n_features_and_metrics('./Logs/logs_times_optimizer_avltree_kernel_cosine_30_it_with_training.txt')
    get_all_n_features_and_metrics('./Logs/logs_times_optimizer_avltree_kernel_sigmoid_30_it_with_training.txt')
    get_all_n_features_and_metrics('./Logs/logs_times_optimizer_avltree_kernel_rbf_30_it_with_training.txt')
    get_all_n_features_and_metrics('./Logs/logs_times_optimizer_avltree_kernel_poly_30_it_with_training.txt')
    get_all_n_features_and_metrics('./Logs/logs_times_optimizer_avltree_kernel_linear_30_it_with_training.txt')


if __name__ == '__main__':
    main()
