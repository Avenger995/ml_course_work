import pandas as pd
from net import build_ensemble

csv_path = ('C:\\Users\\Gubay\\OneDrive\\Documents\\Archive_University\\Мага_3\\ml_course_work\\datasets'
            '\\kdd_10000_labled_modified.csv')
data = pd.read_csv(csv_path)

# Определяем набор гиперпараметров для поиска по сетке
param_grid = {
    'filters_first_layer': [56, 76],
    'filters_second_layer': [63, 93],
    'kernel_size': [1, 3],
    'dropout': [0.3, 0.6],
    'relu_count': [64, 128],
    'algorithm_enum': [1, 3]
}

# Проводим поиск по сетке
best_accuracy = 0
best_params = None

for alg_enum in param_grid['algorithm_enum']:
    for filters1 in param_grid['filters_first_layer']:
        for filters2 in param_grid['filters_second_layer']:
            for kernel in param_grid['kernel_size']:
                for drop in param_grid['dropout']:
                    for r_count in param_grid['relu_count']:
                        print(f"\nTrying: {filters1}, {filters2}, {kernel}, {drop}, {r_count}, {alg_enum}")

                        # Запуск обучения модели для каждой комбинации гиперпараметров
                        accuracy = build_ensemble(data, filters1, filters2, kernel, drop, r_count, alg_enum)

                        # Сравнение результатов и сохранение наилучшей комбинации
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {'filters_first_layer': filters1,
                                           'filters_second_layer': filters2,
                                           'kernel_size': kernel,
                                           'dropout': drop,
                                           'relu_count': r_count,
                                           'algorithm_enum': alg_enum}

print("Best BalancedAccuracy:", best_accuracy)
print("Best Parameters:", best_params)
