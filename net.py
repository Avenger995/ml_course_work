import random

import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from metrics import balanced_accuracy_m, f1_m, recall_m, precision_m
from roc_score import get_roc_auc_score
import numpy as np


def get_algorithm_enum(algorithm_enum) -> str:
    if algorithm_enum <= 1:
        return 'adam'
    if algorithm_enum == 2:
        return 'rmsprop'
    if algorithm_enum >= 3:
        return 'sgd'


def build_model(filters_first_layer, filters_second_layer, kernel_size, dropout, dense_units, algorithm_enum, input_shape):
    model = models.Sequential([
        layers.Conv1D(filters_first_layer, kernel_size, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Dropout(dropout),
        layers.Conv1D(filters_second_layer, kernel_size, padding='same', activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=get_algorithm_enum(algorithm_enum),
                  loss='binary_crossentropy',
                  metrics=['accuracy', recall_m, precision_m, f1_m, balanced_accuracy_m])

    return model


# Среднее предсказание от всех моделей
def ensemble_predict(models, data_x, data_y):
    metrics = [model.evaluate(data_x, data_y, verbose=0) for model in models]
    return np.mean(metrics, axis=0)


def build_ensemble(dataset, filters_first_layer, filters_second_layer, kernel_size, dropout, dense_units, algorithm_enum,
                   is_get_final_models=False, is_plotting=False):
    print('\nStart building ensemble...')
    test_size = 0.2
    data_x = dataset.iloc[:, 0:dataset.shape[1] - 1].values
    data_y = dataset.iloc[:, dataset.shape[1] - 1].values

    scaler = StandardScaler()
    df_x_st_scaled = scaler.fit_transform(data_x)

    data_x = df_x_st_scaled

    # Создание и обучение ансамбля из 5 моделей
    num_models = 5
    models_list = []

    for num_model in range(num_models):
        seed = random.randint(1, 100)
        df_x_train, df_x_test, df_y_train, df_y_test = (
            train_test_split(data_x, data_y, test_size=test_size, random_state=seed))

        df_x_train = tf.cast(df_x_train, dtype=tf.float32)
        df_x_test = tf.cast(df_x_test, dtype=tf.float32)
        df_y_train = tf.cast(df_y_train, dtype=tf.float32)
        df_y_test = tf.cast(df_y_test, dtype=tf.float32)

        input_shape = (df_x_train.shape[1], 1)
        model = build_model(filters_first_layer, filters_second_layer, kernel_size, dropout, dense_units, algorithm_enum,
                            input_shape)
        model.fit(df_x_train, df_y_train, epochs=5, batch_size=32, verbose=0)
        models_list.append(model)

    if is_get_final_models:
        return models_list

    # Оценка ансамбля на тестовых данных
    metrics_data = ensemble_predict(models_list, df_x_test, df_y_test)
    ensemble_accuracy = metrics_data[1]
    ensemble_f1 = metrics_data[2]
    ensemble_precision = metrics_data[3]
    ensemble_recall = metrics_data[4]
    ensemble_balanced_accuracy_score = metrics_data[5]
    roc_auc_score = get_roc_auc_score(models_list,  df_x_test, df_y_test, is_plotting)
    print('Ensemble Accuracy:', ensemble_accuracy)
    print('Ensemble Recall:', ensemble_recall)
    print('Ensemble Precision:', ensemble_precision)
    print('Ensemble F1:', ensemble_f1)
    print('Ensemble ensemble_balanced_accuracy_score:', ensemble_balanced_accuracy_score)
    print('Ensemble ROC_AUC_SCORE:', roc_auc_score)
    return ensemble_balanced_accuracy_score


if __name__ == "__main__":
    csv_path = ('C:\\Users\\Gubay\\OneDrive\\Documents\\Archive_University\\Мага_3\\ml_course_work\\datasets'
                '\\kdd_10000_labled_modified.csv')
    data = pd.read_csv(csv_path)
    result = build_ensemble(data, 23, 83, 4, 0.5265110647958439, 57, 1, is_plotting=True)
