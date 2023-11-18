import random
import tensorflow as tf
from keras import layers, models
from enum import Enum
from sklearn.model_selection import train_test_split
import pandas as pd


class AlgorithmEnum(Enum):
    adam = 1,
    rmsprop = 2,
    sgd = 3


def get_algorithm_enum(algorithm_enum) -> str:
    if algorithm_enum <= 1:
        return 'adam'
    if algorithm_enum == 2:
        return 'rmsprop'
    if algorithm_enum >= 3:
        return 'sgd'


# Определение модели для бэггинга
def build_model(filters_first_layer, filters_second_layer, kernel_size, dropout, algorithm_enum, input_shape):
    model = models.Sequential([
        layers.Conv1D(filters_first_layer, kernel_size, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Dropout(dropout),
        layers.Conv1D(filters_second_layer, kernel_size, padding='same', activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=get_algorithm_enum(algorithm_enum),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Среднее предсказание от всех моделей
def ensemble_predict(models, data):
    predictions = [model.predict(data) for model in models]
    return tf.reduce_mean(predictions, axis=0)


def main(dataset, filters_first_layer, filters_second_layer, kernel_size, dropout, algorithm_enum, is_get_final_models=False):
    test_size = 0.2
    data_x = dataset.iloc[:, 0:dataset.shape[1] - 1]
    data_y = dataset.iloc[:, dataset.shape[1] - 1]

    # Создание и обучение ансамбля из 5 моделей
    num_models = 5
    models_list = []

    for _ in range(num_models):
        seed = random.randint(1, 100)
        df_x_train, df_x_test, df_y_train, df_y_test = (
            train_test_split(data_x, data_y, test_size=test_size, random_state=seed))
        #df_x_train = df_x_train.values.reshape(df_x_train.shape[0], df_x_train.shape[1], 1)
        #df_y_train = df_y_train.values.reshape(df_y_train.shape[0], 1, 1)
        input_shape = (df_x_train.shape[1], 1)
        model = build_model(filters_first_layer, filters_second_layer, kernel_size, dropout, algorithm_enum, input_shape)
        model.fit(df_x_train, df_y_train, epochs=5, batch_size=32, verbose=1)
        models_list.append(model)

    if is_get_final_models:
        return models_list

    # Оценка ансамбля на тестовых данных
    ensemble_predictions = ensemble_predict(models_list, df_x_test)
    ensemble_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(ensemble_predictions, axis=1), df_y_test), tf.float32))
    print('Ensemble Accuracy:', ensemble_accuracy.numpy())

    return ensemble_accuracy.numpy()


if __name__ == "__main__":
    csv_path = ('C:\\Users\\Gubay\\OneDrive\\Documents\\Archive_University\\Мага_3\\ml_course_work\\datasets'
                '\\kdd_10000_labled_modified.csv')
    data = pd.read_csv(csv_path)
    result = main(data, 51, 99, 3, 0.706073411715844, 1)
    #result = main(data, 32, 64, 3, 0.1, 1)
