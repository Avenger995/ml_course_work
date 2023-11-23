import random
import tensorflow as tf
import pandas as pd
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from net import build_model, get_algorithm_enum, ensemble_predict
from roc_score import get_roc_auc_score


def build_autoencoder(cnn_model, filters_first_layer, filters_second_layer, kernel_size, dense_units, algorithm_enum, input_shape):
    encoder = models.Sequential(cnn_model.layers[:-1])

    decoder = models.Sequential([
        layers.Dense(dense_units, activation='relu', input_shape=(dense_units,)),  # Mirror the Dense layer
        layers.Dense((input_shape[0] // 4) * filters_second_layer, activation='relu'),  # Mirror the Flatten layer
        layers.Reshape((input_shape[0] // 4, filters_second_layer)),  # Reshape to mirror the Flatten layer
        layers.UpSampling1D(2),  # Mirror the MaxPooling1D layer
        layers.Conv1DTranspose(filters_second_layer, kernel_size, padding='same', activation='relu'), # Mirror the Conv1D layer
        layers.UpSampling1D(2),  # Mirror the MaxPooling1D layer
        layers.Conv1DTranspose(filters_first_layer, kernel_size, padding='same', activation='relu'), # Mirror the Conv1D layer
        layers.Conv1D(1, kernel_size, padding='same', activation='sigmoid')
    ])
    autoencoder = models.Sequential([
        encoder,
        decoder
    ])

    autoencoder.compile(optimizer=get_algorithm_enum(algorithm_enum), loss='mae')

    return autoencoder


def build_ensemble(cnn, autoencoder, test_size):
    csv_path = ('C:\\Users\\Gubay\\OneDrive\\Documents\\Archive_University\\Мага_3\\ml_course_work\\datasets'
                '\\kdd_10000_labled_modified.csv')
    dataset = pd.read_csv(csv_path)

    data_x = dataset.iloc[:, 0:dataset.shape[1] - 1].values
    data_y = dataset.iloc[:, dataset.shape[1] - 1].values

    scaler = StandardScaler()
    df_x_st_scaled = scaler.fit_transform(data_x)

    data_x = df_x_st_scaled

    # Присвоить средние значения весов слою другой модели
    for i in range(len(cnn.layers) - 1):
        layer_weights = autoencoder.layers[0].layers[i].get_weights()
        cnn.layers[i].set_weights(layer_weights)

    num_cnn = 5
    models_list = []

    for _ in range(num_cnn):
        seed = random.randint(1, 100)
        df_x_train, df_x_test, df_y_train, df_y_test = (
            train_test_split(data_x, data_y, test_size=test_size, random_state=seed))

        df_x_train = tf.cast(df_x_train, dtype=tf.float32)
        df_x_test = tf.cast(df_x_test, dtype=tf.float32)
        df_y_train = tf.cast(df_y_train, dtype=tf.float32)
        df_y_test = tf.cast(df_y_test, dtype=tf.float32)

        cnn.fit(df_x_train, df_y_train, epochs=5, batch_size=32, verbose=1)
        models_list.append(cnn)

        # Оценка ансамбля на тестовых данных
        metrics_data = ensemble_predict(models_list, df_x_test, df_y_test)
        ensemble_accuracy = metrics_data[1]
        ensemble_f1 = metrics_data[2]
        ensemble_precision = metrics_data[3]
        ensemble_recall = metrics_data[4]
        ensemble_balanced_accuracy_score = metrics_data[5]
        roc_auc_score = get_roc_auc_score(models_list, df_x_test, df_y_test, True)
        print('Ensemble Accuracy:', ensemble_accuracy)
        print('Ensemble Recall:', ensemble_recall)
        print('Ensemble Precision:', ensemble_precision)
        print('Ensemble F1:', ensemble_f1)
        print('Ensemble ensemble_balanced_accuracy_score:', ensemble_balanced_accuracy_score)
        print('Ensemble ROC_AUC_SCORE:', roc_auc_score)
        return ensemble_balanced_accuracy_score


def build_ensemble_by_autoencoder(dataset, filters_first_layer, filters_second_layer, kernel_size, dropout, dense_units,
                                  algorithm_enum):
    test_size = 0.2
    data_x = dataset.iloc[:, 0:dataset.shape[1]].values
    input_shape = (data_x.shape[1], 1)

    # Сохранил имена столбцов и последний столбец
    scaler = StandardScaler()
    df_st_scaled = scaler.fit_transform(data_x)

    data_x = df_st_scaled

    print('start create model')
    cnn = build_model(filters_first_layer, filters_second_layer, kernel_size, dropout, dense_units, algorithm_enum, input_shape)
    print('end create model\n')

    df_x_train, df_x_test = train_test_split(data_x, test_size=test_size)

    df_x_train = tf.cast(df_x_train, dtype=tf.float32)
    df_x_test = tf.cast(df_x_test, dtype=tf.float32)

    autoencoder = build_autoencoder(cnn, filters_first_layer, filters_second_layer, kernel_size, dense_units, algorithm_enum, input_shape)
    autoencoder.fit(df_x_train, df_x_train, epochs=25, batch_size=512, verbose=1, validation_data=(df_x_test, df_x_test))

    autoencoder.summary()

    res = build_ensemble(cnn, autoencoder, test_size)
    print(res)


if __name__ == "__main__":
    csv_path = ('C:\\Users\\Gubay\\OneDrive\\Documents\\Archive_University\\Мага_3\\ml_course_work\\datasets'
                '\\kdd_50000_unlabled_modified.csv')
    data = pd.read_csv(csv_path)
    build_ensemble_by_autoencoder(data, 23, 83, 4, 0.5265110647958439, 57, 1)
