# Импортируем необходимые библиотеки
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.layers import Dropout
from sklearn.model_selection import train_test_split


def main():
    # Загружаем данные и разделяем их на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    # Определяем количество экземпляров модели и количество эпох обучения
    num_models = 5
    epochs = 20

    # Создаем список для хранения экземпляров моделей
    models = []

    # Обучаем каждую модель на различном наборе данных
    for i in range(num_models):
        # Создаем экземпляр модели
        model = Sequential()
        # Добавляем сверточные слои
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        # Добавляем полносвязный слой
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        # Добавляем слой Dropout с вероятностью отключения нейронов 0.5
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        # Компилируем модель
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Получаем случайную подвыборку для обучения
        indexes = np.random.choice(X_train.shape[0], size=X_train.shape[0], replace=True)
        X_train_sample = X_train[indexes]
        y_train_sample = y_train[indexes]
        # Обучаем модель на подвыборке
        model.fit(X_train_sample, y_train_sample, epochs=epochs, verbose=0)
        # Добавляем модель в список
        models.append(model)

    # Получаем предсказания для тестового набора данных для каждой модели и объединяем их в единый результат
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)

    # Получаем среднее значение предсказаний
    predictions = np.mean(predictions, axis=0)

    # Оцениваем качество ансамбля на тестовом наборе данных
    score = models[0].evaluate(X_test, y_test, verbose=0)
    print('Accuracy: {:.2f}%'.format(score[1] * 100))

