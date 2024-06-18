import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('path_to_your_model.h5')  # Убедитесь, что путь к модели правильный
classifier = KerasClassifier(model=model)

# Загрузка данных для тестирования (пример с использованием MNIST)
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset('mnist')

# Инициализация атаки
attack = FastGradientMethod(estimator=classifier, eps=0.2)

# Создание атакованных примеров
x_test_adv = attack.generate(x=x_test)

# Оценка модели на атакованных примерах
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f'Accuracy on adversarial test examples: {accuracy * 100:.2f}%')
