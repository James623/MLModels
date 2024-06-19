import numpy as np
import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod
from art.utils import load_mnist
import os

# Выведем текущую рабочую директорию
print("Current working directory:", os.getcwd())

# Загрузите модель
model_path = 'mlmodels/model.h5'
model = tf.keras.models.load_model(model_path)

# Создайте классификатор ART
classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=10,
    input_shape=(28, 28, 1),
    loss_object=tf.keras.losses.CategoricalCrossentropy()
)

# Загрузите данные для тестирования
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

# Выполните атаку
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Оцените модель на уязвимости
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print(f"Accuracy on adversarial test examples: {accuracy * 100}%")
