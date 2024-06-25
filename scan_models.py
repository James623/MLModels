import tensorflow as tf
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
from art.utils import load_mnist

# Путь к модели
model_path = 'mlmodels/mnist_model.h5'

# Загрузка модели
model = tf.keras.models.load_model(model_path)

# Создание KerasClassifier
classifier = KerasClassifier(model=model)

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test), _, _ = load_mnist()

# Создание атакующего метода
attack = FastGradientMethod(estimator=classifier, eps=0.2)

# Создание атакующих примеров
x_test_adv = attack.generate(x=x_test)

# Оценка модели на атакующих примерах
accuracy = classifier._model.evaluate(x_test_adv, y_test)[1]
print(f'Accuracy on adversarial test examples: {accuracy * 100:.2f}%')
