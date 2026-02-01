import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras

os.makedirs("images", exist_ok=True)

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build Dense Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# Predictions
preds = np.argmax(model.predict(x_test), axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Dense Model Confusion Matrix")
plt.savefig("images/confusion_dense.png")
plt.close()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# Wrong Predictions
wrong = np.where(preds != y_test)[0]
plt.figure(figsize=(12,6))
for i in range(12):
    idx = wrong[i]
    plt.subplot(3,4,i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"P:{preds[idx]} T:{y_test[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("images/wrong_preds_dense.png")
plt.close()

# Learning Curve
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title("Dense Model Accuracy")
plt.savefig("images/learning_dense.png")
plt.close()

# Save model
model.save("dense_mnist_model.h5")
