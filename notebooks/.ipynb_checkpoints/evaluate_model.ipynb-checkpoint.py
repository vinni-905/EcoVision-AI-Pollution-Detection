import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ✅ Paths
BASE_DIR = os.path.abspath(os.path.join("..", "dataset", "pollution_data", "Dataset_for_AQI_Classification", "Dataset_for_AQI_Classification"))
train_dir = os.path.join(BASE_DIR, "train_data")
val_dir = os.path.join(BASE_DIR, "val_data")

# ✅ Load the trained model
MODEL_PATH = os.path.join("eco_model.h5")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ✅ Data generator for evaluation
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ✅ Evaluate model
val_loss, val_acc = model.evaluate(val_generator)
print(f"📊 Validation Accuracy: {val_acc * 100:.2f}%")
print(f"📉 Validation Loss: {val_loss:.4f}")

# ✅ Predictions
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# ✅ Classification report
print("\n🧾 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ✅ Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# ✅ Sample predictions visualization
val_generator.reset()
x_batch, y_batch = next(val_generator)
pred_batch = model.predict(x_batch)

plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(x_batch[i])
    true_label = class_labels[np.argmax(y_batch[i])]
    pred_label = class_labels[np.argmax(pred_batch[i])]
    color = "green" if true_label == pred_label else "red"
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
    plt.axis("off")
plt.tight_layout()
plt.show()
