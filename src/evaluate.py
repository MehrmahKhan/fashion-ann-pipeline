import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

d = np.load("data/processed/data.npz")
model = keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(d["x_test"], d["y_test"], verbose=0)
pred = np.argmax(model.predict(d["x_test"], verbose=0), axis=1)

os.makedirs("reports", exist_ok=True)
cm = confusion_matrix(d["y_test"], pred)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.savefig("reports/confusion_matrix.png", dpi=120, bbox_inches="tight")

json.dump({"test_loss": float(loss), "test_accuracy": float(acc)},
          open("metrics.json", "w"), indent=2)
print(f"test_loss={loss:.4f}  test_accuracy={acc:.4f}")
