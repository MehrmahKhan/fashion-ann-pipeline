import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]

d = np.load("data/raw/fashion_mnist_raw.npz")
x_train = (d["x_train"].astype("float32") - 72.94) / 90.02   # standardize
x_test = (d["x_test"].astype("float32") - 72.94) / 90.02

# test_size here = fraction of TRAIN data held out as validation
x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, d["y_train"],
    test_size=params["test_size"], random_state=params["seed"], stratify=d["y_train"],
)

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed(
    "data/processed/data.npz",
    x_train=x_tr, y_train=y_tr, x_val=x_val, y_val=y_val,
    x_test=x_test, y_test=d["y_test"],
)
print("Saved processed:", x_tr.shape, x_val.shape, x_test.shape)
