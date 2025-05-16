import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

# Set the data directory
DATA_DIR = "data"
X, y = [], []
label_list = []
label_dict = {}
current_label = 0

# Scan the data directory for .npy files (except 'labels.npy')
for file in os.listdir(DATA_DIR):
    if file.endswith(".npy") and file != "labels.npy":
        label_name = file.split(".")[0]
        data = np.load(os.path.join(DATA_DIR, file))

        X.append(data)
        y.append([label_name] * len(data))

        if label_name not in label_dict:
            label_dict[label_name] = current_label
            label_list.append(label_name)
            current_label += 1

# Check if data is available
if not X:
    raise RuntimeError("❌ No training data found in the 'data/' folder.")

# Prepare dataset
X = np.concatenate(X)
y = np.concatenate(y).reshape(-1, 1)

# Convert string labels to integers
for i in range(y.shape[0]):
    y[i, 0] = label_dict[y[i, 0]]

y = y.astype("int32")

# One-hot encode labels
y = to_categorical(y)

# Shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Build neural network model
ip = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, batch_size=32)

# Save model and label mapping
model.save("emo_model.h5")
# np.save(os.path.join(DATA_DIR, "labels.npy"), np.array(label_list))
np.save("labels.npy", np.array(label_list))

print("✅ Model training complete and saved as 'model.h5'")
