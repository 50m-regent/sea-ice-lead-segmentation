import keras
import numpy as np

model_dir = "model/ViT_seaice_classifier" # Adjust accordingly. Point to model directory
data_dir = "data/3x3_test_set.npz" # Point to data location

# Load model
vit_model = keras.models.load_model(model_dir) # 3x3 Vision Transformer (ViT) model

# Load test set
test_data = np.load(data_dir)
x_test, y_test = test_data['x'], test_data['y']

# Evaluate model
_, accuracy = vit_model.evaluate(x_test, y_test)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
