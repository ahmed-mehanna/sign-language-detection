import os

WEIGHTS_PATH=os.path.join("..","..","sign_language_detection","ensemble","V1")
KERAS_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"keras_weights","V1.h5")
TORCH_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"pytorch_weights.tar")
