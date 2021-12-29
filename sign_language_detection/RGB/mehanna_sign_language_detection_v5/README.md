# Version 5

This version was built using TensorFlow

- Prepare dataset to read videos and extract a specific number of frames
- Create a DataGenerator to read videos by batch number at a time because we can not read all of the videos at the same time due to the memory size
- Build a model using VGG16 as a start point then concatenate LSTM layer to that model
- Testing this model on 5 classes gaves us a 72% testing accuracy