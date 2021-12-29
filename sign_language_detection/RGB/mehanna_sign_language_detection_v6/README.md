# RGB Version 6

This version contains another model called "resnet(2+1)d_18" and we used this model as a base model with some LSTM and Dropout layers

- Firstlly, prepare our dataset by traverse on the dataset and start selecting some frames and write the on the hard-disk to be used in the future
- Secondlly, build Dataset class to hold pathes of the frames
    - Do some augmentations on the frames
    - Use a given "mean" and "std" to normalize frames's pixels
- Thirdlly, build Data Loader with batch size 1 to use it to load the videos to model
- Fourthlly, build the model by concatenating a Dropout layer then LSTM layer
- This model gaves us a validation accuracy 68% on 3 signs



Another experiment, by trying to use "resnet(2+1)d_18" without any modification we got an 100% validation and training accuracy, but these numbers are totally wrong because we wrote something wrong in the DataLoader part so when we load the videos the DataLoader change its labels to be "1" so the model was training on 1 sign only.