the main Task is to extract keypoints from RGB frames using pose estimation model or mediapipe
and use these keypoints to train a model to predict the images

### Frame extraction and selection
- [ ]  provide information about how we extraced the frames and how many frames

### how and why did we extrac the keypoints

- one of the best techniques used in the video/action classification is poseEstimation as as it got high accuracy  
on many problems but it won't work well on our problem because pose estimation doesn't have hands keypoints 
- [ ] provide images poseestimation model
- a hand detection method is used with pose estimation but it didn't provide high accuracy compared to RGB method
- recently there was a lot of development in the pose estimation  
- we decided to use media pipe a library provided by google and it can extract all body keypoints 
- [ ] provide more information  about mediapipe and how it was used
- [ ] provide images from mediapipe on the dataset
- [ ] provide information of how we extraced the keypoints


# figures of models
- fill the diagram section with diagram of each model describing the architecture


# optimizers and loss function

- [ ] which optimizers and loss function are used and why


# Models Comparison

| Model | n classes |data source| train_acc | Top 1% | Top 5% | 
| ------ | ------ | ------ | ------ | ------ | 
| mediapipe + LSTM (Fn) | 3 | 3 | n | n2 | n3 | 
| mediapipe + LSTM (Fn) | 3 | 3 | n | n2 | n3 | 
| mediapipe + LSTM (Fn) | 3 | 3 | n | n2 | n3 | 
| mediapipe + LSTM (Fn) | 3 | 3 | n | n2 | n3 | 
| mediapipe + LSTM (Fn) | 3 | 3 | n | n2 | n3 | 




# how the real predection work
- [ ] provide info of the techniques used to make the real prediction