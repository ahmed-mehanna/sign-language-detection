# first approach


the first approach here was to use mediapipe and extract keypoints from videos i made and test them before trying on actual data

- prepare mediapipe to extract features from faces (use only the hand and pose and ignore face keypoints )
- start collecting data (small amount ) just 3 signs (30 video for each sign)
- collect only 20 frames for each video
- build LSTM model
- test the model on other part of the data got 88% accuracy (it's very small data for train and test)
- write real time video 

