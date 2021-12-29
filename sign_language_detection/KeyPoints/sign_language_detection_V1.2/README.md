# second approach


after the model done well on small images i collected we decided to try it on the dataset itself but only 3 classes for starter


- use the same prepared keypoints from previous approach
- write method read videos from pathes and convert them into n frames then extract features from them 
(unlike the previous problem we have more frames than we need so we have to choose some of them)
- extract keypoints from all training data of these signs (79 sign trainnig and 4 testing) it's not the full dataset
- save the keypoints to be saved later because it requires a lot of time to extract them
- use the same LSTM model
- train for 600 epoch
- the model is overfitting on the training data since the accuracy 100% and due to small test data we also got 100% but the real time
video classification wasn't good

