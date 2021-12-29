# third approach


to solve the overfitting we needed more data so i decided to try colelct more data and try it on 5 signs till i got all the AUTSL dataset


- same techniques as before for data collectino and extraction
- the video i collected was 250 video 50 for each sign 
- using the same model
- write method read videos from pathes and convert them into n frames then extract features from them 
(unlike the previous problem we have more frames than we need so we have to choose some of them)
- extract keypoints from all training data of these signs (79 sign trainnig and 4 testing) it's not the full dataset
- save the keypoints to be saved later because it requires a lot of time to extract them
- use the same LSTM model
- train and save the best model and stop at 180 epoch the accuracy 88% (more data got better results)
- model is worknig great on real time videos


- next approach try on more data for more signs