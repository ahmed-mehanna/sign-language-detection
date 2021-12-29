# fifth approach


try the first possible improvements (one VS many)

- collecting data is now different
    - train data are all data from one sign and random data from all other signs representing 1,0 classes but data should be balanced
    - validation and testing follow the same approach except data shouldn't be balanced
- extract features from 25 frames and save them  
- try the LSTM model the first one but reduce the layers a little bit since now it's just binary
 the training accuracy is very high 95% but the validation is very low 30%
- try logistic regression for binary classification got 100% on training and less than 25% on validation
- the model has series problem on real time video as it just predict every thing as the sign 



- improvements
- [ ] try change the data distribution maybe there is different way for one vs many
- [ ] use data augmentation
- [ ] try change number of frames
- [ ] try different loss function to work with confusion matrix
- [ ] use data augmentation to generate larger data for training maybe 512 class 0 and 512 class 1 instead of 128,128
- [ ] let the 0 class be random each time so the model couldn't overfit on them
- [ ] try larger LSTM
- [ ] lr schedulre or change optimizer
- [ ] different keypoints 
- [ ] model different than LSTM maybe GCN 
- [ ] try dropout or any regularization method

