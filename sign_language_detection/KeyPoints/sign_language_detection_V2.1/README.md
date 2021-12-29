# fourth approach


now that i got all the AUTSL data set and the model is performing good i will try it on 20 signs


- same techniques as before for data collectino and extraction but only for 20 signs and now we have validation and testing data so collect them
- extract features from all data in these signs (2450 video on training,382 validation,328 testing)
- extract 20 frame form each video 
- try 3 different LSTM architecture on this data and use the best one
- first architecture training accuracy 97% , validation accuracy 76%, (more layer added)
- second architecture training accuracy 75% validation 65%   (with dropout)
- third architecture is the same small one got validation accuracy of 55% 
- the model was slow on real time 



- improvements

- [ ] lr schedulre
- [ ] different keypoints 
- [ ] model different than LSTM maybe GCN 
- [ ] use the motion for extracting like the paper (keypoints and motion toghether)
- [ ] try dropout again but lower
- [ ] use data augmentation
