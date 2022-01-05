# 2.3 

-TASK one (compare between two models on 5 signs with and without Z points)
    - found out that with Z the accuracy improved a little bit


- Task two (try reduce the keypoints and measure accuracy)
    - training speed is improved 
    - accuraccy is improved a little bit
   

- Task 3 use data aug (and change batch size)
    - create around 4500 videos with data augemntation and train on them
    - training accuracy reached 99.5% but validation is 91% 
 
 - Task 4 use data aug with smaller arch
     there was overfitting (the accuracy when so low just one time and when trained on older version it wasn't very good)
     try to understand why this happen
    
    
   
 
 
 
 - the model still confuse between sister and brother so i need to find solution to this(maybe use the data i made once)
 
- Task 5
- collect data for 10 classes (6 aug and 1 original)
- write classifier for 10 signs
- test on two models one on colab and the other on my device
- train each model more than one time from scratch
- save all weight (one of them was deleted because it was bad)
- the model perform very good but each one has some limits



future improvements 
- find a way to work with mediapipe failure to detec movement
- increase number of frames


