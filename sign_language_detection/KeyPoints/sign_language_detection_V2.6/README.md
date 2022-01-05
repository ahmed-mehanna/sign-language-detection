# 2.4


- what happens so far
    - i tried 16 frames instead of 20
    - change read frame function to read n+4 frames and remove first and last two frames
    - change data aug functionto reduce the noise
    - change dataaugemntation so each video has the same aug seed
    - try the V1 model and got 95% 
    - realize that this augmentation is better (it reached 99 faster)
    - V2 architecture is worse than V1 maybe because it's larger so i am going to try another one
    - i tried to reduce keypoints and take previous one and next one won't work well images comes from 0 axis a lot
    - taking only 2 previous ones didn't help a lot also
   
-TODO
    - try V2 & V3 and maybe V4
    - try to fill the missing keypoints and again with all architectures (2.5)
    - try 32 frames instead of 20 frames



