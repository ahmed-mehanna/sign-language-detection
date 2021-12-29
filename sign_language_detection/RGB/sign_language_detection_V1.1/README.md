# sign language detection model (V1)


this is a basic sign language detection
it only detects two signs so far with accuracy of 100% on training and 100% on validation (it doesn't make scene i know) it also contains the model weights 

[Download weights](https://drive.google.com/drive/folders/1RDPSpDedsQX0TLfor0JkkkpzEcA8oN47?usp=sharing)


- the aim is to use VGG model to extract feautres from image and use LSTM to make prediction on the extraced features
- the model has very high accuracy 100% which show something is wrong and i actully want to try it again now to check what's wrong with it
- in other videos i created it always predict another sign unlike the LSTM with mediapipe which managed to generalize

# improvements
    - [ ] we need to try the same model again but on 20 signs and now on all the training data not only part of them i have felling that it might work
    - [ ] use this model for real time predection