# Version 10

In this version we tried another model based on "resnet(2+1)d18", but with some modifications


- We used this model with a pretrained weights was found in the internet for some videos datasets like Kinetics dataset
- We changed all of the activation function from "ReLU" to "SiLU"

- This model gaves us an unexpected results, after 100 epoch of training to classify only 3 signs it gaves us ~100% accuracy in training with ~96% accuracy in validation and ~95% accuracy in testing

- This model architecture will be used as the base model for the near future