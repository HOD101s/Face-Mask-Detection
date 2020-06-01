# Face-Mask-Detection
<p align="center"><img src = "https://user-images.githubusercontent.com/37273226/83437908-30c02000-a45e-11ea-8ed1-4f65a5db0b15.gif"/></p>

In the COVID19 crisis wearing masks is absolutely necessary for public health and controlling the spred of the pandemic. With the power of Deep Learning this system can detect whether a person has worn a mask or not. This model has been trained on the dataset made by [prajnasb](https://github.com/prajnasb/observations)

## Model:
<p align="center"><img src = "https://user-images.githubusercontent.com/37273226/83438743-93fe8200-a45f-11ea-8470-b0ed8b8a2c75.png"/></p>
<br>
Here I have used transfer learning using the MobileNetV2 model which achieved 100% accuracy on the Train, Validation and Test data as can be seen in the notebook

## Implementation:
The Haar Cascades Face Detection algorithm here is used to detect faces which is then passed through my model which estimates whether person is wearing a mask or not.

Final Deployment is done using OpenCV that feeds the webcam data as frames to the model.
