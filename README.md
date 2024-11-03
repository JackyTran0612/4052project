# step 3 - vision




I chose to read "Deep Residual Learning for Image Recognition" by Kaiming He. Deeper neural networks are more difficult to train. The best ImageNet CNN models at the time contained between 16 and 30 layers. The training error of deeper networks was too high. Their team used skip connections in order to train networks much deeper than 30 layers. Skip connections add the output of a previous layer to the layer ahead. This helped with dealing with vanishing gradients, which was a problem in deep networks. Adding the skip function also allows the network to skip layers if it deems the transformation to be unnecessary. They achieved first place in the ILSVRC 2015 competition with their 3.57% top-5 error on ImageNet, and were also able to train networks with 100 or 1000 layers.



I implemented ResNet on the FER-2013 dataset. Looking back, it wasn't the best choice as the dataset is relatively small and I ran into problems with overfitting. 

Train and test 1: (these correspond to the images I placed in the repository)

I used ResNet18 with imagenet default_V1 weights and an adam optimizer. The validation loss hit a low of 0.9889 after 4 epochs and continued to increase after that, even as the training loss decreased, which made me think the model started overfitting.

Train and test 2: 

I used ResNet18 with resnet18_default weights and I changed to an SGD optimizer. I also augmented some of the data to help with the overfitting. The val loss hit a low of 0.9575 after 9 epochs, and my validation accuracy increased from 65% to 68%.

Train and test 3:

I simply continued to train the same weights for another 11 epochs for a total of 21 epochs. The validation loss kept increasing and reached 1.5148 by the last epoch.

Train and test 4:

I added a learning rate scheduler to try and fine tune the weights. 

I didn't observe any significant improvement.


I think that the dataset that I chose was just too small. I will try and train on a larger dataset by the end of the project, and probably try and use a deeper version of ResNet; either ResNet34 or ResNet50

