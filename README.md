# 4052project


For this checkpoint, I decided to read "wav2vec2.0, A Self-Supervised Learning of Speech Representations". 

It talked about how most data, especially audio data, is unlabeled, and so learning representations from unlabeled data and then fine tuning it on labeled data can be very powerful. 

The model consists of a feature encoder, which uses convolutional layers to learn representations from raw audio, 12 transforrmer blocks which use attention to train quicker from the latent feature vectors, and a quantization module, which uses Gumbel softmax to discretize features into speech units.

I trained the pretrained wav2vec2.0 base model with the RAVDESS dataset, which contains labeled speech audio from 12 actors.

To avoid overfitting, I used data augmentation, adding noise and shifting the pitch of the audio, implemented early stopping, weight decay and dropout layers.

I also froze the feature extractor, and froze some layers of the encoder so that I could finetune the model. 

After playing around with the model, the lowest validation loss I could achieve was around 0.93. I believe that this is because the RAVDESS dataset is quite small. The model was quite good at classifying fearful, disgusted, and surprised audio, but did poorly classifying neutral, happy, or calm audio. The accuracy I achieved was 0.6944, which meant that the model classified about 7/10 of the data correctly. I feel like that is reasonable considering the size of the dataset. 


**Statistics**

Accuracy: 0.6944
Confusion Matrix:
[[29  0  4  0  4  0  0  1]
 [ 0 29  1  0  2  5  1  0]
 [ 3  0 28  0  2  0  3  2]
 [ 4  0  0 30  0  0  3  2]
 [ 4  2  0  1 25  2  0  5]
 [ 0  7  0  0  5  7  0  0]
 [ 1  7  3  4  7  0 15  1]
 [ 1  0  1  0  0  0  0 37]]
Classification Report:
              precision    recall  f1-score   support

       angry       0.69      0.76      0.72        38
        calm       0.64      0.76      0.70        38
     disgust       0.76      0.74      0.75        38
     fearful       0.86      0.77      0.81        39
       happy       0.56      0.64      0.60        39
     neutral       0.50      0.37      0.42        19
         sad       0.68      0.39      0.50        38
   surprised       0.77      0.95      0.85        39

    accuracy                           0.69       288
   macro avg       0.68      0.67      0.67       288
weighted avg       0.69      0.69      0.69       288
