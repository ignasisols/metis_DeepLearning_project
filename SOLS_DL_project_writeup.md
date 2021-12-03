# Predicting colon cancer in histopathological images

Ignasi Sols

## **Abstract**

This project aims to predict colon cancer on histopathological images obtained from biopsies.
Histopathology refers to the microscopic examination of tissue in order to study the manifestations of disease. When a colonoscopy is performed, sometimes the doctors find polyps (also known as adenomas). A polyp is a small clump of cells that forms on the lining of the colon. Polyps are benign (non-cancerous) growths, but cancer can start in some types of polyps. For this reason, doctors remove polyps from the colon, and biopsies are performed to determine if they were cancerous. For this project, a convolutional neural network will be trained to predict colon cancer in histopathological images. Hospitals and patients can benefit from a model that accurately predicts colon cancer. NYU Langone Health is the target company for this project.

## **Design**

I have chosen a dataset of colon cancer histopathological images available on Kaggle ("Lung and Colon Cancer Histopathological Images")
* https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images/code 
* https://arxiv.org/abs/1912.12142v1

The dataset has 5,000 images of adenocarcinomas (Tumors) and 5,000 images of benign tissue. These images were already augmented by the dataset authors using the package 'Augmentor'.
The metric that I initially chose for this project was the F score (with beta > 1) to give more importance to recall than to precision. The reason is that I consider false negatives worse than false positives for patient diagnosis. For the CNN, I had to switch to checking both recall and precision, as Keras does not include the F-beta score because it can be misleading when computed in batches rather than globally (for the whole dataset).
The assumptions and risks of this project are (1) There is data bleeding: in the dataset, all the augmented images from a given class are located in the same folder, with no information provided about the original images. This means that, when performing a train/test and validation split, augmented images that were generated from the same original image might be both in the train, validation, and test split. Therefore, there is probably data bleeding. (2) No information regarding the cancer stage of the tumors was provided, which might affect how the model generalizes. 
  
## **Data**
The 10,000 images (5,000 images per condition, already augmented by the dataset authors). I kept the three RGB channels. The images, all with size 768x768 were scaled to 128x128 and binary labeled.

## **Algorithm**
First, an 80/20 train/test split was performed with the sklearn package. 
Then, I developed a baseline (non-Deep Learning) model. I tried both logistic regression and random forests models. First, I scaled the data with the StandardScaler() sklearn function. Next, I reduced the feature dimensionality to 2 features with Principal Component Analysis. After this, I generated a Logistic Regression model (accuracy: 0.74, recall: 0.69, F beta score (beta = 2): 0.70, precision: 0.74). The random forests model performed better (accuracy: 0.80, recall: 0.77, F beta score (beta = 2): 0.78, precision: 0.81), and for this reason, I chose it as a baseline model. 
Next, I generated a CNN without transfer learning. For this, I chose a model with 2 convolutional layers (filters  = 20, kernel_size = 3, activation = 'relu', padding = 'same') for both. Each convolutional layer was followed by a max-pooling step (2x2). Finally, the output was flattened before being passed into two fully connected (dense layers): the first, with 20 nodes (and 'relu' as activation) and the second being the output layer with one node and a sigmoid activation. I introduced an EarlyStopping callback that monitored the validation loss, with patience = 3. The validation loss was 'binary cross-entropy', and the optimizer, 'Adam'.  As a result, I chose epochs = 7 and (accuracy: 0.92, recall: 0.98, precision: 0.87). Finally, I generated a CNN with Transfer Learning. I used mobileNet V2. I used the mobileNet V2 pre-trained weights as a first step in the neural network, removing the top layers of  mobileNetV2 and freezing the rest (such that the weights are fixed and cannot be trained). The output of these pre-trained layers was passed to the same CNN network described above. Total parameters: 2,492,465. Trainable parameters: 234,481. 
The results were: (accuracy: 0.997, recall: 0.9969, precision: 0.9969), providing a significant improvement when comparing with the baseline model and the simple CNN network.


## **Tools**

Numpy and Pandas for data wrangling and computing, sklearn for train/test split and baseline models, Keras for Deep Learning models.

