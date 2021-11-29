# SOLS_DeepLearning_MVP

**Ignasi Sols**

This project aims to predict colon cancer on histopathological images obtained from biopsies. https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images/code 
When a colonoscopy is performed, sometimes the doctors find polyps (also known as adenomas). A polyp is a small clump of cells that forms on the lining of the colon. Polyps are benign (non-cancerous) growths, but cancer can start in some types of polyps. For this reason, doctors remove polyps from the colon, and biopsies are performed to determine if they were cancerous.

To start exploring this goal, I have developed a baseline random forest model:
I first loaded the 10,000 images (5,000 images per condition, already augmented by the dataset authors). I kept the three RGB channels. The images, all with size 768x768 were all scaled to 128x128 and binary labeled.
Next, an 80/20 train/test split was performed with the sklearn package. 
Finally, the train and test data (X_train and X_test) were scaled -for each of the three channels independently-, and dimensionality reduction was performed by applying PCA (n_components = 2). 
I have chosen the Fbeta score (with beta = 5) as the evaluation metric of interest. The reason is that with cancer diagnosis we do care more about recall than precision (we want to care more about having False negatives than False positives).
Finally, I run both a linear regression model (accuracy score = 0.735, Fbeta score = 0.758), and a random forest classifier (n_estimators = 100). The accuracy score was 0.788 and the Fbeta score (with beta = 5) was 0.804. For this reason, I choose the random forest model as the baseline model.

In the next few days, I plan to run a convolutional neural network to improve my baseline model. I plan to apply transfer learning by pre-training the network with a pre-trained convolutional base from an imagenet model. 
