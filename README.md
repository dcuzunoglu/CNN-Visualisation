# CNN-Visualisation
A small convnet trained on University of Oxford 17 flowers dataset using Tensorflow with a Flask webapp interface to feed in images and visualise activations of a hidden layer

## Preprocessing

- Create the following directories in the repo directory: /activation-images, /CSV-files, /datasets, /flower-images, /flower-images-resized, /models.

- Download the Oxford 17 flowers dataset from "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/"

- Then place the flower images under /flower-images directory. 

- Place the labels ("datasplits.mat") under /CSV-files directory. 

- Then, run "preprocessing_utilities.py" to preprocess the data for training. This will populate the /flower-images-resized directory, /CSV-files directory, and /datasets directory.

## Usage

- Train network by calling "convnet.py train". Tensorflow will save model in /models after training is completed. When training completes, this function also saves accuracies, per class accuracies (for multiclass classification), and confusion matrix over the test set recorded at each epoch.

- When model is trained, you can run predictions over the whole test set by calling "convnet.py test". This returns the overall accuracy over the test set. 

- With a trained model, it is also possible to run predictions and get visualisations of filters of first convolutional layer of model on an image. Run "convnet.py activation $IMAGENAME" where $IMAGENAME is an image in /flower-images-resized directory. The output activation image will be saved in /activation-images.

- We can visualise activations over a Flask web server as well. Run "webapp.py" to start server, and navigate to "localhost:5001/cnn\_visualise\_activations/
