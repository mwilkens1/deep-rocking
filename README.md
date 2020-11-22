# Image recognition: classifying guitar models using CNN

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Reflection](#reflection)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python version 3. The final app is live on [deep-rocking.herokuapps.com](deep-rocking.herokuapps.com)

## Project Motivation<a name="motivation"></a>
This project is for me to practice with convolutional neural networks using pytorch. I am applying this technique to a personal interest of mine: guitars. The goal is to build a model that can classify images of guitars into 7 main electric guitar body types: stratocaster, telecaster, les paul, mustang, jazzmaster, PRS SE and the SG. This task should be fairly straightforward because the difference in guitar bodies is quite obvious, although the differences between some bodies are much more subtle. In the future, the application could potentially be improved by giving more precise estimates, i.e. not only the guitar body type but the exact model, the brand and the year in which it was produced. 

The images have been collected from Reverb.com, a website with listings for new and used guitars around the world. The data itself is not available on github due to its filesize. The same holds for the final models. If interested, please send me a message. 

## File Descriptions <a name="files"></a>
* **get_images.ipynb**: Jupyter notebook that collects the images using the Reverb.com API. The API is not straightforward for this purposes because it includes a number of limitations.
* **create_datasets.py**: helper function to create training, validation and testing datasets.
* **CNN_trainer.py**: class that includes functions to train and test a CNN on this data.
* **train_model.ipynb**: Jupyter notebook that trains two different models. The first is a CNN built 'from scratch' and the other one uses the VGG16 parameters for the feature extraction part.
* **net.py**: includes the 'from scratch' model class

## Results<a name="results"></a>
With 7 classes the random probability of getting a class right is just over 14%. The model 'from scratch' gets accuracy in the testing set over 80%, though not for all classes. The transfer learning model gets over 90%.  

The 'from scratch' model is live on [deep-rocking.herokuapps.com](deep-rocking.herokuapps.com). Unfortunately, the VGG16 based model was too big for the free limit on Heroku.

## Reflection<a name="reflection"></a>
Given the accuracy scores, the models work quite well. The classes for which the accuracy is lower are the classes with less images. Including more images for those classes would likely increase accuracy there as well. Further differentiation within classes would be possible, although the differences will become a lot more subtle. What would particulary be a problem in this field are the reissue guitars: modern copies of vintage guitars. These are difficult to distinguish from the vintage ones, but more less expensive. Also, cheaper copies of high quality guitars in general are visually difficult to distinguish. Differences in playability and tone are obviously not captured in images. 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Feel free to use the code. 
