# Machine Learning Pipeline for predicting Radio Galaxies

This set of Notebooks are designed to create a pipeline of Machine Learning models for predicting the detection of Radio Galaxies and their redshift values.  

One initial model will be trained to classify between galaxies and AGN.
This model will be applied to non-labeled sources.
These new sources will be added to the training set and a new model will be
trained that can predict the score a source has of being galaxy or AGN.

Then, a model will be trained to predict if an AGN can have a detection in  
certain radio bands.

Finally, a model will be trained to predict the redshift values of Radio AGN.

Datasets are located in the relative path:

    ../../Catalogs/
