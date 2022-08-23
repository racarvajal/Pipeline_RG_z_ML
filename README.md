# Machine Learning Pipeline for predicting Radio Galaxies

This set of Notebooks are designed to create a pipeline of Machine Learning 
models for predicting the detection of Radio Galaxies and their redshift values.  

## Training of models
One intial model is trained to classify between galaxies and AGN.
A second model is trained to classify between AGN having, or not, 
radio detection above certain limit (given by detections in selected radio surveys).
A third model is trained to predict redshift values for radio-detected AGN.

## Application of pipeline
The pipeline works by applying the first model to obtain a list of predicted AGN.
Then, the second model is applied to these predicted AGN and radio-detected 
sources are predicted.
Finally, the third model is applied to the predicted radio-detected AGN and a 
predicted redshift value is obtained.

## File descriptions
Datasets are located in the relative path (not available in this repository):

    ../../Catalogs/

Description of most files and folders can be seen in `files_naming.txt`.
Plots and images are not included in this repository.
