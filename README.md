# Shoe Brand Image Classification with Deep Neural Networks

### No of free late days used - 1

## Link to Dataset
We have used Nike,Adidas Shoes for Image Classification Dataset from Kaggle and hosted in S3 Bucket.
https://publiclyhosteddata.s3.amazonaws.com/nike-adidas-dataset.zip


## Libraries Used -
- matplotlib
- numpy
- pyspark
- scikit-learn (for visualization)
- tensorflow (only for pre-processing)
- sparkdl (for comparing performance with standard transfer learning models)

## Steps to run -

We have performed Image Classifications using two models. 
1. <u>**Our own two layer DNN model written in pyspark and numpy.**</u><br>
    This model can easily be run on colab. Please use below link to run it -
   https://colab.research.google.com/drive/132KinbZRjHisFtkdmyhjjY5I-he9rVtk         
    
    It will perform the following operations - 
    - Installing necessary libraries
    - Downloading project code from Github
    - Initializing spark session
    - Image Preprocessing
    - Defining model hyper-parameters and necessary functions
    - Model training
    - Model predictions
    - Hyperparameter Tuning

2. <u>**SparkDL model with `inceptionV3` base model with transfer learning on our shoe image datatset.**</u>
    - Since spark-dl is a databricks supported library, and require specific versions of pyspark, tensorflow and other libraies, We have created custom enviroment on local for these.
    - Run `conda env create --file environment.yml -n sparkdl` to create the conda environment from the yaml file.
    - Run `conda activate sparkdl` to activate the environment
    - Run the notebook `nike-addidas-image-classification-with-libraries.ipynb` in this environment, to see results.
   