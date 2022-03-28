# Welcome to my Bootcamp Portfolio!

I am Jov Ermitaño, a data scientist-in-training. On this Github Pages, I present all the Python projects I did in the online Udemy Data Science Bootcamp.

On this page, I present all the fundamental knowledge about Data Science that I have learned from the Bootcamp, categorized into 4: Regression, Clustering, Classification, and Deep Learning.

Disclaimer: I do not own any of the datasets used in these Projects, and I built on the code I learned in the Bootcamp to work on the projects below. All credits and rights reserved to Udemy 365DataScience. I only created this page as my practice for a data science career.



# REGRESSION

## [Project 1: Simple Linear Regression](https://github.com/jovemmanuelre/Simple-Linear-Regression---StatsModels)
I created a simple regression model for a sample real estate dataset using Pandas, Statsmodels, Matplotlib, and Seaborn packages to understand and describe the causal relationship between 'Price' and 'Size' of the real estate properties.

![Regression Summary](images/Linear%20Regressions/Simple/Screen%20Shot%202022-02-04%20at%2012.18.23%20PM.png)

After examining the Linear Regression results of the dataset with 'Price' as the dependent variable, I discovered that the 'Price' of the real estate property increases as the 'Size' does, as shown in the Regression Graph below:

![Graph](images/Linear%20Regressions/Simple/Screen%20Shot%202022-02-08%20at%201.04.08%20PM.png)

## [Project 2: Multiple Linear Regression with Dummies](https://github.com/jovemmanuelre/Multiple-Linear-Regression-w**ith-Dummies)
Using the sample real estate dataset from the Project 1, I performed Multiple Regression using Pandas, Statsmodels, Matplotlib, and Seaborn to understand the causal relationship among 'Price' and 'Size' of the real estate properties, 'Year' they were constructed, and the availability of 'Sea View'. 'Sea View' is categorical variable, so I mapped it into a Dummy Variable in order to perform the regression.

![Regression Summary](images/Linear%20Regressions/Multiple%20w:%20Dummies/Screen%20Shot%202022-02-04%20at%2012.28.32%20PM.png)

With 'Price' as the dependent variable, the Regression results show that 'Size', 'Year', and 'Sea View' explain the variability of the 'Price', as indicated by the R-squared of 0.913 (An R-squared of 1 means that the Regression explains the entire variability of the dataset). I discovered that the 'Price' of the real estate properties is explained by their 'Size', 'Year' they were made, and the availability of 'Sea View' in the units.

## [Project 3: Simple Linear Regression with sklearn](https://github.com/jovemmanuelre/Simple-Regression-sklearn)
Instead of Statsmodels, here I used sklearn, with Pandas, Matplotlib, and Seaborn to create a Simple Linear Regression model and predict the price of the real estate property given its size.

This is the Graph that shows the linear relationship between the 'Price' and 'Size':
![Graph](images/Linear%20Regressions/Screen%20Shot%202022-03-09%20at%204.49.26%20AM.png)

![sklearn code](images/Linear%20Regressions/Screen%20Shot%202022-03-09%20at%204.49.56%20AM.png)

Using sklearn, I created the Regression and could predict the 'Price' of the property given its 'Size' with fewer lines of code than Statsmodels!

## [Project 4: Linear Regression on a Real Business Case](https://github.com/jovemmanuelre/Practical-Case-Example-Regression-with-sklearn)
Here I worked on a real life dataset of cars. I used Numpy, Pandas, Statsmodels, Matplotlib, sklearn, and Seaborn to do the following: preprocess the data, remove the Outliers, check the Ordinary Least Squares (OLS) assumptions, train the model, verify its Residuals, and test the model.

My goal in this project was to predict the 'Price' of a car given its Mileage and Engine Volume and the Year it was made.

After preprocessing the data, I checked the Probability Distribution Functions (PDF) of the Variables 'Price', 'Year', 'Mileage', and 'Engine Volume' to identify and weed out the Outliers and ensure the accuracy of my model.

With Outliers:
![Price](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Before-Price.png)
**Without Outliers:**
![PDF Price](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/PDF_Price.png)

With Outliers:
![Mileage](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Before-Mileage.png)
**Without Outliers:**
![PDF Mileage](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/PDF_Mileage.png)

With Outliers:
![EngineV](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Before-EngineV.png)
**Without Outliers:**
![PDF Engine](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/PDF_EngineV.png)

With Outliers:
![Year](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Before-Year.png)
**Without Outliers:**
![PDF_Year](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/PDF_Year.png)

I then checked the OLS assumptions to be sure that Regression is the right model to create.

I performed the log transformation on the 'Price' to fix heteroscedasticity so that my model won't lose its predictive power on datasets with bigger values.
![Price and Features](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Price%20and%20the%20Features.png)

![Log Price and Features](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Log%20Price%20and%20the%20Features.png)
Now, the PDFs of the Log Price with respect to the features show a Linear Regression Line.

I then trained and tested the model, and achieved a 72% accuracy in predicting the Price given the features mentioned above.

![Residual Graph](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Screen%20Shot%202022-03-09%20at%205.42.23%20AM.png)

As a bonus, I verified that the Residuals and variability of the outcome are normally distributed, then created a table which shows the residuals and the differences between my Predictions and Targets.

![Sorted DF_PF](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Sorted%20DF_PF.png)



# CLUSTERING

## [Project 5: Clustering the Countries](https://github.com/jovemmanuelre/Clustering-Countries)
Using Matplotlib, I clustered the countries in the dataset into 7, according to the data on their latitudes and longitudes. This allowed me to plot their actual location accurately on a graph.

This was the appearance of the dataset on a graph, without clustering them:
![Before Clustering](images/Clustering/Countries/Screen%20Shot%202022-02-04%20at%204.13.26%20PM.png)

Countries clustered into 7:
![After Clustering](images/Clustering/Countries/Screen%20Shot%202022-02-04%20at%204.24.52%20PM.png)

## [Project 6: Clustering the Countries with Categorical Data](https://github.com/jovemmanuelre/Clustering-Countries-Categorical)
This time, I clustered the countries into 7 based on their Continents. Since the names are Categorical Variables, I mapped them into a Dummy Variable in order to create the Clustering on a graph. 

Before:
![Before Categorization](images/Clustering/Clustering%20Categorical%20Data/Screen%20Shot%202022-02-18%20at%207.14.59%20PM.png)


After mapping the Categorical Variable 'Continent' and clustering the countries:
![Clustered Categorized Data](images/Clustering/Clustering%20Categorical%20Data/Screen%20Shot%202022-02-18%20at%207.18.51%20PM.png)

The graph below accurately plots all the countries based on their continents, like a world map.
![Graph](images/Clustering/Clustering%20Categorical%20Data/Screen%20Shot%202022-02-18%20at%207.21.05%20PM.png)

## [Project 7: Basic Dendogram with Heatmap](https://github.com/jovemmanuelre/Basic-Dendogram-with-Heatmap)
Using Seaborn, I created a simple Dendogram with Heatmap using the Latitude and Longitude of the countries to show their similarities and differences in location.

![Dendogram](images/Heatmaps%20and%20Dendograms/Screen%20Shot%202022-02-27%20at%206.52.07%20AM.png)

## [Project 8: Market Segmentation with Clustering](https://github.com/jovemmanuelre/Market-Segmentation-with-Clustering)
My goal in this project was to conduct an exploratory analysis to identify the target customers of the product.

I used sklearn to preprocess the data, then used the Elbow Method to determine the appropriate number of clusters:
![Elbow Method](images/Classification/Market%20Segmentation%20Clustering/Screen%20Shot%202022-02-27%20at%2012.55.44%20PM.png)

I was able to cluster the customers into 4 based on their Loyalty and Satisfaction, as shown below. With the insight from this Graph, suitable marketing techniques may now be designed and used to target these customers.
![Cluster Graph](images/Classification/Market%20Segmentation%20Clustering/Screen%20Shot%202022-02-27%20at%2012.53.56%20PM.png)

## [Project 9: Iris Dataset Clustering Exercise](https://github.com/jovemmanuelre/Clustering-Iris-Dataset.git)
The Iris flower dataset is one of the most popular datasets for machine learning. This real data set is data observed in the real world and consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). 4 features were measured from each sample: sepal length, sepal width, petal length, and petal width.

![Iris Unclustered and Unscaled](images/Clustering/Iris%20Flower%20Dataset/Unclustered%20and%20Unscaled%20-%20Iris.png)
![Unscaled Iris](images/Clustering/Iris%20Flower%20Dataset/Unscaled%20-%20Iris.png)

I started by clustering the dataset into 2, 3, and 5, using the Elbow Method. 
![Iris Elbow Method](images/Clustering/Iris%20Flower%20Dataset/Elbow%20Method%20-%20Iris.png)

However, after examining the other scatter plot (petal length vs petal width), I discovered that in fact the features which actually make the species different are petals and NOT sepals!
![Answers Sepal Size](images/Clustering/Iris%20Flower%20Dataset/Sepal%20Size%20%20Answers.png)
![Answers Petal Size](images/Clustering/Iris%20Flower%20Dataset/Petal%20Size%20Answers.png)

I then compared the scatter plots and learned the Elbow Method, cannot be trusted at all times, as it only shows the potential optimal number of clusters, but doesn't provide the optimal number of clusters for a dataset.

After researching, I confirmed thatthe original dataset has 3 sub-species of the Iris flower. Therefore, the number of clusters is 3.
![Iris Scaled](images/Clustering/Iris%20Flower%20Dataset/Scaled%20-%20Iris.png)
![Table of Iris Dataset Answers and Clustered](images/Clustering/Iris%20Flower%20Dataset/Table%20of%20Iris%20Dataset%20Answers%20and%20Clustered.png)

I finished the project by import the CSV with the correct answers (iris_with_answers.csv) and checked if the clustering worked as expected. 

Note: I am aware that this is not the usual way to work on clustering problems; I could've used classification. as I already had the answers.

# CLASSIFICATION

## [Project 10: Simple and Multivariate Classification](https://github.com/jovemmanuelre/Simple-and-Multivariate-Classification)
My goal in this Project was to create a model that would tell whether the bank marketing strategy was successful. I expanded the model to include other causal factors that might be omitted by the simple logistic model. I used the multivariate logistic regression to include many causal factors such as the ‘interest_rate’, ‘march’, ‘credit’ and ‘previous’ estimators to improve the accuracy of my model.

![Regression Summary](images/Classification/Simple%20and%20Multivariate%20Classification/Screen%20Shot%202022-02-19%20at%208.30.48%20PM.png)
The 'interest rate' indicates the 3-month interest rate between banks and duration indicates the time since the last contact was made with a given consumer. The previous variable shows whether the last marketing campaign was successful with this customer. The 'march' and 'may' are Boolean variables that account for when the call was made to the specific customer and 'credit' shows if the customer has enough credit to avoid defaulting.

![Graph of Subs and Duration](images/Classification/Simple%20and%20Multivariate%20Classification/Screen%20Shot%202022-02-19%20at%208.32.20%20PM.png)

![Optimized Graph](images/Classification/Simple%20and%20Multivariate%20Classification/Screen%20Shot%202022-02-19%20at%208.32.34%20PM.png)

![Confusion Matrix Results](images/Classification/Simple%20and%20Multivariate%20Classification/Screen%20Shot%202022-02-19%20at%208.42.21%20PM.png)
In addition, I used the Confusion Matrix to estimate the accuracy of the model. I compared the the test confusion matrix and the test accuracy with the train confusion matrix and the train accuracy.

## [Project 11: Classification with Binary Predictors](https://github.com/jovemmanuelre/Classification-with-Binary-Predictors)
In this project, what did I do? What was my goal? and How did I do it?

![Optimized Regression Summary](images/Classification/Classification%20with%20Binary%20Predictors/LogIt%20Regression%20Result.png)

I used the Confusion Matrix to compare the accuracy of the test and train data. It was a 90-10 Split.
![Confusion Matrix and Accuracy of Train](images/Classification/Classification%20with%20Binary%20Predictors/Confusion%20Matrix%20and%20Accuracy%20of%20my%20Training%20Model.png)

![Confusion Matrix and Accuracy of Test](images/Classification/Classification%20with%20Binary%20Predictors/Confusion%20Matrix%20and%20Accuracy%20of%20my%20Model.png)

The Misclassification rate is the exact opposite of the accuracy rate of my model:
![Misclassification Rate](images/Classification/Classification%20with%20Binary%20Predictors/Misclassification%20Rate.png)



# DEEP LEARNING

## [Project 12: Basic Neural Network with Numpy](https://github.com/jovemmanuelre/Building-a-Basic-Neural-Network-with-NumPy)
In this project, my goal was to create a basic Neural Network/ML algorithm that must learn that the function that I randomly generated using Numpy. I used Learning Rate and Update Rules from Gradient Descent to create the algorithm.

As a bonus, I used matplotlib for the 2D and 3D graphs to check the Regression:
![3D](images/Deep%20Learning/Building%20a%20Basic%20Neural%20Network%20with%20Numpy/Screen%20Shot%202022-03-02%20at%205.53.04%20AM.png)
![Graph](images/Deep%20Learning/Building%20a%20Basic%20Neural%20Network%20with%20Numpy/Screen%20Shot%202022-03-02%20at%205.53.17%20AM.png)

## [Project 13: Basic Neural Network with Tensorflow](https://github.com/jovemmanuelre/Building-a-Basic-Neural-Network-with-Tensorflow)
In this project I recreated the basic Neural Network I did using Numpy in the previous example, using Tensorflow 2.0.

While extracting the weights, bias, and outputs in my model is inessential, I did extract them in this project to verify whether my answers are correct.

The graph is the same as the one in the previous exercise where I used NumPy:
![Graph](images/Deep%20Learning/Building%20a%20Basic%20Neural%20Network%20with%20Tensorflow/Screen%20Shot%202022-03-05%20at%206.29.31%20PM.png)

## [Project 14: Deep Neural Network for MNIST Classification](https://github.com/jovemmanuelre/Deep-Neural-Network-for-MNIST-Classification)
Known as the "Hello World" of deep learning, the MNIST dataset refers to handwritten digit recognition.  The goal is to write an algorithm that detects which digit is written. Since there are only 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), this is a classification problem with 10 classes. The dataset provides 70,000 images (28x28 pixels) of handwritten digits (1 digit per image). 

After tweaking my hyperparameters, here I used 5 hidden layers with 3000 hidden units each and 10 Epochs to achieve an accuracy of 97.93% percent.


## [Project 15: Audiobooks Case with sklearn and Early Stopping](https://github.com/jovemmanuelre/Deep-Learning-Audiobooks-Case-Preprocessed-and-with-Early-Stopping)
To be updated 

I used data from an Audiobook app to create a machine learning algorithm can predict if a customer will buy again from the Audiobook company, based on the last 2 years of activity and engagement, a customer will convert in the next 6 months. 6 months sounds like a reasonable time. If they don't convert after 6 months, chances are they've gone to a competitor or didn't like the Audiobook way of digesting information. 

This is a classification problem with two classes: won't buy and will buy, represented by 0s and 1s. 

The main idea is that if a customer has a low probability of coming back, there is no reason to spend any money on advertizing to them, thus saving the company some money. Moreover, this model can identify the most important metrics for a customer to come back again. Identifying new customers creates value and growth opportunities.

The inputs are the following: Customer ID, Book length in mins_avg (average of all purchases), Book length in minutes_sum (sum of all purchases), Price Paid_avg (average of all purchases), Price paid_sum (sum of all purchases), Review (a Boolean variable), Review (out of 10), Total minutes listened, Completion (from 0 to 1), Support requests (number), and Last visited minus purchase date (in days).

My model achieved ~80% accuracy in predicting whether a customer will buy again from the Audiobook company.

Each customer in the database has made a purchase at least once,  We want to create a machine learning algorithm based on our available data that can predict if a customer will buy again from the Audiobook company. The main idea is that if a customer has a low probability of coming back, there is no reason to spend any money on advertising to him/her. If we can focus our efforts SOLELY on customers that are likely to convert again, we can make great savings. Moreover, this model can identify the most important metrics for a customer to come back again. Identifying new customers creates value and growth opportunities.

You have a .csv summarizing the data. There are several variables: Customer ID, ), Book length overall (sum of the minute length of all purchases), Book length avg (average length in minutes of all purchases), Price paid_overall (sum of all purchases) ,Price Paid avg (average of all purchases), Review (a Boolean variable whether the customer left a review), Review out of 10 (if the customer left a review, his/her review out of 10, Total minutes listened, Completion (from 0 to 1), Support requests (number of support requests; everything from forgotten password to assistance for using the App), and Last visited minus purchase date (in days).

The targets are a Boolean variable (0 or 1). We are taking a period of 2 years in our inputs, and the next 6 months as targets. So, in fact, we are predicting if: based on the last 2 years of activity and engagement, a customer will convert in the next 6 months. 6 months sounds like a reasonable time. If they don't convert after 6 months, chances are they've gone to a competitor or didn't like the Audiobook way of digesting information. 


This is a classification problem with two classes: won't buy and will buy, represented by 0s and 1s. 

## Project 16: Absenteeism Case
![Grouped Feature Descriptions](images/Deep%20Learning/Absenteeism/Absenteeism%20Feature%20Descriptions%20Grouped.png)
![Feature Descriptions](images/Deep%20Learning/Absenteeism/Absenteeism%20Feature%20Descriptions.png)
![Unprocessed Data](images/Deep%20Learning/Absenteeism/Data%20Summary%20of%20Unprocessed%20Data.png)
![With Month and Day of the Week](images/Deep%20Learning/Absenteeism/With%20Month%20and%20Day%20of%20the%20Week.png)
![With Predictions](images/Deep%20Learning/Absenteeism/With%20Predictions.png)

I created a logistic regression  model that will predict the probability of an individual being excessively absent from work. The logistic regression is a type of a regression model whose dependent variable is binary; that is, the latter can assume one of two values – 0 or 1, True or False, Yes or No. In this case therefore, considering the values of all the features, I wanted to predict whether the dependent variable will take the value of 0 or 1.
 
In addition to numerous machine learning tools and techniques, I specifically used pandas to preprocess the data because it allows me to work with panel data. Moreover, pandas has various tools for handling data in a tabular format (a DataFrame). Finally as a data scientist, I want a more manual way of doing preprocessing because it gives me a higher level of control over my data.

At the end, I stored my work as a Python module called ‘absenteeism_module’ to preserve it in a form suitable for further analysis.
