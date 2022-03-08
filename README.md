## Welcome to my Github Pages!

I am Jov Ermitaño, a data scientist-in-training. On this Github Pages, I present all the projects that I worked on the online Data Science bootcamp course I took on Udemy.

I present all the fundamental knowledge about Data Science I have learned from the course. I categorized the projects into 4: Regression, Clustering, Classification, and Deep Learning. I worked on all these Projects on Python.

Disclaimer: I do not own any of the datasets used in these Projects or all the code I have learned from the bootcamp course, which I edited to suit my learning style. All of this is shared for educational purposes only. All rights are reserved to 365DataScience on Udemy. 


# REGRESSION

## [Project 1: Simple Linear Regression](https://github.com/jovemmanuelre/Simple-Linear-Regression---StatsModels)
I created a simple regression model for a sample real estate dataset using Pandas, Statsmodels, Matplotlib, and Seaborn libraries to understand and describe the causal relationship between 'Price' and 'Size' of the real estate properties.

![Regression Summary](images/Linear%20Regressions/Simple/Screen%20Shot%202022-02-04%20at%2012.18.23%20PM.png)
After examining the Linear Regression results of the dataset with 'Price' as the dependent variable', I discovered that the 'Price' of the real estate property increases as the 'Size' does, as shown in the Regression Graph below:

![Graph](images/Linear%20Regressions/Simple/Screen%20Shot%202022-02-08%20at%201.04.08%20PM.png)

## [Project 2: Multiple Linear Regression with Dummies](https://github.com/jovemmanuelre/Multiple-Linear-Regression-with-Dummies)
Using the sample real estate dataset above, in this Project I performed Multiple Regression using Pandas, Statsmodels, Matplotlib, and Seaborn to understand the causal relationship among 'Price' and 'Size' of the real estate properties, 'Year' they were constructed, and the availability of 'Sea View'. 'Sea View' is categorical variable, so I mapped it into a Dummy Variable in order to perform the regression.

![Regression Summary](images/Linear%20Regressions/Multiple%20w:%20Dummies/Screen%20Shot%202022-02-04%20at%2012.28.32%20PM.png)
With 'Price' as the dependent variable, the Regression results show that 'Size', 'Year', and 'Sea View' explain the variability of the 'Price', as indicated by the R-squared of 0.913 (1 meaning the Regression explains the entire variability of the dataset). I discovered that the 'Price' of the real estate properties is explained the 'Size', 'Year', and 'Sea View'.

## [Project 3: Simple Linear Regression with sklearn](https://github.com/jovemmanuelre/Simple-Regression-sklearn)
Instead of Statsmodels, here I used sklearn, with Pandas, Matplotlib, and Seaborn in Python to create a Simple Linear Regression model and predict the price given the size of the real estate property.

![Graph](images/Linear%20Regressions/Screen%20Shot%202022-03-09%20at%204.49.26%20AM.png)
This is the Graph that shows the linear relationship between 'Price' and 'Size'.

![sklearn code](images/Linear%20Regressions/Screen%20Shot%202022-03-09%20at%204.49.56%20AM.png)
Using sklearn, I performed the Regression and could predict the 'Price' of the property given its 'Size' with fewer lines of code!

## [Project 4: Linear Regression on a Real Business Case](https://github.com/jovemmanuelre/Practical-Case-Example-Regression-with-sklearn)
Whaat is this Real Life Example? What did I create? What are these PDFs? What did I discover/predict? What is the outcome?

I worked on a dataset of cars. It is a real business case. I used Numpy, Pandas, Statsmodels, Matplotlib, sklearn, and Seaborn to predict the 'Price' of a car given its Mileage, Engine Volume, and Year created. I removed the Outliers, checked the OLS assumptions, trained the model, checked for its Residuals, and tested my model (more information in the README of this project's repo).

I checked the Probability Distribution Functions (PDF) of the Variables 'Price', 'Year', 'Mileage', and 'Engine Volume' to identify and weed out the Outliers and ensure the accuracy of my model.

With Outliers:
![Price](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Before-Price.png)
Without Outliers:
![PDF Price](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/PDF_Price.png)

With Outliers:
![Mileage](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Before-Mileage.png)
Without Outliers:
![PDF Mileage](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/PDF_Mileage.png)

With Outliers:
![EngineV](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Before-EngineV.png)
Without Outliers:
![PDF Engine](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/PDF_EngineV.png)

With Outliers:
![Year](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Before-Year.png)
Without Outliers:
![PDF_Year](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/PDF_Year.png)

Then, I performed the log transformation on the 'Price' to fix heteroscedasticity, which loses its predicive power on datasets with bigger values
![Price and Features](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Price%20and%20the%20Features.png)

![Log Price and Features](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Log%20Price%20and%20the%20Features.png)
Now the PDFs show a Linear Regression Line.

![DF_PF](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/DF_PF.png)

I then trained and tested the model and achieved a 72% accuracy.

I also checked the Residuals to verify whether the Residuals and variability of the outcome are normally distributed then created a table which shows the residuals and the differences between my Predictions and Targets.

![Sorted DF_PF](images/Linear%20Regressions/Multiple%20Linear%20Regression%20Practical%20Example%20with%20sklearn/Sorted%20DF_PF.png)


# CLUSTERING

## [Project 5: Clustering the Countries](https://github.com/jovemmanuelre/Clustering-Countries)
I did Clustering in Python using Matplotlib to plot the location of all countries in a graph. Using their respective latitudes and longitudes, I clustered the countries into 7. The graph shows the location of the countries per continent.

![Before Clustering](images/Clustering/Countries/Screen%20Shot%202022-02-04%20at%204.13.26%20PM.png)

![After Clustering](images/Clustering/Countries/Screen%20Shot%202022-02-04%20at%204.24.52%20PM.png)

## [Project 6: Clustering the Countries with Categorical Data](https://github.com/jovemmanuelre/Clustering-Countries-Categorical)
I clustered the countries into 7 categories based on their continents. The names of the continents are categorical data, so I mapped them into numerical data to show their clusters on a graph.
What made this different from the first, and what did I discover?

![Before Categorization](images/Clustering/Clustering%20Categorical%20Data/Screen%20Shot%202022-02-18%20at%207.14.59%20PM.png)

![After Categorization](images/Clustering/Clustering%20Categorical%20Data/Screen%20Shot%202022-02-18%20at%207.15.14%20PM.png)

![Clustered Categorized Data](images/Clustering/Clustering%20Categorical%20Data/Screen%20Shot%202022-02-18%20at%207.18.51%20PM.png)

![Graph](images/Clustering/Clustering%20Categorical%20Data/Screen%20Shot%202022-02-18%20at%207.21.05%20PM.png)

## [Project 7: Basic Dendogram with Heatmap](https://github.com/jovemmanuelre/Basic-Dendogram-with-Heatmap)

![Dendogram](images/Heatmaps%20and%20Dendograms/Screen%20Shot%202022-02-27%20at%206.52.07%20AM.png)

## [Project 8: Market Segmentation with Clustering](https://github.com/jovemmanuelre/Market-Segmentation-with-Clustering)

![Elbow Method](images/Classification/Market%20Segmentation%20Clustering/Screen%20Shot%202022-02-27%20at%2012.55.44%20PM.png)

![Cluster Graph](images/Classification/Market%20Segmentation%20Clustering/Screen%20Shot%202022-02-27%20at%2012.53.56%20PM.png)


# CLASSIFICATION

## [Project 9: Simple and Multivariate Classification](https://github.com/jovemmanuelre/Simple-and-Multivariate-Classification)

![Regression Summary](images/Classification/Simple%20and%20Multivariate%20Classification/Screen%20Shot%202022-02-19%20at%208.30.48%20PM.png)

![Graph of Subs and Duration](images/Classification/Simple%20and%20Multivariate%20Classification/Screen%20Shot%202022-02-19%20at%208.32.20%20PM.png)

![Optimized Graph](images/Classification/Simple%20and%20Multivariate%20Classification/Screen%20Shot%202022-02-19%20at%208.32.34%20PM.png)

![Confusion Matrix Results](images/Classification/Simple%20and%20Multivariate%20Classification/Screen%20Shot%202022-02-19%20at%208.42.21%20PM.png)

## [Project 10: Classification with Binary Predictors](https://github.com/jovemmanuelre/Classification-with-Binary-Predictors)

![Optimized Regression Summary](images/Classification/Classification%20with%20Binary%20Predictors/LogIt%20Regression%20Result.png)

![Confusion Matrix and Accuracy of Train (images/Classification/Classification%20with%20Binary%20Predictors/Confusion%20Matrix%20and%20Accuracy%20of%20my%20Training%20Model.png)

![Confusion Matrix and Accuracy of Test](images/Classification/Classification%20with%20Binary%20Predictors/Confusion%20Matrix%20and%20Accuracy%20of%20my%20Model.png)

![Misclassification Rate](images/Classification/Classification%20with%20Binary%20Predictors/Misclassification%20Rate.png)


# DEEP LEARNING

## [Project 11: Basic Neural Network with Numpy](https://github.com/jovemmanuelre/Building-a-Basic-Neural-Network-with-NumPy)

![3D](images/Deep%20Learning/Building%20a%20Basic%20Neural%20Network%20with%20Numpy/Screen%20Shot%202022-03-02%20at%205.53.04%20AM.png)

![Graph](images/Deep%20Learning/Building%20a%20Basic%20Neural%20Network%20with%20Numpy/Screen%20Shot%202022-03-02%20at%205.53.17%20AM.png)

## [Project 12: Basic Neural Network with Tensorflow](https://github.com/jovemmanuelre/Building-a-Basic-Neural-Network-with-Tensorflow)

![Graph](images/Deep%20Learning/Building%20a%20Basic%20Neural%20Network%20with%20Tensorflow/Screen%20Shot%202022-03-05%20at%206.29.31%20PM.png)

## [Project 13: Deep Neural Network for MNIST Classification](https://github.com/jovemmanuelre/Deep-Neural-Network-for-MNIST-Classification)
I achieved 97.93% accuracy. Where are the images?

## [Project 14: Audiobooks Case with sklearn and Early Stopping](https://github.com/jovemmanuelre/Deep-Learning-Audiobooks-Case-Preprocessed-and-with-Early-Stopping)
Where are the images?
