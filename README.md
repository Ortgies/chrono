# chrono
In this project I collect a dataset of watch offerings from Chrono24, the largest online platform for buying and selling watches. <br>
I use this dataset to develop two models: <br>
- A regression model to predict watch prices based on its characteristics
- A classification model to predict the watch brand based on its characteristics and image data

# Motivation
I wanted to showcase my understanding of regression and classification on a custom dataset. <br>
Instead of using one of the popular datasets like real estate prices and car insurance, it was important to me to examine a problem that hasnt been 'solved' before. <br>
For this reason I collected a dataset on watch features and prices. As a small child I was already fascinated with mechanical watch movements, gears and springs. Later, as a finance student, I got interested in the business side as well. The watch market is very unusual and defies many of the assumptions economists have about markets. <br>

# Data collection
I set up a webscraping algorithm to collect data about recent watch offerings on Chrono24 using Selenium. <br>
The final dataset contains over 60.000 individual watch sales. <br>

# Results
## Price regression
I am able to predict the price of a watch with a high precision. Over 92% of the variation in watch prices can be explained using a random forest model. <br>


| Model  | R² | R² (tuned) |
| ------------- | ------------- | ------------- |
| Linear Regression  | 86.13%  |   |
| Ridge Regression  | 86.13%  |  |
| **Random Forest**  | **92.65%**  | **92.68%**  |
| Gradient Boosting  | 89.54%  |   |

Unsurprisingly, the brand and model are the main determinants of a watches price but there are other significant features as well. Some noteable examples would be the case material, movement or the country of sale. <br>

![Feature importance](https://github.com/Ortgies/chrono/blob/main/graphics/regression.png)

## Model classification
I try out two approaches to classify watch brands. <br>
The first is based on the watch characteristics and the second on the picture accompanying the offer. <br>
In the final step I combine the results of both models into a final predictor. <br>

### Features
Using a random forest classifier, I am able to classify watch brands with a precision of almost 61%. <br>
When gradient boosting is used, accuracy can be increased an additional 0.9%. This comes at the cost of significantly longer run times however. <br>

| Model  | R² | R² (tuned) |
| ------------- | ------------- | ------------- |
| LogisticRegression  | 47.87%  |   |
| KNeighborsClassifier  | 53.67%  |  |
| **RandomForest**  | 60.21%  | **60.79%**  |
| Gradient Boosting  | **61.69%**  |   |

Some of the most imporant features for classification are the watch diameter, the decade of production and the country of sale.

![Feature importance](https://github.com/Ortgies/chrono/blob/main/graphics/classification.png)

### Picture
Based on recognized words in the watch picture, I am able to predict the brand with an accuracy of over 56% using a Random Forest model. <br>

| Model  | R² | R² (tuned) |
| ------------- | ------------- | ------------- |
| LogisticRegression  | 56.56%  |   |
| KNeighborsClassifier  | 40.74%  |  |
| **RandomForest**  | 55.24%  | **56.57%**  |
| Gradient Boosting  | **56.76%**  |   |
### Ensemble
When combining the feature- and picture based regression, the accuracy of classification can be significantly improved. <br>
A stacking classifier, using a logisitic regression is able to predict the correct watch brand in almost 3 out of 4 watches. <br>

| Model  | R² |
| ------------- | ------------- |
| Voting Classifier| 71.10%  |
| **Stacking Classifier**| **73.66%**  |
