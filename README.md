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
I am able to predict the price of a watch with a high precision. Using an ensemble of a Random Forest and a Neural Network I am able to explain 95% of the variance in watch prices. <br>


| Model  | R² |
| ------------- | ------------- |
| Random Forest  | 93.47%  | 
| Neural Network  | 94.61%  |
| **Ensemble**  | **95.03%**  |

Unsurprisingly, the brand, model and reference number are the main determinants of a watches price but there are other significant features as well. Some noteable examples would be the materials, movement and age of the watch.

![Feature importance](https://github.com/Ortgies/chrono/blob/main/graphics/regression_fi.png)

## Brand classification
I try out two approaches to classify watch brands. <br>
The first is based on the watch characteristics and the second on the picture accompanying the offer. <br>
In the final step I combine the results of both models into a final predictor. <br>

### Features
Using a random forest classifier, I am able to classify watch brands with a precision of 70%. <br>
Integrated functions (complications), materials, size and gender are important factors in classifying the watch brand.

![Feature importance](https://github.com/Ortgies/chrono/blob/main/graphics/classification_fi.png)

### Picture
Using a Resnet18 based Neural Network, I am able to correctly classify over 84% of watch pictures.

### Ensemble
When combining both models, accuracy gets another significant boost to 87%.

| Model  | R² |
| ------------- | ------------- |
| Random Forest - Specifications| 70.09%  |
| Neural Network - Images| 84.62%  |
| **Ensemble**| **87.16%**  |
