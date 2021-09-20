# chrono
In this project I collect a dataset of watch offerings from Chrono24, the largest online platform for buying and selling watches. <br>
I use this dataset to develop two models: <br>
- A regression model to predict watch prices based on its characteristics
- A classification model to predict the watch brand based on its characteristics and image data

# Data collection
I set up a webscraping algorithm to collect data about recent watch offerings on Chrono24 using Selenium. <br>
The final dataset contains over 60.000 individual watch sales. <br>

# Results
## Price regression
| Model  | R² | R² (tuned) |
| ------------- | ------------- | ------------- |
| Linear Regression  | 86.13%  |   |
| Ridge Regression  | 86.13%  |  |
| **Random Forest**  | **92.65%**  | **92.68%**  |
| Gradient Boosting  | 89.54%  |   |

![Feature importance](https://github.com/Ortgies/chrono/blob/main/graphics/regression.png)

## Model classification
### Features
| Model  | R² | R² (tuned) |
| ------------- | ------------- | ------------- |
| LogisticRegression  | 47.87%  |   |
| KNeighborsClassifier  | 53.67%  |  |
| **RandomForest**  | 60.21%  | **60.79%**  |
| Gradient Boosting  | **61.69%**  |   |

![Feature importance](https://github.com/Ortgies/chrono/blob/main/graphics/classification.png)

### Picture
| Model  | R² | R² (tuned) |
| ------------- | ------------- | ------------- |
| LogisticRegression  | 56.56%  |   |
| KNeighborsClassifier  | 40.74%  |  |
| **RandomForest**  | 55.24%  | **56.57%**  |
| Gradient Boosting  | **56.76%**  |   |
### Ensemble
| Model  | R² |
| ------------- | ------------- |
| Voting Classifier| 71.10%  |
| **Stacking Classifier**| **73.66%**  |
