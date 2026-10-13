# Neural Network Analysis – Brief Report

## Dataset and Method

The UCI Iris dataset is made up of 150 observations, 3 classes, and 4 numerical features. First the duplicates were removed, bringing the observation count down to 149 observations. Missing values were handled, features were standardized, and the data was split into 80% training and 20% testing.

All 24 combinations across 4 variables were tested. There were three activation functions (logistic, tanh, ReLU), two learning rates (0.01, 0.1), two epoch counts (100, 200), and two hidden-layer configurations (2, 3), using 10 neurons per hidden layer.

## Results

Individually, the best model was Model 3 ("M03"),  using logistic activation, learning rate 0.01, 200 epochs, and 2 hidden layers. It achieved a training accuracy of 98.32%, a test accuracy of 96.67%, and a 0.0730 test log loss. 

At 95.42%, Logistic and tanh tied for the highest average test accuracy.For the same category, ReLU had 93.75%. The Learning rate of 0.01 performed slightly better overall (95.83%) than 0.1 (93.89%).

## Interpretation

ReLU did converge faster, but in the end, Logistic and tanh performed best on average. Test accuracy was NOT consistenly improed by increasing the number of epochs or hidden layers. Several models achieved 100% training accuracy but lower test accuracy. This does suggest possible overfitting.

## Assumptions and Limitations

The last column was treated as the target, with numerical features as inputs. All models used 10 neurons per hidden layer and the same data split. Results are limited by the small test set of 30 observations and a single random seed.