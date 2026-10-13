# Neural Network Analysis - Brief Report

## Dataset and preprocessing

UCI Iris dataset (150 flower observations, four numerical features, three species).
The program reads a public CSV copy from the seaborn-data GitHub repository.
Duplicate/blank rows are removed; missing feature values are imputed with
training-set medians. Features are standardized using training-set statistics
only. A stratified 80/20 train/test split uses random_state=42.

## Experiment setup

All 24 combinations were tested: logistic/tanh/ReLU activation,
learning rates 0.01/0.1, 100/200 epochs, and 2/3 hidden layers.
Each hidden layer contains 10 neurons. The Adam optimizer is trained
incrementally with partial_fit so its state is preserved across epochs.
Training accuracy is recorded after each epoch.
Final errors are multiclass log loss.

## Best individual model

- Model: M03
- Activation: logistic
- Learning rate: 0.01
- Epochs: 200
- Hidden layers: 2
- Training accuracy: 98.32%
- Test accuracy: 96.67%
- Training log loss: 0.0598
- Test log loss: 0.0730

The best model is selected by highest test accuracy,
with lowest test log loss used to break ties.

## Activation-function comparison

- logistic: 95.42% average test accuracy
- tanh: 95.42% average test accuracy
- relu: 93.75% average test accuracy

logistic had the highest average test accuracy across the
eight tested configurations per activation. This result describes
this dataset and these settings, not a universal ranking
of activation functions.

## Learning-rate comparison

- 0.01: 95.83% average test accuracy
- 0.1: 93.89% average test accuracy

Learning rate 0.01 had the highest average test accuracy
across the twelve configurations tested for each learning rate.
The history plots show whether accuracy improved steadily
or fluctuated during training.

## Interpretation and limitations

Review the six plots before submission and add your own observations
about convergence, hidden-layer depth, training duration, and possible
overfitting. Iris has a small test set (30 samples), so differences
of one correctly classified sample change test accuracy by about
3.33 percentage points. Results come from one train/test split
and one random seed.

## Assumptions

The last column is the class label; input features are numeric
or convertible to numeric. Ten neurons per hidden layer are used
consistently across all models. Missing labels are dropped;
missing numeric features are imputed using the training data.
