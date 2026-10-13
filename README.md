
# CS 4375 - Assignment 2: Neural Network Analysis

## Dataset

The experiment uses the Iris dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/53/iris

The program loads a public CSV representation of Iris from:

https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv

This alternate host avoids the SSL certificate error encountered when accessing the original UCI download endpoint.

The dataset itself is not included in this repository.

## Requirements

Python 3.10+ recommended.

Install the required libraries using:

```bash
python -m pip install -r requirements.txt
```

## Running the Program

From the project directory, run:

```bash
python neural_net.py
```

To use a different CSV file:

```bash
python neural_net.py --data "https://example.org/data.csv"
```

If the CSV does not have a header row, add:

```bash
python neural_net.py --no-header
```

The last column must contain the class label. All other columns must contain numerical features.

No computer-specific file paths are hardcoded.

## Hyperparameters

The program evaluates all 24 combinations of:

- Activation functions: logistic, tanh, relu
- Learning rates: 0.01, 0.1
- Maximum iterations/epochs: 100, 200
- Hidden layers: 2, 3

Each hidden layer contains 10 neurons.

The model uses scikit-learn's MLPClassifier with the Adam optimizer.

Training is performed incrementally using partial_fit() so the optimizer maintains its state across epochs.

## Data Preprocessing

The program:

1. Removes empty rows and duplicate records.
2. Handles missing numerical values using training-set medians.
3. Converts feature columns to numerical values.
4. Splits the dataset into 80% training and 20% testing.
5. Standardizes features using StandardScaler.
6. Fits the imputer and scaler using training data only.

The train/test split uses random_state=42 and stratification.

## Generated Results

Running the program creates the following files.

### Results

- results/model_results.csv

  Contains all 24 models and their hyperparameters, training accuracy, test accuracy, training log loss, and test log loss.

- results/brief_report.md

  Contains the experiment summary, best-performing model, activation-function comparisons, learning-rate comparisons, and limitations.

### Graphs

Graph 1:
plots/model_history_part1.png

Training accuracy versus epoch for the eight logistic models (M01-M08).

Graph 2:
plots/model_history_part2.png

Training accuracy versus epoch for the eight tanh models (M09-M16).

Graph 3:
plots/model_history_part3.png

Training accuracy versus epoch for the eight ReLU models (M17-M24).

Graph 4:
plots/activation_comparison.png

Average test accuracy for logistic, tanh, and ReLU.

Each activation is averaged across eight model configurations.

Graph 5:
plots/learning_rate_comparison.png

Average test accuracy for learning rates 0.01 and 0.1.

Each learning rate is averaged across twelve model configurations.

Graph 6:
plots/train_vs_test_accuracy.png

Final training and test accuracy for each of the 24 models.

## Changes from the Professor's Starter Code

The original NeuralNet class and its __init__, preprocess,
and train_evaluate methods have been retained.

The following functionality was added:

- Public CSV dataset loading
- Data preprocessing
- Missing-value handling
- Feature standardization
- Neural network training
- Evaluation of all 24 hyperparameter combinations
- Training accuracy tracking by epoch
- Training and test accuracy calculations
- Training and test log-loss calculations
- Three training-history plots
- Three additional comparison graphs
- CSV results table
- Automatically generated brief report

## Assumptions

- The dataset is a classification dataset.
- The last column contains the target class.
- The remaining columns contain numerical features.
- Each hidden layer contains 10 neurons.
- All models use the same training/testing split.
- A fixed random seed is used for reproducibility.

## Student Information

Add your name and NetID here and to the cover page of your complete submission.
