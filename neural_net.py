
#####################################################################################################################
# Assignment 2: Neural Network Analysis
# Based on the professor's starter code. Changes are documented in README.md.
#####################################################################################################################

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# UCI Iris dataset, served as a public CSV from the seaborn-data repository.
DEFAULT_DATA_URL = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'


class NeuralNet:
    def __init__(self, dataFile, header=True):
        # The public CSV has a header row. --no-header supports UCI's original iris.data format.
        if header:
            self.raw_input = pd.read_csv(dataFile)
        else:
            self.raw_input = pd.read_csv(dataFile, header=None)

    # Preprocess data, including missing values and standardization.
    def preprocess(self):
        self.processed_data = self.raw_input.copy()
        self.processed_data = self.processed_data.replace(r'^\s*$', np.nan, regex=True)
        self.processed_data = self.processed_data.dropna(how='all').drop_duplicates()

        if self.processed_data.shape[1] < 2:
            raise ValueError('Dataset must contain at least one feature and a target column.')

        feature_columns = self.processed_data.columns[:-1]
        target_column = self.processed_data.columns[-1]
        self.processed_data = self.processed_data.dropna(subset=[target_column])

        for column in feature_columns:
            self.processed_data[column] = pd.to_numeric(
                self.processed_data[column], errors='coerce'
            )

        if len(self.processed_data) < 2 or self.processed_data[target_column].nunique() < 2:
            raise ValueError('Dataset must have at least two rows and two target classes.')

        return 0

    # Train and evaluate every required hyperparameter combination.
    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)

        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols - 1)]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Learn imputation and scaling parameters from training data only.
        medians = X_train.median().fillna(0)
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Professor-specified hyperparameters (unchanged).
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200]
        num_hidden_layers = [2, 3]

        results_dir = Path('results')
        plots_dir = Path('plots')

        results_dir.mkdir(exist_ok=True)
        plots_dir.mkdir(exist_ok=True)

        records = []
        histories = []
        classes = np.unique(y_train)

        combinations = itertools.product(
            activations,
            learning_rate,
            max_iterations,
            num_hidden_layers
        )

        # Train all 24 models.
        for model_number, (activation, rate, epochs, layers) in enumerate(
            combinations, start=1
        ):
            model_id = f'M{model_number:02d}'

            model = MLPClassifier(
                hidden_layer_sizes=(10,) * layers,
                activation=activation,
                solver='adam',
                learning_rate_init=rate,
                random_state=42,
                shuffle=True,
            )

            history = []

            # Train one epoch at a time without resetting Adam's state.
            for epoch in range(epochs):
                if epoch == 0:
                    model.partial_fit(X_train, y_train, classes=classes)
                else:
                    model.partial_fit(X_train, y_train)

                history.append(
                    accuracy_score(y_train, model.predict(X_train))
                )

            # Calculate final accuracy and error.
            train_accuracy = accuracy_score(
                y_train, model.predict(X_train)
            )
            test_accuracy = accuracy_score(
                y_test, model.predict(X_test)
            )

            train_loss = log_loss(
                y_train,
                model.predict_proba(X_train),
                labels=model.classes_
            )
            test_loss = log_loss(
                y_test,
                model.predict_proba(X_test),
                labels=model.classes_
            )

            records.append({
                'model': model_id,
                'activation': activation,
                'learning_rate': rate,
                'max_iterations': epochs,
                'hidden_layers': layers,
                'neurons_per_layer': 10,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_error_log_loss': train_loss,
                'test_error_log_loss': test_loss,
            })

            histories.append((
                model_id, activation, rate, epochs, layers, history
            ))

            print(
                f'{model_id} {activation}, lr={rate}, '
                f'epochs={epochs}, layers={layers} '
                f'| train acc={train_accuracy:.3f} '
                f'| test acc={test_accuracy:.3f} '
                f'| test log loss={test_loss:.3f}'
            )

        # Save the table of all 24 models.
        results = pd.DataFrame(records)
        results.to_csv(
            results_dir / 'model_results.csv',
            index=False
        )

        # ---------------------------------------------------------
        # GRAPHS 1-3: TRAINING ACCURACY VS. EPOCH
        # Eight models per graph, grouped by activation function.
        # ---------------------------------------------------------

        for group_index, activation in enumerate(activations, start=1):
            fig, ax = plt.subplots(figsize=(13, 7))

            for model_id, act, rate, epochs, layers, history in histories:
                if act != activation:
                    continue

                label = (
                    f'{model_id}: lr={rate}, '
                    f'{epochs} epochs, {layers} layers'
                )

                ax.plot(
                    range(1, len(history) + 1),
                    history,
                    label=label,
                    linewidth=1.6
                )

            ax.set(
                title=f'Training accuracy by epoch - {activation} activation',
                xlabel='Epoch',
                ylabel='Training accuracy',
                ylim=(0, 1.05)
            )

            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc='lower right')

            fig.tight_layout()
            fig.savefig(
                plots_dir / f'model_history_part{group_index}.png',
                dpi=170
            )
            plt.close(fig)

        # ---------------------------------------------------------
        # GRAPH 4: AVERAGE TEST ACCURACY BY ACTIVATION FUNCTION
        # Each activation averages eight model configurations.
        # ---------------------------------------------------------

        activation_means = (
            results.groupby('activation')['test_accuracy']
            .mean()
            .reindex(activations)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(
            activation_means.index,
            activation_means.values * 100
        )

        ax.bar_label(bars, fmt='%.1f%%', padding=3)

        ax.set(
            title='Average test accuracy by activation function',
            ylabel='Average test accuracy (%)',
            ylim=(0, 105)
        )

        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()

        fig.savefig(
            plots_dir / 'activation_comparison.png',
            dpi=170
        )
        plt.close(fig)

        # ---------------------------------------------------------
        # GRAPH 5: AVERAGE TEST ACCURACY BY LEARNING RATE
        # Each learning rate averages twelve configurations.
        # ---------------------------------------------------------

        rate_means = (
            results.groupby('learning_rate')['test_accuracy']
            .mean()
            .reindex(learning_rate)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(
            [str(rate) for rate in rate_means.index],
            rate_means.values * 100
        )

        ax.bar_label(bars, fmt='%.1f%%', padding=3)

        ax.set(
            title='Average test accuracy by learning rate',
            xlabel='Learning rate',
            ylabel='Average test accuracy (%)',
            ylim=(0, 105)
        )

        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()

        fig.savefig(
            plots_dir / 'learning_rate_comparison.png',
            dpi=170
        )
        plt.close(fig)

        # ---------------------------------------------------------
        # GRAPH 6: TRAINING VS. TEST ACCURACY
        # Two bars for each of the 24 models.
        # ---------------------------------------------------------

        positions = np.arange(len(results))
        bar_width = 0.4

        fig, ax = plt.subplots(figsize=(16, 6))

        ax.bar(
            positions - bar_width / 2,
            results['train_accuracy'] * 100,
            bar_width,
            label='Training'
        )

        ax.bar(
            positions + bar_width / 2,
            results['test_accuracy'] * 100,
            bar_width,
            label='Test'
        )

        ax.set_xticks(
            positions,
            results['model'],
            rotation=45
        )

        ax.set(
            title='Training vs. test accuracy across all 24 models',
            xlabel='Model',
            ylabel='Accuracy (%)',
            ylim=(0, 105)
        )

        ax.legend()
        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()

        fig.savefig(
            plots_dir / 'train_vs_test_accuracy.png',
            dpi=170
        )
        plt.close(fig)

        # ---------------------------------------------------------
        # GENERATE THE BRIEF REPORT
        # ---------------------------------------------------------

        best = results.sort_values(
            ['test_accuracy', 'test_error_log_loss'],
            ascending=[False, True]
        ).iloc[0]

        best_activation = activation_means.idxmax()
        best_rate = rate_means.idxmax()

        activation_summary = '\n'.join(
            f'- {activation}: '
            f'{activation_means[activation] * 100:.2f}% average test accuracy'
            for activation in activations
        )

        rate_summary = '\n'.join(
            f'- {rate}: '
            f'{rate_means[rate] * 100:.2f}% average test accuracy'
            for rate in learning_rate
        )

        report = f'''# Neural Network Analysis - Brief Report

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

- Model: {best['model']}
- Activation: {best['activation']}
- Learning rate: {best['learning_rate']}
- Epochs: {best['max_iterations']}
- Hidden layers: {best['hidden_layers']}
- Training accuracy: {best['train_accuracy'] * 100:.2f}%
- Test accuracy: {best['test_accuracy'] * 100:.2f}%
- Training log loss: {best['train_error_log_loss']:.4f}
- Test log loss: {best['test_error_log_loss']:.4f}

The best model is selected by highest test accuracy,
with lowest test log loss used to break ties.

## Activation-function comparison

{activation_summary}

{best_activation} had the highest average test accuracy across the
eight tested configurations per activation. This result describes
this dataset and these settings, not a universal ranking
of activation functions.

## Learning-rate comparison

{rate_summary}

Learning rate {best_rate} had the highest average test accuracy
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
'''

        (results_dir / 'brief_report.md').write_text(
            report,
            encoding='utf-8'
        )

        print('\nSaved results/model_results.csv and results/brief_report.md.')
        print('Saved six graphs in plots/.')

        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and compare 24 neural network configurations.'
    )

    parser.add_argument(
        '--data',
        default=DEFAULT_DATA_URL,
        help='Public CSV URL or local CSV path'
    )

    parser.add_argument(
        '--no-header',
        action='store_true',
        help='Use if the CSV does not have a header row'
    )

    args = parser.parse_args()

    neural_network = NeuralNet(
        args.data,
        header=not args.no_header
    )

    neural_network.preprocess()
    neural_network.train_evaluate()
