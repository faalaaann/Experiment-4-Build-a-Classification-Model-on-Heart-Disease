# Heart Disease Classification Model

A machine learning experiment that builds and compares three neural-network classifiers using a heart-disease dataset. The notebook demonstrates data cleaning, feature scaling, model training, evaluation, and simple ensemble averaging with TensorFlow/Keras.

> **Dataset note:** The included `heart.csv` contains the columns `age`, `sex`, `cp`, `trestbps`, `chol`, and `fbs`. Because no `target` column is present, the notebook renames `fbs` to `target` and converts it to a binary label. This means the current experiment uses the provided `fbs` values as the prediction target; it should not be interpreted as a clinically validated heart-disease diagnostic model.

## Project contents

- `EX4 classification model.ipynb` — complete data preparation, training, and evaluation workflow
- `heart.csv` — input dataset

## Workflow

1. Load and clean the dataset with pandas.
2. Convert the target to a binary value (`0` or `1`).
3. Split the data into 80% training and 20% testing sets.
4. Standardize the input features with `StandardScaler`.
5. Train three feed-forward neural networks with different hidden-layer sizes:
   - 16 and 8 units
   - 32 and 16 units
   - 8 and 8 units
6. Compare individual model accuracy with an ensemble that averages their predicted probabilities.
7. Evaluate the ensemble using accuracy, a confusion matrix, and a classification report.

## Results

The notebook currently reports the following test-set accuracy values:

| Model | Accuracy |
| --- | ---: |
| Model 1 | 74.59% |
| Model 2 | 78.69% |
| Model 3 | 73.77% |
| Ensemble (average) | 74.59% |

The best individual result is produced by **Model 2**. Results can vary if the dataset, random seeds, TensorFlow version, or training settings change.

## Requirements

- Python 3.9+
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow

Install the dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

## Run the experiment

1. Clone the repository:

   ```bash
   git clone https://github.com/faalaaann/Experiment-4-Build-a-Classification-Model-on-Heart-Disease.git
   cd Experiment-4-Build-a-Classification-Model-on-Heart-Disease
   ```

2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `EX4 classification model.ipynb` and run the cells from top to bottom.

The notebook expects `heart.csv` to be in the same directory.

## Limitations

This is an educational classification experiment, not a medical decision-support system. The dataset is small and imbalanced, and the current label mapping should be reviewed before using the workflow for a genuine heart-disease prediction task. For a clinically meaningful experiment, use a verified dataset with a documented heart-disease outcome column and evaluate additional metrics such as recall, precision, F1-score, and ROC-AUC.
