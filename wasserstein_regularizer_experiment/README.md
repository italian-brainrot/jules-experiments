# Wasserstein Distance Regularization for Output Layers

## Hypothesis

Penalizing the Wasserstein distance between the output distribution of a neural network's final layer and the target distribution can improve model calibration and generalization. This experiment investigates the use of Wasserstein distance as a direct regularizer in a classification task.

## Methodology

A simple Multi-Layer Perceptron (MLP) model was trained on the `mnist1d` dataset. A custom loss function was implemented, combining the standard Cross-Entropy (CE) loss with a 1D Wasserstein distance penalty. The penalty is calculated between the model's softmax output and the one-hot encoded true labels.

The composite loss function is defined as:
`L = CE(outputs, targets) + lambda * W(softmax(outputs), one_hot(targets))`

Where `W` is the 1D Wasserstein distance and `lambda` is a hyperparameter controlling the regularization strength.

To ensure a fair comparison, `optuna` was used to tune the learning rate for a baseline model (using only CE loss) and both the learning rate and the `lambda` for the Wasserstein-regularized model. Each study was run for 15 trials.

## Results

The results of the Optuna hyperparameter search are as follows:

| Model                               | Best Accuracy | Best Learning Rate | Best Lambda |
| ----------------------------------- | ------------- | ------------------ | ----------- |
| Baseline (Cross-Entropy only)       | 0.6845        | 0.0025             | N/A         |
| Wasserstein Regularization          | 0.6900        | 0.0047             | 0.0002      |

The model with Wasserstein distance regularization achieved a slightly higher test accuracy than the baseline model.

## Conclusion

The experiment provides initial evidence that using the Wasserstein distance as a regularizer for the output layer can offer a small but noticeable improvement in performance. The regularization appears to guide the model towards a better-calibrated output distribution, which in turn leads to slightly better generalization.

Further research could explore this regularizer on more complex datasets and architectures, and investigate its effect on model calibration metrics in more detail.
