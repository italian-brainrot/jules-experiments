# Structured Weight Matrices Experiment

This experiment compares standard dense linear layers with several structured alternatives on the MNIST-1D dataset. The goal is to evaluate the parameter efficiency and performance of these structured layers when hyperparameters (specifically the learning rate) are fairly tuned.

## Hypotheses
1. Structured weight matrices (Kronecker product, Low-Rank + Diagonal) can achieve competitive performance with significantly fewer parameters than dense matrices.
2. Different structures will require different optimal learning rates.
3. Structured layers might generalize better or train differently compared to unstructured sparse layers.

## Implemented Layers
- **Dense**: Standard `nn.Linear`.
- **Kronecker**: $W = A \otimes B$, where $A$ and $B$ are smaller matrices.
- **Low-Rank + Diagonal**: $W = U V^T + \text{diag}(d)$, capturing both global low-rank structure and local element-wise shifts.
- **Sparse**: Dense weight matrix multiplied by a fixed random binary mask (10% density).

## Experimental Setup
- **Dataset**: MNIST-1D (40 input features, 10 classes).
- **Architecture**: 2-layer MLP (Hidden size 100).
- **Optimization**: Adam optimizer.
- **Hyperparameter Tuning**: 10 trials of Optuna per architecture to find the best learning rate.
- **Training**: 30 epochs with the best found learning rate.

## Results

| Model | Parameters | Best Learning Rate | Final Test Accuracy |
|-------|------------|--------------------|---------------------|
| Dense | 5110 | 0.018293 | 61.10% |
| Kronecker | 310 | 0.012270 | 43.90% |
| Low-Rank + Diag | 410 | 0.024766 | 38.80% |
| Sparse (10%) | 5110 (511 effective) | 0.013030 | 31.10% |

The results show that the Kronecker product is the most parameter-efficient structure in this setup.

## Visualizations
The convergence plots for all models can be found in `comparison_results.png`.

## Conclusion
- **Parameter Efficiency**: The Kronecker product model is remarkably efficient, achieving ~72% of the dense model's accuracy with only ~6% of the parameters.
- **Structured vs. Random Sparsity**: Structured matrices (Kronecker, Low-Rank + Diag) significantly outperformed the randomly sparse model, despite the sparse model having more effective parameters. This indicates that the specific inductive biases of these structures are beneficial for learning on this dataset.
- **Optimal Hyperparameters**: The best learning rates varied significantly between architectures (from ~0.012 to ~0.025), justifying the use of automated tuning for a fair comparison.
