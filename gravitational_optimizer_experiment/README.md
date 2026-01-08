# Gravitational Optimizer Experiment

This experiment introduces and evaluates the `GravitationalOptimizer`, a novel optimization algorithm inspired by the principles of General Relativity.

## Hypothesis

The core hypothesis is that treating the loss landscape as a dynamic spacetime manifold—where the model's parameters are a particle and the gradients represent the manifold's curvature—could lead to more effective convergence. By calculating a "deflection" to the parameter's trajectory based on this curvature, the optimizer might navigate the loss landscape more efficiently than traditional methods.

## Methodology

### Gravitational Optimizer

The `GravitationalOptimizer` is implemented in `optimizer.py` and inherits from `torch.optim.Optimizer`. It maintains a "velocity" for each parameter, which is analogous to momentum. At each step, it calculates a "deflection" term proportional to the gradient. This deflection is then used to update the velocity, and the new velocity is used to update the parameters.

The update rule is as follows:
1. `deflection = G * gradient`
2. `velocity = beta * velocity + deflection`
3. `parameter = parameter - lr * velocity`

Where `G` is a "gravitational constant" and `beta` is a momentum-like coefficient.

### Comparison

To evaluate the `GravitationalOptimizer`, a comparison was made against the standard Adam optimizer. The `mnist1d` dataset was used for training and validation. A standard Multi-Layer Perceptron (MLP) was used as the model architecture.

To ensure a fair comparison, `optuna` was used to tune the learning rate for both optimizers. For the `GravitationalOptimizer`, `optuna` also tuned the `beta` and `G` hyperparameters. Each optimizer was tuned for 30 trials, and then the best hyperparameters were used to train the model for 20 epochs.

## Results

After running the experiment, the following final validation losses were obtained:

- **Adam:** 1.1746
- **GravitationalOptimizer:** 1.2749

The best hyperparameters found for the `GravitationalOptimizer` were:
- `lr`: 0.0314
- `beta`: 0.9294
- `G`: 3.8914

## Conclusion

The results show that the standard Adam optimizer outperformed the `GravitationalOptimizer` on this task. The hypothesis that the gravitational-inspired update rule would lead to better performance is not supported by this experiment.

While the `GravitationalOptimizer` is an interesting concept, it did not prove to be as effective as Adam in this particular experiment. Further research could explore different formulations of the "deflection" term or other ways of incorporating geometric information into the optimization process.
