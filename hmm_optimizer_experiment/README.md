# HMM Optimizer Experiment

This experiment implements and evaluates a novel optimization algorithm based on Hidden Markov Models (HMMs).

## Algorithm

The algorithm starts with a random search to initialize a population of points. It then iteratively fits an HMM to the points, sorted by their function values, and samples new points from the HMM to explore the search space.

## How to Run

To run the experiment, execute the following command from the root directory of the repository:

```bash
python -m hmm_optimizer_experiment.main
```

This will run the HMM-based optimizer on the Rosenbrock and Ackley test functions and generate visualizations of the optimization paths.
