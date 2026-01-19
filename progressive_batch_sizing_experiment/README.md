# Progressive Batch Sizing Experiment

## Hypothesis

Progressively increasing the batch size during training can lead to both faster convergence and a better final model. The initial small batches will provide a regularizing effect, preventing the model from overfitting to the initial noisy gradients. As training progresses, larger batches will provide more accurate gradient information, allowing the model to fine-tune its parameters and converge to a sharper minimum.

## Methodology

To test this hypothesis, I compared a dynamic batch sizing strategy to two baselines: one with a small, fixed batch size and another with a large, fixed batch size. I used the `mnist1d` dataset for this experiment.

The three training strategies were:
- **Small, fixed batch size:** A baseline model with a batch size of 32.
- **Large, fixed batch size:** A baseline model with a batch size of 512.
- **Progressive batch sizing:** A model trained with a batch size that starts at 32 and progressively increases to 512.

For each of these strategies, I used `Optuna` to find the optimal learning rate. The models were then trained for 100 epochs using the best learning rate, and the validation loss was recorded at each epoch.

## Results

The results of the experiment are summarized in the following plot:

![Progressive Batch Sizing Comparison](progressive_batch_sizing_comparison.png)

As you can see from the plot, the progressive batch sizing strategy achieved the lowest validation loss of the three strategies. It also appeared to be the most resistant to overfitting, as its validation loss increased at a slower rate than the other two strategies after reaching its minimum.

## Conclusion

The results of this experiment suggest that progressively increasing the batch size during training can be an effective strategy for improving model performance. The progressive batch sizing strategy not only achieved a lower validation loss than the fixed-size batch strategies, but it also showed a greater resistance to overfitting. This supports the hypothesis that the initial small batches provide a regularizing effect, while the later, larger batches allow the model to converge to a better minimum.
