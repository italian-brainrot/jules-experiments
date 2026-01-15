
import optuna
import plotly.graph_objects as go
from warm_optimizer_start_experiment.main import train_model, pretrain_and_get_optimizer_state

def objective_cold(trial):
    """Optuna objective for the cold start (control)."""
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    epochs = 30 # A fixed number of epochs for fair comparison

    history = train_model(lr=lr, epochs=epochs, optimizer_state=None)

    # Return the final validation loss as the metric to minimize
    final_loss = history[-1]

    # Store the full history for plotting later
    trial.set_user_attr("history", history)

    return final_loss

def objective_warm(trial):
    """Optuna objective for the warm start (experiment)."""
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    epochs = 30

    # The warm state is generated once and reused for all trials in the study
    optimizer_state = trial.study.user_attrs.get("optimizer_state")
    if optimizer_state is None:
        raise RuntimeError("Optimizer state not found in study's user attributes.")

    history = train_model(lr=lr, epochs=epochs, optimizer_state=optimizer_state)

    final_loss = history[-1]
    trial.set_user_attr("history", history)

    return final_loss

def run_comparison(n_trials=50):
    """
    Runs the full Optuna comparison and generates a plot.
    """
    print("--- Starting Optuna Comparison ---")

    # 1. Run the cold start study
    print("\n--- Tuning Cold Start Optimizer ---")
    study_cold = optuna.create_study(direction='minimize')
    study_cold.optimize(objective_cold, n_trials=n_trials)
    best_trial_cold = study_cold.best_trial
    print(f"Best cold start LR: {best_trial_cold.params['lr']:.6f}, Best Loss: {best_trial_cold.value:.4f}")

    # 2. Pre-train to get the optimizer state for the warm start study
    print("\n--- Generating Optimizer State for Warm Start Study ---")
    # Use the best learning rate from the cold start for pre-training
    pretrain_lr = best_trial_cold.params['lr']
    optimizer_state = pretrain_and_get_optimizer_state(lr=pretrain_lr, epochs=10)

    # 3. Run the warm start study
    print("\n--- Tuning Warm Start Optimizer ---")
    study_warm = optuna.create_study(direction='minimize')
    # Store the generated optimizer state so the objective can access it
    study_warm.set_user_attr("optimizer_state", optimizer_state)
    study_warm.optimize(objective_warm, n_trials=n_trials)
    best_trial_warm = study_warm.best_trial
    print(f"Best warm start LR: {best_trial_warm.params['lr']:.6f}, Best Loss: {best_trial_warm.value:.4f}")

    # 4. Generate and save the comparison plot
    history_cold = best_trial_cold.user_attrs["history"]
    history_warm = best_trial_warm.user_attrs["history"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history_cold, mode='lines', name=f'Best Cold Start (LR={best_trial_cold.params["lr"]:.4f})'))
    fig.add_trace(go.Scatter(y=history_warm, mode='lines', name=f'Best Warm Start (LR={best_trial_warm.params["lr"]:.4f})'))

    fig.update_layout(
        title='Optimizer Warm Start vs. Cold Start Comparison',
        xaxis_title='Epoch',
        yaxis_title='Validation Loss',
        legend_title='Optimizer Initialization'
    )

    plot_path = "warm_optimizer_start_experiment/comparison_plot.html"
    fig.write_html(plot_path)
    print(f"\nComparison plot saved to {plot_path}")

    # Also save as an image
    try:
        img_path = "warm_optimizer_start_experiment/comparison_plot.png"
        fig.write_image(img_path, format="png")
        print(f"Comparison image saved to {img_path}")
    except Exception as e:
        print(f"Could not save image, maybe kaleido is not installed? Error: {e}")


if __name__ == '__main__':
    run_comparison(n_trials=30) # Using a smaller number for a quicker run
