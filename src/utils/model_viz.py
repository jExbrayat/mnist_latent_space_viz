import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(
    loss: list,
    plot_from_n_epoch: int,
    save_path: str = None,
) -> None:
    """Plot loss curve for encoder decoder model

    Args:
        loss (list): training loss
        gathered from history callback

        plot_from_n_epoch (int): epoch from which to plot when there are too many
        save_path (str): path to save plot in .png format
    """

    num_epochs_to_display = len(loss) - plot_from_n_epoch
    step_x_ticks = (
        int(num_epochs_to_display / 10) if int(num_epochs_to_display / 10) >= 1 else 1
    )

    plt.plot(loss[plot_from_n_epoch:], label="training loss")

    plt.legend()
    plt.title("training curve")

    plt.xticks(
        np.arange(0, num_epochs_to_display, step=step_x_ticks),
        np.arange(
            plot_from_n_epoch + 1,
            plot_from_n_epoch + num_epochs_to_display + 1,
            step=step_x_ticks,
        ),
    )

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
