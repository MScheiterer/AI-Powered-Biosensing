import matplotlib.pyplot as plt

def create_plots(n):
    cols = int(n**0.5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]  # Ensure iterable

    for i in range(n):
        analysis(axes[i], i)

    # Hide unused subplots, if any
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def analysis(ax, index):
    """
    Fill a given matplotlib Axes object with data.
    
    Parameters:
    - ax (matplotlib.axes.Axes): The axes to draw on.
    - index (int): Index of the subplot (for differentiation).
    """
    # Example analysis: plotting a sine wave with different phase shifts
    import numpy as np
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x + index)
    ax.plot(x, y)
    ax.set_title(f'Plot {index + 1}')