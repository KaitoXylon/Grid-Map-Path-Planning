import matplotlib.pyplot as plt

def plot_grid(grid, start, goal, path, explored, title="Path"):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray_r')

    for cell in explored:
        ax.plot(cell[1], cell[0], 'yo', markersize=2)
    
    for cell in path:
        ax.plot(cell[1], cell[0], 'ro', markersize=3)

    ax.plot(start[1], start[0], 'gs')  # green start
    ax.plot(goal[1], goal[0], 'bs')   # blue goal
    ax.set_title(title)
    plt.show()
