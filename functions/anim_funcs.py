import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# TODO make function for array of images, series of images

def showim(im):
    im = plt.imshow(a, vmin=0, vmax=1)
    plt.set_cmap('Grays')
    plt.clim(-1,1)
    plt.axis('off')
    plt.show()

# animates an array of images 
def anim_ims(arr, save_path, fps=10, show=False):
    nSeconds = len(arr)//fps
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    a = arr[0]
    im = plt.imshow(a, vmin=0, vmax=1)
    plt.set_cmap('Greys')
    # plt.clim(0,1)
    # plt.clim(-1,1)
    plt.axis('off')

    def animate_func(i):
        if i % fps == 0: print( '.', end ='' )
        im.set_array(arr[i])
        # plt.title(f't={i} ({np.min(arr[i]):.2f}, {np.max(arr[i]):.2f})')
        plt.title(f't = {i}')
        # plt.title(f'     Autoencoder                    Variational Autoencoder')
        return [im]

    anim = animation.FuncAnimation(
                fig, 
                animate_func, 
                frames = (nSeconds * fps),
                interval = (1000 / fps), # in ms
            )
    if show: plt.show()
    if not show: anim.save(save_path, fps=fps)

# animates plotted (x,y) data (2D)
def anim_plot(arr, save_path, fps=10, show=False, train=None):
    nSeconds = len(arr)//fps
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    a = arr[0]
    im = plt.scatter(a[0], a[1], alpha=0.5)
    if train is not None: im=plt.scatter(train[0], train[1], alpha=0.5)
    ax = plt.gca()
    axis_min, axis_max = -2.5, 2.5
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])
    ax.set_aspect('equal', adjustable='box')

    def animate_func(i):
        if i % fps == 0: print( '.', end ='' )
        # im.set_data
        plt.clf()
        ax = plt.gca()
        plt.axis('equal')
        ax.set_xlim([axis_min, axis_max])
        ax.set_ylim([axis_min, axis_max])
        ax.set_aspect('equal', adjustable='box')
        im = plt.scatter(arr[i][0], arr[i][1], alpha=0.5)
        if train is not None: im=plt.scatter(train[0], train[1], alpha=0.5)
        plt.title(f't={i+1} ({np.min(arr[i]):.2f}, {np.max(arr[i]):.2f})')
        return [im]

    anim = animation.FuncAnimation(
                fig, 
                animate_func, 
                frames = (nSeconds * fps),
                interval = (1000 / fps), # in ms
            )
    print('Animation made.')
    if show: plt.show()
    if not show: anim.save(save_path, fps=fps)
