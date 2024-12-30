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
def anim(arr, save_path, fps=10):
    nSeconds = len(arr)//fps
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    # fig = plt.figure( figsize=(8,8) )
    a = arr[0]
    im = plt.imshow(a, vmin=0, vmax=1)
    plt.set_cmap('Grays')
    # plt.clim(-1,1)
    plt.clim(0,1)
    plt.axis('off')

    def animate_func(i):
        if i % fps == 0: print( '.', end ='' )
        im.set_array(arr[i])
        plt.title(f't={i} ({np.min(arr[i]):.2f}, {np.max(arr[i]):.2f})')
        return [im]

    anim = animation.FuncAnimation(
                fig, 
                animate_func, 
                frames = (nSeconds * fps),
                interval = (1000 / fps), # in ms
            )
    anim.save(save_path, fps=fps)

