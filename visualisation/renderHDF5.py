import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Render hdf5 file')
    parser.add_argument('filename',
                        help='filename')
    parser.add_argument('--imshow', default='',
                        help='variable to plot via imshow')
    parser.add_argument('--streamplot', nargs=2, default=[],
                        help='components of vector to plot via streamlines')

    args = parser.parse_args()
    fname = args.filename

    with h5py.File(fname, "r") as hf:
        if args.imshow:
            data = hf[args.imshow][1:-1, 1:-1].T
            im = plt.imshow(data, origin='lower', extent=(0, 1, 0, 1))
            plt.colorbar(im)
        if args.streamplot:
            v1 = hf[args.streamplot[0]][1:-1, 1:-1].T
            v2 = hf[args.streamplot[1]][1:-1, 1:-1].T
            x = np.linspace(0, 1, v1.shape[0])
            y = np.linspace(0, 1, v1.shape[1])
            X, Y = np.meshgrid(x, y)
            plt.streamplot(X, Y, v1, v2)

    plt.show()


main()
