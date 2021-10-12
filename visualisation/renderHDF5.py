import h5py
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Render hdf5 file')
    parser.add_argument('filename',
                        help='filename')
    parser.add_argument('--variable',
                        help='variable to plot')

    args = parser.parse_args()
    fname = args.filename

    with h5py.File(fname, "r") as hf:
        data = hf[args.variable][:, :].T
        im = plt.imshow(data, origin='lower')
        plt.colorbar(im)
        plt.show()


main()
