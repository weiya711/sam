import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import argparse
import os
import dataclasses
import numpy

from pathlib import Path
from util import FormatWriter

from util import TensorCollectionSuiteSparse, ScipyTensorShifter, \
    ScipyMatrixMarketTensorLoader, SuiteSparseTensor, safeCastScipyTensorToInts

SS_PATH = os.getenv('SUITESPARSE_PATH')

ssMtx = dataclasses.make_dataclass("SS", [("name", str), ("nnz", int),
                                          ("dim1", int), ("dim2", int),
                                          ("sparsity", float), ("seg_size", int),
                                          ("crd_size", int)])

fiber_data = dataclasses.make_dataclass("FIBER", [("name", str), ("dcsr0", [int]),
                                          ("dcsr1", [int]), ("dcsc0", [int]),
                                          ("dcsc1", [int]), ])

# UfuncInputCache attempts to avoid reading the same tensor from disk multiple
# times in a benchmark run.
class UfuncInputCache:
    def __init__(self):
        self.lastLoaded = None
        self.lastName = None
        self.tensor = None
        self.other = None

        self.shifter = ScipyTensorShifter()

    def load(self, tensor, suitesparse, cast):
        if self.lastName == str(tensor):
            return self.tensor, self.other
        else:
            if suitesparse:
                self.lastLoaded = tensor.load(ScipyMatrixMarketTensorLoader())
            else:
                self.lastLoaded = tensor.load()
            self.lastName = str(tensor)
            if cast:
                self.tensor = safeCastScipyTensorToInts(self.lastLoaded)
            else:
                self.tensor = self.lastLoaded

            self.other = self.shifter.shiftLastMode(self.lastLoaded)

            return self.tensor, self.other


inputCache = UfuncInputCache()


def stats_tile(filename, args):
    tensor = SuiteSparseTensor(os.path.join(SS_PATH, filename))
    ss_tensor = inputCache.load(tensor, True, False)

    if not isinstance(ss_tensor, numpy.ndarray):
        tensor_list = []
        nx = int(ss_tensor.shape[0] / args.tilewidth) + 1
        ny = int(ss_tensor.shape[1] / args.tileheight) + 1
        for i in range(nx):
            for j in range(ny):
                startx = i * args.tilewidth
                stopx = min((i + 1) * args.tilewidth, ss_tensor.shape[0])
                starty = j * args.tileheight
                stopy = min((j + 1) * args.tileheight, ss_tensor.shape[1])

                if args.debug:
                    print(i, startx, stopx)
                    print(j, starty, stopy)

                sub_tensor = ss_tensor[startx:stopx, starty:stopy]

                name = str(tensor) + "_t" + str(i * ny + j) + "_" + str(args.tilewidth) + "x" + str(args.tileheight)

                if args.debug:
                    print(name)
                    print("NNZ:", sub_tensor.nnz, "\tShape:", sub_tensor.shape)
                    print(sub_tensor.indices)
                    print(sub_tensor.indptr)

                tensor_list.append(ssMtx(name, sub_tensor.nnz,
                                         sub_tensor.shape[0], sub_tensor.shape[1],
                                         float(sub_tensor.nnz) / float(sub_tensor.shape[0] * sub_tensor.shape[1]),
                                         len(sub_tensor.indptr), len(sub_tensor.indices)))

        outname = str(tensor) + "_" + str(args.tilewidth) + "x" + str(args.tileheight)
        dfss = pd.DataFrame(tensor_list)

        hist = dfss.hist(bins=20)
        plt.savefig('./logs/' + outname + '_figure.pdf')

        tile_out_filepath = Path('./logs/' + outname + '.csv')
        tile_out_filepath.parent.mkdir(parents=True, exist_ok=True)
        dfss.to_csv(tile_out_filepath)

def block_mat_tapeout(tensor, glb_size=131072, memtile_size=1024):
    pass

def fiber_stats(ss_tensor, tensor_coo, other_coo, args):
    print("Getting fiber stats...")
    formatWriter = FormatWriter(True)
    print(tensor_coo)
    dcsr = formatWriter.convert_format(tensor_coo, "dcsr")
    dcsc = formatWriter.convert_format(tensor_coo, "dcsc")

    fiber_data_list = []
    fiber_data_list.append([dcsr.seg0[x + 1] - dcsr.seg0[x] for x in range(len(dcsr.seg0)-1)])
    fiber_data_list.append([dcsr.seg1[x + 1] - dcsr.seg1[x] for x in range(len(dcsr.seg1)-1)])
    fiber_data_list.append([dcsc.seg0[x + 1] - dcsc.seg0[x] for x in range(len(dcsc.seg0)-1)])
    fiber_data_list.append([dcsc.seg1[x + 1] - dcsc.seg1[x] for x in range(len(dcsc.seg1)-1)])
    fiber_data(ss_tensor, fiber_data_list[0], fiber_data_list[1], fiber_data_list[2], fiber_data_list[3])

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    counter = 0
    for i in range(2):
        for j in range(2):

            ax = axes[i][j]

            # Plot when we have data
            if counter < len(fiber_data_list):

                ax.hist(fiber_data[counter], color='blue', alpha=0.5,
                        label='{}'.format(counter))

            # Remove axis when we no longer have data
            else:
                ax.set_axis_off()

            counter += 1

    plt.show()

def stats_overall(args):
    print("Processing Overall Stats")

    filenames = sorted([f for f in os.listdir(SS_PATH) if f.endswith(".mtx")])
    if args.num is not None and args.numstop is not None:
        filenames = filenames[args.num:args.numstop]
    elif args.num is not None:
        filenames = filenames[args.num:]
    elif args.numstop is not None:
        filenames = filenames[:args.numstop]

    tensor_list = []
    pbar = tqdm.tqdm(filenames)
    for filename in pbar:
        pbar.set_description("Processing %s" % filename)

        tensor = SuiteSparseTensor(os.path.join(SS_PATH, filename))
        ss_tensor = inputCache.load(tensor, True, False)

        tensor_list.append(ssMtx(str(tensor), ss_tensor.nnz,
                                 ss_tensor.shape[0], ss_tensor.shape[1],
                                 float(ss_tensor.nnz) / float(ss_tensor.shape[0] * ss_tensor.shape[1]),
                                 len(ss_tensor.indptr), len(ss_tensor.indices)))

    dfss = pd.DataFrame(tensor_list)

    hist = dfss.hist(bins=70)
    plt.savefig('./logs/overall_figure.pdf')

    outpath = Path('./logs/overall_out.csv')
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.is_file():
        dfss.to_csv(outpath, mode='a', index=False, header=False)
    else:
        dfss.to_csv(outpath, index=False)


def main():
    parser = argparse.ArgumentParser(description="Process some suitesparse matrix statistics")
    parser.add_argument('-n', '--name', metavar='ssname', type=str, action='store',
                        help='tensor name to run tile analysis on one SS tensor')
    parser.add_argument('--overall', action='store_true',
                        help='Get overall statistics on all matrices (overall). By default false')
    parser.add_argument('--tile', action='store_true',
                        help='Get tile statistics on one or all matrices. By default false')
    parser.add_argument('--fiber', action='store_true',
                        help='Get fiber size statistics on one matrix. By default false')
    parser.add_argument('-nstart', '--num', metavar='N', type=int, action='store',
                        help='Start number of SS matrices to process (--overall=true)')
    parser.add_argument('-nstop', '--numstop', metavar='N', type=int, action='store',
                        help='Stop number of SS matrices to process (--overall=true)')
    parser.add_argument('-tw', '--tilewidth', metavar='N', type=int, action='store', default=64,
                        help='Size of width (1st) tile dimension')
    parser.add_argument('-th', '--tileheight', metavar='N', type=int, action='store', default=64,
                        help='Size of height (2nd) tile dimension')

    parser.add_argument('-g', '--debug', action='store_true', help='Print debug statements')

    args = parser.parse_args()

    log_path = Path('./logs/')
    log_path.mkdir(parents=True, exist_ok=True)
    if args.overall:
        stats_overall(args)
    else:
        print("Processing Tile Stats")
        if args.name is None:
            filenames = sorted([f for f in os.listdir(SS_PATH) if f.endswith(".mtx")])
            if args.num is not None:
                filenames = filenames[:args.num]

            pbar = tqdm.tqdm(filenames)
            for filename in pbar:
                pbar.set_description("Processing %s" % filename)
                stats_tile(filename, args)
        else:
            filename = args.name + ".mtx"
            print("Processing %s" % filename)

            ss_tensor = SuiteSparseTensor(os.path.join(SS_PATH, filename))
            tensor_coo, other_coo = inputCache.load(ss_tensor, True, False)

            if not isinstance(ss_tensor, numpy.ndarray):
                if args.tile:
                    stats_tile(filename, args)
                elif args.fiber:
                    fiber_stats(ss_tensor, tensor_coo, other_coo, args)
            else:
                print("Error: suitesparse file is not a matrix")



if __name__ == '__main__':
    main()
