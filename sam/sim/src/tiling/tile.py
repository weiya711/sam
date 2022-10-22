import numpy as np
import scipy.sparse 
import os
import argparse
import ast
import yaml
import copy 

from itertools import compress
from pathlib import Path
from sam.util import SuiteSparseTensor, InputCacheSuiteSparse 
from sam.sim.src.tiling.process_expr import parse_all, update_dict

SAM_STRS = {"matmul_ikj":"X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss:1,0 -f=C:ss -s=reorder(k,i,j)"}

def print_dict(dd):
    for k,v in dd.items():
        print(k, ":", v)

def get_ivars(names, expr):
    [lhs, rhs] = expr.split("=")

    tree = ast.parse(rhs)
    analyzer = IvarAnalyzer(names, tree)
    return analyzer.get_ivars()

class IvarAnalyzer(ast.NodeVisitor):
    def __init__(self, tensor_names, tree):
        self.tree = tree
        self.tensor_names = tensor_names
        self.name = None
        self.call = False
        self.ivars = {} 

    def visit_Call(self, node):
        self.call = True
        if self.name == node.func.id:
            freevars = []
            for arg in node.args:
                ivar = self.visit(arg)
                if ivar is not None:
                    freevars.append(ivar) 
            self.ivars[self.name] = freevars
        self.call = False


    def visit_Name(self, node):
        if self.call:
            return node.id
        else:
            return None

    def get_ivars(self):
        for name in self.tensor_names:
            self.name = name
            self.visit(self.tree)
        return self.ivars

def parse_sam_input(string):
    sam_str = SAM_STRS[string]

    str_arr = sam_str.split(" ")
    dictionary = parse_all(str_arr, has_quotes=False)    

    # Assume there are no repeat tensors...
    tensors = dictionary["rhs_tensors"]
    permutations = [list(map(int, dictionary[tensor]["perm"])) for tensor in tensors]
    ivars = get_ivars(tensors, str_arr[0]) 
    ivars = [ivars[tensor] for tensor in tensors]
    return tensors, permutations, ivars 
        

# Outputs COO tiles, given a COO tensor
# ASSUME: tensor is a scipy.sparse.coo_matrix
# TODO: new_ivar_order right now is assumed to be one fixed order
#       In the future, will have to take into acocunt all reorderings 
def tile_coo(tensor, ivar_map, split_map, new_ivar_order=None):
    human_readable = False

    tiles = dict()
    order = len(tensor.shape) 

    tensor_points = tensor.todok()
     
    new_shape = []
    for lvl in range(order):
        ivar = ivar_map[lvl] 
        sf = split_map[ivar]
        new_shape.append(sf)

    for crds,val in tensor_points.items():
        point = list(crds)

        new_point = []
        tile_id = []
        for lvl in range(order):
            ivar = ivar_map[lvl] 
            sf = split_map[ivar]


            new_point.append(point[lvl] % sf)

            tile_id.append(int(point[lvl] / sf))

        
        # Add in value to the new_point as well
        new_point.append(val) 
        tile_id = tuple(tile_id) 

        if tile_id in tiles:
            tiles[tile_id].append(new_point)
        else:
            tiles[tile_id] = [new_point]

    # sort the new coo lists
    for key,val in tiles.items():
        if human_readable:
            dok = sorted(val)
        else:
            dok = scipy.sparse.dok_matrix(tuple(new_shape))
            for point in val:
                dok[tuple(point[0:-1])] = point[-1]

        tiles[key] = dok
        
    return tiles

# tensor_names: list of tensor names [B,C,D] (from SAM)
# tensors: list of scipy.sparse.coo_matrix following tensor_names (from SAM)
# permutation_strs: list of permutation_strs [ss01, ss10] following tensor_names (from SAM)
# ivar_strs: list of ivar_strs ["ik", "kj"] following tensor_names (from SAM)
# split_map: dictionary of split factors (from hardware)
def cotile_coo(tensor_names, tensors, permutation_strs, ivar_strs, split_map):
    tiled_tensors = dict()
    
    for i,tensor in enumerate(tensors): 
        tensor_name = tensor_names[i]
        tensor_format = permutation_strs[i]
        ivar_map = dict()
        order = len(tensor.shape)
        for dim in range(order):
            lvl_permutation = tensor_format[dim:dim + 1][0]
            ivar = ivar_strs[i][dim]  
            ivar_map[lvl_permutation] = ivar
        
        tiled_tensors[tensor_name] = tile_coo(tensor, ivar_map, split_map)

    return tiled_tensors


def cotile_multilevel_coo(app_str, hw_config_fname, tensors):
    tensor2 = scipy.sparse.random(tensor.shape[0], tensor.shape[1])
    tensors.append(tensor2)

    names, format_permutations, ivars = parse_sam_input(args.cotile)
    with open(hw_config_fname, "r") as stream:
        try:
            hw_config = yaml.safe_load(stream)
            
            n_levels = hw_config["n_levels"]
            level_names = hw_config["level_names"]
            assert len(level_names) == n_levels

            cotiled = None
            split_map = {}
            for hw_lvl in range(n_levels-1):
                level_names_left = level_names[hw_lvl+1:]

                sf = 1
                for level_name in level_names_left:
                    hw_key = level_name + "_tile_size"
                    tile_size = hw_config[hw_key]
                    sf *= tile_size

                unique_ivars = list(set(sum(ivars, [])))
                for ivar in unique_ivars:
                    split_map[ivar] = sf 

                if cotiled is None:
                    # First iteration of tiling
                    cotiled = cotile_coo(names, tensors, format_permutations, ivars, split_map)
                else:
                    # recursively tile the blocks
                    new_cotiled = {}
                    for i,name in enumerate(names):

                        new_cotiled[name] = {}
                        for tile_id, tile in cotiled[name].items():
                            new_cotiled_temp = cotile_coo(name, [tile.tocoo()], [format_permutations[i]], [ivars[i]], split_map)

                            for kk, vv in copy.deepcopy(new_cotiled_temp)[name].items():
                                new_tile_id = tuple(list(tile_id) + list(kk)) 
                                new_cotiled[name][new_tile_id] = vv
                    cotiled = copy.deepcopy(new_cotiled)

            return cotiled
        except yaml.YAMLError as exc:
            print(exc)
    
 



inputCache = InputCacheSuiteSparse()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tile matrices')
    parser.add_argument("--input_tensor", type=str, default=None)
    parser.add_argument("--gen_tensor", action="store_true")
    parser.add_argument("--cotile", type=str, default=None)
    parser.add_argument("--output_dir_path", type=str, default="./tiles")
    parser.add_argument("--hw_config", type=str, default=None)
    parser.add_argument("--multilevel", action="store_true")

    args = parser.parse_args()

    tensor = None
    cwd = os.getcwd()
    if args.gen_tensor:
        tensor = scipy.sparse.random(16, 16)
    else:
        assert args.input_tensor is not None
        SS_PATH = os.getenv('SUITESPARSE_PATH', default=os.path.join(cwd, 'suitesparse')) 
        print("PATH:", SS_PATH)
        tensor_path = os.path.join(SS_PATH, args.input_tensor + ".mtx")
        ss_tensor = SuiteSparseTensor(tensor_path)
        tensor = inputCache.load(ss_tensor, False)

    split_map = {"i":16, "j":16, "k":16} 


    if args.cotile is None:
        print("ORIG:", tensor)
        print("SPLIT MAP", split_map)
        tiles = tile_coo(tensor, {0:"i", 1:"j"}, split_map)

        print("TILES:")
        print_dict(tiles)
    else:

        if args.multilevel:
            assert args.cotile is not None
            cotiled_tensors = cotile_multilevel_coo(args.cotile, args.hw_config, [tensor]) 
        elif args.cotile is not None:
            tensor2 = scipy.sparse.random(tensor.shape[0], tensor.shape[1])
            names, format_permutations, ivars = parse_sam_input(args.cotile)

            cotiled_tensors = cotile_coo(names, [tensor, tensor2], format_permutations, ivars, split_map)
            # print(cotiled_tensors)


        names = cotiled_tensors.keys()
        output_mtx_name = os.path.join(args.output_dir_path, args.cotile, "mtx")
        output_mtx_path = Path(output_mtx_name)
        output_mtx_path.mkdir(parents=True, exist_ok=True)
        print(os.path.exists(output_mtx_path))
        for name in names:
            for tile_id, tile in cotiled_tensors[name].items():

                [str(item) for item in tile_id]
                filename = "tensor_" + name + "_tile_" + "_".join([str(item) for item in tile_id]) + ".mtx"
                mtx_path_name = os.path.join(output_mtx_name, filename)
                print(tile)
                print(mtx_path_name, cwd)
    #                with open(mtx_path_name, "x") as ff:
                scipy.io.mmwrite(mtx_path_name, tile)
                print(os.path.exists(mtx_path_name))
    #        for tile_id, tile_coo in cotiled_tensors
    #        write_mtx(

