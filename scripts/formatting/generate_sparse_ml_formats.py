import os
import subprocess
import json

sparse_ml_bench = ['gcn']
sparse_ml_path = os.environ['SPARSE_ML_PATH']
out_dir = os.environ['SPARSE_ML_FORMATTED_PATH']
basedir = os.getcwd()

for bench in sparse_ml_bench:
    bench_dir = os.path.join(sparse_ml_path, bench)
    for dataset in os.listdir(bench_dir):
        dataset_dir = os.path.join(bench_dir, dataset)
        for layer in os.listdir(dataset_dir):
            layer_dir = os.path.join(dataset_dir, layer)
            for kernel in os.listdir(layer_dir):
                kernel_dir = os.path.join(layer_dir, kernel)
                print("Formatting Matrices for kernel", kernel, " of the layer", 
                      layer, "of model", bench," trained on dataset", dataset)
                with open(os.path.join(kernel_dir, "info.json"), 'r') as info_file:
                    info = json.load(info_file)
                for tensor in info["input"]:
                    print("Formatting tensor", tensor["name"], "in format", tensor["format"])
                    tensor_name = tensor["name"]
                    tensor_formant = tensor["format"]
                    format_script = os.path.join(basedir, "scripts/formatting/datastructure_sparse_ml.py")
                    formatting_env = os.environ.copy()
                    formatting_env["SPARSE_ML_TENSOR_PATH"] = os.path.join(kernel_dir, tensor_name + ".npy")
                    subprocess.run(["python", 
                                    str(format_script), 
                                    "-n", tensor_name,
                                    "--hw",
                                    "--format", tensor_formant,
                                    "-b", kernel,
                                    "--modelname", bench,
                                    "--datasetname", dataset,
                                    "--layername", layer],
                                    env=formatting_env)
