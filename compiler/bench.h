#ifndef TACO_BENCH_BENCH_H
#define TACO_BENCH_BENCH_H

#include <fstream>
// We're using c++14, so wer're stuck with experimental filesystem.
#include <experimental/filesystem>
#include <tuple>

#include "benchmark/benchmark.h"
#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/index_notation/transformations.h"

#define DIM_EXTRA 10

using namespace taco;

enum SuiteSparseOp {
    SPMV = 1,
    SPMM = 2,
    PLUS3 = 3,
    SDDMM = 4,
    MATTRANSMUL = 5,
    RESIDUAL = 6,
    MMADD = 7,
    MMMUL = 8
};

// std::string opName(SuiteSparseOp op) {
//     switch (op) {
//         case SPMV: {
//             return "spmv";
//         }
//         case SPMM: {
//             return "spmm";
//         }
//         case PLUS3: {
//             return "plus3";
//         }
//         case SDDMM: {
//             return "sddmm";
//         }
//         case MATTRANSMUL: {
//             return "mattransmul";
//         }
//         case RESIDUAL: {
//             return "residual";
//         }
//         case MMADD: {
//             return "mmadd";
//         }
//         case MMMUL: {
//             return "mmmul";
//         }
//         default:
//             return "";
//     }
// }

// Register a benchmark with the following options:
// * Millisecond output display
// * 10 data points
// * Reporting of avg/stddev/median
// * Wall-clock time, rather than CPU time.
#define TACO_BENCH(bench)         \
  BENCHMARK(bench)                \
  ->Unit(benchmark::kMillisecond) \
  ->Repetitions(10)               \
  ->Iterations(1)                 \
  ->ReportAggregatesOnly(true)    \
  ->UseRealTime()

#define GRAPHBLAS_BENCH(bench, times)   \
  BENCHMARK(bench)                \
  ->Unit(benchmark::kMillisecond) \
  ->Repetitions(times)               \
  ->Iterations(1)                 \
  ->ReportAggregatesOnly(false)    \
  ->UseRealTime()

// TACO_BENCH_ARG is similar to TACO_BENCH but allows for passing
// of an arbitrarily typed argument to the benchmark function.
// TODO (rohany): Make this take in only 1 argument.
// TODO (rohany): Don't specify the time here, but do it at the command line.

#define TACO_BENCH_ARG(bench, name, arg)  \
  BENCHMARK_CAPTURE(bench, name, arg)     \
  ->Unit(benchmark::kMicrosecond)         \
  ->Iterations(1)                         \
  ->ReportAggregatesOnly(true)            \
  ->UseRealTime()

#define TACO_BENCH_ARGS(bench, name, ...)       \
  BENCHMARK_CAPTURE(bench, name, __VA_ARGS__)   \
  ->Unit(benchmark::kMicrosecond)               \
  ->Iterations(1)                               \
  ->ReportAggregatesOnly(true)                  \
  ->UseRealTime()

std::string getEnvVar(std::string varname);

std::string getTacoTensorPath();

std::string getValidationOutputPath();

// cleanPath ensures that the input path ends with "/".
std::string cleanPath(std::string path);

taco::TensorBase
loadRandomTensor(std::string name, std::vector<int> dims, float sparsity, taco::Format format, int variant = 0);

taco::TensorBase loadImageTensor(std::string name, int num, taco::Format format, float threshold, int variant = 0);

taco::TensorBase loadMinMaxTensor(std::string name, int order, taco::Format format, int variant = 0);

std::string constructOtherVecKey(std::string tensorName, std::string variant, float sparsity);

std::string constructOtherMatKey(std::string tensorName, std::string variant, std::vector<int> dims, float sparsity);

template<typename T>
taco::Tensor<T> castToType(std::string name, taco::Tensor<double> tensor) {
    taco::Tensor<T> result(name, tensor.getDimensions(), tensor.getFormat());
    std::vector<int> coords(tensor.getOrder());
    for (auto &value: taco::iterate<double>(tensor)) {
        for (int i = 0; i < tensor.getOrder(); i++) {
            coords[i] = value.first[i];
        }
        // Attempt to cast the value to an integer. However, if the cast causes
        // the value to equal 0, then this will ruin the sparsity pattern of the
        // tensor, as the 0 values will get compressed out. So, if a cast would
        // equal 0, insert 1 instead to preserve the sparsity pattern of the tensor.
        if (static_cast<T>(value.second) == T(0)) {
            result.insert(coords, static_cast<T>(1));
        } else {
            result.insert(coords, static_cast<T>(value.second));
        }
    }
    result.pack();
    return result;
}

template<typename T>
taco::Tensor<T> castToTypeZero(std::string name, taco::Tensor<double> tensor) {
    taco::Tensor<T> result(name, tensor.getDimensions(), tensor.getFormat());
    std::vector<int> coords(tensor.getOrder());
    for (auto &value: taco::iterate<double>(tensor)) {
        for (int i = 0; i < tensor.getOrder(); i++) {
            coords[i] = value.first[i];
        }
        // Attempt to cast the value to an integer. However, if the cast causes
        // the value to equal 0, then this will ruin the sparsity pattern of the
        // tensor, as the 0 values will get compressed out. So, if a cast would
        // equal 0, insert 1 instead to preserve the sparsity pattern of the tensor.
        result.insert(coords, static_cast<T>(value.second));
    }
    result.pack();
    return result;
}

template<typename T, typename T2>
taco::Tensor<T> shiftLastMode(std::string name, taco::Tensor<T2> original) {
    taco::Tensor<T> result(name, original.getDimensions(), original.getFormat());
    std::vector<int> coords(original.getOrder());
    for (auto &value: taco::iterate<T2>(original)) {
        for (int i = 0; i < original.getOrder(); i++) {
            coords[i] = value.first[i];
        }
        int lastMode = original.getOrder() - 1;
        // For order 2 tensors, always shift the last coordinate. Otherwise, shift only coordinates
        // that have even last coordinates. This ensures that there is at least some overlap
        // between the original tensor and its shifted counter part.
        if (original.getOrder() <= 2 || (coords[lastMode] % 2 == 0)) {
            coords[lastMode] = (coords[lastMode] + 1) % original.getDimension(lastMode);
        }
        // TODO (rohany): Temporarily use a constant value here.
        result.insert(coords, T(2));
    }
    result.pack();
    return result;
}

template<typename T, typename T2>
taco::Tensor<T> transposeTensor(std::string name, taco::Tensor<T2> original) {
    assert(original.getOrder() == 2);
    std::vector<int> dimensionsT = original.getDimensions();
    std::reverse(dimensionsT.begin(), dimensionsT.end());

    std::vector<int> modeOrderingT(original.getFormat().getModeOrdering());
    std::reverse(modeOrderingT.begin(), modeOrderingT.end());

    taco::Format formatT(original.getFormat().getModeFormatPacks(), modeOrderingT);
    taco::Tensor<T> result(name, dimensionsT, formatT);

    std::vector<int> coords(original.getOrder());
    for (auto &value: taco::iterate<T2>(original)) {
        for (int i = 0; i < original.getOrder(); i++) {
            auto iT = original.getOrder() - 1 - i;
            coords[iT] = value.first[i];
        }

        result.insert(coords, T(value.second));
    }
    result.pack();
    return result;
}

template<typename T, typename T2>
taco::Tensor<T> genOtherVec(std::string name, std::string datasetName, taco::Tensor<T2> original, int mode = 0,
                         float sparsity=0.001, taco::Format format=taco::sparse, bool genNonEmpty = true) {
    int dimension = original.getDimensions().at(mode);
    taco::Tensor<T> result(name, {dimension}, format);

    bool empty = true;
    for (int ii = 0; ii < dimension; ii++) {
        float rand_float = (float) rand() / (float) (RAND_MAX);
        if (rand_float < sparsity) {
            // (owhsu) Setting this number to 2 for now
            result.insert({ii}, T(2));
            empty = false;
        }
    }

    // Insert at least one nonzero
    if (genNonEmpty && empty) {
        result.insert({0}, T(2));
    }

    result.pack();
    taco::write(constructOtherVecKey(datasetName, "vec_mode"+std::to_string(mode), sparsity), result);

    return result;
}

template<typename T, typename T2>
taco::Tensor<T> getOtherVec(std::string name, std::string datasetName, taco::Tensor<T2> original,
                            std::vector<int> dimensions, int mode = 0, float sparsity=0.001) {
    taco::Tensor<T> result;

    taco::Tensor<double> tensor = taco::readDim(constructOtherVecKey(datasetName,
                                                    "vec_mode" + std::to_string(mode), sparsity),
                                                taco::Sparse, dimensions, true);
    result = castToType<T>(name, tensor);
    return result;
}

template<typename T, typename T2>
taco::Tensor<T> genOtherMat(std::string name, std::string datasetName, taco::Tensor<T2> original,
                            std::vector<int> dimensions, std::string filestr, int mode = 0, float sparsity=0.001,
                            taco::Format format = taco::DCSR, bool genNonEmpty = true) {

    taco::Tensor<T> result(name, dimensions, format);

    bool empty = true;
    for (int ii = 0; ii < dimensions[0]; ii++) {
        for (int jj = 0; jj < dimensions[1]; jj++) {
            float rand_float = (float) rand() / (float) (RAND_MAX);
            if (rand_float < sparsity) {
                // (owhsu) Setting this number to 2 for now
                result.insert({ii, jj}, T(2));
                empty = false;
            }
        }
    }

    // Insert at least one nonzero
    if (genNonEmpty && empty) {
      result.insert({0, 0}, T(2));
    }

    result.pack();
    taco::write(constructOtherMatKey(datasetName, "mat_mode"+std::to_string(mode)+"_"+filestr, dimensions, sparsity), result);

    return result;
}


template<typename T, typename T2>
taco::Tensor<T> getOtherMat(std::string name, std::string datasetName, taco::Tensor<T2> original, std::vector<int> dimensions, std::string filestr,
                            int mode = 0, float sparsity=0.001, taco::Format format = taco::DCSR) {
    taco::Tensor<T> result;
    taco::Tensor<double> tensor = taco::readDim(constructOtherMatKey(datasetName, "mat_mode"+std::to_string(mode)+"_"+filestr, dimensions, sparsity),
                                             format, dimensions, true);
    result = castToType<T>(name, tensor);
    return result;
}

// TensorInputCache is a cache for the input to ufunc benchmarks. These benchmarks
// operate on a tensor loaded from disk and the same tensor shifted slightly. Since
// these operations are run multiple times, we can save alot in benchmark startup
// time from caching these inputs.
template<typename T>
struct TensorInputCache {
    template<typename U>
    std::pair<taco::Tensor<T>, taco::Tensor<T>>
    getTensorInput(std::string path, std::string datasetName, U format, bool countNNZ = false, bool includeThird = false,
                   bool includeVec = false, bool includeMat = false, bool genOther = false, bool use_CSR_CSC=false) {
        // See if the paths match.
        // if (this->lastPath == path and this->lastFormat == format) {
        //     // TODO (rohany): Not worrying about whether the format was the same as what was asked for.
        //     return std::make_pair(this->inputTensor, this->otherTensor);
        // }

        // Otherwise, we missed the cache. Load in the target tensor and process it.
        this->lastLoaded = taco::read(path, format);
        this->lastFormat = format;
        // We assign lastPath after lastLoaded so that if taco::read throws an exception
        // then lastPath isn't updated to the new path.
        this->lastPath = path;
        this->inputTensor = castToType<T>("B", this->lastLoaded);
        this->otherTensor = shiftLastMode<T, T>("C", this->inputTensor);

        if (countNNZ) {
            this->nnz = 0;
            for (auto &it: iterate<T>(this->inputTensor)) {
                this->nnz++;
            }
        }
        if (includeThird) {
            this->thirdTensor = shiftLastMode<T, T>("D", this->otherTensor);
            taco::Tensor<T> oTT;
            if (use_CSR_CSC) {
               oTT = this->otherTensor.transpose("C", {1, 0}, CSC);
            } else {
              oTT = this->otherTensor.transpose("C", {1, 0}, DCSC);
            }
            this->otherTensorTrans = oTT;

        }
        if (includeVec and genOther) {
	          std::cout << "Generating OTHER vector for " << datasetName << std::endl;
            this->otherVecFirstMode = genOtherVec<T, T>("C", datasetName, this->inputTensor);
            auto lastMode = this->inputTensor.getDimensions().size() - 1;
            this->otherVecLastMode = genOtherVec<T, T>("D", datasetName, this->inputTensor, lastMode);
        } else if (includeVec) {
            std::vector<int32_t> firstDim;
            std::vector<int32_t> lastDim;
            if (this->inputTensor.getOrder() == 2) {
                firstDim.push_back(this->inputTensor.getDimension(0));
                lastDim.push_back(this->inputTensor.getDimension(1));
            } else {
                firstDim.push_back(this->inputTensor.getDimension(0));
                lastDim.push_back(this->inputTensor.getDimension(2));
            }

            this->otherVecFirstMode = getOtherVec<T, T>("C", datasetName, this->inputTensor, firstDim);
            auto lastMode = this->inputTensor.getDimensions().size() - 1;
            this->otherVecLastMode = getOtherVec<T, T>("D", datasetName, this->inputTensor, lastDim, lastMode);
        }

        if (this->inputTensor.getOrder() > 2 and includeMat and genOther) {
            int DIM1 = this->inputTensor.getDimension(1);
            int DIM2 = this->inputTensor.getDimension(2);

            this->otherMatTTM = genOtherMat<T, T>("C", datasetName, this->inputTensor, {DIM_EXTRA, DIM2}, "ttm", 2);
            this->otherMatMode1MTTKRP = genOtherMat<T, T>("D", datasetName, this->inputTensor, {DIM_EXTRA, DIM1}, "mttkrp", 1);
            this->otherMatMode2MTTKRP = genOtherMat<T, T>("C", datasetName, this->inputTensor, {DIM_EXTRA, DIM2}, "mttkrp", 2);
        } else if (this->inputTensor.getOrder() > 2 and includeMat) {
            int DIM1 = this->inputTensor.getDimension(1);
            int DIM2 = this->inputTensor.getDimension(2);

            this->otherMatTTM = getOtherMat<T, T>("C", datasetName, this->inputTensor, {DIM_EXTRA, DIM2}, "ttm", 2);
            this->otherMatMode1MTTKRP = getOtherMat<T, T>("D", datasetName, this->inputTensor, {DIM_EXTRA, DIM1}, "mttkrp", 1);
            this->otherMatMode2MTTKRP = getOtherMat<T, T>("C", datasetName, this->inputTensor, {DIM_EXTRA, DIM2}, "mttkrp", 2);
        }
        return std::make_pair(this->inputTensor, this->otherTensor);
    }

    taco::Tensor<double> lastLoaded;
    std::string lastPath;
    taco::Format lastFormat;

    taco::Tensor<T> inputTensor;
    taco::Tensor<T> otherTensor;
    taco::Tensor<T> thirdTensor;
    taco::Tensor<T> otherTensorTrans;
    taco::Tensor<T> otherVecFirstMode;
    taco::Tensor<T> otherVecLastMode;

    // FROSTT only
    taco::Tensor<T> otherMatTTM;
    taco::Tensor<T> otherMatMode1MTTKRP;
    taco::Tensor<T> otherMatMode2MTTKRP;

    int64_t nnz;
};

#endif //TACO_BENCH_BENCH_H
