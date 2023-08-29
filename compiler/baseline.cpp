// TACO code for CPU benchmarking SAM

#include <fstream>
// We're using c++14, so wer're stuck with experimental filesystem.
#include <experimental/filesystem>
#include <tuple>
#include <algorithm>

#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/index_notation/transformations.h"

// #include "mkl.h"

using namespace taco;

int WARP_SIZE = 32;
#define DIM_EXTRA 10 

template<int I, class...Ts>
decltype(auto) get(Ts &&... ts) {
    return std::get<I>(std::forward_as_tuple(ts...));
}

 const IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

template<class ...ExtraArgs>
void printTensor(TensorBase tensor, std::string location, std::string benchName, int dim, ExtraArgs &&... extra_args) {
    auto &sparsity = get<0>(extra_args...);
    auto opType = get<1>(extra_args...);

    std::string sparseStr = std::to_string(sparsity);
    sparseStr = sparseStr.substr(2, sparseStr.size());
    std::string filename = location + "/" + benchName + "_" + opType + "_" + \
                          std::to_string(dim) + "_" + sparseStr + "_" + tensor.getName() + ".txt";
    std::ofstream outfile(filename, std::ofstream::out);
    outfile << util::toString(tensor);
    outfile.close();
}

static void applyBenchSizes(benchmark::internal::Benchmark *b) {
    // b->ArgsProduct({{250, 500, 750, 1000, 2500, 5000, 7500, 8000}});
    b->ArgsProduct({{10}});
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
        if (this->lastPath == path and this->lastFormat == format) {
            // TODO (rohany): Not worrying about whether the format was the same as what was asked for.
            return std::make_pair(this->inputTensor, this->otherTensor);
        }

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

TensorInputCache<int64_t> inputCacheInt64;
TensorInputCache<int16_t> inputCacheInt16;
TensorInputCache<float> inputCacheFloat;

std::string cpuBenchKey(std::string tensorName, std::string funcName) {
    return tensorName + "-" + funcName + "-taco";
}

enum FrosttOp {
    TTV = 1,
    TTM = 2,
    MTTKRP = 3,
    INNERPROD = 4,
    PLUS2 = 5
};

std::string opName(FrosttOp op) {
    switch (op) {
        case TTV: {
            return "ttv";
        }
        case TTM: {
            return "ttm";
        }
        case MTTKRP: {
            return "mttkrp";
        }
        case INNERPROD: {
            return "innerprod";
        }
        case PLUS2: {
            return "plus2";
        }
        default:
            return "";
        }
}

static void bench_frostt_unsched(benchmark::State &state, FrosttOp op, int fill_value = 0) {
    bool GEN_OTHER = getEnvVar("GEN") == "ON";
    auto tnsPath = getEnvVar("FROSTT_TENSOR_PATH");

    std::cout << "Running benchmark tensor " << tnsPath << " on expression " << opName(op) << std::endl;


    auto pathSplit = taco::util::split(tnsPath, "/");
    auto filename = pathSplit[pathSplit.size() - 1];
    auto tensorName = taco::util::split(filename, ".")[0];
    state.SetLabel(tensorName);

    // TODO (rohany): What format do we want to do here?
    Tensor<int16_t> frosttTensor, otherShifted;

    std::tie(frosttTensor, otherShifted) = inputCacheInt16.getTensorInput(tnsPath, tensorName, Sparse,
                                                                     false, false, true, true, GEN_OTHER);

    // std::cout << "Running benchmark tensor " << tnsPath << " on expression " << opName(op) << std::endl;

    int DIM0 = frosttTensor.getDimension(0);
    int DIM1 = frosttTensor.getDimension(1);
    int DIM2 = frosttTensor.getDimension(2);

    for (auto _: state) {
        state.PauseTiming();
        Tensor<int16_t> result;
        switch (op) {
            case TTV: {
                result = Tensor<int16_t>("result", {DIM0, DIM1}, DCSR, fill_value);

                Tensor<int16_t> otherVec = inputCacheInt16.otherVecLastMode;

//                IndexVar i, j, k;
                result(i, j) = frosttTensor(i, j, k) * otherVec(k);
                break;
            }
            case INNERPROD: {
                result = Tensor<int16_t>("result");

//                IndexVar i, j, k;
                result() = frosttTensor(i, j, k) * otherShifted(i, j, k);
                break;
            }
            case PLUS2: {
                result = Tensor<int16_t>("result", frosttTensor.getDimensions(), frosttTensor.getFormat(), fill_value);

//                IndexVar i, j, k;
                result(i, j, k) = frosttTensor(i, j, k) + otherShifted(i, j, k);
                break;
            }
            case TTM: {
                // Assume otherMat(0) format is the same as frosttTensor(2)
                result = Tensor<int16_t>("result", {DIM0, DIM1, DIM_EXTRA}, frosttTensor.getFormat(), fill_value);
                Tensor<int16_t> otherMat = inputCacheInt16.otherMatTTM;

//                IndexVar i, j, k, l;
                // TODO: (owhsu) need to pick things for this and MTTKRP...
                result(i, j, k) = frosttTensor(i, j, l) * otherMat(k, l);
                break;
            }
            case MTTKRP: {
                result = Tensor<int16_t>("result", {DIM0, DIM_EXTRA}, DCSR, fill_value);

                Tensor<int16_t> otherMat = inputCacheInt16.otherMatMode1MTTKRP;
                Tensor<int16_t> otherMat1 = inputCacheInt16.otherMatMode2MTTKRP;

//                IndexVar i, j, k, l;
                result(i, j) = frosttTensor(i, k, l) * otherMat(j, k) * otherMat1(j, l);
                break;
            }
            default:
                state.SkipWithError("invalid expression");
                return;
        }
        result.setAssembleWhileCompute(true);
        result.compile();
        state.ResumeTiming();

        result.compute();

        state.PauseTiming();

        if (auto validationPath = getValidationOutputPath(); validationPath != "") {
            auto key = cpuBenchKey(tensorName, opName(op));
            auto outpath = validationPath + key + ".tns";
            taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
        }
        state.ResumeTiming();

    }
}

IndexStmt scheduleSpMVCPU(IndexStmt stmt, int CHUNK_SIZE=16) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .reorder({i0, i1, j})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt schedulePrecompute3D(IndexStmt stmt, IndexExpr precomputedExpr) {
  TensorVar precomputed("precomputed", Type(Float64, {16, 16, 16}), {Dense, Dense, Dense});
  return stmt.precompute(precomputedExpr, {i, j, l} , {i, j, l}, precomputed);
}

IndexStmt schedulePrecompute1D(IndexStmt stmt, IndexExpr precomputedExpr) {
  TensorVar precomputed("precomputed", Type(Float64, {102}), {Dense});
  return stmt.precompute(precomputedExpr, i , i, precomputed);
}

template<typename T>
IndexStmt scheduleSpMMCPU(IndexStmt stmt, Tensor<T> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpGEMMCPU(IndexStmt stmt, bool doPrecompute) {
  Assignment assign = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                          .as<Forall>().getStmt().as<Assignment>();
  TensorVar result = assign.getLhs().getTensorVar();

  stmt = reorderLoopsTopologically(stmt);
  if (doPrecompute) {
    IndexVar j = assign.getLhs().getIndexVars()[1];
    TensorVar w("w", Type(result.getType().getDataType(),
                {result.getType().getShape().getDimension(1)}), taco::dense);
    stmt = stmt.precompute(assign.getRhs(), j, j, w);
  }
  stmt = stmt.assemble(result, AssembleStrategy::Insert, true);
  auto qi_stmt = stmt.as<Assemble>().getQueries();
  IndexVar qi;
  if (isa<Where>(qi_stmt)) {
    qi = qi_stmt.as<Where>().getConsumer().as<Forall>().getIndexVar();
  } else {
    qi = qi_stmt.as<Forall>().getIndexVar();
  }
  stmt = stmt.parallelize(i, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces)
             .parallelize(qi, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces);

  return stmt;
}

IndexStmt scheduleSpAddCPU(IndexStmt stmt) {
  IndexStmt body = stmt.as<Forall>().getStmt().as<Forall>().getStmt();
  if (isa<Forall>(body)) {
    body = body.as<Forall>().getStmt();
  }
  Assignment assign = body.as<Assignment>();
  TensorVar result = assign.getLhs().getTensorVar();

  stmt = reorderLoopsTopologically(stmt);
  stmt = stmt.assemble(result, AssembleStrategy::Insert, true);

  IndexVar qi = stmt.as<Assemble>().getQueries().as<Forall>().getIndexVar();
  stmt = stmt.parallelize(i, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces)
             .parallelize(qi, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces);

  return stmt;
}

template<typename T>
IndexStmt scheduleSDDMMCPU(IndexStmt stmt, Tensor<T> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(k, kpos, B(i,k))
          .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
          .reorder({i0, i1, kpos0, j, kpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(kpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
}

template<typename T>
IndexStmt scheduleTTVCPU(IndexStmt stmt, Tensor<T> B, int CHUNK_SIZE=16) {
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2");
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .reorder({chunk, fpos2, k})
          .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}


IndexStmt scheduleTTVCPUCSR(IndexStmt stmt) {
  TensorVar result = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                         .as<Forall>().getStmt().as<Assignment>().getLhs()
                         .getTensorVar();
  return stmt.assemble(result, AssembleStrategy::Insert)
             .parallelize(i, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces);
}

template<typename T>
IndexStmt scheduleTTMCPU(IndexStmt stmt, Tensor<T> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), kpos("kpos"), kpos1("kpos1"), kpos2("kpos2");
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .pos(k, kpos, B(i,j,k))
          .split(kpos, kpos1, kpos2, UNROLL_FACTOR)
          .reorder({chunk, fpos2, kpos1, l, kpos2})
          .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(kpos2, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);;
}

template<typename T>
IndexStmt scheduleMTTKRPCPU(IndexStmt stmt, Tensor<T> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  int NUM_J = DIM_EXTRA;
  IndexVar i1("i1"), i2("i2");

  IndexExpr precomputeExpr = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Assignment>().getRhs().as<Mul>().getA();
  TensorVar w("w", Type(Float64, {(size_t)NUM_J}), taco::dense);

  stmt = stmt.split(i, i1, i2, CHUNK_SIZE)
    .reorder({i1, i2, k, l, j});
  stmt = stmt.precompute(precomputeExpr, j, j, w);

  return stmt
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

template<typename T>
IndexStmt scheduleMTTKRPPrecomputedCPU(IndexStmt stmt, Tensor<T> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2"), j_pre("j_pre");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

template<typename T>
IndexStmt scheduleMTTKRP4CPU(IndexStmt stmt, Tensor<T> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, m, j})
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

template<typename T>
IndexStmt scheduleMTTKRP5CPU(IndexStmt stmt, Tensor<T> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, m, n, j})
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

static void bench_frostt_sched(benchmark::State &state, FrosttOp op, int fill_value = 0) {
    bool GEN_OTHER = getEnvVar("GEN") == "ON";
    auto tnsPath = getEnvVar("FROSTT_TENSOR_PATH");

    std::cout << "Running benchmark tensor " << tnsPath << " on expression " << opName(op) << std::endl;

    auto pathSplit = taco::util::split(tnsPath, "/");
    auto filename = pathSplit[pathSplit.size() - 1];
    auto tensorName = taco::util::split(filename, ".")[0];
    state.SetLabel(tensorName);

    Format ecsr({Dense, Compressed(ModeFormat::NOT_UNIQUE),
                                  Singleton(ModeFormat::UNIQUE)});
    // TODO (rohany): What format do we want to do here?
    Tensor<int16_t> frosttTensor, otherShifted;
    Tensor<int16_t> frosttTensorEcsr, otherShiftedEcsr;
    if (op == PLUS2) {
        std::tie(frosttTensorEcsr, otherShiftedEcsr) = inputCacheInt16.getTensorInput(tnsPath, tensorName, ecsr,
                                                                                    false, false, true, true, GEN_OTHER, true);

    } else {
      std::tie(frosttTensor, otherShifted) = inputCacheInt16.getTensorInput(tnsPath, tensorName, Sparse,
                                                                            false, false, true, true, GEN_OTHER, true);
    }

    // std::cout << "Running benchmark tensor " << tnsPath << " on expression " << opName(op) << std::endl;

    int DIM0 = frosttTensor.getDimension(0);
    int DIM1 = frosttTensor.getDimension(1);
    int DIM2 = frosttTensor.getDimension(2);

    for (auto _: state) {
        state.PauseTiming();
        Tensor<int16_t> result;
        IndexStmt stmt = IndexStmt();
        switch (op) {
            case TTV: {
                // TODO: Change this to have sparse outputs
                result = Tensor<int16_t>("result", {DIM0, DIM1}, {Dense, Dense}, fill_value);

                Tensor<int16_t> otherVec = inputCacheInt16.otherVecLastMode;

                result(i, j) = frosttTensor(i, j, k) * otherVec(k);

                stmt = result.getAssignment().concretize();
                stmt = scheduleTTVCPU(stmt, frosttTensor);

              break;
            }
            case INNERPROD: {
                result = Tensor<int16_t>("result");

                result() = frosttTensor(i, j, k) * otherShifted(i, j, k);

                stmt = result.getAssignment().concretize();

                // TODO: Generate innerprod schedule
                break;
            }
            case PLUS2: {

                result = Tensor<int16_t>("result", frosttTensor.getDimensions(), ecsr, fill_value);

                result(i, j, k) = frosttTensorEcsr(i, j, k) + otherShiftedEcsr(i, j, k);

                stmt = result.getAssignment().concretize();
                stmt = scheduleSpAddCPU(stmt);
                break;
            }
            case TTM: {
                // Assume otherMat(0) format is the same as frosttTensor(2)
                result = Tensor<int16_t>("result", {DIM0, DIM1, DIM_EXTRA}, Dense, fill_value);
                Tensor<int16_t> otherMat = inputCacheInt16.otherMatTTM;

                Tensor<int16_t> otherMatTrans("C", {DIM2, DIM_EXTRA}, Dense, fill_value);
                std::vector<int> coords(otherMatTrans.getOrder());
                for (auto &value: taco::iterate<int16_t>(otherMat)) {
                  for (int i = 0; i < otherMat.getOrder(); i++) {
                    coords[otherMat.getOrder() - i - 1] = value.first[i];
                  }
                  otherMatTrans.insert(coords, (int16_t)value.second);
                }
                result(i, j, l) = frosttTensor(i, j, k) * otherMatTrans(k, l);

                stmt = result.getAssignment().concretize();
                stmt = scheduleTTMCPU(stmt, frosttTensor);
                break;
            }
            case MTTKRP: {
                result = Tensor<int16_t>("result", {DIM0, DIM_EXTRA}, Dense, fill_value);

                Tensor<int16_t> otherMat1 = inputCacheInt16.otherMatMode1MTTKRP;
                Tensor<int16_t> otherMat2 = inputCacheInt16.otherMatMode2MTTKRP;

                Tensor<int16_t> otherMat1Trans("C", {DIM1, DIM_EXTRA}, Dense, fill_value);

                std::vector<int> coords1(otherMat1Trans.getOrder());
                for (auto &value: taco::iterate<int16_t>(otherMat1)) {
                  for (int i = 0; i < otherMat1.getOrder(); i++) {
                    coords1[otherMat1.getOrder() - i - 1] = value.first[i];
                  }
                  otherMat1Trans.insert(coords1, (int16_t)value.second);
                }

                Tensor<int16_t> otherMat2Trans("D", {DIM2, DIM_EXTRA}, Dense, fill_value);
                std::vector<int> coords2(otherMat1Trans.getOrder());
                for (auto &value: taco::iterate<int16_t>(otherMat2)) {
                  for (int i = 0; i < otherMat2.getOrder(); i++) {
                    coords2[otherMat2.getOrder() - i - 1] = value.first[i];
                  }
                  otherMat2Trans.insert(coords2, (int16_t)value.second);
                }

                result(i, j) = frosttTensor(i, k, l) * otherMat1Trans(k, j) * otherMat2Trans(l, j);

                stmt = result.getAssignment().concretize();
                stmt = scheduleMTTKRPCPU(stmt, frosttTensor);

                break;
            }
            default:
                state.SkipWithError("invalid expression");
                return;
        }
        // result.setAssembleWhileCompute(true);
        taco_iassert(stmt != IndexStmt());
        result.compile(stmt);
        state.ResumeTiming();
        result.assemble();
        result.compute();

        state.PauseTiming();

        if (auto validationPath = getValidationOutputPath(); validationPath != "") {
            auto key = cpuBenchKey(tensorName, opName(op));
            auto outpath = validationPath + key + ".tns";
            taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
        }
        state.ResumeTiming();

    }
}

#define FOREACH_FROSTT_TENSOR(__func__) \
  __func__(facebook, "facebook.tns") \
  __func__(fb1k, "fb1k.tns") \
  __func__(fb10k, "fb10k.tns") \
  __func__(nell-1, "nell-1.tns") \
  __func__(nell-2, "nell-2.tns")

// Other FROSTT tensors that may or may not be too large to load.
// __func__(delicious, "delicious.tns") \
  // __func__(flickr, "flickr.tns") \
  // __func__(patents, "patents.tns") \
  // __func__(reddit, "reddit.tns") \
  // __func__(amazon-reviews, "amazon-reviews.tns") \
  // lbnl-network is fine, but python can't load it.
// __func__(lbnl-network, "lbnl-network.tns") \
 

//#define DECLARE_FROSTT_BENCH(name, path) \

TACO_BENCH_ARGS(bench_frostt_unsched, name/tensor3_ttv, TTV);
TACO_BENCH_ARGS(bench_frostt_unsched, name/tensor3_innerprod, INNERPROD);
TACO_BENCH_ARGS(bench_frostt_unsched, name/tensor3_elemadd_plus2, PLUS2);
TACO_BENCH_ARGS(bench_frostt_unsched, name/tensor3_ttm, TTM);
TACO_BENCH_ARGS(bench_frostt_unsched, name/tensor3_mttkrp, MTTKRP);

//#define DECLARE_FROSTT_BENCH_SCHED(name, path)
// Working:
TACO_BENCH_ARGS(bench_frostt_sched, name/tensor3_ttv, TTV);
// TACO_BENCH_ARGS(bench_frostt_sched, name/tensor3_innerprod, INNERPROD);vim
TACO_BENCH_ARGS(bench_frostt_sched, name/tensor3_ttm, TTM);
TACO_BENCH_ARGS(bench_frostt_sched, name/tensor3_mttkrp, MTTKRP);
// Kind of broken:
// TACO_BENCH_ARGS(bench_frostt_sched, name/tensor3_elemadd_plus2, PLUS2);


//FOREACH_FROSTT_TENSOR(DECLARE_FROSTT_BENCH)
//FOREACH_FROSTT_TENSOR(DECLARE_FROSTT_BENCH_SCHED)

struct SuiteSparseTensors {
    SuiteSparseTensors() {
        auto ssTensorPath = getTacoTensorPath();
        ssTensorPath += "suitesparse/";
        if (std::experimental::filesystem::exists(ssTensorPath)) {
            for (auto &entry: std::experimental::filesystem::directory_iterator(ssTensorPath)) {
                std::string f(entry.path());
                // Check that the filename ends with .mtx.
                if (f.compare(f.size() - 4, 4, ".mtx") == 0) {
                    this->tensors.push_back(entry.path());
                }
            }
        }
    }

    std::vector<std::string> tensors;
};

SuiteSparseTensors ssTensors;

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

std::string opName(SuiteSparseOp op) {
    switch (op) {
        case SPMV: {
            return "spmv";
        }
        case SPMM: {
            return "spmm";
        }
        case PLUS3: {
            return "plus3";
        }
        case SDDMM: {
            return "sddmm";
        }
        case MATTRANSMUL: {
            return "mattransmul";
        }
        case RESIDUAL: {
            return "residual";
        }
        case MMADD: {
            return "mmadd";
        }
        case MMMUL: {
            return "mmmul";
        }
        default:
            return "";
    }
}

static void bench_suitesparse_unsched(benchmark::State &state, SuiteSparseOp op, bool gen=true, int16_t fill_value = 0) {

    bool GEN_OTHER = (getEnvVar("GEN") == "ON" && gen);

    // Counters must be present in every run to get reported to the CSV.
    state.counters["dimx"] = 0;
    state.counters["dimy"] = 0;
    state.counters["nnz"] = 0;
    state.counters["other_sparsity1"] = 0;
    state.counters["other_sparsity1"] = 0;

    auto tensorPath = getEnvVar("SUITESPARSE_TENSOR_PATH");
    // std::cout << "Running " << opName(op) << " " << tensorPath << std::endl;
    if (tensorPath == "") {
        state.error_occurred();
        return;
    }

    auto pathSplit = taco::util::split(tensorPath, "/");
    auto filename = pathSplit[pathSplit.size() - 1];
    auto tensorName = taco::util::split(filename, ".")[0];
    state.SetLabel(tensorName);

    taco::Tensor<int16_t> ssTensor, otherShifted;
    try {
        taco::Format format = op == MATTRANSMUL ? DCSC : DCSR;
        std::tie(ssTensor, otherShifted) = inputCacheInt16.getTensorInput(tensorPath, tensorName, format, true /* countNNZ */,
                                                                     true /* includeThird */, true, false, GEN_OTHER);
    } catch (TacoException &e) {
        // Counters don't show up in the generated CSV if we used SkipWithError, so
        // just add in the label that this run is skipped.
        std::cout << e.what() << std::endl;
        state.SetLabel(tensorName + "/SKIPPED-FAILED-READ");
        return;
    }

    int DIM0 = ssTensor.getDimension(0);
    int DIM1 = ssTensor.getDimension(1);
    
    state.counters["dimx"] = DIM0;
    state.counters["dimy"] = DIM1;
    state.counters["nnz"] = inputCacheInt16.nnz;

    taco::Tensor<int16_t> denseMat1;
    taco::Tensor<int16_t> denseMat2;
    taco::Tensor<int16_t> s1("s1"), s2("s2");
    s1.insert({}, int16_t(2));
    s2.insert({}, int16_t(2));
    if (op == SDDMM) {
        denseMat1 = Tensor<int16_t>("denseMat1", {DIM0, DIM_EXTRA}, Format({dense, dense}));
        denseMat2 = Tensor<int16_t>("denseMat2", {DIM_EXTRA, DIM1}, Format({dense, dense}, {1, 0}));

        // (owhsu) Making this dense matrices of all 1's
        for (int kk = 0; kk < DIM_EXTRA; kk++) {
            for (int ii = 0; ii < DIM0; ii++) {
                denseMat1.insert({ii, kk}, int16_t(2));
            }
            for (int jj = 0; jj < DIM1; jj++) {
                denseMat2.insert({kk, jj}, int16_t(2));
            }
        }
    }

    for (auto _: state) {
        state.PauseTiming();
        Tensor<int16_t> result;
        IndexStmt stmt;

        switch (op) {
            case SPMV: {
                result = Tensor<int16_t>("result", {DIM0}, Format(Sparse), fill_value);
                Tensor<int16_t> otherVec = inputCacheInt16.otherVecFirstMode;

                result(i) = ssTensor(i, j) * otherVec(j);
                break;
            }
            case SPMM: {
                result = Tensor<int16_t>("result", {DIM0, DIM0}, DCSR, fill_value);
                Tensor<int16_t> otherShiftedTrans = inputCacheInt16.otherTensorTrans;

                result(i, j) = ssTensor(i, k) * otherShiftedTrans(k, j);
                
                stmt = result.getAssignment().concretize();
                stmt = reorderLoopsTopologically(stmt);
                // stmt = stmt.assemble(result.getAssignment().getLhs().getTensorVar(), taco::AssembleStrategy::Append);
                break;
            }
            case PLUS3: {
                result = Tensor<int16_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);
                Tensor<int16_t> otherShifted2 = inputCacheInt16.thirdTensor;

                result(i, j) = ssTensor(i, j) + otherShifted(i, j) + otherShifted2(i, j);
                break;
            }
            case SDDMM: {
                result = Tensor<int16_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);

                result(i, j) = ssTensor(i, j) * denseMat1(i, k) * denseMat2(k, j);
                break;
            }
            case RESIDUAL: {
                result = Tensor<int16_t>("result", {DIM0}, Format(Sparse), fill_value);

                Tensor<int16_t> otherVeci = inputCacheInt16.otherVecFirstMode;
                Tensor<int16_t> otherVecj = inputCacheInt16.otherVecLastMode;

                result(i) = otherVeci(i) - ssTensor(i, j) * otherVecj(j);
                break;
            }
            case MMADD: {
                result = Tensor<int16_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);

                result(i, j) = ssTensor(i, j) + otherShifted(i, j);
                break;
            }
            case MMMUL: {
                result = Tensor<int16_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);

                result(i, j) = ssTensor(i, j) * otherShifted(i, j);
                break;
            }
            case MATTRANSMUL: {
                result = Tensor<int16_t>("result", {DIM1}, Format(Sparse), fill_value);



                Tensor<int16_t> otherVeci = inputCacheInt16.otherVecLastMode;
                    Tensor<int16_t> otherVecj = inputCacheInt16.otherVecFirstMode;

                    result(i) = s1() * ssTensor(j, i) * otherVecj(j) + s2() * otherVeci(i);
                    break;
                }
                default:
                    state.SkipWithError("invalid expression");
                    return;
            }
            
            if (op == SPMM) {
                result.compile(stmt);
                state.ResumeTiming();
                result.assemble();
            }
            else {
                result.setAssembleWhileCompute(true);
                result.compile();
                state.ResumeTiming();
            }

            result.compute();

            state.PauseTiming();
            if (auto validationPath = getValidationOutputPath(); validationPath != "") {
                auto key = cpuBenchKey(tensorName, opName(op));
                auto outpath = validationPath + key + ".tns";
                taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
            }
            state.ResumeTiming();

        }
    }

    // The first app is set to true to generate both mode0 and mode1 vector
    // generation.
    TACO_BENCH_ARGS(bench_suitesparse_unsched, mat_mattransmul, MATTRANSMUL, true);
    TACO_BENCH_ARGS(bench_suitesparse_unsched, mat_elemadd3_plus3, PLUS3, false);
    TACO_BENCH_ARGS(bench_suitesparse_unsched, mat_sddmm, SDDMM, false);
    TACO_BENCH_ARGS(bench_suitesparse_unsched, mat_elemadd_mmadd, MMADD, false);
    TACO_BENCH_ARGS(bench_suitesparse_unsched, matmul_spmm, SPMM, false);
    TACO_BENCH_ARGS(bench_suitesparse_unsched, mat_elemmul, MMMUL, false);
    TACO_BENCH_ARGS(bench_suitesparse_unsched, vecmul_spmv, SPMV, false);
    TACO_BENCH_ARGS(bench_suitesparse_unsched, mat_residual, RESIDUAL, false);

static void bench_suitesparse_sched(benchmark::State &state, SuiteSparseOp op, bool gen=true, int fill_value = 0) {

  bool GEN_OTHER = (getEnvVar("GEN") == "ON" && gen);

  // Counters must be present in every run to get reported to the CSV.
  state.counters["dimx"] = 0;
  state.counters["dimy"] = 0;
  state.counters["nnz"] = 0;
  state.counters["other_sparsity1"] = 0;
  state.counters["other_sparsity1"] = 0;

  auto tensorPath = getEnvVar("SUITESPARSE_TENSOR_PATH");
  // std::cout << "Running " << opName(op) << " " << tensorPath << std::endl;
  if (tensorPath == "") {
    state.error_occurred();
    return;
  }

  auto pathSplit = taco::util::split(tensorPath, "/");
  auto filename = pathSplit[pathSplit.size() - 1];
  auto tensorName = taco::util::split(filename, ".")[0];
  state.SetLabel(tensorName);

  taco::Tensor<int64_t> ssTensor, otherShifted;
  try {
    taco::Format format = op == MATTRANSMUL ? DCSC : DCSR;
    std::tie(ssTensor, otherShifted) = inputCacheInt64.getTensorInput(tensorPath, tensorName, format, true /* countNNZ */,
                                                                 true /* includeThird */, true, false, GEN_OTHER);
  } catch (TacoException &e) {
    // Counters don't show up in the generated CSV if we used SkipWithError, so
    // just add in the label that this run is skipped.
    std::cout << e.what() << std::endl;
    state.SetLabel(tensorName + "/SKIPPED-FAILED-READ");
    return;
  }

  int DIM0 = ssTensor.getDimension(0);
  int DIM1 = ssTensor.getDimension(1);

  state.counters["dimx"] = DIM0;
  state.counters["dimy"] = DIM1;
  state.counters["nnz"] = inputCacheInt64.nnz;

  taco::Tensor<int16_t> denseMat1;
  taco::Tensor<int16_t> denseMat2;
  taco::Tensor<int16_t> s1("s1"), s2("s2");
  s1.insert({}, int16_t(2));
  s2.insert({}, int16_t(2));
  if (op == SDDMM) {
    denseMat1 = Tensor<int16_t>("denseMat1", {DIM0, DIM_EXTRA}, Format({dense, dense}));
    denseMat2 = Tensor<int16_t>("denseMat2", {DIM_EXTRA, DIM1}, Format({dense, dense}, {1, 0}));

    // (owhsu) Making this dense matrices of all 1's
    for (int kk = 0; kk < DIM_EXTRA; kk++) {
      for (int ii = 0; ii < DIM0; ii++) {
        denseMat1.insert({ii, kk}, int16_t(1));
      }
      for (int jj = 0; jj < DIM1; jj++) {
        denseMat2.insert({kk, jj}, int16_t(1));
      }
    }
  }

  for (auto _: state) {
    state.PauseTiming();
    Tensor<int16_t> result;
    IndexStmt stmt;

    switch (op) {
      case SPMV: {
        result = Tensor<int16_t>("result", {DIM0}, Format(Sparse), fill_value);
        Tensor<int64_t> otherVec = inputCacheInt64.otherVecLastMode;

        result(i) = ssTensor(i, j) * otherVec(j);
        break;
      }
      case SPMM: {
        result = Tensor<int16_t>("result", {DIM0, DIM0}, DCSR, fill_value);
        Tensor<int64_t> otherShiftedTrans = inputCacheInt64.otherTensorTrans;

        result(i, k) = ssTensor(i, j) * otherShiftedTrans(j, k);

        stmt = result.getAssignment().concretize();
        stmt = scheduleSpMMCPU(stmt, ssTensor);
        // stmt = stmt.assemble(result.getAssignment().getLhs().getTensorVar(), taco::AssembleStrategy::Append);
        break;
      }
      case PLUS3: {
        result = Tensor<int16_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);
        Tensor<int64_t> otherShifted2 = inputCacheInt64.thirdTensor;

        IndexVar i, j, k, l;
        result(i, j) = ssTensor(i, j) + otherShifted(i, j) + otherShifted2(i, j);
        break;
      }
      case SDDMM: {
        result = Tensor<int16_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);

        IndexVar i, j, k;
        result(i, j) = ssTensor(i, j) * denseMat1(i, k) * denseMat2(k, j);
        break;
      }
      case RESIDUAL: {
        result = Tensor<int16_t>("result", {DIM0}, Format(Sparse), fill_value);

        Tensor<int64_t> otherVeci = inputCacheInt64.otherVecFirstMode;
        Tensor<int64_t> otherVecj = inputCacheInt64.otherVecLastMode;

        IndexVar i, j, k;
        result(i) = otherVeci(i) - ssTensor(i, j) * otherVecj(j);
        break;
      }
      case MMADD: {
        result = Tensor<int16_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);

        IndexVar i, j, k;
        result(i, j) = ssTensor(i, j) + otherShifted(i, j);
        break;
      }
      case MMMUL: {
        result = Tensor<int16_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);

        IndexVar i, j, k;
        result(i, j) = ssTensor(i, j) * otherShifted(i, j);
        break;
      }
      case MATTRANSMUL: {
        result = Tensor<int16_t>("result", {DIM1}, Format(Sparse), fill_value);



        Tensor<int16_t> otherVeci = inputCacheInt64.otherVecLastMode;
        Tensor<int16_t> otherVecj = inputCacheInt64.otherVecFirstMode;

        IndexVar i, j;
        result(i) = s1() * ssTensor(j, i) * otherVecj(j) + s2() * otherVeci(i);
        break;
      }
      default:
        state.SkipWithError("invalid expression");
        return;
    }

    result.compile(stmt);

    state.ResumeTiming();
    result.assemble();
    result.compute();
    state.PauseTiming();

    if (auto validationPath = getValidationOutputPath(); validationPath != "") {
      auto key = cpuBenchKey(tensorName, opName(op));
      auto outpath = validationPath + key + ".tns";
      taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
    }
    state.ResumeTiming();

  }
}

// The first app is set to true to generate both mode0 and mode1 vector
// generation.
TACO_BENCH_ARGS(bench_suitesparse_sched, mat_mattransmul, MATTRANSMUL, true);
TACO_BENCH_ARGS(bench_suitesparse_sched, vecmul_spmv, SPMV, false);
TACO_BENCH_ARGS(bench_suitesparse_sched, mat_elemadd3_plus3, PLUS3, false);
TACO_BENCH_ARGS(bench_suitesparse_sched, mat_sddmm, SDDMM, false);
TACO_BENCH_ARGS(bench_suitesparse_sched, mat_residual, RESIDUAL, false);
TACO_BENCH_ARGS(bench_suitesparse_sched, mat_elemadd_mmadd, MMADD, false);
// TODO: need to fix for DCSC for this
TACO_BENCH_ARGS(bench_suitesparse_sched, matmul_spmm, SPMM, false);
TACO_BENCH_ARGS(bench_suitesparse_sched, mat_elemmul, MMMUL, false);

template<typename T>
static void print_array(T * array, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << (T) array[i] << ", ";
  }
  std::cout << std::endl;
}

IndexStmt scheduleTTMGPU(IndexStmt stmt, Tensor<double> B, int NNZ_PER_WARP=8*32, int BLOCK_SIZE=256, int CO_FACTOR=4) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar jk("jk"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");

  return stmt.reorder({i, j, k, l})
          .fuse(j, k, jk)
          .fuse(i, jk, f)
          .pos(f, fpos, B(i, j, k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(l, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, CO_FACTOR, BoundType::MaxExact)
          .reorder({block, warp, nnz, thread, dense_val})
          .unroll(dense_val, CO_FACTOR)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

IndexStmt scheduleTTVGPU(IndexStmt stmt, Tensor<double> B, IndexExpr precomputedExpr, int NNZ_PER_WARP=8*32, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar jk("jk"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), fpos2("fpos2"), thread("thread"), thread_nz("thread_nz"), thread_nz_pre("thread_nz_pre");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(thread_nz)}), taco::dense);

  return stmt.fuse(j, k, jk)
          .fuse(i, jk, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, fpos2, NNZ_PER_WARP)
          .split(fpos2, thread, thread_nz, NNZ_PER_WARP/WARP_SIZE)
          .reorder({block, warp, thread, thread_nz})
          .precompute(precomputedExpr, thread_nz, thread_nz_pre, precomputed)
          .unroll(thread_nz_pre, NNZ_PER_WARP/WARP_SIZE)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

IndexStmt scheduleMTTKRPGPU(IndexStmt stmt, Tensor<double> B, int NNZ_PER_WARP=16, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar kl("kl"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  return stmt.reorder({i,k,l,j})
          .fuse(k, l, kl)
          .fuse(i, kl, f)
          .pos(f, fpos, B(i, k, l))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(j, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, 1, BoundType::MaxExact)
          .reorder({block, warp, dense_val, thread, nnz})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

static void bench_frostt_sched_gpu(benchmark::State &state, SuiteSparseOp op, bool gen=true, int fill_value = 0) {

  if (!should_use_CUDA_codegen()) {
    return;
  }

  bool GEN_OTHER = getEnvVar("GEN") == "ON";
  auto tnsPath = getEnvVar("FROSTT_TENSOR_PATH");

  std::cout << "Running benchmark tensor " << tnsPath << " on expression " << opName(op) << std::endl;

  auto pathSplit = taco::util::split(tnsPath, "/");
  auto filename = pathSplit[pathSplit.size() - 1];
  auto tensorName = taco::util::split(filename, ".")[0];
  state.SetLabel(tensorName);

  Format ecsr({Dense, Compressed(ModeFormat::NOT_UNIQUE),
               Singleton(ModeFormat::UNIQUE)});
  // TODO (rohany): What format do we want to do here?
  Tensor<int16_t> frosttTensor, otherShifted;
  Tensor<int16_t> frosttTensorEcsr, otherShiftedEcsr;
  if (op == PLUS2) {
    std::tie(frosttTensorEcsr, otherShiftedEcsr) = inputCacheInt16.getTensorInput(tnsPath, tensorName, ecsr,
                                                                                  false, false, true, true, GEN_OTHER, true);

  } else {
    std::tie(frosttTensor, otherShifted) = inputCacheInt16.getTensorInput(tnsPath, tensorName, Sparse,
                                                                          false, false, true, true, GEN_OTHER, true);
  }

  // std::cout << "Running benchmark tensor " << tnsPath << " on expression " << opName(op) << std::endl;

  int DIM0 = frosttTensor.getDimension(0);
  int DIM1 = frosttTensor.getDimension(1);
  int DIM2 = frosttTensor.getDimension(2);

  for (auto _: state) {
    state.PauseTiming();
    Tensor<int16_t> result;
    IndexStmt stmt = IndexStmt();
    switch (op) {
      case TTV: {
        // TODO: Change this to have sparse outputs
        result = Tensor<int16_t>("result", {DIM0, DIM1}, {Dense, Dense}, fill_value);

        Tensor<int16_t> otherVec = inputCacheInt16.otherVecLastMode;

        auto precomputedExpr =  frosttTensor(i, j, k) * otherVec(k);
        result(i, j) = precomputedExpr;

        stmt = result.getAssignment().concretize();
        stmt = scheduleTTVGPU(stmt, frosttTensor, precomputedExpr);

        break;
      }
      case TTM: {
        result = Tensor<int16_t>("result", {DIM0, DIM1, DIM_EXTRA}, Dense, fill_value);
        Tensor<int16_t> otherMat = inputCacheInt16.otherMatTTM;

        Tensor<int16_t> otherMatTrans("C", {DIM2, DIM_EXTRA}, Dense, fill_value);
        std::vector<int> coords(otherMatTrans.getOrder());
        for (auto &value: taco::iterate<int16_t>(otherMat)) {
          for (int i = 0; i < otherMat.getOrder(); i++) {
            coords[otherMat.getOrder() - i - 1] = value.first[i];
          }
          otherMatTrans.insert(coords, (int16_t)value.second);
        }
        result(i, j, l) = frosttTensor(i, j, k) * otherMatTrans(k, l);

        stmt = result.getAssignment().concretize();
        stmt = scheduleTTMGPU(stmt, frosttTensor);
        break;
      }
      case MTTKRP: {
        result = Tensor<int16_t>("result", {DIM0, DIM_EXTRA}, Dense, fill_value);

        Tensor<int16_t> otherMat1 = inputCacheInt16.otherMatMode1MTTKRP;
        Tensor<int16_t> otherMat2 = inputCacheInt16.otherMatMode2MTTKRP;

        Tensor<int16_t> otherMat1Trans("C", {DIM1, DIM_EXTRA}, Dense, fill_value);

        std::vector<int> coords1(otherMat1Trans.getOrder());
        for (auto &value: taco::iterate<int16_t>(otherMat1)) {
          for (int i = 0; i < otherMat1.getOrder(); i++) {
            coords1[otherMat1.getOrder() - i - 1] = value.first[i];
          }
          otherMat1Trans.insert(coords1, (int16_t)value.second);
        }

        Tensor<int16_t> otherMat2Trans("D", {DIM2, DIM_EXTRA}, Dense, fill_value);
        std::vector<int> coords2(otherMat1Trans.getOrder());
        for (auto &value: taco::iterate<int16_t>(otherMat2)) {
          for (int i = 0; i < otherMat2.getOrder(); i++) {
            coords2[otherMat2.getOrder() - i - 1] = value.first[i];
          }
          otherMat2Trans.insert(coords2, (int16_t)value.second);
        }

        result(i, j) = frosttTensor(i, k, l) * otherMat1Trans(k, j) * otherMat2Trans(l, j);

        stmt = result.getAssignment().concretize();
        stmt = scheduleMTTKRPGPU(stmt, frosttTensor);

        break;
      }
      default:
        state.SkipWithError("invalid expression");
        return;
    }
    // result.setAssembleWhileCompute(true);

    taco_iassert(stmt != IndexStmt());
    result.compile(stmt);
    state.ResumeTiming();
    result.assemble();
    result.compute();

    state.PauseTiming();

    if (auto validationPath = getValidationOutputPath(); validationPath != "") {
      auto key = cpuBenchKey(tensorName, opName(op));
      auto outpath = validationPath + key + ".tns";
      taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
    }
    state.ResumeTiming();

  }
}

// The first app is set to true to generate both mode0 and mode1 vector
// generation.
TACO_BENCH_ARGS(bench_frostt_sched_gpu, tensor3_ttv, TTV, false);
TACO_BENCH_ARGS(bench_frostt_sched_gpu, tensor3_ttm, TTM, false);
TACO_BENCH_ARGS(bench_frostt_sched_gpu, tensor3_mttkrp, MTTKRP, false);

// static void bench_suitesparse_mkl(benchmark::State &state, SuiteSparseOp op, bool gen=true, int fill_value = 0) {
//   std::cout << "START BENCHMARK" << std::endl;
//   bool GEN_OTHER = (getEnvVar("GEN") == "ON" && gen);
// 
//   // Counters must be present in every run to get reported to the CSV.
//   state.counters["dimx"] = 0;
//   state.counters["dimy"] = 0;
//   state.counters["nnz"] = 0;
//   state.counters["other_sparsity1"] = 0;
//   state.counters["other_sparsity1"] = 0;
// 
//   auto tensorPath = getEnvVar("SUITESPARSE_TENSOR_PATH");
//   // std::cout << "Running " << opName(op) << " " << tensorPath << std::endl;
//   if (tensorPath == "") {
//     std::cout << "BENCHMARK ERROR" << std::endl;
//     state.error_occurred();
//     return;
//   }
// 
//   auto pathSplit = taco::util::split(tensorPath, "/");
//   auto filename = pathSplit[pathSplit.size() - 1];
//   auto tensorName = taco::util::split(filename, ".")[0];
//   state.SetLabel(tensorName);
// 
//   taco::Tensor<float> ssTensor, otherShifted;
//   try {
//     taco::Format format = op == MATTRANSMUL ? CSC : CSR;
//     std::tie(ssTensor, otherShifted) = inputCacheFloat.getTensorInput(tensorPath, tensorName, format, true /* countNNZ */,
//                                                                  true /* includeThird */, true, false, GEN_OTHER, true);
//   } catch (TacoException &e) {
//     // Counters don't show up in the generated CSV if we used SkipWithError, so
//     // just add in the label that this run is skipped.
//     std::cout << e.what() << std::endl;
//     state.SetLabel(tensorName + "/SKIPPED-FAILED-READ");
//     return;
//   }
// 
//   int DIM0 = ssTensor.getDimension(0);
//   int DIM1 = ssTensor.getDimension(1);
// 
//   state.counters["dimx"] = DIM0;
//   state.counters["dimy"] = DIM1;
//   state.counters["nnz"] = inputCacheFloat.nnz;
// 
//   taco::Tensor<float> denseMat1;
//   taco::Tensor<float> denseMat2;
//   taco::Tensor<float> s1("s1"), s2("s2");
//   s1.insert({}, float(2));
//   s2.insert({}, float(2));
//   if (op == SDDMM) {
//     denseMat1 = Tensor<int16_t>("denseMat1", {DIM0, DIM_EXTRA}, Format({dense, dense}));
//     denseMat2 = Tensor<int16_t>("denseMat2", {DIM_EXTRA, DIM1}, Format({dense, dense}, {1, 0}));
// 
//     // (owhsu) Making this dense matrices of all 1's
//     for (int kk = 0; kk < DIM_EXTRA; kk++) {
//       for (int ii = 0; ii < DIM0; ii++) {
//         denseMat1.insert({ii, kk}, float(2));
//       }
//       for (int jj = 0; jj < DIM1; jj++) {
//         denseMat2.insert({kk, jj}, float(2));
//       }
//     }
//   }
// 
//   MKL_INT NNZ = inputCacheFloat.nnz;
// 
// //  int *ptrB = (int*)malloc(sizeof(int) * DIM0);
// //  int *idxB = (int*)malloc(sizeof(int) * NNZ);
// //  float* valsB = (float*)malloc(sizeof(float)*NNZ);
// 
//   int *ptrB;
//   int *idxB;
//   float* valsB;
// 
//   std::cout << "GOT HERE" << std::endl;
//   getCSRArrays(ssTensor, &ptrB, &idxB, &valsB);
// 
//   sparse_matrix_t csrB;
//   mkl_sparse_s_create_csr(&csrB, SPARSE_INDEX_BASE_ZERO, DIM0, DIM1,
//                           ptrB, ptrB + 1, idxB, valsB);
// 
//   print_array(ptrB, DIM0 + 1);
//   print_array(idxB, NNZ);
//   print_array(valsB, NNZ);
// 
//   std::cout << "sparse_s_create_csr PASSED" << std::endl;
//   matrix_descr descrB;
//   descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
// 
//   int* ptrBCheck;
//   int* idxBCheck;
//   float* valsBCheck;
//   int rowsCheck, colsCheck;
//   sparse_index_base_t indextype;
//   mkl_sparse_s_export_csr(csrB, &indextype, &rowsCheck, &colsCheck, &ptrBCheck, &ptrBCheck + 1, &idxBCheck, &valsBCheck);
//   print_array(ptrBCheck, rowsCheck + 1);
//   print_array(idxBCheck, NNZ);
//   print_array(valsBCheck, NNZ);
// 
// 
// 
//   for (auto _: state) {
//     state.PauseTiming();
//     Tensor<float> result;
//     IndexStmt stmt;
// 
//     switch (op) {
//       case SPMV: {
// 
//         if (false) {
//           std::cout << "Entered SPMV Case" << std::endl;
// 
//           Tensor<float> otherVecj = inputCacheFloat.otherVecLastMode;
// 
//           std::cout << "otherVecj: " << otherVecj << otherVecj.getStorage() << std::endl;
// 
// //        const float* x = (float *)otherVecj.getStorage().getValues().getData();
// //        std::cout << "x: " << x << ", " << x << std::endl;
// 
//           auto x = (float *) malloc(sizeof(float) * DIM1);
// 
//           std::fill(x, x + DIM1, 0.0);
//           for (auto &value: taco::iterate<float>(otherVecj)) {
//             for (int i = 0; i < otherVecj.getOrder(); i++) {
//               std::cout << "x: (" << i << ", " << value.first[i] << ")" << std::endl;
//               x[i] = (float) value.first[i];
//             }
//           }
// 
//           float *y;
//           // = (float  *)malloc(sizeof(float) * DIM0);
//           std::fill(y, y + DIM0, 0.0);
// 
//           std::cout << "y: " << y << std::endl;
//           print_array(y, DIM0);
// //        std::cout << csrB << std::endl;
// 
//           std::cout << "mkl_sparse_optimize ENTERING" << std::endl;
//           // mkl_sparse_optimize(csrB);
//           std::cout << "mkl_sparse_optimize PASSED" << std::endl;
//           state.ResumeTiming();
//           auto mkl_code = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrB, descrB, x, 0.0, y);
//           state.PauseTiming();
//           std::cout << "mkl_sparse_s_mv PASSED" << std::endl;
//         } else {
//           // Since SpMV is seg faulting, do SpMM with a size jx1 matrix...
//           int *ptrC;
//           int *idxC;
//           float* valsC;
// 
//           Tensor<float> otherVecj = inputCacheFloat.otherVecLastMode;
// 
//           // FIXME: (owhsu) this isn't correct yet
//           ptrC[0] = 0;
//           ptrC[1] = otherVecj.getDimension(0);
//           idxC[0] = 0;
// 
//           sparse_matrix_t csrC;
//           mkl_sparse_s_create_csc(&csrC, SPARSE_INDEX_BASE_ZERO, DIM1, DIM0,
//                                   ptrC, ptrC + 1, idxC, valsC);
//           std::cout << "sparse_s_create_csc C PASSED" << std::endl;
// 
//           auto result = (float *)malloc(sizeof(float) * DIM0 * DIM1);
// 
//           state.ResumeTiming();
//           auto mkl_code = mkl_sparse_s_spmmd(SPARSE_OPERATION_NON_TRANSPOSE, csrB, csrC, SPARSE_LAYOUT_ROW_MAJOR, result, DIM0);
//           state.PauseTiming();
// 
//           free(result);
//         }
// 
// 
//       }
//       case SPMM: {
//         int *ptrC;
//         int *idxC;
//         float* valsC;
// 
//         getCSCArrays(inputCacheFloat.otherTensorTrans, &ptrC, &idxC, &valsC);
// 
//         sparse_matrix_t csrC;
//         mkl_sparse_s_create_csc(&csrC, SPARSE_INDEX_BASE_ZERO, DIM1, DIM0,
//                                 ptrC, ptrC + 1, idxC, valsC);
// 
//         float * result; // = (float *)malloc(sizeof(float) * DIM0 * DIM1);
// 
//         state.ResumeTiming();
//         auto mkl_code = mkl_sparse_s_spmmd(SPARSE_OPERATION_NON_TRANSPOSE, csrB, csrC, SPARSE_LAYOUT_ROW_MAJOR, result, DIM0);
//         std::cout << mkl_code << std::endl;
//         state.PauseTiming();
// 
//         mkl_sparse_destroy(csrC);
//         break;
//       }
//       case PLUS3: {
//         // Turn C tensor into mkl CSR
//         int *ptrC;
//         int *idxC;
//         float* valsC;
// 
//         std::cout << "GOT HERE, Tensor C" << std::endl;
//         getCSRArrays(otherShifted, &ptrC, &idxC, &valsC);
// 
//         sparse_matrix_t csrC;
// 
//         mkl_sparse_s_create_csr(&csrC, SPARSE_INDEX_BASE_ZERO, DIM1, DIM0,
//                                 ptrC, ptrC + 1, idxC, valsC);
// 
// //        int *ptrD = (int*)malloc(sizeof(int) * DIM0);
// //        int *idxD = (int*)malloc(sizeof(int) * NNZ);
// //        auto* valsD = (float*)malloc(sizeof(float)*NNZ);
//         int *ptrD;
//         int *idxD;
//         float* valsD;
// 
//         // Turn D tensor into mkl CSR
//         std::cout << "GOT HERE, Tensor D" << std::endl;
//         getCSRArrays(inputCacheFloat.thirdTensor, &ptrD, &idxD, &valsD);
// 
//         sparse_matrix_t csrD;
//         mkl_sparse_s_create_csr(&csrC, SPARSE_INDEX_BASE_ZERO, DIM1, DIM0,
//                                 ptrD, ptrD + 1, idxD, valsD);
//         std::cout << "mkl_sparse_s_create_csr, PASSED" << std::endl;
// 
//         sparse_matrix_t csrBC;
//         sparse_matrix_t csrResult;
// 
//         state.ResumeTiming();
//         auto mkl_code = mkl_sparse_s_add(SPARSE_OPERATION_NON_TRANSPOSE, csrB, 1.0, csrC, &csrBC);
//         std::cout << "First mkl_sparse_s_add call code: " <<  mkl_code << std::endl;
//         mkl_code = mkl_sparse_s_add(SPARSE_OPERATION_NON_TRANSPOSE, csrBC, 1.0, csrD, &csrResult);
//         std::cout << "Second mkl_sparse_s_add call code: " <<  mkl_code << std::endl;
//         state.PauseTiming();
// 
//         mkl_sparse_destroy(csrC);
//         mkl_sparse_destroy(csrD);
//         mkl_sparse_destroy(csrBC);
//         mkl_sparse_destroy(csrResult);
//         break;
//       }
//       case SDDMM: {
//       }
//       case RESIDUAL: {
//       }
//       case MMADD: {
//       }
//       case MMMUL: {
//       }
//       case MATTRANSMUL: {
//       }
//       default:
//         state.SkipWithError("invalid expression");
//         return;
//     }
// 
//     mkl_sparse_destroy(csrB);
// 
//     if (auto validationPath = getValidationOutputPath(); validationPath != "") {
//       auto key = cpuBenchKey(tensorName, opName(op));
//       auto outpath = validationPath + key + ".tns";
//       taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
//     }
//     state.ResumeTiming();
// 
//   }
// }

// SpMV seg faults in libmkl for some reason
// TACO_BENCH_ARGS(bench_suitesparse_mkl, vecmul_spmv_mkl, SPMV, false)->UseRealTime();
//TACO_BENCH_ARGS(bench_suitesparse_mkl, matmul_spmm_mkl, SPMM, false)->UseRealTime();
//TACO_BENCH_ARGS(bench_suitesparse_mkl, mat_plus3_mkl, PLUS3, false)->UseRealTime();
