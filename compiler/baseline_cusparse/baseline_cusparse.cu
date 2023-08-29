/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <math.h>            // fabs
#include <stdlib.h>           // EXIT_FAILURE
#include <taco.h>
#include <taco/storage/file_io_mtx.h>
#include <taco/format.h>
#include <taco/index_notation/transformations.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include "benchmark/benchmark.h"
#include "../bench.h"
#include <tuple>
// #include "benchmark/include/benchmark/benchmark.h"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

bool float_compare(float f1, float f2, float pct){
    if((f1 == 0.0f) && (f2 == 0.0f)){
        return true;
    }
    auto div_by = f2;
    if(f2 == 0.0f){
        div_by = f1;
    }
    auto percent_diff = (fabs(f1 - f2) / div_by) * 100.0;
    return percent_diff < pct;
}

int spgemm(taco::Tensor<float> & tensorA, taco::Tensor<float> & tensorB, benchmark::State * state){

    bool benching = (state != nullptr);

    if(benching) {state->PauseTiming();}

    std::cout << "Calculating SPGEMM" << std::endl;
    auto dims = tensorA.getDimensions();

    // Get A and its arrays to analyze sizes/etc
    int * rowptrA;
    int * colidxA;
    float * valsA;

    taco::getCSRArrays(tensorA, &rowptrA, &colidxA, &valsA);

    auto storage = tensorA.getStorage();
    auto index = storage.getIndex();
    auto rowptrArr = index.getModeIndex(1).getIndexArray(0);
    auto colidxArr = index.getModeIndex(1).getIndexArray(1);
    auto rowptrsize = rowptrArr.getSize();
    auto numrowsA = rowptrsize - 1;
    auto coloffsetsize = colidxArr.getSize();
    auto valssize = coloffsetsize;

    int NUM_I = dims[0];
    int NUM_K = dims[1];
    int NUM_J = dims[0];

    // tensorB is transposed in CSR
    taco::Tensor<float> tensorB_csc = tensorB.transpose("tensorB_transposed_csc", {0, 1}, taco::CSC);

    int * rowptrB;
    int * colidxB;
    float * valsB;

    taco::getCSRArrays(tensorB, &rowptrB, &colidxB, &valsB);

    /*
        Compute the output of spgemm
    */
    taco::IndexVar i, j, k;

    taco::Tensor<float> expected("expected", {NUM_I, NUM_J}, taco::CSR);
    expected(i, k) = tensorA(i, j) * tensorB_csc(j, k);
    expected.compile();
    expected.assemble();
    expected.compute();

    // Remove 0's or else the GPU stuff doesn't work due to mismatches in allocs.
    // auto expected_no_zeros = expected.removeExplicitZeros(taco::CSR);
    auto expected_no_zeros = expected.removeExplicitZeros(taco::CSR);

    // Get the information/ptrs from C for comparison and GPU calls
    int * rowptrC;
    int * colidxC;
    float * valsC;

    taco::getCSRArrays(expected_no_zeros, &rowptrC, &colidxC, &valsC);

    auto storageC = expected_no_zeros.getStorage();
    auto indexC = storageC.getIndex();

    auto rowptrArrC = indexC.getModeIndex(1).getIndexArray(0);
    auto colidxArrC = indexC.getModeIndex(1).getIndexArray(1);

    auto rowptrsizeC = rowptrArrC.getSize();
    auto coloffsetsizeC = colidxArrC.getSize();
    auto valssizeC = coloffsetsizeC;

    // Host problem definition
    #define   A_NUM_ROWS dims[0]   // C compatibility
    const int A_num_rows = dims[0];
    const int A_num_cols = dims[1];
    const int A_nnz      = valssize;
    const int B_num_rows = dims[1];
    const int B_num_cols = dims[0];
    const int B_nnz      = valssize;

    int  * hA_csrOffsets = rowptrA;
    int  * hA_columns    = colidxA;
    float* hA_values     = valsA;
    int  * hB_csrOffsets = rowptrB;
    int  * hB_columns    = colidxB;
    float* hB_values     = valsB;
    int   *hC_csrOffsets = rowptrC;
    int   *hC_columns    = colidxC;
    float *hC_values     = valsC;

    const int C_nnz       = valssizeC;
    #define   C_NUM_NNZ valssizeC   // C compatibility
    float               alpha       = 1.0f;
    float               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // Device memory management: Allocate and copy A, B
    int   *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
        *dC_csrOffsets, *dC_columns;
    float *dA_values, *dB_values, *dC_values;
    // allocate A
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                        (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float)) )
    // allocate B
    CHECK_CUDA( cudaMalloc((void**) &dB_csrOffsets,
                        (B_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_columns, B_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dB_values,  B_nnz * sizeof(float)) )
    // allocate C offsets
    CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,
                        (A_num_rows + 1) * sizeof(int)) )
    // copy A
    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
    // CHECK_CUDA( cudaMemcpy(dA_csrOffsets, &hA_csrOffsets,
                        (A_num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                        A_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    // copy B
    CHECK_CUDA( cudaMemcpy(dB_csrOffsets, hB_csrOffsets,
                        (B_num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_columns, hB_columns, B_nnz * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_values, hB_values,
                        B_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                    dA_csrOffsets, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                    dB_csrOffsets, dB_columns, dB_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                    dC_csrOffsets, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    if(benching) {state->ResumeTiming();}

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, NULL) )

    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, dBuffer1) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                            spgemmDesc, &bufferSize2, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                        &alpha, matA, matB, &beta, matC,
                                        computeType, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize2, dBuffer2) )
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                        &C_nnz1) )
    // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float)) )

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    int   hC_csrOffsets_tmp[A_NUM_ROWS + 1];
    int   hC_columns_tmp[C_nnz1];
    float hC_values_tmp[C_nnz1];
    CHECK_CUDA( cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
                        (A_num_rows + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_columns_tmp, dC_columns, C_nnz1 * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_values_tmp, dC_values, C_nnz1 * sizeof(float),
                        cudaMemcpyDeviceToHost) )

    if(benching) {state->PauseTiming();}

    taco::Tensor<float> out_gpu = taco::makeCSR("output_from_gpu", {dims[0], dims[0]}, &hC_csrOffsets_tmp[0], &hC_columns_tmp[0], &hC_values_tmp[0]);

    // Need to squeeze out some zeros
    auto out_gpu_no_zeros = out_gpu.removeExplicitZeros(taco::CSR);

    int * rowptr_out;
    int * colidx_out;
    float * vals_out;

    taco::getCSRArrays(out_gpu_no_zeros, &rowptr_out, &colidx_out, &vals_out);

    int correct = 1;
    for (int i = 0; i < A_num_rows + 1; i++) {
        if (rowptr_out[i] != hC_csrOffsets[i]) {
            std::cout << "ROWS GPU: " << rowptr_out[i] << " COMPARED TO CPU: " << hC_csrOffsets[i] << std::endl;
            correct = 0;
            break;
        }
    }

    for (int i = 0; i < C_nnz1; i++) {
        if (colidx_out[i] != hC_columns[i]){
            std::cout << "COL GPU: " << colidx_out[i] << " COMPARED TO CPU: " << hC_columns[i] << std::endl;
            correct = 0;
            break;
        }
        else if(!float_compare(vals_out[i], hC_values[i], 0.01f)) { // direct floating point
            std::cout << "VAL GPU: " << vals_out[i] << " COMPARED TO CPU: " << hC_values[i] << std::endl;
            correct = 0;                         // comparison is not reliable
            break;
        }
    }
    if (correct)
        printf("spgemm_example test PASSED\n");
    else {
        printf("spgemm_example test FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB_csrOffsets) )
    CHECK_CUDA( cudaFree(dB_columns) )
    CHECK_CUDA( cudaFree(dB_values) )
    CHECK_CUDA( cudaFree(dC_csrOffsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
    return EXIT_SUCCESS;
}

int sddmm(std::string mat_path){

    std::cout << "Calculating SDDMM" << std::endl;
    taco::TensorBase tb_denseA_double = taco::readMTX(mat_path, {taco::Dense, taco::Dense});
    auto dims = tb_denseA_double.getDimensions();

    //  Have double A, need to convert to float first...
    auto storage_A_double_arr = tb_denseA_double.getStorage().getValues();
    auto valsAsize = storage_A_double_arr.getSize();
    auto valsA_double = (double *) storage_A_double_arr.getData();
    float * valsA = (float *) malloc(valsAsize * sizeof(float));
    for(int i_ = 0; i_ < valsAsize; i_++){
        valsA[i_] = (float) valsA_double[i_];
    }

    taco::Tensor<float> tb_denseA(dims, {taco::Dense, taco::Dense}, 0);
    tb_denseA.setName("tb_denseA");
    for(int x = 0; x < dims[0]; x++){
        for(int y = 0; y < dims[1]; y++){
            tb_denseA.insert({x, y}, valsA[x * dims[1] + y]);
        }
    }
    taco::Tensor<float> tb_denseB = tb_denseA.transpose("tb_denseB", {1, 0}, {taco::Dense, taco::Dense});
    auto valsB = (float *) tb_denseB.getStorage().getValues().getData();

    // Compute C as A*B so dimensions match
    taco::IndexVar i, j, k;

    taco::Tensor<float> sparseC("tb_sparseC", {dims[0], dims[0]}, taco::CSR);
    sparseC(i, k) = tb_denseA(i, j) * tb_denseB(j, k);
    sparseC.compile();
    sparseC.assemble();
    sparseC.compute();

    // Remove 0's or else the GPU stuff doesn't work due to mismatches in allocs.
    auto sparseC_no_zeros = sparseC.removeExplicitZeros(taco::CSR);

    int * rowptrC;
    int * colidxC;
    float * valsC;

    taco::getCSRArrays(sparseC_no_zeros, &rowptrC, &colidxC, &valsC);

    auto storage = sparseC_no_zeros.getStorage();
    auto index = storage.getIndex();

    auto colidxArr = index.getModeIndex(1).getIndexArray(1);
    auto coloffsetsize = colidxArr.getSize();
    auto valsCsize = coloffsetsize;

    float * valsC_all_ones = (float *) malloc(valsCsize * sizeof(float));
    for(int z = 0; z < valsCsize; z++){
        valsC_all_ones[z] = 1.0f;
    }

    taco::Tensor<float> sparseC_no_zeros_sparsity_pattern = taco::makeCSR("tb_sparseC_sp_patt", {dims[0], dims[0]},
                                                                          rowptrC, colidxC, valsC_all_ones);

    // Compute result
    taco::IndexVar i_, j_, k_;
    taco::Tensor<float> expected("expected", {dims[0], dims[0]}, taco::CSR);
    expected(i_,j_) = sparseC_no_zeros(i_,j_) * tb_denseA(i_,k_) * tb_denseB(k_,j_);
    auto stmt = expected.getAssignment().concretize();
    stmt = stmt.reorder({i_, j_, k_});
    expected.compile(stmt);
    expected.assemble();
    expected.compute();

    // Get final output values
    auto vals_expected = (float *) expected.getStorage().getValues().getData();

    // std::cout << tb_denseA << std::endl;
    // std::cout << tb_denseB << std::endl;
    // std::cout << sparseC << std::endl;
    // std::cout << sparseC_no_zeros_sparsity_pattern << std::endl;
    // std::cout << expected << std::endl;

    // Host problem definition
    int   A_num_rows   = dims[0];
    int   A_num_cols   = dims[1];
    int   B_num_rows   = dims[1];
    int   B_num_cols   = dims[0];
    int   C_nnz        = valsCsize;
    int   lda          = A_num_cols;
    int   ldb          = B_num_cols;
    int   A_size       = lda * A_num_rows;
    int   B_size       = ldb * B_num_rows;
    float *hA          = valsA;
    float *hB          = valsB;
    int   *hC_offsets  = rowptrC;
    int   *hC_columns  = colidxC;
    float *hC_values   = valsC;
    float *hC_result   = vals_expected;

    thrust::device_vector<float> dC_values_mul(valsC, valsC + valsCsize);
    // float hA[]         = { 1.0f,   2.0f,  3.0f,  4.0f,
    //                     5.0f,   6.0f,  7.0f,  8.0f,
    //                     9.0f,  10.0f, 11.0f, 12.0f,
    //                     13.0f, 14.0f, 15.0f, 16.0f };
    // float hB[]         = {  1.0f,  2.0f,  3.0f,
    //                         4.0f,  5.0f,  6.0f,
    //                         7.0f,  8.0f,  9.0f,
    //                     10.0f, 11.0f, 12.0f };
    // int   hC_offsets[] = { 0, 3, 4, 7, 9 };
    // int   hC_columns[] = { 0, 1, 2, 1, 0, 1, 2, 0, 2 };
    // float hC_values[]  = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    //                     0.0f, 0.0f, 0.0f, 0.0f };
    // float hC_result[]  = { 70.0f, 80.0f, 90.0f,
    //                     184.0f,
    //                     246.0f, 288.0f, 330.0f,
    //                     334.0f, 450.0f };

    float alpha        = 1.0f;
    float beta         = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dC_offsets, *dC_columns;
    float *dC_values, *dB, *dA;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_offsets,
                        (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size * sizeof(float),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_offsets, hC_offsets,
                        (A_num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_columns, hC_columns, C_nnz * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_values, hC_values, C_nnz * sizeof(float),
                        cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix C in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, C_nnz,
                                    dC_offsets, dC_columns, dC_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSDDMM_bufferSize(
                                handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute preprocess (optional)
    CHECK_CUSPARSE( cusparseSDDMM_preprocess(
                                handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )
    // execute SpMM
    CHECK_CUSPARSE( cusparseSDDMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    thrust::device_vector<float> dC_values_thrust(dC_values, dC_values + valsCsize);
    //--------------------------------------------------------------------------

    // Create output vector
    thrust::device_vector<float> final_out(valsCsize);
    // device result check
    thrust::transform(dC_values_mul.begin(), dC_values_mul.end(), dC_values_thrust.begin(), dC_values_thrust.begin(),
                  thrust::multiplies<float>());

    // CHECK_CUDA( cudaMemcpy(hC_values, dC_values, C_nnz * sizeof(float),
    //                     cudaMemcpyDeviceToHost) )
    thrust::host_vector<float> final_out_local = dC_values_thrust;

    int correct = 1;
    for (int i = 0; i < C_nnz; i++) {
        if (!float_compare(final_out_local[i], hC_result[i], 0.01f)) {
            std::cout << "VALS GPU: " << final_out_local[i] << " COMPARED TO CPU: " << hC_result[i] << std::endl;
            correct = 0; // direct floating point comparison is not reliable
            break;
        }
    }
    if (correct)
        printf("sddmm_csr_example test PASSED\n");
    else
        printf("sddmm_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC_offsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
    return EXIT_SUCCESS;
}

int spmv(taco::Tensor<float> tensorA, taco::Tensor<float> tensorB_csr, benchmark::State * state){

    bool benching = (state != nullptr);

    if(benching) {state->PauseTiming();}

    std::cout << "Calculating SPMv" << std::endl;
    auto dims = tensorA.getDimensions();

    // Get A and its arrays to analyze sizes/etc
    int * rowptrA;
    int * colidxA;
    float * valsA;

    taco::getCSRArrays(tensorA, &rowptrA, &colidxA, &valsA);

    auto storage = tensorA.getStorage();
    auto index = storage.getIndex();
    auto rowptrArr = index.getModeIndex(1).getIndexArray(0);
    auto colidxArr = index.getModeIndex(1).getIndexArray(1);
    auto rowptrsize = rowptrArr.getSize();
    auto numrowsA = rowptrsize - 1;
    auto coloffsetsize = colidxArr.getSize();
    auto valssize = coloffsetsize;
    // Create the float version
    taco::Tensor<float> tensorB = tensorB_csr.transpose("tensorB_dense", {0}, taco::Dense);

    // Create random vector and zero vector
    float * zeroVec = (float *) malloc(dims[0] * sizeof(float));
    for(auto i = 0; i < dims[0]; i++){
        zeroVec[i] = 0.0f;
    }

    taco::IndexVar i, j;
    // Use taco to compute result
    taco::Tensor<float> expected("expected", {dims[0]}, {taco::Dense});
    expected(i) = tensorA(i, j) * tensorB(j);
    expected.compile();
    expected.assemble();
    expected.compute();

    auto vals_expected = (float *) expected.getStorage().getValues().getData();

    auto vec_vals = (float *) tensorB.getStorage().getValues().getData();

   // Host problem definition
    const int A_num_rows      = dims[0];
    const int A_num_cols      = dims[1];
    const int A_nnz           = valssize;
    int       *hA_csrOffsets = rowptrA;
    int       *hA_columns    = colidxA;
    float     *hA_values     = valsA;
    float     *hX            = vec_vals;
    float     *hY            = zeroVec;
    float     *hY_result     = vals_expected;

    float     alpha           = 1.0f;
    float     beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )

    if(benching) {state->ResumeTiming();}

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
                           cudaMemcpyDeviceToHost) )

    if(benching) {state->PauseTiming();}

    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (!float_compare(hY[i], hY_result[i], 0.01f)) { // direct floating point comparison is not
            correct = 0;             // reliable
            break;
        }
    }
    if (correct)
        printf("spmv_csr_example test PASSED\n");
    else
        printf("spmv_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    return EXIT_SUCCESS;

}

int mmadd(std::string mat_path){
    std::cout << "Calculating MMADD" << std::endl;
    taco::TensorBase tb = taco::readMTX(mat_path, taco::CSR);
    auto dims = tb.getDimensions();

    // Get A and its arrays to analyze sizes/etc
    int * rowptrA;
    int * colidxA;
    double * valsA_pre;

    taco::getCSRArrays(tb, &rowptrA, &colidxA, &valsA_pre);

    auto storage = tb.getStorage();
    auto index = storage.getIndex();
    auto rowptrArr = index.getModeIndex(1).getIndexArray(0);
    auto colidxArr = index.getModeIndex(1).getIndexArray(1);
    auto rowptrsize = rowptrArr.getSize();
    auto numrowsA = rowptrsize - 1;
    auto coloffsetsize = colidxArr.getSize();
    auto valssize = coloffsetsize;

    float * valsA = (float *) malloc(valssize * sizeof(float));
    for(int i_ = 0; i_ < valssize; i_++){
        valsA[i_] = (float) valsA_pre[i_];
    }

    // Create the float version
    taco::Tensor<float> tb_floatA = taco::makeCSR("tb_float", dims, rowptrA, colidxA, valsA);
    taco::Tensor<float> tb_floatB = tb_floatA.transpose("tb_floatB", {0, 1}, taco::CSR);

    int * rowptrB;
    int * colidxB;
    float * valsB;
    taco::getCSRArrays(tb_floatB, &rowptrB, &colidxB, &valsB);

    int NUM_I = dims[0];
    int NUM_K = dims[1];

    /*
        Compute the output of mmadd
    */
    taco::IndexVar i, k;

    taco::Tensor<float> expected("expected", {NUM_I, NUM_K}, taco::CSR);
    expected(i, k) = tb_floatA(i, k) + tb_floatB(i, k);
    expected.compile();
    expected.assemble();
    expected.compute();

    // Remove 0's or else the GPU stuff doesn't work due to mismatches in allocs.
    auto expected_no_zeros = expected.removeExplicitZeros(taco::CSR);

    // Get the information/ptrs from C for comparison and GPU calls
    int * rowptrC;
    int * colidxC;
    float * valsC;

    taco::getCSRArrays(expected_no_zeros, &rowptrC, &colidxC, &valsC);

    auto storageC = expected_no_zeros.getStorage();
    auto indexC = storageC.getIndex();

    auto rowptrArrC = indexC.getModeIndex(1).getIndexArray(0);
    auto colidxArrC = indexC.getModeIndex(1).getIndexArray(1);

    auto rowptrsizeC = rowptrArrC.getSize();
    auto coloffsetsizeC = colidxArrC.getSize();
    auto valssizeC = coloffsetsizeC;

    // // Host problem definition
    const int A_num_rows = dims[0];
    const int A_num_cols = dims[1];
    const int A_nnz      = valssize;
    const int B_num_rows = dims[1];
    const int B_num_cols = dims[0];
    const int B_nnz      = valssize;

    int  * hA_csrOffsets = rowptrA;
    int  * hA_columns    = colidxA;
    float* hA_values     = valsA;
    int  * hB_csrOffsets = rowptrB;
    int  * hB_columns    = colidxB;
    float* hB_values     = valsB;
    int   *hC_csrOffsets = rowptrC;
    int   *hC_columns    = colidxC;
    float *hC_values     = valsC;

    // const int C_nnz       = valssizeC;
    // #define   C_NUM_NNZ valssizeC   // C compatibility
    float               alpha       = 1.0f;
    float               beta        = 1.0f;
    // cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    // cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    // cudaDataType        computeType = CUDA_R_32F;
    // //--------------------------------------------------------------------------
    // // Device memory management: Allocate and copy A, B
    int   *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
        *dC_csrOffsets, *dC_columns;
    float *dA_values, *dB_values, *dC_values;
    // allocate A
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                        (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float)) )
    // allocate B
    CHECK_CUDA( cudaMalloc((void**) &dB_csrOffsets,
                        (B_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_columns, B_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dB_values,  B_nnz * sizeof(float)) )
    // allocate C offsets
    CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,
                        (A_num_rows + 1) * sizeof(int)) )
    // copy A
    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                        (A_num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                        A_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    // copy B
    CHECK_CUDA( cudaMemcpy(dB_csrOffsets, hB_csrOffsets,
                        (B_num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_columns, hB_columns, B_nnz * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_values, hB_values,
                        B_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    // //--------------------------------------------------------------------------
    // // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    cusparseMatDescr_t matA_nosp, matB_nosp, matC_nosp;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format

    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                    dA_csrOffsets, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                    dB_csrOffsets, dB_columns, dB_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                    dC_csrOffsets, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // Create mat descriptor
    CHECK_CUSPARSE( cusparseCreateMatDescr( &matA_nosp) )
    CHECK_CUSPARSE( cusparseCreateMatDescr( &matB_nosp) )
    CHECK_CUSPARSE( cusparseCreateMatDescr( &matC_nosp) )

    CHECK_CUSPARSE( cusparseScsrgeam2_bufferSizeExt(handle, dims[0], dims[1],
                                                    &alpha,
                                                    matA_nosp, valssize,
                                                    dA_values, dA_csrOffsets, dA_columns,
                                                    &beta,
                                                    matB_nosp, valssize,
                                                    dB_values, dB_csrOffsets, dB_columns,
                                                    matC_nosp,
                                                    dC_values, dC_csrOffsets, dC_columns,
                                                    &bufferSize1) )

    std::cout << "This many in outputs for C " << bufferSize1 << std::endl;
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )

    int nnzC;
    int *nnzptr = &nnzC;

    cusparseXcsrgeam2Nnz(handle, dims[0], dims[1],
            matA_nosp, valssize, dA_csrOffsets, dA_columns,
            matB_nosp, valssize, dB_csrOffsets, dB_columns,
            matC_nosp, dC_csrOffsets, nnzptr,
            dBuffer1);

    nnzC = *nnzptr;

    std::cout << "Number nonzeros in output according to cuda land..." << nnzC << std::endl;

    // // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, nnzC * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  nnzC * sizeof(float)) )

    cusparseScsrgeam2(handle, dims[0], dims[1],
            &alpha,
            matA_nosp, valssize,
            dA_values, dA_csrOffsets, dA_columns,
            &beta,
            matB_nosp, valssize,
            dB_values, dB_csrOffsets, dB_columns,
            matC_nosp,
            dC_values, dC_csrOffsets, dC_columns,
            dBuffer1);

    // // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    // //--------------------------------------------------------------------------
    // // device result check
    int   hC_csrOffsets_tmp[dims[0] + 1];
    int   hC_columns_tmp[nnzC];
    float hC_values_tmp[nnzC];
    CHECK_CUDA( cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
                        (A_num_rows + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_columns_tmp, dC_columns, nnzC * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_values_tmp, dC_values, nnzC * sizeof(float),
                        cudaMemcpyDeviceToHost) )

    int correct = 1;
    for (int i = 0; i < A_num_rows + 1; i++) {
        if (hC_csrOffsets_tmp[i] != hC_csrOffsets[i]) {
            std::cout << "ROWS GPU: " << hC_csrOffsets_tmp[i] << " COMPARED TO CPU: " << hC_csrOffsets[i] << std::endl;
            correct = 0;
            break;
        }
    }

    for (int i = 0; i < nnzC; i++) {
        if (hC_columns_tmp[i] != hC_columns[i]){
            std::cout << "COL GPU: " << hC_columns_tmp[i] << " COMPARED TO CPU: " << hC_columns[i] << std::endl;
            correct = 0;
            break;
        }
        else if(!float_compare(hC_values_tmp[i], hC_values[i], 0.01f)) { // direct floating point
            std::cout << "VAL GPU: " << hC_values_tmp[i] << " COMPARED TO CPU: " << hC_values[i] << std::endl;
            std::cout << fabs(hC_values_tmp[i] - hC_values[i]) << std::endl;
            // std::cout << percent_diff << std::endl;
            correct = 0;                         // comparison is not reliable
            break;
        }
    }
    if (correct)
        printf("spgemm_example test PASSED\n");
    else {
        printf("spgemm_example test FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB_csrOffsets) )
    CHECK_CUDA( cudaFree(dB_columns) )
    CHECK_CUDA( cudaFree(dB_values) )
    CHECK_CUDA( cudaFree(dC_csrOffsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
    return EXIT_SUCCESS;
}

int plus3(taco::Tensor<float> tensorA, taco::Tensor<float> tensorB, taco::Tensor<float> tensorD, benchmark::State * state){

    bool benching = (state != nullptr);

    if(benching) {state->PauseTiming();}

    std::cout << "Calculating PLUS3" << std::endl;
    auto dims = tensorA.getDimensions();

    // Get A and its arrays to analyze sizes/etc
    int * rowptrA;
    int * colidxA;
    float * valsA;

    taco::getCSRArrays(tensorA, &rowptrA, &colidxA, &valsA);

    auto storage = tensorA.getStorage();
    auto index = storage.getIndex();
    auto rowptrArr = index.getModeIndex(1).getIndexArray(0);
    auto colidxArr = index.getModeIndex(1).getIndexArray(1);
    auto rowptrsize = rowptrArr.getSize();
    auto numrowsA = rowptrsize - 1;
    auto coloffsetsize = colidxArr.getSize();
    auto valssize = coloffsetsize;

    int * rowptrB;
    int * colidxB;
    float * valsB;
    taco::getCSRArrays(tensorB, &rowptrB, &colidxB, &valsB);

    int * rowptrD;
    int * colidxD;
    float * valsD;
    taco::getCSRArrays(tensorD, &rowptrD, &colidxD, &valsD);

    int NUM_I = dims[0];
    int NUM_K = dims[1];

    /*
        Compute the output of PLUS3
    */
    taco::IndexVar i, k;

    taco::Tensor<float> expected("expected", {NUM_I, NUM_K}, taco::CSR);
    expected(i, k) = tensorA(i, k) + tensorB(i, k) + tensorD(i, k);
    expected.compile();
    expected.assemble();
    expected.compute();

    // Remove 0's or else the GPU stuff doesn't work due to mismatches in allocs.
    auto expected_no_zeros = expected.removeExplicitZeros(taco::CSR);

    // Get the information/ptrs from C for comparison and GPU calls
    int * rowptrC;
    int * colidxC;
    float * valsC;

    taco::getCSRArrays(expected_no_zeros, &rowptrC, &colidxC, &valsC);

    auto storageC = expected_no_zeros.getStorage();
    auto indexC = storageC.getIndex();

    auto rowptrArrC = indexC.getModeIndex(1).getIndexArray(0);
    auto colidxArrC = indexC.getModeIndex(1).getIndexArray(1);

    auto rowptrsizeC = rowptrArrC.getSize();
    auto coloffsetsizeC = colidxArrC.getSize();
    auto valssizeC = coloffsetsizeC;

    // // Host problem definition
    const int A_num_rows = dims[0];
    const int A_num_cols = dims[1];
    const int A_nnz      = valssize;
    const int B_num_rows = dims[0];
    const int B_num_cols = dims[1];
    const int B_nnz      = valssize;
    const int D_num_rows = dims[0];
    const int D_num_cols = dims[1];
    const int D_nnz      = valssize;

    int  * hA_csrOffsets = rowptrA;
    int  * hA_columns    = colidxA;
    float* hA_values     = valsA;
    int  * hB_csrOffsets = rowptrB;
    int  * hB_columns    = colidxB;
    float* hB_values     = valsB;
    int  * hD_csrOffsets = rowptrD;
    int  * hD_columns    = colidxD;
    float* hD_values     = valsD;
    int   *hC_csrOffsets = rowptrC;
    int   *hC_columns    = colidxC;
    float *hC_values     = valsC;

    float               alpha       = 1.0f;
    float               beta        = 1.0f;
    // //--------------------------------------------------------------------------
    // // Device memory management: Allocate and copy A, B, C, D, E
    int   *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
        *dC_csrOffsets, *dC_columns, *dD_csrOffsets, *dD_columns, *dE_csrOffsets, *dE_columns;
    float *dA_values, *dB_values, *dC_values, *dD_values, *dE_values;
    // allocate A
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                        (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float)) )
    // allocate B
    CHECK_CUDA( cudaMalloc((void**) &dB_csrOffsets,
                        (B_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_columns, B_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dB_values,  B_nnz * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dD_csrOffsets,
                        (D_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dD_columns, D_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dD_values,  D_nnz * sizeof(float)) )
    // allocate C offsets
    CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,
                        (A_num_rows + 1) * sizeof(int)) )
    // allocate E offsets
    CHECK_CUDA( cudaMalloc((void**) &dE_csrOffsets,
                        (A_num_rows + 1) * sizeof(int)) )
    // copy A
    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                        (A_num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
    // CHECK_CUDA( cudaMemcpy(dA_values, &hA_values,
                        A_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    // copy B
    CHECK_CUDA( cudaMemcpy(dB_csrOffsets, hB_csrOffsets,
                        (B_num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_columns, hB_columns, B_nnz * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_values, hB_values,
                        B_nnz * sizeof(float), cudaMemcpyHostToDevice) )

    // copy D
    CHECK_CUDA( cudaMemcpy(dD_csrOffsets, hD_csrOffsets,
                        (D_num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dD_columns, hD_columns, D_nnz * sizeof(int),
                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dD_values, hD_values,
                        D_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    // //--------------------------------------------------------------------------
    // // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC, matD, matE;
    cusparseMatDescr_t matA_nosp, matB_nosp, matC_nosp, matD_nosp, matE_nosp;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format

    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                    dA_csrOffsets, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                    dB_csrOffsets, dB_columns, dB_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                    dC_csrOffsets, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matD, A_num_rows, B_num_cols, 0,
                                    dD_csrOffsets, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matE, A_num_rows, B_num_cols, 0,
                                    dE_csrOffsets, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // Create mat descriptor
    CHECK_CUSPARSE( cusparseCreateMatDescr( &matA_nosp) )
    CHECK_CUSPARSE( cusparseCreateMatDescr( &matB_nosp) )
    CHECK_CUSPARSE( cusparseCreateMatDescr( &matC_nosp) )
    CHECK_CUSPARSE( cusparseCreateMatDescr( &matD_nosp) )
    CHECK_CUSPARSE( cusparseCreateMatDescr( &matE_nosp) )

    if(benching) {state->ResumeTiming();}

    // First computation...C = A + B
    CHECK_CUSPARSE( cusparseScsrgeam2_bufferSizeExt(handle, dims[0], dims[1],
                                                    &alpha,
                                                    matA_nosp, valssize,
                                                    dA_values, dA_csrOffsets, dA_columns,
                                                    &beta,
                                                    matB_nosp, valssize,
                                                    dB_values, dB_csrOffsets, dB_columns,
                                                    matC_nosp,
                                                    dC_values, dC_csrOffsets, dC_columns,
                                                    &bufferSize1) )

    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )

    int nnzC;
    int *nnzptr = &nnzC;

    cusparseXcsrgeam2Nnz(handle, dims[0], dims[1],
            matA_nosp, valssize, dA_csrOffsets, dA_columns,
            matB_nosp, valssize, dB_csrOffsets, dB_columns,
            matC_nosp, dC_csrOffsets, nnzptr,
            dBuffer1);

    nnzC = *nnzptr;

    // // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, nnzC * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  nnzC * sizeof(float)) )

    cusparseScsrgeam2(handle, dims[0], dims[1],
            &alpha,
            matA_nosp, valssize,
            dA_values, dA_csrOffsets, dA_columns,
            &beta,
            matB_nosp, valssize,
            dB_values, dB_csrOffsets, dB_columns,
            matC_nosp,
            dC_values, dC_csrOffsets, dC_columns,
            dBuffer1);

    // Second computation...E = C + D
    CHECK_CUSPARSE( cusparseScsrgeam2_bufferSizeExt(handle, dims[0], dims[1],
                                                    &alpha,
                                                    matC_nosp, nnzC,
                                                    dC_values, dC_csrOffsets, dC_columns,
                                                    &beta,
                                                    matD_nosp, valssize,
                                                    dD_values, dD_csrOffsets, dD_columns,
                                                    matE_nosp,
                                                    dE_values, dE_csrOffsets, dE_columns,
                                                    &bufferSize2) )

    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    int nnzE;
    int *nnzptrE = &nnzE;

    cusparseXcsrgeam2Nnz(handle, dims[0], dims[1],
            matC_nosp, nnzC, dC_csrOffsets, dC_columns,
            matD_nosp, valssize, dD_csrOffsets, dD_columns,
            matE_nosp, dE_csrOffsets, nnzptrE,
            dBuffer2);

    nnzE = *nnzptrE;

    // // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) &dE_columns, nnzE * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dE_values,  nnzE * sizeof(float)) )

    cusparseScsrgeam2(handle, dims[0], dims[1],
            &alpha,
            matC_nosp, nnzC,
            dC_values, dC_csrOffsets, dC_columns,
            &beta,
            matD_nosp, valssize,
            dD_values, dD_csrOffsets, dD_columns,
            matE_nosp,
            dE_values, dE_csrOffsets, dE_columns,
            dBuffer2);

    // // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matD) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matE) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    // //--------------------------------------------------------------------------
    // // device result check
    int   hE_csrOffsets_tmp[dims[0] + 1];
    int   hE_columns_tmp[nnzE];
    float hE_values_tmp[nnzE];
    CHECK_CUDA( cudaMemcpy(hE_csrOffsets_tmp, dE_csrOffsets,
                        (A_num_rows + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hE_columns_tmp, dE_columns, nnzE * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hE_values_tmp, dE_values, nnzE * sizeof(float),
                        cudaMemcpyDeviceToHost) )

    if(benching) {state->PauseTiming();}

    taco::Tensor<float> out_gpu = taco::makeCSR("output_from_gpu", {dims[0], dims[1]}, &hE_csrOffsets_tmp[0], &hE_columns_tmp[0], &hE_values_tmp[0]);

    // Need to squeeze out some zeros
    auto out_gpu_no_zeros = out_gpu.removeExplicitZeros(taco::CSR);

    int * rowptr_out;
    int * colidx_out;
    float * vals_out;

    taco::getCSRArrays(out_gpu_no_zeros, &rowptr_out, &colidx_out, &vals_out);

    int correct = 1;
    for (int i = 0; i < A_num_rows + 1; i++) {
        if (rowptr_out[i] != hC_csrOffsets[i]) {
            std::cout << "ROWS GPU: " << rowptr_out[i] << " COMPARED TO CPU: " << hC_csrOffsets[i] << std::endl;
            correct = 0;
            break;
        }
    }

    for (int i = 0; i < nnzE; i++) {
        if (colidx_out[i] != hC_columns[i]){
            std::cout << "COL GPU: " << colidx_out[i] << " COMPARED TO CPU: " << hC_columns[i] << std::endl;
            correct = 0;
            break;
        }
        else if(!float_compare(vals_out[i], hC_values[i], 0.01f)) { // direct floating point
            std::cout << "VAL GPU: " << vals_out[i] << " COMPARED TO CPU: " << hC_values[i] << std::endl;
            std::cout << fabs(vals_out[i] - hC_values[i]) << std::endl;
            correct = 0;                         // comparison is not reliable
            break;
        }
    }
    if (correct)
        printf("plus3 test PASSED\n");
    else {
        printf("plus3 test FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB_csrOffsets) )
    CHECK_CUDA( cudaFree(dB_columns) )
    CHECK_CUDA( cudaFree(dB_values) )
    CHECK_CUDA( cudaFree(dC_csrOffsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
    CHECK_CUDA( cudaFree(dD_csrOffsets) )
    CHECK_CUDA( cudaFree(dD_columns) )
    CHECK_CUDA( cudaFree(dD_values) )
    CHECK_CUDA( cudaFree(dE_csrOffsets) )
    CHECK_CUDA( cudaFree(dE_columns) )
    CHECK_CUDA( cudaFree(dE_values) )
    return EXIT_SUCCESS;
}

TensorInputCache<float> inputCacheFloat;

static void cusparse_benchmark(benchmark::State &state, SuiteSparseOp op, bool gen=true, int fill_value = 0) {

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

  taco::Tensor<float> tensorA, tensorB, tensorB_pre, tensorC;
//   try {
//     // taco::Format format = op == MATTRANSMUL ? DCSC : DCSR;
//     // std::tie(ssTensor, otherShifted) = inputCacheFloat.getTensorInput(tensorPath, tensorName, format, true /* countNNZ */,
//                                                                 //  true /* includeThird */, true, false, GEN_OTHER);
//     // std::string mat_path = "/home/max/Documents/SPARSE/GPU/mats/Zhao1/Zhao1.mtx";
//     // std::string mat_path = "/home/max/Documents/SPARSE/GPU/mats/fake/fake.mtx";
//     std::tie(tensorA, tensorB_pre) = inputCacheFloat.getTensorInput(mat_path, Zhao1, taco::CSR,
//                                                                        false, false, false, false, true);
//     tensorB = tensorB_pre.transpose("tensorB_transposed", {1, 0}, taco::CSR);

//   } catch (TacoException &e) {
//     // Counters don't show up in the generated CSV if we used SkipWithError, so
//     // just add in the label that this run is skipped.
//     std::cout << e.what() << std::endl;
//     state.SetLabel(tensorName + "/SKIPPED-FAILED-READ");
//     return;
//   }

//   taco::Tensor<int16_t> denseMat1;
//   taco::Tensor<int16_t> denseMat2;
//   taco::Tensor<int16_t> s1("s1"), s2("s2");
//   s1.insert({}, int16_t(2));
//   s2.insert({}, int16_t(2));
//   if (op == SDDMM) {
//     denseMat1 = Tensor<int16_t>("denseMat1", {DIM0, DIM_EXTRA}, Format({dense, dense}));
//     denseMat2 = Tensor<int16_t>("denseMat2", {DIM_EXTRA, DIM1}, Format({dense, dense}, {1, 0}));

//     // (owhsu) Making this dense matrices of all 1's
//     for (int kk = 0; kk < DIM_EXTRA; kk++) {
//       for (int ii = 0; ii < DIM0; ii++) {
//         denseMat1.insert({ii, kk}, int16_t(1));
//       }
//       for (int jj = 0; jj < DIM1; jj++) {
//         denseMat2.insert({kk, jj}, int16_t(1));
//       }
//     }
//   }
    // tensorPath = "/home/max/Documents/SPARSE/GPU/mats/relat3/relat3.mtx";
    // tensorName = "relat3";
    // tensorPath = "/home/max/Documents/SPARSE/GPU/mats/Zhao1/Zhao1.mtx";
    // tensorName = "Zhao1";

  for (auto _: state) {
    state.PauseTiming();
    switch (op) {
      case SPMV: {
        state.PauseTiming();
        std::tie(tensorA, tensorB_pre) = inputCacheFloat.getTensorInput(tensorPath, tensorName, taco::CSR,
                                                                       false, false, /* Include vec*/true,
                                                                       false, false);
        tensorB = inputCacheFloat.otherVecLastMode;
        int DIM0 = tensorA.getDimensions()[0];
        int DIM1 = tensorA.getDimensions()[1];

        state.counters["dimx"] = DIM0;
        state.counters["dimy"] = DIM1;
        state.counters["nnz"] = inputCacheFloat.nnz;

        state.ResumeTiming();
        spmv(tensorA, tensorB, &state);
        state.PauseTiming();
        break;
      }
      case SPMM: {
        state.PauseTiming();
        std::tie(tensorA, tensorB_pre) = inputCacheFloat.getTensorInput(tensorPath, tensorName, taco::CSR,
                                                                        false, false, false, false, true);
        tensorB = tensorB_pre.transpose("tensorB_transposed_csr", {1, 0}, taco::CSR);

        int DIM0 = tensorA.getDimensions()[0];
        int DIM1 = tensorA.getDimensions()[1];

        state.counters["dimx"] = DIM0;
        state.counters["dimy"] = DIM1;
        state.counters["nnz"] = inputCacheFloat.nnz;
        state.ResumeTiming();
        spgemm(tensorA, tensorB, &state);
        state.PauseTiming();
        // stmt = stmt.assemble(result.getAssignment().getLhs().getTensorVar(), taco::AssembleStrategy::Append);
        break;
      }
      case PLUS3: {
        state.PauseTiming();
        std::tie(tensorA, tensorB) = inputCacheFloat.getTensorInput(tensorPath, tensorName, taco::CSR,
                                                                    false, true, /* Include vec*/false,
                                                                    false, false);
        tensorC = inputCacheFloat.thirdTensor;

        int DIM0 = tensorA.getDimensions()[0];
        int DIM1 = tensorA.getDimensions()[1];

        state.counters["dimx"] = DIM0;
        state.counters["dimy"] = DIM1;
        state.counters["nnz"] = inputCacheFloat.nnz;

        state.ResumeTiming();
        plus3(tensorA, tensorB, tensorC, nullptr);
        state.PauseTiming();
        break;
      }
      default:
        state.SkipWithError("invalid expression");
        return;
    }

    // result.compile(stmt);

    // state.ResumeTiming();
    // result.assemble();
    // result.compute();
    // state.PauseTiming();

    // if (auto validationPath = getValidationOutputPath(); validationPath != "") {
    //   auto key = cpuBenchKey(tensorName, opName(op));
    //   auto outpath = validationPath + key + ".tns";
    //   taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
    // }
    // state.ResumeTiming();

  }
}

TACO_BENCH_ARGS(cusparse_benchmark, spmm_ , SPMM, true);
TACO_BENCH_ARGS(cusparse_benchmark, spmv_ , SPMV, true);
TACO_BENCH_ARGS(cusparse_benchmark, plus3_ , PLUS3, true);


#ifdef BENCH_CUSPARSE

BENCHMARK_MAIN();

#else

int main(int argc, char *argv[]) {

    if(argc > 3){
        std::cout << "BAD ARGS..." << std::endl;
        return 0;
    }

    std::string default_mat = "bcsstk01";

    if(argc > 1){
        default_mat = argv[1];
    }

    int default_op = 1;

    if(argc == 3){
        default_op = std::stoi(argv[2]);
    }

    assert(default_op >= 1);
    // assert((default_op >= 1) && (default_op <= 4));

    // Get mat path
    char * mat_path_base_ptr = std::getenv("MAT_PATH");
    assert(mat_path_base_ptr != NULL);
    std::string mat_path_base = mat_path_base_ptr;

    // Load in the matrix
    std::string mat_path = mat_path_base + "/" + default_mat + "/" + default_mat + ".mtx";
    std::cout << "Using path: " << mat_path << std::endl;

    TensorInputCache<float> inputCache;
    taco::Tensor<float> tensorA, tensorB_pre, tensorB, tensorC;

    // return 0;

    switch(default_op){
        // SPGEMM
        case 1:
            std::tie(tensorA, tensorB_pre) = inputCache.getTensorInput(mat_path, default_mat, taco::CSR,
                                                                       false, false, false, false, true);
            tensorB = tensorB_pre.transpose("tensorB_transposed_csr", {1, 0}, taco::CSR);
            return spgemm(tensorA, tensorB, nullptr);
        // SDDMM
        case 2:
            return sddmm(mat_path);
        // SPMV
        case 3:
            std::tie(tensorA, tensorB_pre) = inputCache.getTensorInput(mat_path, default_mat, taco::CSR,
                                                                       false, false, /* Include vec*/true,
                                                                       false, false);
            tensorB = inputCache.otherVecLastMode;

            return spmv(tensorA, tensorB, nullptr);
        // MMADD
        case 4:
            return mmadd(mat_path);
        // PLUS3
        case 5:
            std::tie(tensorA, tensorB) = inputCache.getTensorInput(mat_path, default_mat, taco::CSR,
                                                                       false, true, /* Include vec*/false,
                                                                       false, true);
            tensorC = inputCache.thirdTensor;
            // std::cout << tensorC << std::endl;
            return plus3(tensorA, tensorB, tensorC, nullptr);
        default:
            std::cout << "Invalid OP selected..." << std::endl;
    }

    return 0;

}

#endif