#include <stdio.h>
#define BLOCK_SIZE 512

__global__ void spmv_csr_kernel(unsigned int dim, unsigned int *csrRowPtr, 
    unsigned int *csrColIdx, float *csrData, float *inVector, 
    float *outVector) {

    // INSERT KERNEL CODE HERE
	int row = blockDim.x*blockIdx.x+threadIdx.x;
	if (row < dim){
	float dot = 0;
	int row_start = csrRowPtr[row];
	int row_end = csrRowPtr[row+1];
	for (int jj = row_start; jj < row_end; jj++)
		dot += csrData[jj]*inVector[csrColIdx[jj]];
	outVector[row] += dot;
	}

}

__global__ void spmv_jds_kernel(unsigned int dim, unsigned int *jdsRowPerm, 
    unsigned int *jdsRowNNZ, unsigned int *jdsColStartIdx, 
    unsigned int *jdsColIdx, float *jdsData, float* inVector,
    float *outVector) {

     //INSERT KERNEL CODE HERE
	int row = blockDim.x*blockIdx.x+threadIdx.x;
	if (row < dim){
	float temp = 0.0;
	for (int jj = 0; jj < jdsRowNNZ[row]; jj++){
		unsigned int idx = row + jdsColStartIdx[jj];
		temp += jdsData[idx]*inVector[jdsColIdx[idx]];
	}
	outVector[jdsRowPerm[row]] = temp;
	}
}

void spmv_csr(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
    float *csrData, float *inVector, float *outVector) {

    // INSERT CODE HERE
	spmv_csr_kernel<<<((dim+ BLOCK_SIZE - 1)/ BLOCK_SIZE), BLOCK_SIZE>>>(dim, csrRowPtr, csrColIdx, csrData, inVector, outVector);

}

void spmv_jds(unsigned int dim, unsigned int *jdsRowPerm, unsigned int *jdsRowNNZ, 
    unsigned int *jdsColStartIdx, unsigned int *jdsColIdx, float *jdsData, 
    float* inVector, float *outVector) {

    // INSERT CODE HERE
	spmv_jds_kernel<<<((dim+ BLOCK_SIZE - 1)/ BLOCK_SIZE), BLOCK_SIZE>>>(dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData, inVector, outVector);
}






