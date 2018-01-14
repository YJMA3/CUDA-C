/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	// Multiply the two matrices
	
	// Calculate row index of the P element and M
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	
	// Calculate column index of P element and N
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	
	if ((Row < MATRIX_SIZE) && (Col < MATRIX_SIZE)) {
	float Pvalue = 0.0;
	
	// each thread computes one element of the block sub-matrix
	for(int k = 0; k < MATRIX_SIZE; ++k){
	Pvalue += M.elements[Row*MATRIX_SIZE+k] * N.elements[k*MATRIX_SIZE+Col];
	}
	P.elements[Row*MATRIX_SIZE+Col] = Pvalue;
}


}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
