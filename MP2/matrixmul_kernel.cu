/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

#define TILE_WIDTH 16

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	__shared__ float M_s[TILE_WIDTH][TILE_WIDTH];
	__shared__ float N_s[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	//int w = M.width/TILE_WIDTH;
	//int r = M.width % TILE_WIDTH;
	
	//if (r){w++;}
	
	// Identify the row and column of the P_d element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	// Loop over the M_ and N_ tiles to compute the P_d element
	for (int m = 0; m < (TILE_WIDTH + M.width-1)/TILE_WIDTH; ++m) {
		// Collaborative loading of M_d and N_d tiles into shared memory
		if (Row < M.height && m*TILE_WIDTH+tx < M.width)
          M_s[ty][tx] = M.elements[Row*M.width + m*TILE_WIDTH+tx];
       else
          M_s[ty][tx] = 0;
       __syncthreads();
       if (Col < N.width && m*TILE_WIDTH+ty < N.height)
          N_s[ty][tx] = N.elements[(m*TILE_WIDTH+ty)*N.width+Col];
       else
          N_s[ty][tx] = 0;
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
		Pvalue += M_s[ty][k] * N_s[k][tx];
		__syncthreads();
	}
	if (Row < P.height && Col < P.width)
		P.elements[Row*P.width+Col] = Pvalue;

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
