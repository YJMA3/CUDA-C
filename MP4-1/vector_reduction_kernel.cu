#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

#define BLOCK_SIZE 512
// **===--------------------- Modify this function -----------------------===**
//! @param g_data  input data in global memory
//                  result is expected in index 0 of g_data
//! @param n        input number of elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(unsigned int *g_data, int n)
{
	__shared__ unsigned int partialSum[2*BLOCK_SIZE];
	unsigned int t = threadIdx.x;  
	unsigned int i = 2 * blockIdx.x * BLOCK_SIZE;
	
	
	if (i + t < n)
       partialSum[t] = g_data[i+t];
    else
       partialSum[t] = 0;
    if (i + BLOCK_SIZE + t < n)
       partialSum[BLOCK_SIZE + t] = g_data[i + BLOCK_SIZE + t];
    else
       partialSum[BLOCK_SIZE + t] = 0;
	   

	for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) 
	{ 
		__syncthreads(); 
		if (t < stride) 
			partialSum[t] += partialSum[t+stride]; 
	} 
	
	if (t == 0){ 
		g_data[blockIdx.x] = partialSum[0];
	}

}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
