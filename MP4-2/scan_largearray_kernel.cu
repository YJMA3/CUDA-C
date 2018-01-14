#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define TILE_SIZE 1024
// You can use any other block size you wish.
#define BLOCK_SIZE 256



// Kernel Functions

//Exclusive Scan
__global__ void Ex_Scan_Kernel(unsigned int *blockSum, unsigned int *output_Kernel, unsigned int *input_Kernel, int numElements) {
	__shared__ unsigned int scan_array[2*BLOCK_SIZE];
	unsigned int i = 2 * blockIdx.x * BLOCK_SIZE;
	unsigned int t = threadIdx.x;
	unsigned int ls_element;
	
	if (i + t < numElements) {
		scan_array[t] = input_Kernel[i+t];
	}
	else
       scan_array[t] = 0; 
    if (i + BLOCK_SIZE + t < numElements)
       scan_array[BLOCK_SIZE + t] = input_Kernel[i + BLOCK_SIZE + t];
    else
       scan_array[BLOCK_SIZE + t] = 0;
       
    //Save the last element of each block for further use
    __syncthreads();
    if(t==0) 
        ls_element = scan_array[2*blockDim.x-1];
    __syncthreads();
	

	//Start reduction
	int stride = 1;
    while(stride <= BLOCK_SIZE)
    {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < 2*BLOCK_SIZE)
            scan_array[index] += scan_array[index-stride];
        stride = stride*2;
        __syncthreads();
    }
	
	//Start postscan
	if (t == 0){
		scan_array[2*blockDim.x-1] = 0;
		if(blockIdx.x ==0)
            blockSum[0] = 0;
	}
	
	stride = BLOCK_SIZE;
	while(stride > 0) 
    {     
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < 2* BLOCK_SIZE) 
        {
            float temp = scan_array[index];
            scan_array[index] += scan_array[index-stride]; 
            scan_array[index-stride] = temp; 
        } 
        stride = stride / 2;
        __syncthreads();
    } 
	
	//Write the scan result back to the array
	if(i + t < numElements){
		output_Kernel[i + t] = scan_array[t];
	}
	else{
		output_Kernel[i + t] = 0;
	}
	if(i + BLOCK_SIZE + t < numElements){
		output_Kernel[i + BLOCK_SIZE + t] = scan_array[t + BLOCK_SIZE];
	}
	else{
		output_Kernel[i + BLOCK_SIZE + t] = 0;
	}
	
	if (t == 0) {
		blockSum[blockIdx.x] = scan_array[2*blockDim.x-1] + ls_element;
	}
	
}

//Vector Addition for the final result
__global__ void vector_addition(unsigned int *outArray, unsigned int *inArray, int numElements)
{
    __shared__ unsigned int addition;
    int index = 2* blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if(threadIdx.x==0) {
		addition = inArray[blockIdx.x];
	}
    __syncthreads();
    
    if(index < numElements)
    {
        outArray[index] += addition;
        outArray[index + BLOCK_SIZE] += addition;

    }
}

//Make a recursive function of exclusive scan
void Recursive_Ex_Scan(unsigned int *outArray, int numElements)
{
    unsigned int *blockSum;

    int GRID_SIZE_Re = ceil(numElements/(2.0*BLOCK_SIZE));
    cudaMalloc( (void**) &blockSum, sizeof(unsigned int) * (GRID_SIZE_Re+1));
   
    //Exclusive Scan the array
    Ex_Scan_Kernel<<<GRID_SIZE_Re, BLOCK_SIZE>>>(blockSum, outArray, outArray, numElements);

    //Do recursive exclusive scan and return the final result
    if(GRID_SIZE_Re > 1)
    {
        Recursive_Ex_Scan(blockSum, GRID_SIZE_Re);
        vector_addition<<<GRID_SIZE_Re , BLOCK_SIZE>>>(outArray,blockSum,numElements);
    }
	cudaFree(blockSum);
}


// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE
void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{

	int GRID_SIZE = ceil(numElements/(2.0*BLOCK_SIZE));
	//int num = numElements;

    unsigned int *blockSum;    
    cudaMalloc( (void**) &blockSum, sizeof(unsigned int) * GRID_SIZE);
    
    Ex_Scan_Kernel<<<GRID_SIZE, BLOCK_SIZE>>>(blockSum, outArray, inArray, numElements);
    
    if(GRID_SIZE > 1)
    {	   
		Recursive_Ex_Scan(blockSum, GRID_SIZE);
		vector_addition<<<GRID_SIZE, BLOCK_SIZE>>>(outArray, blockSum, numElements);
	}
	cudaFree(blockSum);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
