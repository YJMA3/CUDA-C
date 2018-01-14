#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"


/*__device__  void atomicADD(uint32_t *address, uint32_t val) {
    unsigned int *address_as_ull = (unsigned int *)address;
    unsigned int old = *address_as_ull, assumed;
    do {
        if(old>=UINT8_MAX)    
            break;
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed+val);
    } while(old != assumed);
}*/

__global__ void opt_2dhisto_kernel (uint32_t *input_data, int size, uint32_t *input_bins)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x ;
	int stride = blockDim.x*gridDim.x;
	while (id < size && input_bins[input_data[id]] < 255){
		atomicAdd (&input_bins[input_data[id]], 1);
		id += stride;
	}
	
	/*int id = blockIdx.x * blockDim.x + threadIdx.x ;
	int stride = blockDim.x*gridDim.x;
	while (id < size && input_bins[input_data[id]] < 255){
		atomicADD(&input_bins[input_data[id]], 1);
		id += stride;
	}*/
	/*__shared__ unsigned int histo_private[256];
		
		if (threadIdx.x < 256) histo_private[threadIdx.x] = 0;
		__syncthreads();

		int id = blockIdx.x * blockDim.x + threadIdx.x ;
		int stride = blockDim.x*gridDim.x;
		while (id < size){
			atomicADD (&histo_private[input_data[id]], 1);
			id += stride;
		}
		__syncthreads();

	  if (threadIdx.x < 256) 
		 atomicAdd(&(input_bins[threadIdx.x]), histo_private[threadIdx.x]);
	  __syncthreads();*/
	
}

void opt_2dhisto(uint32_t *input_data, uint32_t *input_bins)
{
    /* This function should only contain grid setup 
       code and a call to the GPU histogramming kernel. 
       Any memory allocations and transfers must be done 
       outside this function */
	int size = INPUT_HEIGHT * INPUT_WIDTH;
	// Zero out all the bins
    cudaMemset(input_bins, 0, sizeof(uint32_t) * HISTO_WIDTH);

    opt_2dhisto_kernel<<<((size+ BLOCK_SIZE - 1)/ BLOCK_SIZE), BLOCK_SIZE>>>(input_data, size, input_bins);
    cudaDeviceSynchronize(); 
    
}

/* Include below the implementation of any other functions you need */


// Copy host data to device
void CopyToDevice(uint32_t *device_data, uint32_t *host_data, uint32_t input_height, uint32_t input_width, int element_size)
{
    const size_t padded = (input_width + 128) & 0xFFFFFF80;
    size_t row = input_width * element_size;

    for(int i=0; i<input_height; i++)
    {
        cudaMemcpy(device_data, host_data, row, cudaMemcpyHostToDevice);
        device_data += input_width;
        host_data += (padded);
    }
}


// Copy device data to host
void CopyToHost(uint32_t *host_data, uint32_t *device_data, int element_size)
{
    cudaMemcpy(host_data, device_data, element_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < HISTO_WIDTH * HISTO_HEIGHT; i++)
        if(host_data[i] > 255){
			host_data[i] = 255;
		}
		else{
			host_data[i] = host_data[i];
		}
}


// Memory Allocation
uint8_t *AllocateDeviceMemory(int histo_width, int histo_height, int element_size)
{
    uint8_t *d_memory;
    cudaMalloc((void **)&d_memory, histo_width * histo_height * element_size);
    return d_memory;
}
