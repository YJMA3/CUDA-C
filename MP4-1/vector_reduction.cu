#ifdef _WIN32
#  define NOMINMAX 
#endif

#define NUM_ELEMENTS 1000000
#define BLOCK_SIZE 512

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, kernels
#include "vector_reduction_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(unsigned int*, char* file_name);
unsigned int computeOnDevice(unsigned int* h_data, int array_mem_size);

extern "C" 
void computeGold( unsigned int* reference, unsigned int* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int num_elements = NUM_ELEMENTS;
    int errorM = 0;

    const unsigned int array_mem_size = sizeof( unsigned int) * num_elements;

    // allocate host memory to store the input data
    unsigned int* h_data = (unsigned int*) malloc( array_mem_size);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Read the input data array from the given file.
    switch(argc-1)
    {      
        case 1:  // One Argument
            errorM = ReadFile(h_data, argv[1]);
            if(errorM != num_elements)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                //h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
		h_data[i] = rand()%1000;
            }
        break;  
    }
    // compute reference solution
    unsigned int reference = 0;  
    computeGold(&reference , h_data, num_elements);
    
    // **===-------- Modify the body of this function -----------===**
    unsigned int result = computeOnDevice(h_data, num_elements);
    // **===-----------------------------------------------------------===**


    // We can use an epsilon of 0 since values are integral and in a range 
    // that can be exactly represented
    unsigned int epsilon = 0;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    //printf( "device: %f  host: %f\n", result, reference);
    printf( "device: %d  host: %d\n", result, reference);
    // cleanup memory
    free( h_data);
}

// Read a vector into M (already allocated) from file
int ReadFile(unsigned int* V, char* file_name)
{
    unsigned int data_read = NUM_ELEMENTS;
    FILE* input = fopen(file_name, "r");
    unsigned i = 0;
    for (i = 0; i < data_read; i++) 
        fscanf(input, "%d", &(V[i]));
    return data_read;
}

// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimentions, excutes kernel function, and copy result of reduction back
// to h_data.
// Note: unsigned int* h_data is both the input and the output of this function.
unsigned int computeOnDevice(unsigned int* h_data, int num_elements)
{
	unsigned int *d_data;
    cudaMalloc( (void**) &d_data, sizeof(unsigned int) * num_elements);

	cudaMemcpy(d_data, h_data, sizeof(unsigned int) * num_elements, cudaMemcpyHostToDevice);  	
	
	// Setup the execution configuration
	int GRID_SIZE = ceil(num_elements/(BLOCK_SIZE*2.0));
	int num = num_elements;
	
	
	while (num > 1){
		reduction<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, num);
		num = GRID_SIZE;
		GRID_SIZE = ceil(num/(BLOCK_SIZE*2.0));		
	}
		
	cudaMemcpy(h_data, d_data, sizeof(int) * num, cudaMemcpyDeviceToHost);  
	
	// placeholder
	cudaFree(d_data);
	return h_data[0];

}
