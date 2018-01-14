#ifndef OPT_KERNEL
#define OPT_KERNEL
#define BLOCK_SIZE 512

void opt_2dhisto(uint32_t *input_data, uint32_t *input_bins);

/* Include below the function headers of any other functions that you implement */

uint8_t *AllocateDeviceMemory(int histo_width, int histo_height, int element_size);

void CopyToDevice(uint32_t *device_data, uint32_t *host_data, uint32_t input_height, uint32_t input_width, int element_size);

void CopyToHost(uint32_t *host, uint32_t *device, int size);

#endif
