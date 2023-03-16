#include <omp.h>

#define NUN_THREADS 8

void compress_int4_weight(void *weight, void *out, int n, int m)
{
    #pragma omp parallel for num_threads(NUN_THREADS)
	for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            (*(unsigned char*)(out + sizeof(unsigned char) * (i * m + j))) |= ((*(unsigned char*)(weight + sizeof(unsigned char) * (i * (m << 1) + (j << 1)))) << 4);
            (*(unsigned char*)(out + sizeof(unsigned char) * (i * m + j))) |= (((*(unsigned char*)(weight + sizeof(unsigned char) * (i * (m << 1) + ((j << 1) | 1)))) & 15));
        }
    }
}

void extract_int8_weight_to_float(void *weight, void *scale_list, void *out, int n, int m)
{
    #pragma omp parallel for num_threads(NUN_THREADS)
	for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
            (*(float*)(out + sizeof(float) * (i * m + j))) = (*(float*)(scale_list + sizeof(float) * i)) * (*(char*)(weight + sizeof(char) * (i * m + j)));
    }
}

void extract_int4_weight_to_float(void *weight, void *scale_list, void *out, int n, int m)
{
    #pragma omp parallel for num_threads(NUN_THREADS)
	for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            (*(float*)(out + sizeof(float) * (i * (m << 1) + (j << 1)))) = (*(float*)(scale_list + sizeof(float) * i)) * ((*(char*)(weight + sizeof(char) * (i * m + j))) >> 4);
            (*(float*)(out + sizeof(float) * (i * (m << 1) + ((j << 1) | 1)))) = (*(float*)(scale_list + sizeof(float) * i)) * (((char)((*(unsigned char*)(weight + sizeof(char) * (i * m + j))) << 4))>> 4);
        }
    }
}
