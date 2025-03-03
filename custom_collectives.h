#ifndef CUSTOM_COLLECTIVES_H
#define CUSTOM_COLLECTIVES_H

#include <mpi.h>

void custom_allreduce_sum(int *local_array, int *global_sum, int num_elem, int rank, int size);

/* Note: 
 * 1. recv_data_ptr is a pointer to the recv_data array (which itself is a pointer to element recv_data[0]).
 * 2. return type of _many2many function is an int which is the length of the recv_data array after many2many function is performend.
 * */

int custom_many2many(int *send_data, int *sendcounts, int** recv_data_ptr, int rank, int size);

#endif
