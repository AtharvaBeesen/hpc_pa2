#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cassert>
#include <fstream>

#include "custom_collectives.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* initialize the variables needed */
  int* send_data;
  int* sendcounts;
  int* recv_data_custom = nullptr;

  /* read the input file PE 0 and send all the processors the data they need */
  sendcounts = new int[size];
  
  /* make sure the command line arguments are 3 in number */
  if (argc < 2 || argc > 3) {
    if (rank == 0) std::cout << "Usage: " << argv[0] << " <input> (OPTIONAL: <output.txt>)" << std::endl;
    MPI_Finalize();
    return 1;
  }

  int sdata_size_in_file[size];
   std::ifstream input_file(argv[1]);

  if (rank == 0) {
    /* make sure the first integer in file is same as the size */
    int size_in_file;
    input_file >> size_in_file;
    assert(size_in_file == size);

    /* put the second lines in secondcounts array */
    for (int i = 0; i < size; ++i) input_file >> sendcounts[i];

    sdata_size_in_file[0] = 0;
    for (int i = 0; i < size; i++) sdata_size_in_file[0] += sendcounts[i];

    int tmp[size];
    for (int pe = 1; pe < size; pe++) {
      sdata_size_in_file[pe] = 0;
      for (int i = 0; i < size; i++) input_file >> tmp[i];
      for (int i = 0; i < size; i++) sdata_size_in_file[pe] += tmp[i];
      MPI_Send(tmp, size, MPI_INT, pe, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(sendcounts, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  /* allocate enough space for send_data array */
  int send_data_size = 0;
  for (int i = 0; i < size; ++i) send_data_size += sendcounts[i];
  send_data = new int[send_data_size];

  if (rank == 0) {
    /* fill your own send_data */
    for (int i = 0; i < send_data_size; ++i) input_file >> send_data[i];
    
    /* send the send_data of other processors to them */
    for (int pe = 1; pe < size; pe++) {
      int remote_sdata_size = sdata_size_in_file[pe];
      int* tmp = new int[remote_sdata_size];
      for (int i = 0; i < remote_sdata_size; ++i) input_file >> tmp[i];
      MPI_Send(tmp, remote_sdata_size, MPI_INT, pe, 0, MPI_COMM_WORLD);
      delete[] tmp;
    }
  } else {
    MPI_Recv(send_data, send_data_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  /* Call the custom function */
  double start2 = MPI_Wtime();
  int recv_num_custom = custom_many2many(send_data, sendcounts, &recv_data_custom, rank, size);
  double end2 = MPI_Wtime();
  double local_custom_time = end2 - start2, global_custom_time;
  MPI_Allreduce(&local_custom_time, &global_custom_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) printf("Time taken by custom function: %f\n", global_custom_time);

  if (argc == 2) {
    MPI_Finalize();
    return 0;
  }

  /* collect the information in PE 0 and check the output with the sample output */
  std::ifstream output_file(argv[2]);
  if (rank == 0) {
    int* sample_outline = new int[recv_num_custom];
  	/* read the first line and and put it in sample_outline vector */
    for (int i = 0; i < recv_num_custom; ++i) output_file >> sample_outline[i];
    /* check the output with the sample output */
    for (int i = 0; i < recv_num_custom; ++i) {
      if (recv_data_custom[i] != sample_outline[i]) {
        std::cout << "Wrong Output " << std::endl;
        exit(1);
      }
    }
    delete[] sample_outline;

    /* moving on to other PEs */
    for (int pe = 1; pe < size; pe++) {
      int remote_recv_count;
      MPI_Recv(&remote_recv_count, 1, MPI_INT, pe, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int* remote_recv_data = new int[remote_recv_count];
      MPI_Recv(remote_recv_data, remote_recv_count, MPI_INT, pe, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* reallocate enough memory in sample_outline */
      
      sample_outline = new int[remote_recv_count];

      /* read the next line and and put it in sample_outline vector */
      for (int i = 0; i < remote_recv_count; ++i) output_file >> sample_outline[i];

      /* check the output with the sample output */
      for (int i = 0; i < remote_recv_count; ++i) {
        if (remote_recv_data[i] != sample_outline[i]) {
          std::cout << "Wrong Output " << std::endl;
          exit(1);
      }
    }

      delete[] remote_recv_data;
      delete[] sample_outline;
    }
  } else {
    MPI_Send(&recv_num_custom, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(recv_data_custom, recv_num_custom, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  /* If the code has come this far, output is correct */
  if (rank == 0) std::cout << "Correct Output" << std::endl;

  /* free all dynamically allocated memory */
  delete[] send_data;
  delete[] sendcounts;
  delete[] recv_data_custom;

  /* MPI Finalize */
  MPI_Finalize();
  
  return 0;
}
