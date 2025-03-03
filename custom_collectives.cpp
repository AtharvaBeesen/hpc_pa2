#include <iostream> 
#include <mpi.h>

#include "custom_collectives.h"


/*
CUSTOM MANY2MANY
IDEA:
Send sendcounts to each other
Figure out sizes of input data + offsets and use this to create recv_data_ptr and create new array of offsets
(Then when you recv data, recv data into the specific offset of the recv_data_ptr)

THEN DO THE shit I started doing below off sending and recieving order based on rank
*/

int custom_many2many(int *send_data, int *sendcounts, int** recv_data_ptr, int rank, int size) {
    // ----------------------------------------------------------------
    //We need to know how many elements each other process will send to us.
    //From our perspective, the number of elements that process j sends to us is the jth element in their sendcount j
    //Imma create an array recv_counts of size size so that recv_counts[i] = number of elements process i sends to us.
    int* recv_counts = new int[size];
    recv_counts[rank] = sendcounts[rank]; // we send sendcounts[rank] to ourselves

    //(tag 1 for exchanging counts - we used 0 for hypercubic)
    //Loop over all processes and, for each process i != rank, exchange our info so we can fill in recv_counts
    //To avoid deadlock, we use the rank ordering:
    // - If our rank is lower than i, we send our count for process i first, then receive.
    // - If our rank is higher than i, we receive first, then send.
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;  // Skip self
        if (rank < i) {
            //Rank is lower than i.
            //Send our count for process i to process i.
            MPI_Send(&sendcounts[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            //Recv from process i its count for destination rank - i.e. how much its about to send to us
            MPI_Recv(&recv_counts[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            //Our rank is higher than i.
            //Same shit in reverse order
            MPI_Recv(&recv_counts[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&sendcounts[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    }


    //Now recv_counts[i] holds the number of elements that process i sends to us.

    // ----------------------------------------------------------------
    //Now calculate the total number of elements that we will receive.
    //Also, we need to create an offsets array so that data from process i will be stored starting at offset[i] in our final receive buffer.
    //I.e. recall that data needs to be ordered, so we need to know offset.

    int total_recv = 0; //Total info recieved
    int* offsets = new int[size];
    for (int i = 0; i < size; i++) {
        offsets[i] = total_recv; //For messages from offsets[i]. we need to start from the amount of messages that have been sent before
        total_recv += recv_counts[i]; //Also increment total_recv
    }

    // Now actually build out recv data pointer
    *recv_data_ptr = new int[total_recv];
    int* recv_data = *recv_data_ptr; // Imma Use this to iterate over recv data ptr

    // ----------------------------------------------------------------
    //If we have sendcounts as [3,2]
    // And we have send_data as [1,2,3,4,5]
    //Then we send 1,2,3 to proc0 and 4,5 to proc 1
    //In order to facilitate this, we need to create local offsets into send_data too.
    int* local_offsets = new int[size]; // Offsets
    int local_total = 0; // Total data we are sending(should equal the sum total of sendcounts)
    for (int i = 0; i < size; i++) { //Offset for sending data to i is the amt of data sent before we get to i.
        local_offsets[i] = local_total;
        local_total += sendcounts[i];
    }

    // ----------------------------------------------------------------
    //Now actually perform Many-to-Many Data Exchange
    //We now exchange the actual data. For each process i (from 0 to size-1). RANK GIVEN AS 0 TO N-1
    //Need to send the data chunk destined for process i and receive the data chunk that i sending to us
    //Bcos of deadlocks, again, order the communication so that smaller ranks send then recieve.
    //Tag = 2 - used 1 for counts and 0 for hypercubic
    for (int i = 0; i < size; i++) {
        if (i == rank) {
            //For self: simply copy the data from our local send buffer to our receive buffer.
            //The data intended for ourselves is in send_data starting at local_offsets[rank]
            //and has sendcounts[rank] elements. We copy it to recv_data starting at offsets[rank].
            int prev_index = local_offsets[rank];
            int new_index = offsets[rank];
            for (int j = 0; j < sendcounts[rank]; j++) {
                recv_data[new_index + j] = send_data[prev_index + j]; // Iterate through based on offset and deep copy over
            }
        } else {
            //Order the send/receive based on rank for communications w other procs
            if (rank < i) {
                //If our rank is lower than i, send our data first and then receive.]
                //Send -> send_data + local_offset so we know where to start from
                //Sendcounts tells us how much to send
                //MPT int is variable type
                //i = destination/source (other proc)
                //Tag = 2 - used 1 for counts and 0 for hypercubic
                MPI_Send(send_data + local_offsets[i], sendcounts[i], MPI_INT, i, 2, MPI_COMM_WORLD);
                MPI_Recv(recv_data + offsets[i], recv_counts[i], MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // If our rank is higher than i, same stuff reverse order.
                MPI_Recv(recv_data + offsets[i], recv_counts[i], MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_data + local_offsets[i], sendcounts[i], MPI_INT, i, 2, MPI_COMM_WORLD);
            }
        }
    }

    // ----------------------------------------------------------------
    //Clean up n free temporary arrays that are no longer needed.
    delete[] recv_counts;
    delete[] offsets;
    delete[] local_offsets;

    //Return the total number of elements received.
    return total_recv; // length of recv data arr
}




void custom_allreduce_sum(int *local, int *global, int num_elem, int rank, int size) {
  //CHECK IF WE ARE AN EXTRA PROC SHOULD WE SKIP HYPERCUBIC


  // Damn rank and size given

  // // First find rank and size
  // int rank, size;
  // MPI_Comm_rank(comm, &rank); // Get curr rank and size
  // MPI_Comm_size(comm, &size);

  //In order to perform a hypercube reduction, we need a power of two processors.
  //Need to first determine the largest 2^k (m) less than or equal to size.
  //This is our "main" group for the hypercube reduction.
  int main = 1;
  while (main * 2 <= size) {
      main *= 2;
  }
  //Now need to figure out number of processors more than this.
  int extra_count = size - main;

  //Eextra processors (ranks m to size-1)
  //Idea is to send data from these "extra" processors to a designated processor within the main hypercube group
  //Each extra processor w rank r sends to process (r - m) - so each main processors has to deal with at most one extra proc

  //--------------------------------------------------------------------------
  if (rank >= main) { // If the current processor is an extra one.
      int designated = rank - main;  // Map extra process to a main group process
      //Send the local array to the designated process.
      MPI_Send(local, num_elem, MPI_INT, designated, 0, MPI_COMM_WORLD);
      //After the main group computes the global sum, receive the final result
      MPI_Recv(global, num_elem, MPI_INT, designated, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // This occurs at the very end of fn
      //Extra processes are done after receiving the global sum - no longer need to deal with these as their numbers have been shipped off
      return;
  } else {
      //Main group proc
      //If this proc is designated to receive data from an extra process - i.e. it has rank 0 to extra_count-1
      //Then run that reception first.
      if (rank < extra_count) { // Check as per above
          int source = rank + main;  // The extra process it is recieving stuff from.
          int *temp_buf = new int[num_elem];
          //Receive the extra process's local arr
          MPI_Recv(temp_buf, num_elem, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          //Add the received data element-wise into current proc local array.
          for (int i = 0; i < num_elem; i++) {
              local[i] += temp_buf[i];
          }
          delete [] temp_buf;
      }
  }

  //--------------------------------------------------------------------------
  //Now time to actually run Hypercube reduction among the main group (ranks 0 to m-1)
  //log2(m) rounds.
  //In each round, each process exchanges its current partial sum with a partner (found via XORing the rank with 2^i for round i)
  //THen sum arrays
  int rounds = 0;
  int temp = main;
  while (temp > 1) { // Log base 2 finder lol
      rounds++;
      temp /= 2;
  }

  for (int i = 0; i < rounds; i++) { //Iterate over rounds
      //Find partner
      int partner = rank ^ (1 << i);
      
      //Temp buffer to hold partner's data.
      int *recv_buf = new int[num_elem];
 
      //Only allowed Send + Recv, so to avoid deadlock with blocking calls, order send/recv based on rank.
      if (rank < partner) {
          //Rrank is lower -> send first then receive.
          MPI_Send(local, num_elem, MPI_INT, partner, 0, MPI_COMM_WORLD);
          MPI_Recv(recv_buf, num_elem, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
          //Else -> receive first then send.
          MPI_Recv(recv_buf, num_elem, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Send(local, num_elem, MPI_INT, partner, 0, MPI_COMM_WORLD);
      }
      
      //Element-wise addition: combine received partial sum into our local sum.
      //The local sum array keeps getting filled
      for (int j = 0; j < num_elem; j++) {
          local[j] += recv_buf[j];
      }
      delete [] recv_buf;
  }

  //--------------------------------------------------------------------------
  //At this point, every process in the main group has the complete global sum.
  //Deep copy result into the global array.
  for (int i = 0; i < num_elem; i++) {
      global[i] = local[i];
  }

  //--------------------------------------------------------------------------
  //Now neeed to send the global result to extra processes (if any) -> recall we taked about this earlier when we were making extra procs
  //The designated main group processes send the final result to their corresponding extra proc.
  if (rank < extra_count) {
      int dest = rank + main;  // Corresponding to this main process.
      MPI_Send(global, num_elem, MPI_INT, dest, 0, MPI_COMM_WORLD);
  }

  //Now all procs should have the overall global sum
}
