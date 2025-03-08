#include <iostream>
#include <mpi.h>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(argc < 3){
        if(rank == 0)
            std::cout << "Usage: " << argv[0] 
                      << " <test_type: allreduce or m2m> <input_file> (OPTIONAL: <output_file>)" 
                      << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    std::string test_type = argv[1];
    
    if(test_type == "allreduce"){
        // ******************** ALLREDUCE TEST ********************
        int LEN = 0;
        std::ifstream file(argv[2]);
        if(!file.is_open()){
            if(rank == 0)
                std::cerr << "Error: could not open file " << argv[2] << std::endl;
            MPI_Finalize();
            return 1;
        }
        
        // Rank 0 reads the number of processors and the problem size.
        if(rank == 0){
            int file_num_pes;
            file >> file_num_pes;
            assert(file_num_pes == size);
            file >> LEN;
        }
        MPI_Bcast(&LEN, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        int* local_array = new int[LEN];
        int* global_sum_mpi = new int[LEN];
        
        // Distribute the local arrays: Rank 0 reads its own array and then sends each subsequent line.
        if(rank == 0){
            for(int i = 0; i < LEN; i++){
                file >> local_array[i];
            }
            int* tmp = new int[LEN];
            for(int pe = 1; pe < size; pe++){
                for(int i = 0; i < LEN; i++){
                    file >> tmp[i];
                }
                MPI_Send(tmp, LEN, MPI_INT, pe, 0, MPI_COMM_WORLD);
            }
            delete[] tmp;
        } else {
            MPI_Recv(local_array, LEN, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Time the MPI_Allreduce call.
        double start = MPI_Wtime();
        MPI_Allreduce(local_array, global_sum_mpi, LEN, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        double end = MPI_Wtime();
        double local_time = end - start, global_time;
        MPI_Allreduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(rank == 0)
            printf("MPI_Allreduce time: %f seconds\n", global_time);
        
        // Optionally, write the output to file.
        if(argc == 4 && rank == 0){
            std::ofstream out(argv[3]);
            for(int i = 0; i < LEN; i++){
                out << global_sum_mpi[i] << " ";
            }
            out << std::endl;
            out.close();
        }
        
        delete[] local_array;
        delete[] global_sum_mpi;
    }
    else if(test_type == "m2m"){
        // ******************** MANY-TO-MANY TEST ********************
        // This test uses MPI_Alltoallv.
        // The input file format is as follows:
        //   1. The first integer: number of processors.
        //   2. Next, 'size' lines each containing 'size' integers: the send count array for each processor.
        //   3. Then, 'size' lines each containing the concatenated send data for that processor.
        
        int file_num_pes = 0;
        std::vector<int> local_sendcounts(size, 0);
        std::vector<int> local_senddata;
        std::vector< std::vector<int> > all_sendcounts;
        std::vector< std::vector<int> > all_senddata;
        
        if(rank == 0){
            std::ifstream file(argv[2]);
            if(!file.is_open()){
                std::cerr << "Error: could not open file " << argv[2] << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            file >> file_num_pes;
            assert(file_num_pes == size);
            
            // Read the send count arrays for each processor.
            all_sendcounts.resize(size, std::vector<int>(size,0));
            for(int i = 0; i < size; i++){
                for(int j = 0; j < size; j++){
                    file >> all_sendcounts[i][j];
                }
            }
            // Read the send data for each processor.
            all_senddata.resize(size);
            for(int i = 0; i < size; i++){
                int total = 0;
                for(int j = 0; j < size; j++){
                    total += all_sendcounts[i][j];
                }
                all_senddata[i].resize(total);
                for(int j = 0; j < total; j++){
                    file >> all_senddata[i][j];
                }
            }
            file.close();
            
            // For Rank 0, set local data.
            local_sendcounts = all_sendcounts[0];
            local_senddata = all_senddata[0];
            
            // Distribute the send count array and data for each other processor.
            for(int proc = 1; proc < size; proc++){
                MPI_Send(all_sendcounts[proc].data(), size, MPI_INT, proc, 0, MPI_COMM_WORLD);
                int total = 0;
                for(int j = 0; j < size; j++){
                    total += all_sendcounts[proc][j];
                }
                MPI_Send(all_senddata[proc].data(), total, MPI_INT, proc, 1, MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(local_sendcounts.data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int total = 0;
            for(int j = 0; j < size; j++){
                total += local_sendcounts[j];
            }
            local_senddata.resize(total);
            MPI_Recv(local_senddata.data(), total, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Compute send displacements for MPI_Alltoallv.
        std::vector<int> sdispls(size, 0);
        sdispls[0] = 0;
        for(int i = 1; i < size; i++){
            sdispls[i] = sdispls[i-1] + local_sendcounts[i-1];
        }
        
        // Exchange send counts to determine receive counts.
        std::vector<int> recvcounts(size, 0);
        MPI_Alltoall(local_sendcounts.data(), 1, MPI_INT,
                     recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // Compute receive displacements.
        std::vector<int> rdispls(size, 0);
        rdispls[0] = 0;
        for(int i = 1; i < size; i++){
            rdispls[i] = rdispls[i-1] + recvcounts[i-1];
        }
        int total_recv = 0;
        for(int i = 0; i < size; i++){
            total_recv += recvcounts[i];
        }
        std::vector<int> recv_data(total_recv, 0);
        
        // Time the MPI_Alltoallv call.
        double start = MPI_Wtime();
        MPI_Alltoallv(local_senddata.data(), local_sendcounts.data(), sdispls.data(), MPI_INT,
                      recv_data.data(), recvcounts.data(), rdispls.data(), MPI_INT, MPI_COMM_WORLD);
        double end = MPI_Wtime();
        double local_time = end - start, global_time;
        MPI_Allreduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(rank == 0)
            printf("MPI_Alltoallv time: %f seconds\n", global_time);
        
        // Optionally, write the received data to an output file.
        if(argc == 4 && rank == 0){
            std::ofstream out(argv[3]);
            for(int i = 0; i < total_recv; i++){
                out << recv_data[i] << " ";
            }
            out << std::endl;
            out.close();
        }
    }
    else{
        if(rank == 0)
            std::cerr << "Unknown test type. Use 'allreduce' or 'm2m'." << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    MPI_Finalize();
    return 0;
}
