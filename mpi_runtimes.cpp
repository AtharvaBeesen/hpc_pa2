#include <iostream>
#include <mpi.h>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 3){
        if(rank == 0)
            cerr << "Usage: " << argv[0] 
                 << " <test_type: allreduce or m2m> <input_file> (OPTIONAL: <output_file>)" 
                 << endl;
        MPI_Finalize();
        return 1;
    }

    string test_type = argv[1];

    if(test_type == "allreduce"){
        // ---------------- Allreduce Test ----------------
        // Expected input file format:
        //   Line 1: number of processors (should equal MPI size)
        //   Line 2: number of elements per processor (n)
        //   Next p lines: each line contains n space-separated integers

        int p = 0, n = 0;
        int *full_data = nullptr; // Only used by rank 0

        if(rank == 0){
            ifstream infile(argv[2]);
            if(!infile.is_open()){
                cerr << "Error: could not open file " << argv[2] << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            infile >> p;
            assert(p == size);
            infile >> n;
            full_data = new int[p * n];
            for(int i = 0; i < p * n; i++){
                infile >> full_data[i];
            }
            infile.close();
        }

        // Broadcast the number of elements per processor.
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Each process gets its local array via MPI_Scatter.
        int *local_array = new int[n];
        MPI_Scatter(full_data, n, MPI_INT, local_array, n, MPI_INT, 0, MPI_COMM_WORLD);

        // Allocate memory for the global sum result.
        int *global_sum = new int[n];

        // Time the MPI_Allreduce call.
        double start = MPI_Wtime();
        MPI_Allreduce(local_array, global_sum, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        double end = MPI_Wtime();
        double local_time = end - start, max_time;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(rank == 0)
            cout << "MPI_Allreduce time: " << max_time << " seconds" << endl;

        // Optionally, write the output to a file.
        if(argc == 4 && rank == 0){
            ofstream outfile(argv[3]);
            for(int i = 0; i < n; i++){
                outfile << global_sum[i] << " ";
            }
            outfile << endl;
            outfile.close();
        }

        delete[] local_array;
        delete[] global_sum;
        if(rank == 0)
            delete[] full_data;
    }
    else if(test_type == "m2m"){
        // ---------------- Many-to-Many Test ----------------
        // Expected input file format:
        //   Line 1: number of processors (should equal MPI size)
        //   Next p lines: each line contains p integers (the sendcount array for that process)
        //   Next p lines: each line contains the concatenated send data for that process.
        int file_num = 0;
        vector<int> local_sendcounts(size, 0);
        vector<int> local_senddata;

        if(rank == 0){
            ifstream infile(argv[2]);
            if(!infile.is_open()){
                cerr << "Error: could not open file " << argv[2] << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            infile >> file_num;
            assert(file_num == size);
            // Read sendcount arrays for all processes.
            vector<vector<int>> all_sendcounts(size, vector<int>(size, 0));
            for(int i = 0; i < size; i++){
                for(int j = 0; j < size; j++){
                    infile >> all_sendcounts[i][j];
                }
            }
            // Read send data arrays for all processes.
            vector<vector<int>> all_senddata(size);
            for(int i = 0; i < size; i++){
                int total = 0;
                for(int j = 0; j < size; j++){
                    total += all_sendcounts[i][j];
                }
                all_senddata[i].resize(total);
                for(int k = 0; k < total; k++){
                    infile >> all_senddata[i][k];
                }
            }
            infile.close();
            // For rank 0, set local data.
            local_sendcounts = all_sendcounts[0];
            local_senddata = all_senddata[0];
            // Send the data to other processes.
            for(int proc = 1; proc < size; proc++){
                MPI_Send(all_sendcounts[proc].data(), size, MPI_INT, proc, 0, MPI_COMM_WORLD);
                int total = 0;
                for(int j = 0; j < size; j++){
                    total += all_sendcounts[proc][j];
                }
                MPI_Send(all_senddata[proc].data(), total, MPI_INT, proc, 1, MPI_COMM_WORLD);
            }
        }
        else{
            MPI_Recv(local_sendcounts.data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int total = 0;
            for(int j = 0; j < size; j++){
                total += local_sendcounts[j];
            }
            local_senddata.resize(total);
            MPI_Recv(local_senddata.data(), total, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Compute send displacements.
        vector<int> sdispls(size, 0);
        sdispls[0] = 0;
        for(int i = 1; i < size; i++){
            sdispls[i] = sdispls[i - 1] + local_sendcounts[i - 1];
        }

        // Exchange sendcounts with all processes to determine receive counts.
        vector<int> recvcounts(size, 0);
        MPI_Alltoall(local_sendcounts.data(), 1, MPI_INT,
                     recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Compute receive displacements.
        vector<int> rdispls(size, 0);
        rdispls[0] = 0;
        for(int i = 1; i < size; i++){
            rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
        }
        int total_recv = 0;
        for(int i = 0; i < size; i++){
            total_recv += recvcounts[i];
        }
        vector<int> recv_data(total_recv, 0);

        // Time the MPI_Alltoallv call.
        double start = MPI_Wtime();
        MPI_Alltoallv(local_senddata.data(), local_sendcounts.data(), sdispls.data(), MPI_INT,
                      recv_data.data(), recvcounts.data(), rdispls.data(), MPI_INT,
                      MPI_COMM_WORLD);
        double end = MPI_Wtime();
        double local_time = end - start, max_time;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(rank == 0)
            cout << "MPI_Alltoallv time: " << max_time << " seconds" << endl;

        // Optionally, write the received data to an output file.
        if(argc == 4 && rank == 0){
            ofstream outfile(argv[3]);
            for(int i = 0; i < total_recv; i++){
                outfile << recv_data[i] << " ";
            }
            outfile << endl;
            outfile.close();
        }
    }
    else{
        if(rank == 0)
            cerr << "Unknown test type. Use 'allreduce' or 'm2m'." << endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
