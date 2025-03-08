#include <iostream>
#include <mpi.h>
#include <cassert>
#include <fstream>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check command-line arguments.
    if (argc < 2 || argc > 3) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <input_file> (OPTIONAL: <output_file>)" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int n = 0;  // Number of elements per processor.
    int p = 0;  // Number of processors (should equal 'size').
    int *full_data = nullptr;  // Only used by rank 0.
    int *local_array = nullptr;  // Local array for each process.

    if (rank == 0) {
        std::ifstream infile(argv[1]);
        if (!infile.is_open()) {
            std::cerr << "Error: could not open file " << argv[1] << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        infile >> p;
        assert(p == size); // Make sure the file's processor count matches.
        infile >> n;
        // Allocate space for the entire input (p * n integers).
        full_data = new int[p * n];
        for (int i = 0; i < p * n; i++) {
            infile >> full_data[i];
        }
        infile.close();
    }

    // Broadcast the problem size 'n' to all processes.
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process allocates space for its local array.
    local_array = new int[n];

    // Scatter the full data from rank 0 so each process gets its own local array.
    MPI_Scatter(full_data, n, MPI_INT, local_array, n, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate space for the result.
    int *global_sum = new int[n];

    // Time the MPI_Allreduce call.
    double start = MPI_Wtime();
    MPI_Allreduce(local_array, global_sum, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double local_mpi_time = end - start;
    
    // Optionally, gather the maximum time across processes.
    double max_mpi_time;
    MPI_Reduce(&local_mpi_time, &max_mpi_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "MPI_Allreduce time: " << max_mpi_time << " seconds" << std::endl;
        // Optionally write the output to file.
        if (argc == 3) {
            std::ofstream outfile(argv[2]);
            for (int i = 0; i < n; i++) {
                outfile << global_sum[i] << " ";
            }
            outfile << std::endl;
            outfile.close();
        }
    }

    // Clean up.
    delete[] local_array;
    delete[] global_sum;
    if (rank == 0)
        delete[] full_data;

    MPI_Finalize();
    return 0;
}
