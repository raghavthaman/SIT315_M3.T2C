#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <cmath>
#include <string>
#include <sstream>

// Standard namespace aliases for cleaner, more readable code
using std::cout;
using std::endl;
using std::left;
using std::right;
using std::setw;
using std::string;
using std::vector;

// Small tolerance for floating-point comparison when checking correctness.
// 1e-3 is safe for this problem because all input values are small integers (0-9).
static const float EPS = 1e-3f;

// Test matrix sizes we will run. These were chosen to be large enough to see
// meaningful timing differences while still finishing quickly on a typical student machine.
// 2048 is our "stress" case that will be used later for the OpenCL stress test requirement.
static const std::vector<int> kSizes = {256, 512, 1024};
static const int kStressSize = 2048;   // Change only this if you want a bigger stress test.

// -----------------------------------------------------------------------------
// Helper struct that describes how matrix rows are distributed across MPI ranks.
// We use this because N might not be perfectly divisible by the number of processes.
// This gives every rank a fair share (some ranks may get one extra row).
// -----------------------------------------------------------------------------
struct Layout
{
    vector<int> rowsPerRank;   // how many rows each rank is responsible for
    vector<int> counts;        // number of floats each rank will send/receive (rows * N)
    vector<int> displs;        // starting offset in the global matrix for each rank
};

// Builds the row distribution layout for Scatterv/Gatherv.
// This is a clean, general-purpose way to handle uneven workloads in MPI.
Layout buildLayout(int n, int worldSize)
{
    Layout layout;
    layout.rowsPerRank.resize(worldSize);
    layout.counts.resize(worldSize);
    layout.displs.resize(worldSize);

    int base = n / worldSize;
    int rem = n % worldSize;
    int offset = 0;

    for (int r = 0; r < worldSize; ++r)
    {
        int rows = base + (r < rem ? 1 : 0);        // give the first 'rem' ranks one extra row
        layout.rowsPerRank[r] = rows;
        layout.counts[r] = rows * n;
        layout.displs[r] = offset;
        offset += layout.counts[r];
    }

    return layout;
}

// Fills a matrix with random integers 0-9 (cast to float).
// We use a deterministic seed (based on matrix size) so every run produces the same
// input data - this makes correctness checking reproducible across machines.
void fillRandomMatrix(vector<float>& m, int n, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 9);

    for (int i = 0; i < n * n; ++i)
    {
        m[i] = static_cast<float>(dist(gen));
    }
}

// Pure sequential matrix multiplication (i * j * k triple loop).
// This is our trusted CPU baseline used for both timing and correctness verification.
void sequentialMultiply(const vector<float>& A, const vector<float>& B, vector<float>& C, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
            {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Compares two matrices element-wise. Returns true only if every value matches
// within the allowed floating-point tolerance. Used to prove the parallel result
// is numerically identical to the sequential version.
bool compareMatrices(const vector<float>& X, const vector<float>& Y)
{
    if (X.size() != Y.size()) return false;

    for (size_t i = 0; i < X.size(); ++i)
    {
        if (std::fabs(X[i] - Y[i]) > EPS)
        {
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Core MPI matrix multiplication routine (the actual parallel work).
// This function is called by EVERY rank. It contains the complete communication
// pattern: Bcast (full B matrix) + Scatterv (rows of A) + local computation + Gatherv.
// Timing is measured only around the computation phase (after data is distributed).
// -----------------------------------------------------------------------------
double runMPICase(
    int n,
    int rank,
    int worldSize,
    const Layout& layout,
    vector<float>& A,
    vector<float>& B,
    vector<float>& C)
{
    int localRows = layout.rowsPerRank[rank];
    vector<float> localA(localRows * n);           // only the rows this rank owns
    vector<float> localC(localRows * n, 0.0f);     // result buffer for this rank

    // Broadcast the entire B matrix to every rank (every process needs the full right-hand matrix)
    MPI_Bcast(B.data(), n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Scatter the rows of A using variable counts/displacements so the workload is balanced
    MPI_Scatterv(
        rank == 0 ? A.data() : nullptr,
        layout.counts.data(),
        layout.displs.data(),
        MPI_FLOAT,
        localA.data(),
        layout.counts[rank],
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    // Synchronise all ranks before starting the timed section
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Each rank multiplies only its local slice of A with the full B matrix
    for (int i = 0; i < localRows; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
            {
                sum += localA[i * n + k] * B[k * n + j];
            }
            localC[i * n + j] = sum;
        }
    }

    // Gather the partial results back into the full C matrix on rank 0
    MPI_Gatherv(
        localC.data(),
        layout.counts[rank],
        MPI_FLOAT,
        rank == 0 ? C.data() : nullptr,
        layout.counts.data(),
        layout.displs.data(),
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    // Return the slowest rank's time (so the reported MPI time is wall-clock time)
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return maxElapsed;
}

// Prints the nice table header for the results (only rank 0 does output)
void printHeader()
{
    cout << left
         << setw(12) << "Case"
         << setw(12) << "N"
         << setw(12) << "Procs"
         << setw(16) << "Seq(s)"
         << setw(16) << "MPI(s)"
         << setw(14) << "Speedup"
         << setw(10) << "Verify"
         << endl;
    cout << string(82, '-') << endl;
}

// Prints one row of the results table with proper formatting
void printRow(const string& label, int n, int procs, double seqTime, double mpiTime, bool ok)
{
    double speedup = seqTime / mpiTime;

    cout << left
         << setw(12) << label
         << setw(12) << n
         << setw(12) << procs
         << setw(16) << std::fixed << std::setprecision(6) << seqTime
         << setw(16) << std::fixed << std::setprecision(6) << mpiTime
         << setw(14) << std::fixed << std::setprecision(2) << speedup
         << setw(10) << (ok ? "PASS" : "FAIL")
         << endl;
}

// -----------------------------------------------------------------------------
// Main program entry point.
// Demonstrates the full MPI-only baseline as required by Task M3.T2C.
// Runs multiple matrix sizes, prints a clear performance table, and performs
// correctness verification against the sequential implementation.
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (rank == 0)
    {
        cout << "=== MPI-only Matrix Multiplication (Baseline) ===" << endl;
        cout << "MPI processes used: " << worldSize << endl;
        cout << "Test sizes: 256, 512, 1024 | Stress size: " << kStressSize << endl;
        cout << "This version uses Scatterv/Gatherv for perfect load balance." << endl;
        printHeader();
    }

    std::vector<int> sizes = kSizes;
    sizes.push_back(kStressSize);   // stress test is run last

    for (size_t idx = 0; idx < sizes.size(); ++idx)
    {
        int n = sizes[idx];
        string label = (n == kStressSize) ? "stress" : "test";

        Layout layout = buildLayout(n, worldSize);

        vector<float> A;
        vector<float> B(n * n);
        vector<float> C;
        vector<float> Cref;

        if (rank == 0)
        {
            A.resize(n * n);
            C.resize(n * n);
            Cref.resize(n * n);

            // Generate the same random matrices on every run for reproducibility
            fillRandomMatrix(A, n, 1000u + static_cast<unsigned>(n));
            fillRandomMatrix(B, n, 2000u + static_cast<unsigned>(n));

            // Run the trusted sequential version (used for both timing and verification)
            double seqStart = MPI_Wtime();
            sequentialMultiply(A, B, Cref, n);
            double seqEnd = MPI_Wtime();

            // Run the MPI parallel version
            double mpiTime = runMPICase(n, rank, worldSize, layout, A, B, C);

            // Verify that the parallel result matches the sequential result
            bool ok = compareMatrices(C, Cref);

            printRow(label, n, worldSize, seqEnd - seqStart, mpiTime, ok);
        }
        else
        {
            // Non-root ranks only participate in communication and local computation
            runMPICase(n, rank, worldSize, layout, A, B, C);
        }
    }

    MPI_Finalize();
    return 0;
}