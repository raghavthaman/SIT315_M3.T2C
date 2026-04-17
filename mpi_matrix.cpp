#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <cmath>
#include <string>
#include <sstream>

using std::cout; using std::endl; using std::left; using std::right;
using std::setw; using std::string; using std::vector;

static const float EPS = 1e-3f; // tolerance for floating-point comparison (safe for 0-9 inputs)

static const std::vector<int> kSizes = {256, 512, 1024};
static const int kStressSize = 2048; // stress test size

struct Layout {
    vector<int> rowsPerRank;
    vector<int> counts;
    vector<int> displs;
};

// Build row distribution for Scatterv/Gatherv (handles N not divisible by worldSize)
Layout buildLayout(int n, int worldSize) {
    Layout layout;
    layout.rowsPerRank.resize(worldSize);
    layout.counts.resize(worldSize);
    layout.displs.resize(worldSize);
    int base = n / worldSize;
    int rem = n % worldSize;
    int offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        int rows = base + (r < rem ? 1 : 0);
        layout.rowsPerRank[r] = rows;
        layout.counts[r] = rows * n;
        layout.displs[r] = offset;
        offset += layout.counts[r];
    }
    return layout;
}

// Fill matrix with random integers 0-9 (deterministic seed for reproducibility)
void fillRandomMatrix(vector<float>& m, int n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 9);
    for (int i = 0; i < n * n; ++i) {
        m[i] = static_cast<float>(dist(gen));
    }
}

// Pure sequential matrix multiplication (trusted baseline)
void sequentialMultiply(const vector<float>& A, const vector<float>& B, vector<float>& C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

// Compare two matrices element-wise within EPS tolerance
bool compareMatrices(const vector<float>& X, const vector<float>& Y) {
    if (X.size() != Y.size()) return false;
    for (size_t i = 0; i < X.size(); ++i)
        if (std::fabs(X[i] - Y[i]) > EPS) return false;
    return true;
}

// Core MPI matrix multiplication routine (Bcast + Scatterv + local compute + Gatherv)
double runMPICase(int n, int rank, int worldSize, const Layout& layout,
                  vector<float>& A, vector<float>& B, vector<float>& C) {
    int localRows = layout.rowsPerRank[rank];
    vector<float> localA(localRows * n);
    vector<float> localC(localRows * n, 0.0f);

    MPI_Bcast(B.data(), n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, layout.counts.data(),
                 layout.displs.data(), MPI_FLOAT, localA.data(),
                 layout.counts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int i = 0; i < localRows; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
                sum += localA[i * n + k] * B[k * n + j];
            localC[i * n + j] = sum;
        }

    MPI_Gatherv(localC.data(), layout.counts[rank], MPI_FLOAT,
                rank == 0 ? C.data() : nullptr, layout.counts.data(),
                layout.displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return maxElapsed;
}

// Print table header (rank 0 only)
void printHeader() {
    cout << left << setw(12) << "Case" << setw(12) << "N" << setw(12) << "Procs"
         << setw(16) << "Seq(s)" << setw(16) << "MPI(s)" << setw(14) << "Speedup"
         << setw(10) << "Verify" << endl;
    cout << string(82, '-') << endl;
}

// Print one results row
void printRow(const string& label, int n, int procs, double seqTime, double mpiTime, bool ok) {
    double speedup = seqTime / mpiTime;
    cout << left << setw(12) << label << setw(12) << n << setw(12) << procs
         << setw(16) << std::fixed << std::setprecision(6) << seqTime
         << setw(16) << std::fixed << std::setprecision(6) << mpiTime
         << setw(14) << std::fixed << std::setprecision(2) << speedup
         << setw(10) << (ok ? "PASS" : "FAIL") << endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (rank == 0) {
        cout << "=== MPI-only Matrix Multiplication (Baseline) ===" << endl;
        cout << "MPI processes used: " << worldSize << endl;
        cout << "Test sizes: 256, 512, 1024 | Stress size: " << kStressSize << endl;
        printHeader();
    }

    std::vector<int> sizes = kSizes;
    sizes.push_back(kStressSize);

    for (size_t idx = 0; idx < sizes.size(); ++idx) {
        int n = sizes[idx];
        string label = (n == kStressSize) ? "stress" : "test";

        Layout layout = buildLayout(n, worldSize);

        vector<float> A, B(n * n), C, Cref;

        if (rank == 0) {
            A.resize(n * n);
            C.resize(n * n);
            Cref.resize(n * n);

            fillRandomMatrix(A, n, 1000u + static_cast<unsigned>(n));
            fillRandomMatrix(B, n, 2000u + static_cast<unsigned>(n));

            double seqStart = MPI_Wtime();
            sequentialMultiply(A, B, Cref, n);
            double seqEnd = MPI_Wtime();

            double mpiTime = runMPICase(n, rank, worldSize, layout, A, B, C);

            bool ok = compareMatrices(C, Cref);
            printRow(label, n, worldSize, seqEnd - seqStart, mpiTime, ok);
        } else {
            runMPICase(n, rank, worldSize, layout, A, B, C);
        }
    }

    MPI_Finalize();
    return 0;
}
