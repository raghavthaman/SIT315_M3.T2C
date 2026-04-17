#define CL_TARGET_OPENCL_VERSION 120

#include <mpi.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <cmath>
#include <string>
#include <sstream>
#include <cstring>
#include <stdexcept>

using std::cout; using std::endl; using std::left; using std::setw; using std::string; using std::vector;

static const float EPS = 1e-3f;
static const std::vector<int> kSizes = {256, 512, 1024};
static const int kStressSize = 2048;
static const size_t LOCAL_X = 16, LOCAL_Y = 16;

struct Layout { vector<int> rowsPerRank, counts, displs; };
struct OpenCLState { cl_platform_id platform = nullptr; cl_device_id device = nullptr; cl_context context = nullptr; cl_command_queue queue = nullptr; cl_program program = nullptr; cl_kernel kernel = nullptr; string deviceName, deviceType; };

// Build row distribution for Scatterv/Gatherv
Layout buildLayout(int n, int worldSize) {
    Layout layout; layout.rowsPerRank.resize(worldSize); layout.counts.resize(worldSize); layout.displs.resize(worldSize);
    int base = n / worldSize, rem = n % worldSize, offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        int rows = base + (r < rem ? 1 : 0);
        layout.rowsPerRank[r] = rows; layout.counts[r] = rows * n; layout.displs[r] = offset; offset += layout.counts[r];
    }
    return layout;
}

// Fill matrix with random 0-9 (deterministic)
void fillRandomMatrix(vector<float>& m, int n, unsigned seed) {
    std::mt19937 gen(seed); std::uniform_int_distribution<int> dist(0,9);
    for (int i = 0; i < n*n; ++i) m[i] = static_cast<float>(dist(gen));
}

// Sequential matrix multiplication (baseline)
void sequentialMultiply(const vector<float>& A, const vector<float>& B, vector<float>& C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += A[i*n+k] * B[k*n+j];
            C[i*n+j] = sum;
        }
}

// Compare matrices within tolerance
bool compareMatrices(const vector<float>& X, const vector<float>& Y) {
    if (X.size() != Y.size()) return false;
    for (size_t i = 0; i < X.size(); ++i) if (std::fabs(X[i]-Y[i]) > EPS) return false;
    return true;
}

// Round up for OpenCL global size
size_t roundUp(size_t value, size_t multiple) {
    if (multiple == 0) return value;
    size_t rem = value % multiple;
    return rem == 0 ? value : value + (multiple - rem);
}

string deviceTypeToString(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_GPU) return "GPU";
    if (type & CL_DEVICE_TYPE_CPU) return "CPU";
    if (type & CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
    return "OTHER";
}

void checkCLError(cl_int err, const string& where) {
    if (err != CL_SUCCESS) {
        std::ostringstream oss; oss << "OpenCL error at " << where << " (code " << err << ")";
        throw std::runtime_error(oss.str());
    }
}

void printBuildLog(cl_program program, cl_device_id device) {
    size_t logSize = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    vector<char> log(logSize+1, '\0');
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    cout << "\nOpenCL build log:\n" << log.data() << endl;
}

// Initialise OpenCL
OpenCLState initOpenCL(int rank) {
    OpenCLState st; cl_int err = CL_SUCCESS;
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    bool found = false; cl_device_type chosenType = 0;
    for (cl_uint p = 0; p < numPlatforms && !found; ++p) {
        cl_platform_id plat = platforms[p]; cl_uint numDevices = 0; cl_device_id dev = nullptr;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &numDevices) == CL_SUCCESS && numDevices > 0) {
            st.platform = plat; st.device = dev; chosenType = CL_DEVICE_TYPE_GPU; found = true;
        } else if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 1, &dev, &numDevices) == CL_SUCCESS && numDevices > 0) {
            st.platform = plat; st.device = dev; chosenType = CL_DEVICE_TYPE_CPU; found = true;
        }
    }
    if (!found) throw std::runtime_error("No OpenCL device found.");

    char nameBuf[256] = {0};
    clGetDeviceInfo(st.device, CL_DEVICE_NAME, sizeof(nameBuf), nameBuf, nullptr);
    st.deviceName = nameBuf; st.deviceType = deviceTypeToString(chosenType);

    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)st.platform, 0};
    st.context = clCreateContext(props, 1, &st.device, nullptr, nullptr, &err);
    st.queue = clCreateCommandQueue(st.context, st.device, 0, &err);

    const char* kernelSource = R"CLC(
    __kernel void matrix_mult(__global const float* A, __global const float* B, __global float* C, const int N, const int localRows) {
        int row = get_global_id(0), col = get_global_id(1);
        if (row < localRows && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) sum += A[row*N+k] * B[k*N+col];
            C[row*N+col] = sum;
        }
    }
    )CLC";

    size_t srcLen = std::strlen(kernelSource);
    st.program = clCreateProgramWithSource(st.context, 1, &kernelSource, &srcLen, &err);
    clBuildProgram(st.program, 1, &st.device, nullptr, nullptr, nullptr);
    if (clGetProgramBuildInfo(st.program, st.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &srcLen) == CL_SUCCESS && srcLen > 1)
        printBuildLog(st.program, st.device);  // only print if log exists
    st.kernel = clCreateKernel(st.program, "matrix_mult", &err);

    if (rank == 0) {
        cout << "MPI + OpenCL Matrix Multiplication\nSelected device: " << st.deviceName << " (" << st.deviceType << ")\n"
             << "Sizes: 256, 512, 1024, stress=" << kStressSize << "\nKernel config: local size = " << LOCAL_X << " x " << LOCAL_Y << "\n";
    }
    return st;
}

void releaseOpenCL(OpenCLState& st) {
    if (st.kernel) clReleaseKernel(st.kernel);
    if (st.program) clReleaseProgram(st.program);
    if (st.queue) clReleaseCommandQueue(st.queue);
    if (st.context) clReleaseContext(st.context);
}

// Run one MPI+OpenCL case
double runOpenCLCase(int n, int rank, int worldSize, const Layout& layout, const OpenCLState& st, vector<float>& A, vector<float>& B, vector<float>& C) {
    int localRows = layout.rowsPerRank[rank];
    vector<float> localA(localRows*n), localC(localRows*n, 0.0f);

    MPI_Bcast(B.data(), n*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank==0?A.data():nullptr, layout.counts.data(), layout.displs.data(), MPI_FLOAT, localA.data(), layout.counts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    cl_int err; cl_mem bufA = clCreateBuffer(st.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*localA.size(), localA.data(), &err);
    cl_mem bufB = clCreateBuffer(st.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*B.size(), B.data(), &err);
    cl_mem bufC = clCreateBuffer(st.context, CL_MEM_WRITE_ONLY, sizeof(float)*localC.size(), nullptr, &err);

    clSetKernelArg(st.kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(st.kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(st.kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(st.kernel, 3, sizeof(int), &n);
    clSetKernelArg(st.kernel, 4, sizeof(int), &localRows);

    size_t local[2] = {LOCAL_X, LOCAL_Y};
    size_t global[2] = {roundUp(localRows, LOCAL_X), roundUp(n, LOCAL_Y)};

    MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
    clEnqueueNDRangeKernel(st.queue, st.kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
    clFinish(st.queue);
    clEnqueueReadBuffer(st.queue, bufC, CL_TRUE, 0, sizeof(float)*localC.size(), localC.data(), 0, nullptr, nullptr);
    MPI_Gatherv(localC.data(), layout.counts[rank], MPI_FLOAT, rank==0?C.data():nullptr, layout.counts.data(), layout.displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    clReleaseMemObject(bufA); clReleaseMemObject(bufB); clReleaseMemObject(bufC);
    return maxElapsed;
}

void printHeader() {
    cout << left << setw(12) << "Case" << setw(12) << "N" << setw(12) << "Procs"
         << setw(16) << "Seq(s)" << setw(16) << "MPI+OCL(s)" << setw(14) << "Speedup" << setw(10) << "Verify" << endl
         << std::string(82, '-') << endl;
}

void printRow(const string& label, int n, int procs, double seqTime, double oclTime, bool ok) {
    cout << left << setw(12) << label << setw(12) << n << setw(12) << procs
         << setw(16) << std::fixed << std::setprecision(6) << seqTime
         << setw(16) << std::fixed << std::setprecision(6) << oclTime
         << setw(14) << std::fixed << std::setprecision(2) << (seqTime/oclTime)
         << setw(10) << (ok ? "PASS" : "FAIL") << endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    try {
        OpenCLState ocl = initOpenCL(rank);
        if (rank == 0) { cout << "MPI processes used: " << worldSize << endl; printHeader(); }

        std::vector<int> sizes = kSizes; sizes.push_back(kStressSize);

        for (int n : sizes) {
            string label = (n == kStressSize) ? "stress" : "test";
            Layout layout = buildLayout(n, worldSize);
            vector<float> A, B(n*n), C, Cref;

            if (rank == 0) {
                A.resize(n*n); C.resize(n*n); Cref.resize(n*n);
                fillRandomMatrix(A, n, 3000u + n);
                fillRandomMatrix(B, n, 4000u + n);

                double seqStart = MPI_Wtime();
                sequentialMultiply(A, B, Cref, n);
                double seqEnd = MPI_Wtime();

                double oclTime = runOpenCLCase(n, rank, worldSize, layout, ocl, A, B, C);
                printRow(label, n, worldSize, seqEnd-seqStart, oclTime, compareMatrices(C, Cref));
            } else {
                runOpenCLCase(n, rank, worldSize, layout, ocl, A, B, C);
            }
        }
        releaseOpenCL(ocl);
    } catch (const std::exception& ex) {
        if (rank == 0) std::cerr << "Error: " << ex.what() << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
