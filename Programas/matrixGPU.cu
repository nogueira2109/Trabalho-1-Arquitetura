#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>


using namespace std;


template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(double *C, double *A, double *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    double Csub = 0.0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}
void constantInit(double *data, int size, double val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
void matrixMultiplyFloat(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB,bool bSave)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, 2.f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;



    float incial,fim;
   double tTime;

   
    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(1);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(1);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(1);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(1);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(1);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(1);
    }

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");


    incial = clock();
    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16)
	{
    	matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
   	}else{
        matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
   	}   
            
          
    

    printf("done\n");

    cudaDeviceSynchronize();

    /*
    int nIter = 300;

    for (int j = 0; j < nIter; j++)
    {
        if (block_size == 16)
        {
            matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else
        {
            matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    printf("done\n");

    cudaDeviceSynchronize();
    */
    // Copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    cout << "Resposta do primeiro valor da matriz: "<<h_C[0] << endl;
    fim = clock();
    tTime = ((fim-incial)/(float)CLOCKS_PER_SEC)/8;

    FILE *f = fopen("log.csv", "a");
    fprintf(f, "%d;%d;%f;Float;GPU;%f\n",dimsC.x,block_size,tTime,h_C[0] );

    //string log = to_string(dimsC.x)+";"+to_string(block_size)+";"+to_string(tTime)+";Float;GPU";
	
	fclose(f);

    printf("tempo de processamento %f\n", tTime);

    if(bSave){
    	printf("Salvando a matriz\n");
	    ofstream saida("saida.csv");
	    saida << dimsC.x << " " << dimsC.x <<" "<<dimsC.x*dimsC.x << endl;
	    for (int i = 0; i < dimsC.x; ++i)
	    {
	    	for (int j = 0; j < dimsC.x; ++j)
	    	{
	    		saida << i+1 << " " << j+1 << " " <<h_C[i*dimsC.x+j] << endl;
	    	}
	    }

    saida.close();
    }
    
    
    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

   
}

void matrixMultiplyDouble(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB, bool bSave )
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(double) * size_A;
    double *h_A = (double *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(double) * size_B;
    double *h_B = (double *)malloc(mem_size_B);

    // Initialize host memory
    constantInit(h_A, size_A, 1.0);
    constantInit(h_B, size_B, 2.0);

    // Allocate device memory
    double *d_A, *d_B, *d_C;



    double incial,fim;
   double tTime;

   
    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(double);
    double *h_C = (double *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(1);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(1);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(1);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(1);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(1);
    }

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");


   incial = clock();
    if (block_size == 16)
	{
    	matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
   	}else{
        matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
   	}
            
          
    
    printf("done\n");

    cudaDeviceSynchronize();

    /*
    int nIter = 300;

    for (int j = 0; j < nIter; j++)
    {
        if (block_size == 16)
        {
            matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else
        {
            matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    printf("done\n");

    cudaDeviceSynchronize();
	*/
    // Copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    cout << "Resposta do primeiro valor da matriz: "<<h_C[0] << endl;
    fim = clock();
    tTime = ((fim-incial)/(float)CLOCKS_PER_SEC)/8;


    //Log para processamento em Double
    FILE *f = fopen("log.csv", "a");
    fprintf(f, "%d;%d;%f;Double;GPU;%lf\n",dimsC.x,block_size,tTime,h_C[0]  );

    //string log = to_string(dimsC.x)+";"+to_string(block_size)+";"+to_string(tTime)+";Float;GPU";
	
	fclose(f);
	
	//string log = to_string(dimsC.x)+";"+to_string(block_size)+";"+to_string(tTime)+";Double;GPU";
	
    printf("tempo de processamento %f\n", tTime);

     if(bSave){
    	printf("Salvando a matriz\n");
	    ofstream saida("saida.csv");
	    saida << dimsC.x << " " << dimsC.x <<" "<<dimsC.x*dimsC.x << endl;
	    for (int i = 0; i < dimsC.x; ++i)
	    {
	    	for (int j = 0; j < dimsC.x; ++j)
	    	{
	    		saida << i+1 << " " << j+1 << " " <<h_C[i*dimsC.x+j] << endl;
	    	}
	    }

    saida.close();
    }
    
    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

   
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
	int N = -1;
	int T = -1;
	bool bDouble = false;
	bool bSave = false;
    printf("[Matrix Multiply Using CUDA] - Starting...\n");


    if (argc > 1) {
		for (int i = 0; i < argc; i++) {
			if (strcmp(argv[i], "-n") == 0) {
				N = atoi(argv[++i]);
			}else if (strcmp(argv[i], "-t") == 0) {
				T = atoi(argv[++i]);
			}else if (strcmp(argv[i], "-d") == 0) {
				bDouble = true;
			}
			else if (strcmp(argv[i], "-save") == 0) {
				bSave = true;
			}
		}
	}
	else {
		cout << "Without parameters!!!!" << endl;
		cout << "Use:\n\
		-t ${NUM} \tset size of tiles\n\
		-n ${NUM} \tset size of matrix\n\
		-save \t save the resulting matrix \n\
		-d \t set matrix to double precision \n\
		\n\n\
		Example: ./matrixGPU  -n 2048 -t 32\n\n\
		By defaul, this program running with single precision!!!!"<< endl;
		return 1;
	}
	

    
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;
    block_size = T;

    printf("Size of block: %d\n",block_size );

    dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
    dim3 dimsB(5*4*block_size, 5*2*block_size, 1);

    if(N == -1 || T == -1){
    	cout << deviceProp.major << endl;
		cout << "Tamanho da matrix não definido!!!!" << endl;
		return 1;
	}else{
		dimsA.x = N;
		dimsA.y = N;
		dimsB.x = N;
		dimsB.y = N;
	}
    
    
    if(bDouble){
    	printf("MatrixA(%d,%d), MatrixB(%d,%d) in DOUBLE\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    	matrixMultiplyDouble(argc, argv, block_size, dimsA, dimsB, bSave);	
    }else{
    	printf("MatrixA(%d,%d), MatrixB(%d,%d) in FLOAT\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    	matrixMultiplyFloat(argc, argv, block_size, dimsA, dimsB,bSave);	
    }
    
}
