#include <iostream>
#include <algorithm>
#include <chrono>
#include "bmp/EasyBMP.h"

#define M_PI 3.14159265358979323846
#define TILE_X 16
#define TILE_Y 16


using namespace std;
// 1D Gaussian kernel array values of a fixed size (make sure the number > filter size d)
__constant__ float cGaussian[64];
// Initialize texture memory to store the input
texture<float, cudaTextureType2D, cudaReadModeElementType> inTexture;

/* 
   GAUSSIAN IN 1D FOR SPATIAL DIFFERENCE
   Here, exp(-[(x_centre - x_curr)^2 + (y_centre - y_curr)^2]/(2*sigma*sigma)) can be broken down into ...
   exp[-(x_centre - x_curr)^2 / (2*sigma*sigma)] * exp[-(y_centre - y_curr)^2 / (2*sigma*sigma)] 
   i.e, 2D gaussian -> product of two 1D Gaussian
   A constant Gaussian 1D array can be initialzed to store the gaussian values
   Eg: For a kernel size 5, the pixel difference array will be ...
   [-2, -1, 0, 1 , 2] for which the gaussian kernel is applied
*/
void updateGaussian(int r,double sd)
{
	float fGaussian[64];
	for (int i = 0; i < 2*r +1 ; i++)
	{
		float x = i - r;
		fGaussian[i] = expf(-(x*x) / (2 * sd*sd));
	}
	cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*r + 1));
}

// Gaussian function for range difference
__device__ inline double gaussian(float x, double sigma)
{
	return __expf(-(powf(x, 2)) / (2 * powf(sigma, 2))) ;
}

double hostGaussian(float x, double sigma)
{
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) ;
}

// Bilateral filter kernel
__global__ void cudaBilateral(float* input, float* output, int width, int height) {
    int r = 1;
    double sI = 10;
    double sS = 1000;
	// Initialize global Tile indices along x,y and xy
	int txIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int tyIndex = blockDim.y * blockIdx.y + threadIdx.y;

	// If within image size
	if ((txIndex < width) && (tyIndex < height))
	{
		double iFiltered = 0;
		double wP = 0;
		// Get the centre pixel value
		float centrePx = tex2D(inTexture, txIndex, tyIndex);
		// Iterate through filter size from centre pixel
		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				// Get the current pixe; value
				float currPx = tex2D(inTexture, txIndex + dx, tyIndex + dy);
				// Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)
				double w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian(centrePx - currPx, sI);
				iFiltered += w * currPx;
				wP += w;				
			}
		}
		output[tyIndex*width + txIndex] = iFiltered / wP;
	}
}


void hostBilateral(float* output, unsigned int rows, unsigned int cols) {
    int r = 1;
    double sI = 10;
    double sS = 1000;
    for (int i = 0; i<rows; i++) {
        for (int j = 0; j<cols; j++){
            double iFiltered = 0;
            double wP = 0;
            // Get the centre pixel value
            float centrePx = output[i*cols + j];
            // Iterate through filter size from centre pixel
            for (int dy = -r; dy <= r; dy++) {
                for (int dx = -r; dx <= r; dx++) {
                    int currentRow = (i + dx)*cols;
                
                    if (currentRow < 0) {
                       currentRow = 1;
                    }
                    if (currentRow == rows) {
                        currentRow = rows -2;
                    }
                    float currPx = output[currentRow + j+dy];
                    // Get the current pixel value
                    // Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)
                    double w = hostGaussian(dy+r, sS) * hostGaussian(dy + r, sS) * hostGaussian(centrePx - currPx, sI);
                    iFiltered += w * currPx;
                    wP += w;				
                }
            }
            output[i*cols + j] = iFiltered / wP;
        }
    }
    
}

float *readLikeGrayScale(char *filePathInput, unsigned int *rows, unsigned int *cols) {
    BMP Input;
    Input.ReadFromFile(filePathInput);
    *rows = Input.TellHeight();
    *cols = Input.TellWidth();
    float *grayscale = (float *)calloc(*rows * *cols, sizeof(float));
    for (int j = 0; j < *rows; j++)
    {
        for (int i = 0; i < *cols; i++)
        {
            float gray = (float)floor(0.299 * Input(i, j)->Red +
                                      0.587 * Input(i, j)->Green +
                                      0.114 * Input(i, j)->Blue);
            grayscale[j * *cols + i] = gray;
        }
    }
    return grayscale;
}

void writeImage(char *filePath, float *grayscale, unsigned int rows, unsigned int cols) {
    BMP Output;
    Output.SetSize(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            RGBApixel pixel;
            pixel.Red = grayscale[i * cols + j];
            pixel.Green = grayscale[i * cols + j];
            pixel.Blue = grayscale[i * cols + j];
            pixel.Alpha = 0;
            Output.SetPixel(j, i, pixel);
        }
    }
    Output.WriteToFile(filePath);
}

int main() {
    float *grayscale, *grayscale2 = 0;
    unsigned int rows, cols;

    grayscale = readLikeGrayScale("lena.bmp", &rows, &cols);
    grayscale2 = readLikeGrayScale("lena.bmp", &rows, &cols);
    auto t1 = chrono::high_resolution_clock::now();

    hostBilateral(grayscale2, rows, cols);

    auto t2 = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
    cout << "Host duration:" << duration << endl;


    writeImage("CPU.bmp", grayscale2, rows, cols);

    writeImage("afterRead.bmp", grayscale, rows, cols);
    cudaEvent_t start;
    cudaEventCreate(&start);

    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0,
                              cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    cudaMallocArray(&cuArray, &channelDesc, cols, rows);

    cudaMemcpyToArray(cuArray, 0, 0, grayscale, rows * cols * sizeof(float),
                                      cudaMemcpyHostToDevice);
                                      
    
    int r = 1;
    double sS = 10;
    updateGaussian(r,sS);
    inTexture.addressMode[0] = cudaAddressModeWrap;
    inTexture.addressMode[1] = cudaAddressModeWrap;
    inTexture.filterMode = cudaFilterModeLinear;

    cudaBindTextureToArray(inTexture, cuArray, channelDesc);

    float *dev_grayscale, *dev_output, *output;
    output = (float *)calloc(rows * cols, sizeof(float));
    cudaMalloc(&dev_output, rows * cols * sizeof(float));
    cudaMalloc(&dev_grayscale, rows * cols * sizeof(float));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
                 (rows + dimBlock.y - 1) / dimBlock.y);
    cudaBilateral<<<dimGrid, dimBlock>>>(dev_grayscale, dev_output, cols, rows);
    cudaMemcpy(output, dev_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, NULL);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    float msec = msecTotal;
    cout << "Device duration: " << msec << endl;
    writeImage("result.bmp", output, rows, cols);
    cudaFreeArray(cuArray);
    cudaFree(dev_output);
    return 0;
}