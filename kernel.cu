#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "bmp/EasyBMP.h"

#define PI 3.14159265358979
#define WINDOW_SIZE 3
#define WINDOW_LENGHT WINDOW_SIZE *WINDOW_SIZE

using namespace std;

texture<float, cudaTextureType2D, cudaReadModeElementType> tex;

__global__ void cudaBilateral(float *output, int imageWidth, int imageHeight)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (!(row < imageHeight && col < imageWidth)) {
        return;
    }

    float filter[WINDOW_LENGHT] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int x = 0; x < WINDOW_SIZE; x++)
    {
        for (int y = 0; y < WINDOW_SIZE; y++)
        {
            filter[x * WINDOW_SIZE + y] = tex2D(tex, col + y - 1, row + x - 1);
        }
    }
    for (int i = 0; i < WINDOW_LENGHT; i++)
    {
        for (int j = i + 1; j < WINDOW_LENGHT; j++)
        {
            
        }
    }
    output[row * imageWidth + col] = filter[(int)(WINDOW_LENGHT / 2)];
}

float *readLikeGrayScale(char *filePathInput, unsigned int *rows, unsigned int *cols)
{
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

void writeImage(char *filePath, float *grayscale, unsigned int rows, unsigned int cols)
{
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


float g(int x, int y, int i, int j) {
    return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

float f(float x, float sigma) {
    return exp(-(pow(x, 2))/(2 * pow(sigma, 2))) / (2 * PI * pow(sigma, 2));
}

float r(float x, float x1, float sigma) {
    return exp(pow(f(x, sigma) - f(x1, sigma), 2) / sigma);
}

void hostBilateral(float* output, unsigned int rows, unsigned int cols) {
    float sigma = 16.0;
    cout << rows<< endl;
    cout << cols << endl;
    for (int i = 0; i<rows; i++) {
        for (int j = 0; j<cols; j++){
                int h =0, k = 0;
                for (int x = -1; x<WINDOW_SIZE-1; x++) {
                    for (int y = -1; y< WINDOW_SIZE-1; y++) {
                        int currentRow = i + x * rows;
                        int currentColumn = j+y;
                        if (currentRow < 0 || currentRow > rows) {
                            currentRow = i-x;
                        }
                        if (currentColumn < 0 || currentColumn > rows) {
                            currentColumn = j - y;
                        }
                        
                        h += f(output[currentRow + currentColumn], sigma) + g(x, y, i, j) + r(output[i + j], output[currentRow +currentRow], sigma);
                        k += g(x, y, i, j) + r(output[i + j], output[currentRow + currentColumn], sigma);
                    }
                }
                h /= k;
                // cout << output[i*rows+j] << endl;
                cout << h << endl;
                output[i*rows+j] = h;
        }
    }
    
}

int main()
{
    float *grayscale = 0;
    unsigned int rows, cols;

    grayscale = readLikeGrayScale("lena.bmp", &rows, &cols);
    hostBilateral(grayscale, rows, cols);
    writeImage("afterRead.bmp", grayscale, rows, cols);
    // cudaChannelFormatDesc channelDesc =
    //     cudaCreateChannelDesc(32, 0, 0, 0,
    //                           cudaChannelFormatKindFloat);
    // cudaArray *cuArray;
    // cudaMallocArray(&cuArray, &channelDesc, cols, rows);

    // cudaMemcpyToArray(cuArray, 0, 0, grayscale, rows * cols * sizeof(float),
    //                                   cudaMemcpyHostToDevice);

    // tex.addressMode[0] = cudaAddressModeWrap;
    // tex.addressMode[1] = cudaAddressModeWrap;
    // tex.filterMode = cudaFilterModeLinear;

    // cudaBindTextureToArray(tex, cuArray, channelDesc));

    // float *dev_output, *output;
    // output = (float *)calloc(rows * cols, sizeof(float));
    // cudaMalloc(&dev_output, rows * cols * sizeof(float));

    // dim3 dimBlock(16, 16);
    // dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
    //              (rows + dimBlock.y - 1) / dimBlock.y);
    // cudaBilateral<<<dimGrid, dimBlock>>>(dev_output, cols, rows);
    // cudaMemcpy(output, dev_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // writeImage("result.bmp", output, rows, cols);
    // cudaFreeArray(cuArray);
    // cudaFree(dev_output);
    return 0;
}

