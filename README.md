# About app

The application allow you to confirm bilateral algorithm on your GPU and to compare the result with multiplication completed on CPU

# How to execute

Before start the app please follow this [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

---
Note: you need to have compatible GPU, [list](https://developer.nvidia.com/cuda-gpus) of allowed GPUs
---

After completeing of all prerequests you can complile it with the following commands:

`make`
And then: `make bilateral`
## Results

Image input:

![Input image](lena.bmp)

Image output:

![Output image](result.bmp)

CPU time: 517ms
GPU time: 27.1246ms