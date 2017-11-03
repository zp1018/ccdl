

#pragma once

#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cudart.lib")

#ifdef USE_CUDNN
#pragma comment(lib, "cudnn.lib")
#endif
