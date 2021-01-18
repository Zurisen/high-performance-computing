#ifndef __MANDELGPU_H
#define __MANDELGPU_H

void __global__ mandel_gpu(int width, int height, int *image, int max_iter,int blk_size);

#endif

