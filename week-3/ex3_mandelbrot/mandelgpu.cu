void __global__ mandel_gpu(int disp_width, int disp_height, int *image, int max_iter, int blk_size) {

    double  scale_real, scale_imag;
    double  x, y, u, v, u2, v2;

    scale_real = 3.5 / (double)disp_width;
    scale_imag = 3.5 / (double)disp_height;

    int j = blockIdx.y * blockDim.y + threadIdx.y;  // WIDTH
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // HEIGHT
    int idx = i * disp_height + j;
    
    // check if inside picture
    if (i >= disp_width || j >= disp_height) return;

    //
    x = ((double)i * scale_real) - 2.25;
    y = ((double)j * scale_imag) - 1.75;

    u    = 0.0;
    v    = 0.0;
    u2   = 0.0;
    v2   = 0.0;
    int iter = 0;

    while ( u2 + v2 < 4.0 &&  iter < max_iter ) {
        v = 2 * v * u + y;
        u = u2 - v2 + x;
        u2 = u*u;
        v2 = v*v;
        iter ++;
    }

    // if we exceed max_iter, reset to zero
    iter = iter == max_iter ? 0 : iter;

    image[idx] = iter;  
}


