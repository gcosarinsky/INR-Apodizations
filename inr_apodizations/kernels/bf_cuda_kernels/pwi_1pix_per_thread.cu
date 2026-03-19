__device__ void compute_sample_index(float *x_rx, float *xf, float *zf, float *c1, float *bfd,
                                     float *fs, int *ns, float *t1, float *t2, float *t,
                                     unsigned int *k, float *ap_dyn) {
    *t2 = hypotf(*x_rx - *xf, *zf) / *c1;
    *ap_dyn = *zf/(fabsf(*x_rx - *xf) + FLT_EPSILON) > *bfd ;  // Apodización dinámica
    *t = *t1 + *t2;
    *t = *t * (*t > 0 ? 1 : 0);  /* First sample must be 0 !!! */
    *k = min((unsigned int)floorf(*t * (*fs)), *ns - 2); /* resto 2 para evitar que k+1 = ns */
}


extern "C" __global__ void pwi_1pix_per_thread(
                               const int *int_params,
                               const float *float_params,
                               const float *angles,
                               const short *matrix,
                               const short *matrix_imag,
                               float *img,
                               float *img_imag,
                               float *cohe) {

    unsigned short iz = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned short ix = blockIdx.y * blockDim.y + threadIdx.y;

    // Obtener los parámetros enteros y flotantes
    // int_params
    int nel = int_params[N_ELEMENTS];
    int nang = int_params[N_ANGLES];
    int ns = int_params[N_SAMPLES];
    int nx = int_params[NX];
    int nz = int_params[NZ];
    if (iz >= nz || ix >= nx) return;  // Verificar límites de los índices

    // float params
    float fs = float_params[FS];
    float c1 = float_params[C1];
    float pitch = float_params[PITCH];
    float x0_roi = float_params[X0_ROI];
    float z0_roi = float_params[Z0_ROI];
    float x_step = float_params[X_STEP];
    float z_step = float_params[Z_STEP];
    float t_start = float_params[T_START];
    float x0 = float_params[X_0];
    float bfd = float_params[BFD];

    float xf = x0_roi + x_step * ix;
    float zf = z0_roi + z_step * iz;  // Z POSITIVE DOWNWARDS
    float x_rx, wave_source;
    float t1, t2;
    float t, dt, temp, theta, ap_dyn;
    unsigned int k, k0 = 0;
    float a, b, q = 0, q_imag = 0, w = 0, w_imag = 0;

    unsigned int f_idx = iz * nx + ix;

    for (unsigned short i = 0; i < nang; i++) {

        theta = angles[i];
        wave_source = x0 * (theta < 0 ? 1 : -1);
        t1 = ((xf - wave_source) * sinf(theta) + zf * cosf(theta)) / c1 - t_start;
        x_rx = -x0;  // Inicializar x_rx para el primer elemento
        for (unsigned short e = 0; e < nel; e++) {
            compute_sample_index(&x_rx, &xf, &zf, &c1, &bfd, &fs, &ns, &t1, &t2, &t, &k, &ap_dyn);
            dt = t * fs - k;

            temp = (float)matrix[k0 + k];
            a = ((float)matrix[k0 + k + 1] - temp) * dt + temp;
            q += a * ap_dyn;

            temp = (float)matrix_imag[k0 + k];
            b = ((float)matrix_imag[k0 + k + 1] - temp) * dt + temp;
            q_imag += b * ap_dyn;

            temp = hypotf(a, b) + FLT_EPSILON;  /* módulo del "fasor" */
            /* se suman las componentes de los fasores para cada A-scan */
            w += a / temp;
            w_imag += b / temp;

            k0 += ns;
            x_rx += pitch;  // Incrementar x_rx para cada elemento
        }

    }

    img[f_idx] = q ;
    img_imag[f_idx] = q_imag ;
    cohe[f_idx] = hypotf(w, w_imag) ;
}



extern "C" __global__ void pwi_gather_delayed_samples(
                               const int *int_params,
                               const float *float_params,
                               const float *angles,
                               const short *matrix,
                               const short *matrix_imag,
                               float2 *delayed_samples) {

    unsigned short iz = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned short ix = blockIdx.y * blockDim.y + threadIdx.y;

    // Obtener los parámetros enteros y flotantes
    // int_params
    int nel = int_params[N_ELEMENTS];
    int nang = int_params[N_ANGLES];
    int ns = int_params[N_SAMPLES];
    int nx = int_params[NX];
    int nz = int_params[NZ];
    if (iz >= nz || ix >= nx) return;  // Verificar límites de los índices

    // float params
    float fs = float_params[FS];
    float c1 = float_params[C1];
    float pitch = float_params[PITCH];
    float x0_roi = float_params[X0_ROI];
    float z0_roi = float_params[Z0_ROI];
    float x_step = float_params[X_STEP];
    float z_step = float_params[Z_STEP];
    float t_start = float_params[T_START];
    float x0 = float_params[X_0];
    float bfd = float_params[BFD];

    float xf = x0_roi + x_step * ix;
    float zf = z0_roi + z_step * iz;  // Z POSITIVE DOWNWARDS
    float x_rx, wave_source;
    float t1, t2;
    float t, dt, temp, theta, ap_dyn;
    unsigned int k, k0 = 0, n_pix = nx * nz;
    float a, b ;

    unsigned int idx = iz * nx + ix;

    for (unsigned short i = 0; i < nang; i++) {

        theta = angles[i];
        wave_source = x0 * (theta < 0 ? 1 : -1);
        t1 = ((xf - wave_source) * sinf(theta) + zf * cosf(theta)) / c1 - t_start;
        x_rx = -x0;  // Inicializar x_rx para el primer elemento

        for (unsigned short e = 0; e < nel; e++) {
            compute_sample_index(&x_rx, &xf, &zf, &c1, &bfd, &fs, &ns, &t1, &t2, &t, &k, &ap_dyn);
            dt = t * fs - k;

            temp = (float)matrix[k0 + k];
            a = ((float)matrix[k0 + k + 1] - temp) * dt + temp;


            temp = (float)matrix_imag[k0 + k];
            b = ((float)matrix_imag[k0 + k + 1] - temp) * dt + temp;

            delayed_samples[idx].x = a ;
            delayed_samples[idx].y = b ;

            idx += n_pix ;
            k0 += ns;
            x_rx += pitch;  // Incrementar x_rx para cada elemento
        }
    }
}



extern "C" __global__ void pwi_gather_delayed_samples_points(
                               const int *int_params,
                               const float *float_params,
                               const float *angles,
                               const short *matrix,
                               const short *matrix_imag,
                               const float *points_x,    // Lista de coordenadas x
                               const float *points_z,    // Lista de coordenadas z
                               const int n_points,       // Número total de puntos
                               float2 *delayed_samples)  // Output: [n_angles, n_elements, n_points]
{
    // Cada thread procesa un punto
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (point_idx >= n_points) return;

    // int_params
    int nel = int_params[N_ELEMENTS];
    int nang = int_params[N_ANGLES];
    int ns = int_params[N_SAMPLES];

    // float params
    float fs = float_params[FS];
    float c1 = float_params[C1];
    float pitch = float_params[PITCH];
    float t_start = float_params[T_START];
    float x0 = float_params[X_0];
    float bfd = float_params[BFD];

    // Obtener coordenadas del punto actual
    float xf = points_x[point_idx];
    float zf = points_z[point_idx];

    float x_rx, wave_source;
    float t1, t2, t, dt, temp, theta, ap_dyn;
    unsigned int k, k0;
    float a, b;

    int output_idx = point_idx;  // Índice base en delayed_samples

    for (unsigned short i = 0; i < nang; i++) {
        theta = angles[i];
        wave_source = x0 * (theta < 0 ? 1 : -1);
        t1 = ((xf - wave_source) * sinf(theta) + zf * cosf(theta)) / c1 - t_start;
        
        x_rx = -x0;
        k0 = i * nel * ns;  // Offset para este ángulo en matrix

        for (unsigned short e = 0; e < nel; e++) {
            compute_sample_index(&x_rx, &xf, &zf, &c1, &bfd, &fs, &ns, 
                               &t1, &t2, &t, &k, &ap_dyn);
            dt = t * fs - k;

            // Interpolación lineal
            temp = (float)matrix[k0 + k];
            a = ((float)matrix[k0 + k + 1] - temp) * dt + temp;

            temp = (float)matrix_imag[k0 + k];
            b = ((float)matrix_imag[k0 + k + 1] - temp) * dt + temp;

            // Aplicar apodización dinámica
            a *= ap_dyn;
            b *= ap_dyn;

            // Guardar en output: [angle, element, point]
            delayed_samples[output_idx].x = a;
            delayed_samples[output_idx].y = b;

            output_idx += n_points;
            k0 += ns;
            x_rx += pitch;
        }
    }
}