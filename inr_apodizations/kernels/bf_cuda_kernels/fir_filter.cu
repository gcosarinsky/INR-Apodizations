__device__ void shift_left(short *x, int taps) {
    for (int j = 0; j < taps; j++) {
        x[j] = x[j + 1];
    }
    x[taps] = 0;
}


__device__ float multi_sum(short *x, float *coef, int taps) {
    float q = 0;
    for (int j = 0; j < (taps + 1); j++) {
        q += coef[taps - j] * x[j];
    }
    // Clamp accumulated value within the range of a signed short
    q = fmaxf(fminf(q, 32767.0f), -32768.0f);
    return q;
}


extern "C" __global__ void fir_filter(const int *int_params, const short *datain, const float *coef_g, short *dataout)
    {

    // Compute global thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Thread index

    int nel = int_params[N_ELEMENTS];
    int ns = int_params[N_SAMPLES];
    int taps = int_params[TAPS];

    // Bounds check
    if (tid >= nel * ns) return;

    //extern __shared__ float coef[];
    float coef[MAX_FIR_SIZE] ;
    short x[MAX_FIR_SIZE];

    // Index of first sample of the A-scan in dataout
    int i = ns * tid;

    /* copy filter coefficients into private memory */
    unsigned short l0 = taps/2 ;
    unsigned short lmax = taps + 1 ;
    for (unsigned short l=0; l < lmax; l++) {
        coef[l] = coef_g[l] ;
    }

    // Compute transient
    for (unsigned short l = 0; l <= l0; l++) {
        x[l + l0] = datain[i + l];
    }

    // Compute first output sample
    dataout[i] = (short)rintf(multi_sum(x, coef, taps));

    // Continuar hasta traer la última muestra del A-scan
    lmax = ns - l0;
    for (unsigned short l = 1; l < lmax; l++) {
        shift_left(x, taps);
        x[taps] = datain[i + l + l0];
        dataout[i + l] = (short)rintf(multi_sum(x, coef, taps));
    }

    for (int l = lmax; l < ns; l++) {
        shift_left(x, taps);
        dataout[i + l] = (short)rintf(multi_sum(x, coef, taps));
    }

    /* Force these values for beamforming border handling */
    dataout[i] = 0 ;
    dataout[i + ns - 2] = 0 ;
    dataout[i + ns - 1] = 0 ;
}
