__device__ void shift_izq(short *x, int taps) {
    for (int j = 0; j < taps; j++) {
        x[j] = x[j + 1];
    }
    x[taps] = 0;
}


__device__ float multisum(short *x, float *coef, int taps) {
    float q = 0;
    for (int j = 0; j < (taps + 1); j++) {
        q += coef[taps - j] * x[j];
    }
    // Limitar el valor acumulado dentro del rango de un short
    q = fmaxf(fminf(q, 32767.0f), -32768.0f);
    return q;
}


extern "C" __global__ void fir_filter(const int *int_params, const short *datain, const float *coef_g, short *dataout)
    {

    // Calcular índices globales del hilo
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Índice de thread

    int nel = int_params[N_ELEMENTS];
    int ns = int_params[N_SAMPLES];
    int taps = int_params[TAPS];

    // Verificar límites
    if (tid >= nel * ns) return;

    //extern __shared__ float coef[];
    float coef[MAX_FIR_SIZE] ;
    short x[MAX_FIR_SIZE];

    // Índice de primer sample del ascan en dataout
    int i = ns * tid;

    /* copiar coeficientes del filtro en memoria privada */
    unsigned short l0 = taps/2 ;
    unsigned short lmax = taps + 1 ;
    for (unsigned short l=0; l < lmax; l++) {
        coef[l] = coef_g[l] ;
    }

    // Calcular transitorio
    for (unsigned short l = 0; l <= l0; l++) {
        x[l + l0] = datain[i + l];
    }

    // Calcular primera muestra de salida
    dataout[i] = (short)rintf(multisum(x, coef, taps));

    // Continuar hasta traer la última muestra del A-scan
    lmax = ns - l0;
    for (unsigned short l = 1; l < lmax; l++) {
        shift_izq(x, taps);
        x[taps] = datain[i + l + l0];
        dataout[i + l] = (short)rintf(multisum(x, coef, taps));
    }

    for (int l = lmax; l < ns; l++) {
        shift_izq(x, taps);
        dataout[i + l] = (short)rintf(multisum(x, coef, taps));
    }

    /* forzar esto para el tema de los bordes en el beamforming */
    dataout[i] = 0 ;
    dataout[i + ns - 2] = 0 ;
    dataout[i + ns - 1] = 0 ;
}
