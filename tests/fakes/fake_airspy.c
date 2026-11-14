/* Fausse bibliothèque libairspy pour les essais : mêmes signatures que airspy.h,
   produit du bruit + une porteuse à +100 kHz du centre.  Ne remplace pas un essai sur le vrai récepteur. */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

struct airspy_device { int streaming; uint32_t rate, freq; uint8_t bias, gain; pthread_t th;
                       int (*cb)(void*); void* ctx; int type; };
typedef struct { struct airspy_device* device; void* ctx; void* samples; int sample_count;
                 uint64_t dropped_samples; int sample_type; } airspy_transfer_t;
static struct airspy_device dev0;
uint32_t fake_last_freq(void) { return dev0.freq; }
uint8_t fake_last_bias(void) { return dev0.bias; }
uint8_t fake_last_gain(void) { return dev0.gain; }

int airspy_open(struct airspy_device** d) { dev0.rate = 10000000; *d = &dev0; return 0; }
int airspy_close(struct airspy_device* d) { return 0; }
int airspy_set_sample_type(struct airspy_device* d, int t) { d->type = t; return 0; }
int airspy_get_samplerates(struct airspy_device* d, uint32_t* buf, const uint32_t len) {
    if (len == 0) { buf[0] = 2; return 0; }
    buf[0] = 10000000; if (len > 1) buf[1] = 2500000; return 0; }
int airspy_set_samplerate(struct airspy_device* d, uint32_t r) { d->rate = r; return 0; }
int airspy_set_freq(struct airspy_device* d, const uint32_t f) { d->freq = f; return 0; }
int airspy_set_linearity_gain(struct airspy_device* d, uint8_t v) { d->gain = v; return 0; }
int airspy_set_sensitivity_gain(struct airspy_device* d, uint8_t v) { d->gain = 100 + v; return 0; }
int airspy_set_rf_bias(struct airspy_device* d, uint8_t v) { d->bias = v; return 0; }
static void* run(void* arg) {
    struct airspy_device* d = arg; enum { N = 65536 }; static float buf[2 * N]; uint64_t n = 0;
    while (d->streaming) {
        for (int i = 0; i < N; i++, n++) {
            double ph = 2 * M_PI * 100000.0 * (double)n / d->rate;
            buf[2*i] = 0.1f * cos(ph) + 0.002f * (rand() / (float)RAND_MAX - 0.5f);
            buf[2*i+1] = 0.1f * sin(ph) + 0.002f * (rand() / (float)RAND_MAX - 0.5f);
        }
        airspy_transfer_t t = { d, d->ctx, buf, N, 7, d->type };
        if (d->cb(&t) != 0) break;
        usleep((useconds_t)(1e6 * N / d->rate));
    }
    return 0;
}
int airspy_start_rx(struct airspy_device* d, int (*cb)(void*), void* ctx) {
    d->cb = cb; d->ctx = ctx; d->streaming = 1; return pthread_create(&d->th, 0, run, d); }
int airspy_stop_rx(struct airspy_device* d) { d->streaming = 0; pthread_join(d->th, 0); return 0; }
