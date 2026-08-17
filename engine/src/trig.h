/* trig.h — sin/cos that give the SAME BITS on every platform.
 *
 * WHY THIS EXISTS, and it is not premature. Linux and Windows builds of this
 * engine produced different output. Weights were bit-identical, lrintf matched,
 * thread count was irrelevant, and the kernels agreed with their reference on
 * both. Tracing FNV checksums through the forward pass put the first divergence
 * at `rope_q` -- and only at pos != 0, where the rotation is not the identity.
 *
 * The cause, found by comparing all 64 RoPE frequency indices instead of a
 * sample of 8: at i=7, sinf(1*fr) is 0x3e6023da on glibc and 0x3e6023d9 on
 * msvcrt. One ULP. Neither is wrong -- IEEE-754 requires correct rounding for
 * +,-,*,/ and sqrt, and explicitly does NOT for transcendentals, so every libm
 * is free to differ in the last place. That one bit enters through RoPE,
 * propagates through attention, and eventually flips a near-tie argmax.
 *
 * ⚠ THE EARLIER PROBE SAID "IDENTICAL" AND WAS WRONG. It compared expf, powf,
 * sinf, cosf and sqrtf at eight arbitrary points and found every bit equal, so
 * libm was ruled out. It exercised the FUNCTIONS but not the ARGUMENTS the
 * engine actually uses -- the same defect as a test that never reaches the code
 * path it claims to cover. Sampling is what made it a false negative.
 *
 * THE FIX. Compute sin and cos from +, -, * and / only, which IEEE-754 DOES
 * require to be correctly rounded, so every conforming platform must produce
 * identical bits. Cody-Waite range reduction to [-pi/4, pi/4], then a Taylor
 * series in double, rounded to float at the end.
 *
 * ACCURACY. On |r| <= pi/4 the first omitted sine term is r^13/6227020800 which
 * is below 1.6e-11 -- far under a float ULP, so the result is at least as close
 * to the true value as libm's. Measured effect on the engine is in the README.
 */
#ifndef COLI_TRIG_H
#define COLI_TRIG_H

#include <math.h>

/* pi/2 split so that x - k*PIO2_HI is exact for the k values RoPE produces.
 * A single-constant reduction loses bits for large arguments, and RoPE's are
 * large: pos can reach the context length. */
#define COLI_PIO2_HI 1.57079632673412561417e+00   /* high 33 bits of pi/2 */
#define COLI_PIO2_MID 6.07710050650619224932e-11
#define COLI_PIO2_LO 2.02226624879595063154e-21

static inline void coli_sincos(double x, double *sout, double *cout) {
    /* quadrant */
    double q = x * (2.0/3.14159265358979323846);
    double k = (q >= 0.0) ? (double)(long long)(q + 0.5) : (double)(long long)(q - 0.5);

    /* Cody-Waite: subtract pi/2 in three exact pieces. */
    double r = x - k*COLI_PIO2_HI;
    r = r - k*COLI_PIO2_MID;
    r = r - k*COLI_PIO2_LO;

    double r2 = r*r;
    /* sin(r), Taylor to r^11 */
    double s = r*(1.0 + r2*(-1.0/6.0 + r2*(1.0/120.0 + r2*(-1.0/5040.0
             + r2*(1.0/362880.0 + r2*(-1.0/39916800.0))))));
    /* cos(r), Taylor to r^10 */
    double c = 1.0 + r2*(-0.5 + r2*(1.0/24.0 + r2*(-1.0/720.0
             + r2*(1.0/40320.0 + r2*(-1.0/3628800.0)))));

    /* quadrant fix-up: k mod 4 selects which of (s,c,-s,-c) each output is */
    long long ik = (long long)k & 3;
    switch (ik) {
        case 0: *sout =  s; *cout =  c; break;
        case 1: *sout =  c; *cout = -s; break;
        case 2: *sout = -s; *cout = -c; break;
        default:*sout = -c; *cout =  s; break;
    }
}

/* powf(theta, -2i/hd) is the other transcendental RoPE calls. It is computed
 * ONCE per (i, head_dim, theta) and is not position-dependent, so it is hoisted
 * into a table by the caller rather than made exact here -- but it must still be
 * computed identically, so it goes through exp/log built from the same
 * primitives. exp2/log2 on a power-of-ten base is not exactly representable
 * either way; what matters is that both platforms take the SAME path. */
static inline double coli_pow(double base, double e) {
    /* base^e = exp2(e * log2(base)); both from +,-,*,/ via a Taylor kernel. */
    /* log2(base): frexp to m in [0.5,1), then log(m) series around 1. */
    int ex; double m = frexp(base, &ex);          /* frexp is exact: it only moves the exponent */
    /* log(m) for m in [0.5,1): use atanh form, z = (m-1)/(m+1) */
    double z = (m - 1.0)/(m + 1.0), z2 = z*z;
    double lg = 2.0*z*(1.0 + z2*(1.0/3.0 + z2*(1.0/5.0 + z2*(1.0/7.0
              + z2*(1.0/9.0 + z2*(1.0/11.0 + z2*(1.0/13.0)))))));
    double log2b = (double)ex + lg*1.4426950408889634074;
    double t = e * log2b;
    /* exp2(t) = 2^n * 2^f, f in [-0.5,0.5]; 2^f = exp(f*ln2) by Taylor */
    double n = (t >= 0.0) ? (double)(long long)(t + 0.5) : (double)(long long)(t - 0.5);
    double f = (t - n) * 0.69314718055994530942;
    double e2 = 1.0 + f*(1.0 + f*(1.0/2.0 + f*(1.0/6.0 + f*(1.0/24.0 + f*(1.0/120.0
              + f*(1.0/720.0 + f*(1.0/5040.0 + f*(1.0/40320.0))))))));
    return ldexp(e2, (int)n);                     /* ldexp is exact for the same reason */
}

#endif
