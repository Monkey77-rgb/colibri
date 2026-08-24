#!/usr/bin/env python3
"""Generate shaders/gemm_i4_tile.comp for a given register-tile shape.

The batched GEMM's speed is dominated by its tile shape, and the optimum is not
derivable -- it was measured, twice, and MOVED between the two sweeps because the
first was taken while the engine was still stalled on PCIe (see the header of the
generated shader). So the shape has to stay easy to re-sweep rather than being
hand-edited, and the inner loop has to be emitted rather than written: at TN=16
it is 64 dot products with distinct register names, which is not something to
maintain by hand.

  usage: gen_tile.py TM TN TSR TSC [KC]
  constraint: (TSR/TM) * (TSC/TN) == 256 threads

The doc comment at the top of the existing shader is preserved verbatim -- it
carries the measurement history and must not be regenerated away.
"""
import sys, os

TM, TN, TSR, TSC = (int(x) for x in sys.argv[1:5])
KC = int(sys.argv[5]) if len(sys.argv) > 5 else 128
DB = int(sys.argv[6]) if len(sys.argv) > 6 else 0   # 1 = prefetch next chunk into registers
WG = 256
if (TSR//TM)*(TSC//TN) != WG:
    sys.exit(f"(TSR/TM)*(TSC/TN) must be {WG}, got {(TSR//TM)*(TSC//TN)}")

here = os.path.dirname(os.path.abspath(__file__))
shader = os.path.join(here, "..", "shaders", "gemm_i4_tile.comp")
src = open(shader).read()
head = src[:src.index("#define TSR ")]

acc = [f"c{a}_{b}" for a in range(TM) for b in range(TN)]
sm  = [f"s{a}_{b}" for a in range(TM) for b in range(TN)]
D = lambda t, ns: "\n".join("    " + t + " " + ", ".join(f"{ns[a*TN+b]}=0" for b in range(TN)) + ";" for a in range(TM))

DIRECT_STAGE = """        int kbase = chunk * KC;
        for (int idx = lid; idx < TSR * (KC/4); idx += WG) {
            int rr = idx / (KC/4), jj = idx % (KC/4);
            int r  = r0 + rr;
            As[rr][jj] = (r < p.n) ? x_packed[r * xwords + kbase/4 + jj] : 0u;
        }
        for (int idx = lid; idx < TSC * u4pc; idx += WG) {
            int o = idx / u4pc, q = idx % u4pc;
            int og = o0 + o;
            if (og < p.O) {
                uvec4 wv4 = w_packed4[og * wuvec4 + chunk*u4pc + q];
                for (int c = 0; c < 4; c++) {
                    uint wv = wv4[c], lo = 0u, hi = 0u;
                    for (int k = 0; k < 4; k++) { int v = int((wv >> (4*k)) & 0xFu) - 8;
                        lo |= (uint(v) & 0xFFu) << (8*k); }
                    for (int k = 0; k < 4; k++) { int v = int((wv >> (4*(k+4))) & 0xFu) - 8;
                        hi |= (uint(v) & 0xFFu) << (8*k); }
                    Bs[o][q*8 + c*2] = lo; Bs[o][q*8 + c*2 + 1] = hi;
                }
                Sw[o][q] = w_bscale[og * wnb + chunk*u4pc + q];
            } else {
                for (int c = 0; c < 8; c++) Bs[o][q*8 + c] = 0u;
                Sw[o][q] = 0.0;
            }
        }
        for (int idx = lid; idx < TSR * (KC/16); idx += WG) {
            int rr = idx / (KC/16), jj = idx % (KC/16);
            int r  = r0 + rr;
            Sx[rr][jj] = (r < p.n) ? x_scale[r * p.nblk + chunk*(KC/16) + jj] : 0.0;
        }"""

DB_DECL_SRC = """
/* DOUBLE BUFFERING WITHOUT cp.async. Vulkan has no asynchronous copy, but the
 * same latency hiding falls out of ORDER: issue chunk k+1's global loads into
 * registers BEFORE computing on chunk k. Their results are not consumed until
 * the next iteration, so the compiler can keep them in flight across the whole
 * compute block instead of stalling at a barrier. Without it the kernel eats
 * full global latency once per chunk, 32 times for a 4096-wide row.
 *
 * The cost is register pressure -- roughly 30 more live values on top of TM*TN
 * accumulators, which at TN=16 is already 64. If it spills this is SLOWER, which
 * is why it is a switch and not a rewrite. */
#define AREG  (TSR*(KC/4)/WG)
#define BREG  (TSC*(KC/32)/WG)
#define SXREG (TSR*(KC/16)/WG)
    uint  aReg[AREG];
    uvec4 bReg[BREG];
    float swReg[BREG];
    float sxReg[SXREG];

#define LOAD_CHUNK(ch) { \\
    for (int t = 0; t < AREG; t++) { int idx = lid + t*WG; \\
        int rr = idx / (KC/4), jj = idx % (KC/4); int r = r0 + rr; \\
        aReg[t] = (r < p.n) ? x_packed[r * xwords + ((ch)*KC)/4 + jj] : 0u; } \\
    for (int t = 0; t < BREG; t++) { int idx = lid + t*WG; \\
        int o = idx / u4pc, q = idx % u4pc; int og = o0 + o; \\
        bReg[t]  = (og < p.O) ? w_packed4[og*wuvec4 + (ch)*u4pc + q] : uvec4(0u); \\
        swReg[t] = (og < p.O) ? w_bscale[og*wnb + (ch)*u4pc + q] : 0.0; } \\
    for (int t = 0; t < SXREG; t++) { int idx = lid + t*WG; \\
        int rr = idx / (KC/16), jj = idx % (KC/16); int r = r0 + rr; \\
        sxReg[t] = (r < p.n) ? x_scale[r * p.nblk + (ch)*(KC/16) + jj] : 0.0; } }

    LOAD_CHUNK(0)
"""

DB_STAGE_SRC = """        barrier();
        for (int t = 0; t < AREG; t++) { int idx = lid + t*WG;
            As[idx / (KC/4)][idx % (KC/4)] = aReg[t]; }
        for (int t = 0; t < BREG; t++) {
            int idx = lid + t*WG; int o = idx / u4pc, q = idx % u4pc;
            uvec4 wv4 = bReg[t];
            for (int c = 0; c < 4; c++) {
                uint wv = wv4[c], lo = 0u, hi = 0u;
                for (int k = 0; k < 4; k++) { int v = int((wv >> (4*k)) & 0xFu) - 8;
                    lo |= (uint(v) & 0xFFu) << (8*k); }
                for (int k = 0; k < 4; k++) { int v = int((wv >> (4*(k+4))) & 0xFu) - 8;
                    hi |= (uint(v) & 0xFFu) << (8*k); }
                Bs[o][q*8 + c*2] = lo; Bs[o][q*8 + c*2 + 1] = hi;
            }
            Sw[o][q] = swReg[t];
        }
        for (int t = 0; t < SXREG; t++) { int idx = lid + t*WG;
            Sx[idx / (KC/16)][idx % (KC/16)] = sxReg[t]; }
        barrier();
        if (chunk + 1 < chunks) LOAD_CHUNK(chunk + 1)
"""

DB_DECL  = DB_DECL_SRC if DB else ""
DB_STAGE = DB_STAGE_SRC if DB else DIRECT_STAGE

body = f"""#define TSR {TSR}
#define TSC {TSC}
#define KC  {KC}
#define TM  {TM}
#define TN  {TN}
#define WG  {WG}

layout(local_size_x = WG) in;

layout(std430, binding = 0) readonly  buffer W  {{ uvec4 w_packed4[]; }};
layout(std430, binding = 1) readonly  buffer WS {{ float w_bscale[];  }};
layout(std430, binding = 2) readonly  buffer X  {{ uint  x_packed[];  }};
layout(std430, binding = 3) readonly  buffer XS {{ float x_scale[];   }};
layout(std430, binding = 4) readonly  buffer XM {{ int   x_sum[];     }};
layout(std430, binding = 5) writeonly buffer Y  {{ float y[];         }};

layout(push_constant) uniform P {{ int I; int O; int n; int nblk; int tile; int outs; }} p;

/* +1 pads the row stride: without it the bank of [row][j] is j % 32 regardless
 * of row, so a half-warp reading different rows at one j hits ONE bank. */
shared uint  As[TSR][KC/4 + 1];
shared uint  Bs[TSC][KC/4 + 1];
shared float Sw[TSC][KC/32];
shared float Sx[TSR][KC/16];

void main() {{
    int otiles = (p.O + TSC - 1) / TSC;
    int gid = int(gl_WorkGroupID.x);
    int o0  = (gid % otiles) * TSC;
    int r0  = (gid / otiles) * TSR;
    if (r0 >= p.n) return;
    int lid = int(gl_LocalInvocationID.x);
    int tx  = lid % (TSC/TN);
    int ty  = lid / (TSC/TN);
    int xwords = p.I / 4;
    int wuvec4 = p.I / 32;
    int wnb    = p.I / 32;

{D("float", acc)}

    int chunks = p.I / KC;
    int u4pc   = KC / 32;
{DB_DECL}
    for (int chunk = 0; chunk < chunks; chunk++) {{
{DB_STAGE}
        barrier();
        for (int sub = 0; sub < KC/16; sub++) {{
{D("int", sm)}
            int rb = ty*TM, cb = tx*TN;
            for (int w = 0; w < 4; w++) {{
                int j = sub*4 + w;
{"                " + " ".join(f"int x{a}=int(As[rb+{a}][j]);" for a in range(TM))}
{"                " + " ".join(f"int b{b}=int(Bs[cb+{b}][j]);" for b in range(TN))}
{chr(10).join("                " + " ".join(f"{sm[a*TN+b]}=dotPacked4x8AccSatEXT(b{b},x{a},{sm[a*TN+b]});" for b in range(TN)) for a in range(TM))}
            }}
{"            " + " ".join(f"float w{b}=Sw[cb+{b}][sub/2];" for b in range(TN))}
{"            " + " ".join(f"float y{a}=Sx[rb+{a}][sub];" for a in range(TM))}
{chr(10).join("            " + " ".join(f"{acc[a*TN+b]}+=w{b}*y{a}*float({sm[a*TN+b]});" for b in range(TN)) for a in range(TM))}
        }}
        barrier();
    }}

    float outv[TM*TN] = float[TM*TN]({",".join(acc)});
    for (int a = 0; a < TM; a++) {{
        int r = r0 + ty*TM + a;
        if (r >= p.n) continue;
        for (int b = 0; b < TN; b++) {{
            int o = o0 + tx*TN + b;
            if (o < p.O) y[r * p.O + o] = outv[a*TN + b];
        }}
    }}
}}
"""
open(shader, "w").write(head + body)
print(f"gemm_i4_tile.comp: TM={TM} TN={TN} tile={TSR}x{TSC} KC={KC}")
