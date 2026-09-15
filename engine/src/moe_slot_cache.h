/* moe_slot_cache.h -- the GPU expert slot cache POLICY, and nothing else
 * (2026-09-14). Pure bookkeeping over integers so tests/test_slot_cache.c can
 * exercise it without a GPU; model.cpp owns the IO (store fetch, repack, slot
 * fill) and the compute.
 *
 * The design is FreeToken's (FlashML-org, python/freetoken/moe/offload_kernels.py,
 * Apache-2.0): a flat set of VRAM slots each holding one (layer, expert); per
 * layer per decode step the routed experts are classified hit/miss; at most
 * `cap` misses are fetched into slots this step (the GPU computes hits + fetched,
 * the CPU the rest); which misses get fetched is by RECENCY -- the miss that was
 * active most recently before this step goes first, so recurring misses win
 * slots over one-off ones -- or, as the control, by lowest expert id; the slot
 * given up is the least-recently-USED one that this step is not itself using.
 * cap == 0 is the pre-existing behaviour: whatever was pinned at load stays. */
#pragma once
#include <stdint.h>
#include <vector>
#include <algorithm>

struct ColiSlotCache {
    int NL = 0, NE = 0, nslots = 0;
    std::vector<int>      slot_key;        /* [nslots] layer*NE+e, or -1 = empty */
    std::vector<uint64_t> slot_last_use;   /* [nslots] step of last hit/fill */
    std::vector<int>      key_slot;        /* [NL*NE] slot index or -1 */
    std::vector<uint64_t> key_last_active; /* [NL*NE] last step the expert was routed (0 = never) */
    uint64_t step = 0;
    /* stats over expert-calls (one routed expert of one token on one layer) */
    uint64_t hits = 0, misses = 0, fetches = 0, evictions = 0;

    void init(int layers, int experts, int slots) {
        NL = layers; NE = experts; nslots = slots;
        slot_key.assign(slots, -1); slot_last_use.assign(slots, 0);
        key_slot.assign((size_t)layers*experts, -1); key_last_active.assign((size_t)layers*experts, 0);
        step = 0; hits = misses = fetches = evictions = 0;
    }
    int key(int l, int e) const { return l*NE + e; }
    int slot_of(int l, int e) const { return key_slot[(size_t)key(l,e)]; }

    /* Bind key -> slot after the caller has FILLED the slot. Evicts whatever the
     * slot held. Counts a fetch only when `fetched` (the load-time pin is not one). */
    void commit(int slot, int l, int e, int fetched) {
        int k = key(l,e);
        if (slot_key[slot] >= 0) { key_slot[(size_t)slot_key[slot]] = -1; evictions++; }
        if (key_slot[(size_t)k] >= 0) slot_key[(size_t)key_slot[(size_t)k]] = -1;   /* re-fill of a key already elsewhere: unlikely, kept consistent */
        slot_key[slot] = k; key_slot[(size_t)k] = slot; slot_last_use[slot] = step;
        if (fetched) fetches++;
    }
    /* Mark a slot whose fill FAILED as empty (contents undefined). */
    void abandon(int slot) {
        if (slot_key[slot] >= 0) { key_slot[(size_t)slot_key[slot]] = -1; evictions++; }
        slot_key[slot] = -1; slot_last_use[slot] = 0;
    }

    /* One layer of one decode step. sel[K] = routed expert ids. Advances `step`
     * once per (layer) call -- the recency clock is per layer-call, which is
     * what makes "active most recently before this step" well-defined. Fills
     * hit_slot[k] (slot for a hit, else -1) and fetch_plan: (k index into sel,
     * victim slot) pairs the caller must fill+commit (or abandon) IN ORDER.
     * policy: 0 = recency (default), 1 = lowest expert id (control). */
    /* fetchable[k] (may be nullptr = all): a miss with fetchable[k]==0 is never
     * planned. model.cpp passes "all three matrices are RAM-resident" under
     * COLI_MOE_GPU_RAMONLY, so a promotion never pays a disk read -- measured
     * 2026-09-14 (goss8 s1_rec): fetching store-misses cost 11.9 ms each and
     * halved throughput. */
    void plan(int l, const int *sel, int K, int cap, int policy,
              int *hit_slot, std::vector<std::pair<int,int>> &fetch_plan, const int *fetchable = nullptr) {
        step++;
        fetch_plan.clear();
        std::vector<int> miss; miss.reserve(K);
        std::vector<uint64_t> prev((size_t)K);
        for (int k = 0; k < K; k++) {
            int kk = key(l, sel[k]);
            prev[(size_t)k] = key_last_active[(size_t)kk];
            key_last_active[(size_t)kk] = step;
            int s = key_slot[(size_t)kk];
            hit_slot[k] = s;
            if (s >= 0) { hits++; slot_last_use[s] = step; }
            else { misses++; if (!fetchable || fetchable[k]) miss.push_back(k); }
        }
        if (cap <= 0 || miss.empty() || nslots == 0) return;
        if (policy == 1) std::sort(miss.begin(), miss.end(), [&](int a, int b){ return sel[a] < sel[b]; });
        else std::stable_sort(miss.begin(), miss.end(), [&](int a, int b){
                 if (prev[(size_t)a] != prev[(size_t)b]) return prev[(size_t)a] > prev[(size_t)b];   /* most recently active first */
                 return sel[a] < sel[b]; });
        int nf = (int)miss.size() < cap ? (int)miss.size() : cap;
        std::vector<char> taken((size_t)nslots, 0);
        for (int i = 0; i < nf; i++) {
            /* LRU victim among slots not used (hit or assigned) this step */
            int best = -1; uint64_t bu = ~0ull;
            for (int s = 0; s < nslots; s++) {
                if (taken[(size_t)s] || slot_last_use[s] == step) continue;
                if (slot_key[s] < 0) { best = s; break; }               /* an empty slot first */
                if (slot_last_use[s] < bu) { bu = slot_last_use[s]; best = s; }
            }
            if (best < 0) break;
            taken[(size_t)best] = 1;
            fetch_plan.push_back(std::make_pair(miss[(size_t)i], best));
        }
    }
};
