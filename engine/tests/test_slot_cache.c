/* test_slot_cache -- the GPU expert slot cache policy (src/moe_slot_cache.h),
 * no GPU. Each property has a control: the same scenario with the policy
 * flipped, or a deliberately wrong expectation, must fail. */
#include <stdio.h>
#include "../src/moe_slot_cache.h"
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } } while (0)
int main(void) {
    ColiSlotCache c; c.init(2, 8, 3);
    int hs[4]; std::vector<std::pair<int,int>> fp;
    /* 1. cold cache, cap 0: everything misses, nothing is planned */
    int s1[4] = {1,2,3,4};
    c.plan(0, s1, 4, 0, 0, hs, fp);
    CHECK(c.misses == 4 && c.hits == 0 && fp.empty(), "cold cap0: hits %llu misses %llu plan %zu", (unsigned long long)c.hits, (unsigned long long)c.misses, fp.size());
    /* 2. cap 2, recency: experts 1..4 all last active at step 1 (tie) -> lowest ids 1,2 fetched into empty slots */
    c.plan(0, s1, 4, 2, 0, hs, fp);
    CHECK(fp.size() == 2 && s1[fp[0].first] == 1 && s1[fp[1].first] == 2, "cap2 tie -> lowest ids: got %zu plans, ids %d %d", fp.size(), fp.size()>0?s1[fp[0].first]:-1, fp.size()>1?s1[fp[1].first]:-1);
    for (auto &p : fp) c.commit(p.second, 0, s1[p.first], 1);
    CHECK(c.slot_of(0,1) >= 0 && c.slot_of(0,2) >= 0 && c.fetches == 2, "commit bound the keys");
    /* 3. recency beats id: make expert 7 active (step 4), then route {5,7} with cap 1 -> 7 (recent) fetched, not 5 (lowest) */
    int s7[1] = {7}; c.plan(0, s7, 1, 0, 0, hs, fp);
    int s57[2] = {5,7}; c.plan(0, s57, 2, 1, 0, hs, fp);
    CHECK(fp.size() == 1 && s57[fp[0].first] == 7, "recency: expected 7, got %d", fp.size()?s57[fp[0].first]:-1);
    /* control: lowest-id policy on the identical state must pick 5 */
    { ColiSlotCache d = c; std::vector<std::pair<int,int>> fpd; d.plan(0, s57, 2, 1, 1, hs, fpd);
      CHECK(fpd.size() == 1 && s57[fpd[0].first] == 5, "control lowest-id: expected 5, got %d", fpd.size()?s57[fpd[0].first]:-1); }
    /* commit the recency plan: 7 lands in the third (empty) slot at step 4 */
    for (auto &p : fp) c.commit(p.second, 0, s57[p.first], 1);
    CHECK(c.slot_of(0,7) >= 0 && c.slot_of(0,5) < 0, "7 resident, 5 not");
    /* 4. LRU victim: slots hold 1 (used step 2), 2 (step 2), 7 (step 4); touch 2 (step 5), then fetch 5 (step 6) -> victim is 1's slot */
    int s2[1] = {2}; c.plan(0, s2, 1, 0, 0, hs, fp); CHECK(hs[0] >= 0 && c.hits == 1, "2 is a hit");
    int s5[1] = {5}; c.plan(0, s5, 1, 1, 0, hs, fp);
    int slot1 = c.slot_of(0,1);
    CHECK(fp.size() == 1 && fp[0].second == slot1, "LRU victim should be expert 1's slot %d, got %d", slot1, fp.size()?fp[0].second:-1);
    for (auto &p : fp) c.commit(p.second, 0, s5[p.first], 1);
    CHECK(c.slot_of(0,1) < 0 && c.slot_of(0,5) >= 0 && c.evictions == 1, "eviction recorded");
    /* 5. a slot hit this step is never a victim: route {2,7,5} (all hits) + 3 with cap 1 -> no free victim -> no plan */
    int s3[4] = {2,7,5,3}; c.plan(0, s3, 4, 1, 0, hs, fp);
    CHECK(fp.empty() && hs[0]>=0 && hs[1]>=0 && hs[2]>=0 && hs[3]<0, "no victim among slots used this step: plan %zu", fp.size());
    /* 6. layers are independent keys */
    CHECK(c.slot_of(1,2) < 0, "layer 1 expert 2 is not resident");
    /* 7. abandon empties */
    c.abandon(c.slot_of(0,2)); CHECK(c.slot_of(0,2) < 0, "abandon");
    /* 8. eligibility mask: route {1,6} into a cache where both miss, cap 2, mask {0,1} -> only 6 planned; control mask {1,1} plans both */
    { ColiSlotCache e; e.init(1, 8, 4); int s16[2] = {1,6}; int m01[2] = {0,1}, m11[2] = {1,1};
      e.plan(0, s16, 2, 2, 0, hs, fp, m01); CHECK(fp.size() == 1 && s16[fp[0].first] == 6, "mask: expected only 6, got %zu plans", fp.size());
      ColiSlotCache e2; e2.init(1, 8, 4); e2.plan(0, s16, 2, 2, 0, hs, fp, m11); CHECK(fp.size() == 2, "control mask all: expected 2 plans, got %zu", fp.size()); }
    /* apparatus control: a wrong expectation must be caught */
    int before = fails; CHECK(c.slot_of(0,7) < 0, "(expected failure) 7 is resident"); int caught = fails > before; fails = before;
    printf("%s (apparatus control %s)\n", fails ? "FAIL" : "PASS", caught ? "can fail" : "INERT");
    return (fails == 0 && caught) ? 0 : 1;
}
