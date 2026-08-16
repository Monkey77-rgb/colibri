#include "../src/cpu_features.h"
#include <stdio.h>
int main(void){
  char b[256]; coli_cpu_describe(b,sizeof b);
  printf("cpu: %s\n", b);
  for(int l=1;l<=3;l++){ uint64_t c=coli_cache_bytes(l);
    printf("L%d: %llu bytes (%.1f MiB)%s\n",l,(unsigned long long)c,c/1048576.0,c?"":"  <- UNKNOWN"); }
  return 0; }
