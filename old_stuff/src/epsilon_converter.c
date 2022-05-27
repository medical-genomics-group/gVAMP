// Author : E. Orliac
//          DCSR, UNIL
// Date   : 2019/06/04
// Purpose: Converts a binary epsilon file to txt.
//          Allows for sanity check on estimates retrieval
//          during development phase.
// Compil.: icc code.c -o exe
// Run    : ./exe path_to_epsilon_file [> out.txt]
// Note   : by design the .eps only contains last saved iteration!
//          format in: uint,uint,double[i] i~[0,n_indiv]
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char * argv[]) {
    printf("__CONVERT EPSILON FILES TO TEXT__\n");

    if (argc != 2) {
        printf("Wrong number of arguments passed: %d; expected: 1 (path to .eps file to convert)!\n", argc - 1);
        exit(1);
    }

    FILE* fh = fopen(argv[1], "rb");
    if (fh == NULL) {
        printf("Error opening file: %s\n", argv[1]);
        exit(1);
    }

    uint iter = 0;
    fread(&iter, sizeof(uint), 1, fh);
    printf("iteration %d was last logged into epsilon file.\n", iter);

    uint N = 0;
    fread(&N, sizeof(uint), 1, fh);
    printf("%d individuals were processed.\n", N);

    double eps;
    for (uint i=0; i<N; ++i) {
        fseek(fh, sizeof(uint) + sizeof(uint) + sizeof(double) * i, SEEK_SET);
        fread(&eps, sizeof(double), 1, fh);
        printf("%5d/%7d = %20.11f\n", iter, i, eps);
    }
    
    fclose(fh);

    return 0;
}
