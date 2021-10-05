#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char * argv[]) {
    printf("__CHECK BETA VALUES__\n");

    if (argc != 4) {
        printf("Wrong number of arguments passed: %d; expected 3 (path to .bet file, marker ID, and iteration)!\n", argc - 1);
        exit(1);
    }

    FILE* fh = fopen(argv[1], "rb");
    if (fh == NULL) {
        printf("Error opening file: %s\n", argv[1]);
        exit(1);
    }

    uint marker = atoi(argv[2]);
    uint iter   = atoi(argv[3]);

    uint M = 0;
    fread(&M, sizeof(uint), 1, fh);
    printf("%d markers were processed.\n", M);

    double beta;
    fseek(fh, sizeof(uint) + sizeof(double) * (M * iter + marker), SEEK_SET);

    fread(&beta, sizeof(double), 1, fh);
    printf("Beta for marker %7d = %15.10f\n", marker, beta);

    fclose(fh);

    return 0;
}
