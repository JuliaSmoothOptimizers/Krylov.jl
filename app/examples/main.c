#include <stdio.h>
#include <krylov.h>
#include <julia_init.h>

int main(int argc, char **argv) {
  init_julia(argc, argv);

  int i, n, m, nnz;

  /* Symmetric and positive definite linear system */
  n = 5;
  m = 5;
  nnz = 15;
  int irn1[15] = {1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5};
  int jcn1[15] = {1, 3, 4, 5, 2, 1, 3, 5, 1, 4, 5, 1, 3, 4, 5};
  double val1[15] = {53.0, 8.0, 4.0, 3.0, 10.0, 8.0, 6.0, 8.0, 4.0, 26.0, 5.0, 3.0, 8.0, 5.0, 14.0};
  double b1[5] = {108.0, 20.0, 66.0, 133.0, 117.0};
  double x1[5];

  cg_sparse(n, m, nnz, irn1, jcn1, val1, b1, x1);

  printf("Expected result with CG is x = 1.00000 2.00000 3.00000 4.00000 5.00000\n");
  printf("Computed result with CG is x = ");
  for(i=0; i<5; i++){
    printf("%7.5f ", x1[i]);
  }
  printf("\n");

  /* Least-squares problem */
  n = 7;
  m = 5;
  nnz = 13;
  int irn2[13] = {1, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7};
  int jcn2[13] = {1, 3, 5, 2, 3, 5, 1, 4, 4, 5, 2, 1, 3};
  double val2[13] = {1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 4.0, 1.0, 5.0, 1.0, 3.0, 6.0, 1.0};
  double b2[7] = {22.0, 5.0, 13.0, 8.0, 25.0, 5.0, 9.0};
  double x2[5];

  lsmr_sparse(n, m, nnz, irn2, jcn2, val2, b2, x2);

  printf("Expected result with LSMR is x = 1.00000 2.00000 3.00000 4.00000 5.00000\n");
  printf("Computed result with LSMR is x = ");
  for(i=0; i<5; i++){
    printf("%7.5f ", x2[i]);
  }
  printf("\n");

  /* Least-norm problem */
  n = 5;
  m = 7;
  nnz = 13;
  int irn3[13] = {1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5};
  int jcn3[13] = {3, 5, 7, 1, 4, 6, 2, 6, 5, 6, 3, 4, 7};
  double val3[13] = {2.0, 3.0, 5.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0};
  double b3[5] = {56.0, 21.0, 16.0, 22.0, 25.0};
  double x3[7];

  craig_sparse(n, m, nnz, irn3, jcn3, val3, b3, x3);
  
  printf("Expected result with CRAIG is x = 1.00000 2.00000 3.00000 4.00000 5.00000 6.00000 7.00000\n");
  printf("Computed result with CRAIG is x = ");
  for(i=0; i<7; i++){
    printf("%7.5f ", x3[i]);
  }
  printf("\n");

  shutdown_julia(0);
  return 0;
}
