int cg_dense(int n, int m, double *A, double *b, double *x);
int cg_sparse(int n, int m, int nnz, int *irn, int *jcn, double *val, double *b, double *x);

int lsmr_dense(int n, int m, double *A, double *b, double *x);
int lsmr_sparse(int n, int m, int nnz, int *irn, int *jcn, double *val, double *b, double *x);

int craig_dense(int n, int m, double *A, double *b, double *x);
int craig_sparse(int n, int m, int nnz, int *irn, int *jcn, double *val, double *b, double *x);
