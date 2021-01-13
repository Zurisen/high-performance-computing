N_DEFAULT=10;
N = N_DEFAULT;
iter_max = 30000;
tolerance=1e-3;
start_T=10;
u = zeros(N,N,N);
uOld = zeros(N,N,N);
uSwap = zeros(N,N,N);
f = zeros(N,N,N);

%/* get the parameters from the command line */
gridSpace = 4*(N^(-2));


%/* Initialize the 3d arrays */
u=init_3d(start_T, N, u);
uOld=init_3d(start_T, N, uOld);
f=init_f(N, f);

%/* Jacobi method */
[u, d, iter]= jacobi(u, uOld, uSwap, f, N, iter_max, gridSpace, tolerance);

%/* Gauss Seidel method */
%[u, d, iter]= gauss_seidel(u, uOld, uSwap, f, N, iter_max, gridSpace, tolerance);