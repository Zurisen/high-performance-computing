% jacobi.c - Poisson problem in 3d
function [uNew, d, iter] = gauss_seidel(uNew, uOld,uSwap, f, N, iter_max, gridSpace, tolerance) 
    invCube = 1/6;
    d=10e+5;
    iter = 0;
    formatSpec = 'iteration: %8.8f / distance %8.8f \n';
    while or(iter < iter_max , d > tolerance)
        iter=iter+1;
        d = 0.0;
        uSwap = uNew;
        uNew = uOld;
        uOld = uSwap;

        for i = 2:N-1
            for j = 2:N-1
                for k = 2:N-1

                    %/* Compute update of uNew */
                    uNew(i,j,k) = invCube*(uNew(i-1,j,k) + uOld(i+1,j,k) + uNew(i,j-1,k) + uOld(i,j+1,k) + uNew(i,j,k-1) + uOld(i,j,k+1) + gridSpace*f(i,j,k));

                    %/* Compute new d */
                    %/* NOTE: maybe this can be parallelized better if splitted into a different nested loop */
                    d =d + abs(uNew(i,j,k) - uOld(i,j,k));
                end
            end
        end
        
        fprintf(formatSpec,iter,d)
    end
end


