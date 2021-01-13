function [u] = init_3d(start_T, N, u)

	% initilaize boundaries with bound conditions
	
	%NOTE: Initial conditions in the corner of the box are not well defined
    %u = zeros(N,N,N);
    for  w = 1:N
        for v = 1:N
            u(1,w,v) = 20.0;
            u(N-1,w,v) = 20.0;
            u(w,1,v) = 0.0;
            u(w,N-1,v) = 20.0;
            u(w,v,1) = 20.0;
            u(w,v,N-1) = 20.0;
        end
    end
	
	% Fill inside of the cube with starting temperature 
    
    for i = 2:N-1
        for j = 2:N-1
            for k = 2:N-1
                u(i,j,k) = start_T;
            end
        end
    end
end


