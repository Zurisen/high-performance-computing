function [f] = init_f(N, f)

	rangX = round(5*N/16);
	rangY = round(0.25*N);
	rangMinZ = round(N/6);
	rangMaxZ = round(0.5*N);

	% Fill radiator f with zeros 
    for i = rangX:N
        for j = rangY: N
            for k = 1:N
                f(i,j,k) = 0;
            end
        end
    end


	% Fill radiator f with limit
    for r = 1:rangX
        for s = 1:rangY
            for t = rangMinZ:rangMaxZ
              f(r,s,t) = 200.0;
            end
        end
    end
end