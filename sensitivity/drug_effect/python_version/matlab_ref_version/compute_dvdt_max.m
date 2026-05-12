function dvdt = compute_dvdt_max(V, T)
%dvdt = compute_dvdt_max(V, T)
%   Compute the maximum upstroke velocity

dvdt = max((V(2:end)-V(1:end-1))./(T(2:end)-T(1:end-1)));

end

