function APD = compute_APD(V, T, factor)
%APD = compute_APD(V, T, factor)
% Compute the action potential duration at factor percent repolarization

T_half = max(T)/2;
[~, idx_T_half] = min(abs(T-T_half));

% Set up threshold 
[V_max, max_idx] = max(V(1:idx_T_half));
V_min = min(V);

th = V_min + (1-factor/100)*(V_max-V_min);

% Find start time
t_start = 0;
for n=1:min(max_idx, length(T)-1)
    if V(n+1) > th && V(n) < th % n is lower point
        idx1 = n;
        v_u = V(idx1); v_o = V(idx1 + 1);
        t_u = T(idx1); t_o = T(idx1 + 1);
        t_start = (t_o-t_u)/(v_o-v_u)*(th - (t_o*v_u - t_u*v_o)/(t_o-t_u));
        break
    end
end

% Find end time
t_end = inf;
for n=max(2, max_idx):length(T)
    try
        if V(n-1) > th && V(n) < th % n is lower point
            idx2 = n;
            v_u = V(idx2); v_o = V(idx2 - 1);
            t_u = T(idx2); t_o = T(idx2 - 1);
            t_end = (t_o-t_u)/(v_o-v_u)*(th - (t_o*v_u - t_u*v_o)/(t_o-t_u));
            break
        end
    catch
        disp(n)
    end
end

APD = t_end - t_start;

end

