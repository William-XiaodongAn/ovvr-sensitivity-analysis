function H = cost_terms(v, model)
%H = cost_terms(v, model)


terms = zeros(length(model.w), 1);
V = model.V1;

if model.w(1) > 0
    terms(1) = model.w(1)*norm(v-V)/norm(V);
else
    terms(1) = 0;
end
    
%%%% H_2 and H_3: Peaks
if model.w(2) > 0 || model.w(3) > 0
    [tp_v, vp_v] = find_peak(model.t_sample, v);
    [tp_V, vp_V] = find_peak(model.t_sample, model.V);
    terms(2) = model.w(2)*abs(tp_v - tp_V)/abs(tp_V);
    terms(3) = model.w(3)*abs(vp_v - vp_V)/abs(vp_V);
else
    terms(2) = 0;
    terms(3) = 0;
end
    
%%%% H_4: Max norm
if model.w(4) > 0
    terms(4) = model.w(4)*(max(abs(v-V))/max(abs(V)));
else
    terms(4) = 0;
end

    
%%%% H_5: dvdt_max
if model.w(5) > 0 
    dvdt_v = compute_dvdt_max(v, model.t_sample);
    dvdt_V = compute_dvdt_max(V, model.t_sample);
    terms(5) = model.w(5)*abs(dvdt_v -dvdt_V)/abs(dvdt_V);
else
    terms(5) = 0;
end


% H_6 - H_21: APD
APDs = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90];
APD_idx = 5 + (1:length(APDs));

for n = 1:length(APDs)
    if model.w(APD_idx(n)) > 0
        APD_v = compute_APD(v, model.t_sample, APDs(n));
        APD_V = compute_APD(V, model.t_sample, APDs(n));
        terms(APD_idx(n)) = model.w(APD_idx(n))*abs(APD_v - APD_V)/abs(APD_V);
    else
        terms(APD_idx(n)) = 0;
    end
    
    
end
      
% H_23: Integral of AP at 30 percent
if model.w(23) > 0
    int_30_v = compute_integral(v, model.t_sample, 30);
    int_30_V = compute_integral(V, model.t_sample, 30);
    terms(23) = model.w(23)*abs(int_30_v - int_30_V)/abs(int_30_V);
else
    terms(23) = 0;
end

H = sum(terms);


end

