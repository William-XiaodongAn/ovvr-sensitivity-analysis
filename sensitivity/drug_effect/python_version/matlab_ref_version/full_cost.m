function H = full_cost(x, model)

model.current_lambda = x;

% Compute solution
[s, t] = compute_solution(model, [0, model.period], x);
v = s(:, model.V_idx);

% Interpolate solution to the model's t_sample
v = interp1(t, v, model.t_sample);

% Compute cost
H = cost_terms(v, model);



end


