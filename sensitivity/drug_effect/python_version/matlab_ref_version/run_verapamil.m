% Run the singular value decomposition analysis to compute the
% identifiability index of the ten Tusscher model with verapamil
clear all;

% Set up model
Tstop = 500; % Simulation time (in ms)
[model, m] = tentusscher_model();
model.init_parameters = @tentusscher_init_parameters_verapamil;
model.init_states = @tentusscher_init_states_verapamil;
model.opt_idx = [m.Na_idx; m.bNa_idx; m.NaK_idx; m.Kr_idx; m.pK_idx; ...
    m.Ks_idx; m.to_idx; m.K1_idx; m.CaL_idx;  m.NaCa_idx; m.pCa_idx; m.bCa_idx];
names = {'Na', 'bNa', 'NaK', 'Kr', 'pK', 'Ks', 'to', 'K1','CaL', 'NaCa', 'pCa', 'bCa'};
model.t_stim = 100;
lambda = zeros(length(model.opt_idx), 1);
model.num_cycles = 20;
param = model.init_parameters();

% Compute the solution
[s, t_orig] = compute_solution(model, [0, Tstop], lambda);
dt = 0.1;
t = (0:dt:Tstop)';
s = interp1(t_orig, s, t);
V_base = s(:, model.V_idx);

% Compute the currents
currents = tentusscher_currents(s, param);

A = [currents.i_Na, currents.i_bNa, currents.i_NaK, currents.i_Kr, ...
    currents.i_pK, currents.i_Ks, currents.i_to, currents.i_K1, ...
    currents.i_CaL, currents.i_NaCa, currents.i_pCa, currents.i_bCa];

% Singular value decomposition
[~,S,V] = svd(A);
q=size(A); m=q(1); n=q(2);
VV=V';
sigma = diag(S(1:n, 1:n));

% Compute perturbation effect on biomarkers along eigenvectors
eps_values = [-0.5, 0.5];
c = cost_function_index();
model.w = zeros(c.num_w, 1);
model.w(c.APD30_idx) = 1;
model.w(c.APD50_idx) = 1;
model.w(c.APD80_idx) = 1;
model.w(c.dvdt_idx) = 1;
model.w(c.v_idx) = 1;
model.V1 = V_base;
model.period = Tstop;
model.t_sample = t;
model.dt = dt;

H_all = zeros(12, length(eps_values));
for i = 1:length(eps_values)
    eps = eps_values(i);
    H_values = zeros(length(names), 1);
    for n=1:length(names)
        lambda = eps*V(:, n);
        H = full_cost(lambda, model);
        H_values(n) = H;
    end

    H_all(:, i) = H_values;
end


% Determine the nullspace
delta = 0.25;
nullspace = [];
for n=1:length(sigma)
    if max(H_all(n,:)) < delta
        nullspace = [nullspace, n];
    end
end

% Compute identifiability index
s_idx = zeros(length(sigma), 1);
for k=1:length(sigma)
    e = zeros(length(sigma), 1);
    e(k) = 1;
    Pe = zeros(length(sigma), 1);
    for j=nullspace
        Pe = Pe + V(k, j)*V(:, j);
    end
    s_idx(k) = norm(e-Pe);
end

[S_sorted, idx_sorted] = sort(s_idx);
S_sorted = flip(S_sorted);
idx_sorted = flip(idx_sorted);
names = names(idx_sorted);
s_idx = s_idx(idx_sorted);

for k=1:length(names)
    fprintf('%s: %.2f\n', names{k}, s_idx(k))
end