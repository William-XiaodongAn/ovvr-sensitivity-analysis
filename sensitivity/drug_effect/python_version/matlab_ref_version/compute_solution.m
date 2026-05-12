function [s, t] = compute_solution(model, t_sample, lambda)
%[s, t] = compute_solution(model, t_sample, lambda) 
% Compute an action potential for the new value of lambda
%
% Input arguments:
%     model: object containging information about the model used in the minimization
%     t_sample: points in time to compute the solution for (only last element is used)
%     lambda: adjustment factors for the parameters in model.opt_idx
%
% Output arguments:
%     s: state variables of the solution
%     t: time points for the computed solution

% Set up default parameters and initial conditions
param = model.init_parameters();
states = model.init_states();

% Update parameters
param(model.opt_idx) = (1+lambda).*param(model.opt_idx);
states = update_initial_conditions(states, param, model);
 

% Run a simulation for the time before stimulation
if model.t_stim > 0
    param(model.stim_idx) = 0;
    [t, s] = ode15s(@model.rhs, [0, model.t_stim], states, model.options, param);
    states = s(end,:)';
else
    t = 0;
    s = states';
end

% Continue the simulation from the time of stimulation
Tstop = t_sample(end) - model.t_stim;
param(model.stim_idx) = model.stim_amp;
[t_new, s_new] = ode15s(@model.rhs, [0, Tstop], states, model.options, param);
t = [t; t(end) + t_new(2:end)];
s = [s; s_new(2:end,:)];

    
end

