function [model, m] = tentusscher_model()
%[model, m] = tentusscher_model()
% Set up the model

model.init_states = @tentusscher_init_states;
model.init_parameters = @tentusscher_init_parameters;
[~, param_names] = tentusscher_init_parameters;
model.rhs = @tentusscher_rhs;
model.V_idx = 18;
m = tentusscher_idx();
model.name = 'tentusscher';
model.stim_idx = find(strcmp('stim_amplitude', param_names));
model.stim_start_idx = find(strcmp('stim_start', param_names));
model.stim_amp = 40.0;
model.options = [];

end

