function states = update_initial_conditions(states, param, model)
%states = update_initial_conditions(states, param, model)

% Run a simulation
N = model.num_cycles;
if isfield(model, 'pace_length')
    Tstop = model.pace_length;
else
    Tstop = 1000;
end

param(model.stim_idx) = model.stim_amp;
T=[];
Y=[];

for n=1:N
    
    % Adjust last stimulation to keep a constant cycle length
    if n==N
        Tstop = max(Tstop - model.t_stim, 0);
    end
    
    if Tstop > 0
        [T_new, Y_new] = ode15s(@model.rhs, [0, Tstop], states, model.options, param);
        states = Y_new(end,:)';
    end
    if n> 1
        T = [T; T(end)+T_new];
    else
        T = [T; T_new];
    end
    Y = [Y; Y_new];
        
    
end
    
end

