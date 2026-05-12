function [states, varargout] = tentusscher_init_states_verapamil()
  % % Default state values for ODE model: tentusscher_2006_epi with
  % verapamil
  
  load('tentusscher_init_verapamil.mat')


  if nargout == 2

    % --- State names --- 
    state_names = cell(19, 1);

    % --- Xr1 gate ---
    state_names{1} = 'Xr1';

    % --- Xr2 gate ---
    state_names{2} = 'Xr2';

    % --- Xs gate ---
    state_names{3} = 'Xs';

    % --- m gate ---
    state_names{4} = 'm';

    % --- h gate ---
    state_names{5} = 'h';

    % --- j gate ---
    state_names{6} = 'j';

    % --- d gate ---
    state_names{7} = 'd';

    % --- f gate ---
    state_names{8} = 'f';

    % --- F2 gate ---
    state_names{9} = 'f2';

    % --- FCass gate ---
    state_names{10} = 'fCass';

    % --- s gate ---
    state_names{11} = 's';

    % --- r gate ---
    state_names{12} = 'r';

    % --- Calcium dynamics ---
    state_names{13} = 'Ca_i';
    state_names{14} = 'R_prime';
    state_names{15} = 'Ca_SR';
    state_names{16} = 'Ca_ss';

    % --- Sodium dynamics ---
    state_names{17} = 'Na_i';

    % --- Membrane ---
    state_names{18} = 'V';

    % --- Potassium dynamics ---
    state_names{19} = 'K_i';
    varargout(1) = {state_names};
  end
end