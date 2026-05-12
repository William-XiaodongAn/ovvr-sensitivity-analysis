function [parameters, varargout] = tentusscher_init_parameters_cisapride()
  % % Default parameter values for ODE model: tentusscher_2006_epi with
  % cisapride

  if nargout < 1 || nargout > 2
    error('Expected 1-2 output arguments.');
  end

  % --- Default parameters values --- 
  parameters = zeros(53, 1);

  % --- Reversal potentials ---
  parameters(1) = 0.03; % P_kna;

  % --- Inward rectifier potassium current ---
  parameters(2) = 5.405; % g_K1;

  % --- Rapid time dependent potassium current ---
  parameters(3) = 0.153*0.5; % g_Kr;

  % --- Slow time dependent potassium current ---
  parameters(4) = 0.392; % g_Ks;

  % --- Fast sodium current ---
  parameters(5) = 14.838; % g_Na;

  % --- Sodium background current ---
  parameters(6) = 0.00029; % g_bna;

  % --- L_type Ca current ---
  parameters(7) = 3.98e-05; % g_CaL;

  % --- Calcium background current ---
  parameters(8) = 0.000592; % g_bca;

  % --- Transient outward current ---
  parameters(9) = 0.294; % g_to;

  % --- Sodium potassium pump current ---
  parameters(10) = 40; % K_mNa;
  parameters(11) = 1; % K_mk;
  parameters(12) = 2.724; % P_NaK;

  % --- Sodium calcium exchanger current ---
  parameters(13) = 1000; % K_NaCa;
  parameters(14) = 0.1; % K_sat;
  parameters(15) = 1.38; % Km_Ca;
  parameters(16) = 87.5; % Km_Nai;
  parameters(17) = 2.5; % alpha;
  parameters(18) = 0.35; % gamma;

  % --- Calcium pump current ---
  parameters(19) = 0.0005; % K_pCa;
  parameters(20) = 0.1238; % g_pCa;

  % --- Potassium pump current ---
  parameters(21) = 0.0146; % g_pK;

  % --- Calcium dynamics ---
  parameters(22) = 0.2; % Buf_c;
  parameters(23) = 10; % Buf_sr;
  parameters(24) = 0.4; % Buf_ss;
  parameters(25) = 2; % Ca_o;
  parameters(26) = 1.5; % EC;
  parameters(27) = 0.001; % K_buf_c;
  parameters(28) = 0.3; % K_buf_sr;
  parameters(29) = 0.00025; % K_buf_ss;
  parameters(30) = 0.00025; % K_up;
  parameters(31) = 0.00036; % V_leak;
  parameters(32) = 0.102; % V_rel;
  parameters(33) = 0.001094; % V_sr;
  parameters(34) = 5.468e-05; % V_ss;
  parameters(35) = 0.0038; % V_xfer;
  parameters(36) = 0.006375; % Vmax_up;
  parameters(37) = 0.15; % k1_prime;
  parameters(38) = 0.045; % k2_prime;
  parameters(39) = 0.06; % k3;
  parameters(40) = 0.005; % k4;
  parameters(41) = 2.5; % max_sr;
  parameters(42) = 1; % min_sr;

  % --- Sodium dynamics ---
  parameters(43) = 140; % Na_o;

  % --- Membrane ---
  parameters(44) = 0.185; % Cm;
  parameters(45) = 96485.3415; % F;
  parameters(46) = 8314.472; % R;
  parameters(47) = 310; % T;
  parameters(48) = 0.016404; % V_c;
  parameters(49) = 40; % stim_amplitude;
  parameters(50) = 1; % stim_duration;
  parameters(51) = 1000; % stim_period;
  parameters(52) = 0; % stim_start;

  % --- Potassium dynamics ---
  parameters(53) = 5.4; % K_o;

  if nargout == 2

    % --- Parameter names --- 
    parameter_names = cell(53, 1);

    % --- Reversal potentials ---
    parameter_names{1} = 'P_kna';

    % --- Inward rectifier potassium current ---
    parameter_names{2} = 'g_K1';

    % --- Rapid time dependent potassium current ---
    parameter_names{3} = 'g_Kr';

    % --- Slow time dependent potassium current ---
    parameter_names{4} = 'g_Ks';

    % --- Fast sodium current ---
    parameter_names{5} = 'g_Na';

    % --- Sodium background current ---
    parameter_names{6} = 'g_bna';

    % --- L_type Ca current ---
    parameter_names{7} = 'g_CaL';

    % --- Calcium background current ---
    parameter_names{8} = 'g_bca';

    % --- Transient outward current ---
    parameter_names{9} = 'g_to';

    % --- Sodium potassium pump current ---
    parameter_names{10} = 'K_mNa';
    parameter_names{11} = 'K_mk';
    parameter_names{12} = 'P_NaK';

    % --- Sodium calcium exchanger current ---
    parameter_names{13} = 'K_NaCa';
    parameter_names{14} = 'K_sat';
    parameter_names{15} = 'Km_Ca';
    parameter_names{16} = 'Km_Nai';
    parameter_names{17} = 'alpha';
    parameter_names{18} = 'gamma';

    % --- Calcium pump current ---
    parameter_names{19} = 'K_pCa';
    parameter_names{20} = 'g_pCa';

    % --- Potassium pump current ---
    parameter_names{21} = 'g_pK';

    % --- Calcium dynamics ---
    parameter_names{22} = 'Buf_c';
    parameter_names{23} = 'Buf_sr';
    parameter_names{24} = 'Buf_ss';
    parameter_names{25} = 'Ca_o';
    parameter_names{26} = 'EC';
    parameter_names{27} = 'K_buf_c';
    parameter_names{28} = 'K_buf_sr';
    parameter_names{29} = 'K_buf_ss';
    parameter_names{30} = 'K_up';
    parameter_names{31} = 'V_leak';
    parameter_names{32} = 'V_rel';
    parameter_names{33} = 'V_sr';
    parameter_names{34} = 'V_ss';
    parameter_names{35} = 'V_xfer';
    parameter_names{36} = 'Vmax_up';
    parameter_names{37} = 'k1_prime';
    parameter_names{38} = 'k2_prime';
    parameter_names{39} = 'k3';
    parameter_names{40} = 'k4';
    parameter_names{41} = 'max_sr';
    parameter_names{42} = 'min_sr';

    % --- Sodium dynamics ---
    parameter_names{43} = 'Na_o';

    % --- Membrane ---
    parameter_names{44} = 'Cm';
    parameter_names{45} = 'F';
    parameter_names{46} = 'R';
    parameter_names{47} = 'T';
    parameter_names{48} = 'V_c';
    parameter_names{49} = 'stim_amplitude';
    parameter_names{50} = 'stim_duration';
    parameter_names{51} = 'stim_period';
    parameter_names{52} = 'stim_start';

    % --- Potassium dynamics ---
    parameter_names{53} = 'K_o';
    varargout(1) = {parameter_names};
  end
end