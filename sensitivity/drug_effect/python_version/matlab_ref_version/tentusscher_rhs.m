function [values] = tentusscher_rhs(t, states, parameters)
  % Compute the right hand side of the tentusscher_2006_epi ODE

  % Assign states
  if length(states)~=19
    error('Expected the states array to be of size 19.');
  end
  Xr1=states(1); Xr2=states(2); Xs=states(3); m=states(4); h=states(5);...
    j=states(6); d=states(7); f=states(8); f2=states(9); fCass=states(10);...
    s=states(11); r=states(12); Ca_i=states(13); R_prime=states(14);...
    Ca_SR=states(15); Ca_ss=states(16); Na_i=states(17); V=states(18);...
    K_i=states(19);

  % Assign parameters
  if length(parameters)~=53
    error('Expected the parameters array to be of size 53.');
  end
  P_kna=parameters(1); g_K1=parameters(2); g_Kr=parameters(3);...
    g_Ks=parameters(4); g_Na=parameters(5); g_bna=parameters(6);...
    g_CaL=parameters(7); g_bca=parameters(8); g_to=parameters(9);...
    K_mNa=parameters(10); K_mk=parameters(11); P_NaK=parameters(12);...
    K_NaCa=parameters(13); K_sat=parameters(14); Km_Ca=parameters(15);...
    Km_Nai=parameters(16); alpha=parameters(17); gamma=parameters(18);...
    K_pCa=parameters(19); g_pCa=parameters(20); g_pK=parameters(21);...
    Buf_c=parameters(22); Buf_sr=parameters(23); Buf_ss=parameters(24);...
    Ca_o=parameters(25); EC=parameters(26); K_buf_c=parameters(27);...
    K_buf_sr=parameters(28); K_buf_ss=parameters(29); K_up=parameters(30);...
    V_leak=parameters(31); V_rel=parameters(32); V_sr=parameters(33);...
    V_ss=parameters(34); V_xfer=parameters(35); Vmax_up=parameters(36);...
    k1_prime=parameters(37); k2_prime=parameters(38); k3=parameters(39);...
    k4=parameters(40); max_sr=parameters(41); min_sr=parameters(42);...
    Na_o=parameters(43); Cm=parameters(44); F=parameters(45);...
    R=parameters(46); T=parameters(47); V_c=parameters(48);...
    stim_amplitude=parameters(49); stim_duration=parameters(50);...
    stim_period=parameters(51); stim_start=parameters(52); K_o=parameters(53);

  % Init return args
  values = zeros(19, 1);

  % Expressions for the Reversal potentials component
  E_Na = R*T*log(Na_o/Na_i)/F;
  E_K = R*T*log(K_o/K_i)/F;
  E_Ks = R*T*log((K_o + Na_o*P_kna)/(P_kna*Na_i + K_i))/F;
  E_Ca = 0.5*R*T*log(Ca_o/Ca_i)/F;

  % Expressions for the Inward rectifier potassium current component
  alpha_K1 = 0.1/(1 + 6.14421235333e-06*exp(0.06*V - 0.06*E_K));
  beta_K1 = (0.367879441171*exp(0.1*V - 0.1*E_K) + 3.06060402008*exp(0.0002*V...
    - 0.0002*E_K))/(1 + exp(0.5*E_K - 0.5*V));
  xK1_inf = alpha_K1/(alpha_K1 + beta_K1);
  i_K1 = 0.430331482912*g_K1*sqrt(K_o)*(-E_K + V)*xK1_inf;

  % Expressions for the Rapid time dependent potassium current component
  i_Kr = 0.430331482912*g_Kr*sqrt(K_o)*(-E_K + V)*Xr1*Xr2;

  % Expressions for the Xr1 gate component
  xr1_inf = 1.0/(1 + exp(-26/7 - V/7));
  alpha_xr1 = 450/(1 + exp(-9/2 - V/10));
  beta_xr1 = 6/(1 + 13.5813245226*exp(0.0869565217391*V));
  tau_xr1 = alpha_xr1*beta_xr1;
  values(1) = (-Xr1 + xr1_inf)/tau_xr1;

  % Expressions for the Xr2 gate component
  xr2_inf = 1.0/(1 + exp(11/3 + V/24));
  alpha_xr2 = 3/(1 + exp(-3 - V/20));
  beta_xr2 = 1.12/(1 + exp(-3 + V/20));
  tau_xr2 = alpha_xr2*beta_xr2;
  values(2) = (-Xr2 + xr2_inf)/tau_xr2;

  % Expressions for the Slow time dependent potassium current component
  i_Ks = g_Ks*Xs^2*(-E_Ks + V);

  % Expressions for the Xs gate component
  xs_inf = 1.0/(1 + exp(-5/14 - V/14));
  alpha_xs = 1400/sqrt(1 + exp(5/6 - V/6));
  beta_xs = 1.0/(1 + exp(-7/3 + V/15));
  tau_xs = 80 + alpha_xs*beta_xs;
  values(3) = (-Xs + xs_inf)/tau_xs;

  % Expressions for the Fast sodium current component
  i_Na = g_Na*m^3*(-E_Na + V)*h*j;

  % Expressions for the m gate component
  m_inf = (1 + 0.00184221158117*exp(-0.110741971207*V))^(-2);
  alpha_m = 1.0/(1 + exp(-12 - V/5));
  beta_m = 0.1/(1 + exp(7 + V/5)) + 0.1/(1 + exp(-1/4 + V/200));
  tau_m = alpha_m*beta_m;
  values(4) = (-m + m_inf)/tau_m;

  % Expressions for the h gate component
  h_inf = (1 + 15212.5932857*exp(0.134589502019*V))^(-2);
  alpha_h = ((V < -40)*(4.43126792958e-07*exp(-0.147058823529*V)) + ~(V <...
    -40)*(0));
  beta_h = ((V < -40)*(310000*exp(0.3485*V) + 2.7*exp(0.079*V)) + ~(V <...
    -40)*(0.77/(0.13 + 0.0497581410839*exp(-0.0900900900901*V))));
  tau_h = 1.0/(alpha_h + beta_h);
  values(5) = (-h + h_inf)/tau_h;

  % Expressions for the j gate component
  j_inf = (1 + 15212.5932857*exp(0.134589502019*V))^(-2);
  alpha_j = ((V < -40)*((37.78 + V)*(-25428*exp(0.2444*V) -...
    6.948e-06*exp(-0.04391*V))/(1 + 50262745826.0*exp(0.311*V))) + ~(V <...
    -40)*(0));
  beta_j = ((V < -40)*(0.02424*exp(-0.01052*V)/(1 +...
    0.0039608683399*exp(-0.1378*V))) + ~(V < -40)*(0.6*exp(0.057*V)/(1 +...
    0.0407622039784*exp(-0.1*V))));
  tau_j = 1.0/(alpha_j + beta_j);
  values(6) = (-j + j_inf)/tau_j;

  % Expressions for the Sodium background current component
  i_b_Na = g_bna*(-E_Na + V);

  % Expressions for the L_type Ca current component
  i_CaL = 4*g_CaL*F^2*(-15 + V)*(-Ca_o + 0.25*Ca_ss*exp(F*(-30 +...
    2*V)/(R*T)))*d*f*f2*fCass/(R*T*(-1 + exp(F*(-30 + 2*V)/(R*T))));

  % Expressions for the d gate component
  d_inf = 1.0/(1 + 0.344153786865*exp(-0.133333333333*V));
  alpha_d = 0.25 + 1.4/(1 + exp(-35/13 - V/13));
  beta_d = 1.4/(1 + exp(1 + V/5));
  gamma_d = 1.0/(1 + exp(5/2 - V/20));
  tau_d = alpha_d*beta_d + gamma_d;
  values(7) = (-d + d_inf)/tau_d;

  % Expressions for the f gate component
  f_inf = 1.0/(1 + exp(20/7 + V/7));
  tau_f = 20 + 180/(1 + exp(3 + V/10)) + 200/(1 + exp(13/10 - V/10)) +...
    1102.5*exp(-(27 + V)^2/225);
  values(8) = (-f + f_inf)/tau_f;

  % Expressions for the F2 gate component
  f2_inf = 0.33 + 0.67/(1 + exp(5 + V/7));
  tau_f2 = 31/(1 + exp(5/2 - V/10)) + 80/(1 + exp(3 + V/10)) + 562*exp(-(27 +...
    V)^2/240);
  values(9) = (-f2 + f2_inf)/tau_f2;

  % Expressions for the FCass gate component
  fCass_inf = 0.4 + 0.6/(1 + 400.0*Ca_ss^2);
  tau_fCass = 2 + 80/(1 + 400.0*Ca_ss^2);
  values(10) = (-fCass + fCass_inf)/tau_fCass;

  % Expressions for the Calcium background current component
  i_b_Ca = g_bca*(-E_Ca + V);

  % Expressions for the Transient outward current component
  i_to = g_to*(-E_K + V)*r*s;

  % Expressions for the s gate component
  s_inf = 1.0/(1 + exp(4 + V/5));
  tau_s = 3 + 5/(1 + exp(-4 + V/5)) + 85*exp(-(45 + V)^2/320);
  values(11) = (-s + s_inf)/tau_s;

  % Expressions for the r gate component
  r_inf = 1.0/(1 + exp(10/3 - V/6));
  tau_r = 0.8 + 9.5*exp(-(40 + V)^2/1800);
  values(12) = (-r + r_inf)/tau_r;

  % Expressions for the Sodium potassium pump current component
  i_NaK = K_o*P_NaK*Na_i/((K_mNa + Na_i)*(K_mk + K_o)*(1 +...
    0.0353*exp(-F*V/(R*T)) + 0.1245*exp(-0.1*F*V/(R*T))));

  % Expressions for the Sodium calcium exchanger current component
  i_NaCa = K_NaCa*(Ca_o*Na_i^3*exp(F*gamma*V/(R*T)) -...
    alpha*Na_o^3*Ca_i*exp(F*(-1 + gamma)*V/(R*T)))/((1 + K_sat*exp(F*(-1 +...
    gamma)*V/(R*T)))*(Ca_o + Km_Ca)*(Km_Nai^3 + Na_o^3));

  % Expressions for the Calcium pump current component
  i_p_Ca = g_pCa*Ca_i/(K_pCa + Ca_i);

  % Expressions for the Potassium pump current component
  i_p_K = g_pK*(-E_K + V)/(1 + 65.4052157419*exp(-0.167224080268*V));

  % Expressions for the Calcium dynamics component
  i_up = Vmax_up/(1 + K_up^2/Ca_i^2);
  i_leak = V_leak*(-Ca_i + Ca_SR);
  i_xfer = V_xfer*(-Ca_i + Ca_ss);
  kcasr = max_sr - (max_sr - min_sr)/(1 + EC^2/Ca_SR^2);
  Ca_i_bufc = 1.0/(1 + Buf_c*K_buf_c/(K_buf_c + Ca_i)^2);
  Ca_sr_bufsr = 1.0/(1 + Buf_sr*K_buf_sr/(K_buf_sr + Ca_SR)^2);
  Ca_ss_bufss = 1.0/(1 + Buf_ss*K_buf_ss/(K_buf_ss + Ca_ss)^2);
  values(13) = (V_sr*(-i_up + i_leak)/V_c - Cm*(-2*i_NaCa + i_b_Ca +...
    i_p_Ca)/(2*F*V_c) + i_xfer)*Ca_i_bufc;
  k1 = k1_prime/kcasr;
  k2 = k2_prime*kcasr;
  O = Ca_ss^2*R_prime*k1/(k3 + Ca_ss^2*k1);
  values(14) = k4*(1 - R_prime) - Ca_ss*R_prime*k2;
  i_rel = V_rel*(-Ca_ss + Ca_SR)*O;
  values(15) = (-i_leak - i_rel + i_up)*Ca_sr_bufsr;
  values(16) = (V_sr*i_rel/V_ss - V_c*i_xfer/V_ss -...
    Cm*i_CaL/(2*F*V_ss))*Ca_ss_bufss;

  % Expressions for the Sodium dynamics component
  values(17) = Cm*(-i_Na - i_b_Na - 3*i_NaCa - 3*i_NaK)/(F*V_c);

  % Expressions for the Membrane component
  i_Stim = ((t - stim_period*floor(t/stim_period) <= stim_duration +...
    stim_start & t - stim_period*floor(t/stim_period) >=...
    stim_start)*(-stim_amplitude) + ~(t - stim_period*floor(t/stim_period) <=...
    stim_duration + stim_start & t - stim_period*floor(t/stim_period) >=...
    stim_start)*(0));
  values(18) = -i_CaL - i_K1 - i_Kr - i_Ks - i_Na - i_NaCa - i_NaK - i_Stim -...
    i_b_Ca - i_b_Na - i_p_Ca - i_p_K - i_to;

  % Expressions for the Potassium dynamics component
  values(19) = Cm*(-i_K1 - i_Kr - i_Ks - i_Stim - i_p_K - i_to +...
    2*i_NaK)/(F*V_c);
end