function [currents] = tentusscher_currents(states, parameters)
  % Compute the right hand side of the tentusscher_2006_epi ODE

  % Assign states
  Xr1=states(:,1); Xr2=states(:,2); Xs=states(:,3); m=states(:,4); h=states(:,5);...
    j=states(:,6); d=states(:,7); f=states(:,8); f2=states(:,9); fCass=states(:,10);...
    s=states(:,11); r=states(:,12); Ca_i=states(:,13); R_prime=states(:,14);...
    Ca_SR=states(:,15); Ca_ss=states(:,16); Na_i=states(:,17); V=states(:,18);...
    K_i=states(:,19);

  % Assign parameters
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


  % Expressions for the Reversal potentials component
  E_Na = R.*T.*log(Na_o./Na_i)./F;
  E_K = R.*T.*log(K_o./K_i)./F;
  E_Ks = R.*T.*log((K_o + Na_o.*P_kna)./(P_kna.*Na_i + K_i))./F;
  E_Ca = 0.5.*R.*T.*log(Ca_o./Ca_i)./F;

  % Expressions for the Inward rectifier potassium current component
  alpha_K1 = 0.1./(1 + 6.14421235333e-06.*exp(0.06.*V - 0.06.*E_K));
  beta_K1 = (0.367879441171.*exp(0.1.*V - 0.1.*E_K) + 3.06060402008.*exp(0.0002.*V...
    - 0.0002.*E_K))./(1 + exp(0.5.*E_K - 0.5.*V));
  xK1_inf = alpha_K1./(alpha_K1 + beta_K1);
  i_K1 = 0.430331482912.*g_K1.*sqrt(K_o).*(-E_K + V).*xK1_inf;
  
  currents.i_K1 = i_K1;

  % Expressions for the Rapid time dependent potassium current component
  i_Kr = 0.430331482912.*g_Kr.*sqrt(K_o).*(-E_K + V).*Xr1.*Xr2;
  
  currents.i_Kr = i_Kr;


  % Expressions for the Slow time dependent potassium current component
  i_Ks = g_Ks.*Xs.^2.*(-E_Ks + V);
  
  currents.i_Ks = i_Ks;


  % Expressions for the Fast sodium current component
  i_Na = g_Na.*m.^3.*(-E_Na + V).*h.*j;
  
  currents.i_Na = i_Na;


  % Expressions for the Sodium background current component
  i_b_Na = g_bna.*(-E_Na + V);
  
  currents.i_bNa = i_b_Na;
  

  % Expressions for the L_type Ca current component
  i_CaL = 4.*g_CaL.*F.^2.*(-15 + V).*(-Ca_o + 0.25.*Ca_ss.*exp(F.*(-30 +...
    2.*V)./(R.*T))).*d.*f.*f2.*fCass./(R.*T.*(-1 + exp(F.*(-30 + 2.*V)./(R.*T))));

  currents.i_CaL = i_CaL;


  % Expressions for the Calcium background current component
  i_b_Ca = g_bca.*(-E_Ca + V);
  
  currents.i_bCa = i_b_Ca;

  % Expressions for the Transient outward current component
  i_to = g_to.*(-E_K + V).*r.*s;
  
  currents.i_to = i_to;
  

  % Expressions for the Sodium potassium pump current component
  i_NaK = K_o.*P_NaK.*Na_i./((K_mNa + Na_i).*(K_mk + K_o).*(1 +...
    0.0353.*exp(-F.*V./(R.*T)) + 0.1245.*exp(-0.1.*F.*V./(R.*T))));

  currents.i_NaK = i_NaK;

  % Expressions for the Sodium calcium exchanger current component
  i_NaCa = K_NaCa.*(Ca_o.*Na_i.^3.*exp(F.*gamma.*V./(R.*T)) -...
    alpha.*Na_o.^3.*Ca_i.*exp(F.*(-1 + gamma).*V./(R.*T)))./((1 + K_sat.*exp(F.*(-1 +...
    gamma).*V./(R.*T))).*(Ca_o + Km_Ca).*(Km_Nai.^3 + Na_o.^3));

  currents.i_NaCa = i_NaCa;

  % Expressions for the Calcium pump current component
  i_p_Ca = g_pCa.*Ca_i./(K_pCa + Ca_i);
  
  currents.i_pCa = i_p_Ca;

  % Expressions for the Potassium pump current component
  i_p_K = g_pK.*(-E_K + V)./(1 + 65.4052157419.*exp(-0.167224080268.*V));
  
  currents.i_pK = i_p_K;

  % Expressions for the Calcium dynamics component
  i_up = Vmax_up./(1 + K_up.^2./Ca_i.^2);
  i_leak = V_leak.*(-Ca_i + Ca_SR);
  i_xfer = V_xfer.*(-Ca_i + Ca_ss);
  kcasr = max_sr - (max_sr - min_sr)./(1 + EC.^2./Ca_SR.^2);
  k1 = k1_prime./kcasr;
  O = Ca_ss.^2.*R_prime.*k1./(k3 + Ca_ss.^2.*k1);
  i_rel = V_rel.*(-Ca_ss + Ca_SR).*O;
  
  currents.i_up = i_up;
  currents.i_leak = i_leak;
  currents.i_xfer = i_xfer;
  currents.i_rel = i_rel;
end