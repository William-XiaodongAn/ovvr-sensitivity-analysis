function [tT, all_idx, g_idx] = tentusscher_idx()
%[tT, all_idx, g_idx] = tentusscher_idx() Set up indices for the maturation map of the tT06 model

tT.Cm_idx = 44;
tT.Vc_idx = 48;
tT.Vsr_idx = 33;
tT.Vss_idx = 34;
tT.Na_idx = 5;
tT.CaL_idx = 7;
tT.to_idx = 9;
tT.Ks_idx = 4;
tT.Kr_idx = 3;
tT.K1_idx = 2;
tT.NaCa_idx = 13;
tT.NaK_idx = 12;
tT.pCa_idx = 20;
tT.pK_idx = 21;
tT.bNa_idx = 6;
tT.bCa_idx = 8;
tT.NaL_idx = 54;
tT.leak_idx = 31;
tT.up_idx = 36;
tT.rel_idx = 32;
tT.xfer_idx = 35;

all_idx = [tT.Cm_idx; tT.Vc_idx; tT.Vsr_idx; tT.Vss_idx; tT.Na_idx; tT.CaL_idx; tT.to_idx; ...
    tT.Ks_idx; tT.Kr_idx; tT.K1_idx; tT.NaCa_idx; tT.NaK_idx; tT.pCa_idx; tT.pK_idx; tT.bNa_idx; ...
    tT.bCa_idx; tT.NaL_idx; tT.leak_idx; tT.up_idx; tT.rel_idx; tT.xfer_idx];

g_idx = [tT.Na_idx; tT.CaL_idx; tT.to_idx; tT.Ks_idx; tT.Kr_idx; tT.K1_idx; ...
    tT.NaCa_idx; tT.NaK_idx; tT.pCa_idx; tT.pK_idx; tT.bNa_idx; tT.bCa_idx; ...
    tT.NaL_idx];
end

