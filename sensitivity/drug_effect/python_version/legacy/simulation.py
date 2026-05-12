import numpy as np
import pickle
import math

# Load drug parameters
def get_drug_data(drug_name):
    with open('drug_dict.pkl', 'rb') as f:
        drug_dict = pickle.load(f)
        if drug_name not in drug_dict:
            print(f"Warning: Drug '{drug_name}' not found in drug_dict. Returning empty parameters.")
            return False
        return drug_dict.get(drug_name, {})

def update_conductivity(C, EFTPC, EFTPC_multiplier, IC50, h):
    if IC50 == 0 or IC50 is None:
        return C
    return C * (1 / (1 + (EFTPC * EFTPC_multiplier / IC50)**h))

def run_tnnp_simulation(
    drug_name,
    pacing_period,
    drug_concentration_multiplier,
    total_time=None,
    perturb_multipliers=None,
    initial_wait=20000,
    record_pre=100,
    record_post=500,
    record_interval=0.1,
):
    if total_time is None:
        # Match the browser simulation in 2D-TNNP-pacing-general-5xdrug/app/main.js:
        # it records from initial_wait - 100 ms until initial_wait + 500 ms.
        total_time = initial_wait + record_post

    # Constants
    Ko=5.4; Cao=2.0; Nao=140.0
    Vc=0.016404; Vsr=0.001094; Vss=0.00005468
    Bufc=0.2; Kbufc=0.001; Bufsr=10.
    Kbufsr=0.3; Bufss=0.4; Kbufss=0.00025
    Vmaxup=0.006375; Kup=0.00025; Vrel=0.102
    k3=0.060; k4=0.005
    k1prime=0.15; k2prime=0.045
    EC=1.5; maxsr=2.5; minsr=1.
    Vleak=0.00036; Vxfer=0.0038
    RR=8314.3; FF=96486.7; TT=310.0
    CAPACITANCE=0.185
    C_m=1.0

    # Cell type specific conductivities (EPI)
    Gks=0.392; Gto=0.294

    Gkr=0.153; pKNa=0.03        
    GK1=5.405; alphanaca=2.5
    GNa=14.838; GbNa=0.00029     
    KmK=1.0; KmNa=40.0
    knak=2.724; GCaL=0.00003980  
    GbCa=0.000592
    knaca=1000.; KmNai=87.5       
    KmCa=1.38; ksat=0.1
    n=0.35; GpCa=0.1238      
    KpCa=0.0005; GpK=0.0146

    inverseVcF2=1./(2.*Vc*FF)
    inverseVcF=1./(Vc*FF)
    inversevssF2=1./(2.*Vss*FF)
    rtof=(RR*TT)/FF
    fort=1./rtof
    KmNai3=KmNai*KmNai*KmNai
    Nao3=Nao*Nao*Nao
    Gkrfactor=math.sqrt(Ko/5.4)

    # Base conductance multipliers
    C_CaL=1.0; C_pCa=1.0; C_bCa=1.0; C_leak=1.0; C_up=1.0
    C_xfer=1.0; C_rel=1.0; C_Na=1.0; C_bNa=1.0; C_NaK=1.0
    C_NaCa=1.0; C_K1=1.0; C_to=1.0; C_Kr=1.0; C_Ks=1.0; C_pK=1.0

    # Apply drug effects
    drug_data = get_drug_data(drug_name)
    EFTPC = drug_data.get('EFTPCmax', 0.0)
    
    for channel, C_var in [('INa', 'C_Na'), ('IKr', 'C_Kr'), ('ICaL', 'C_CaL'), 
                           ('IKs', 'C_Ks'), ('Ito', 'C_to'), ('IK1', 'C_K1')]:
        if channel in drug_data:
            ic50 = drug_data[channel].get('IC50', 0)
            h = drug_data[channel].get('h', 1.0)
            if h is None: h = 1.0
            if ic50 > 0:
                new_C = update_conductivity(locals()[C_var], EFTPC, drug_concentration_multiplier, ic50, h)
                locals()[C_var] = new_C
                # Specifically update the local variables
                if C_var == 'C_Na': C_Na = new_C
                elif C_var == 'C_Kr': C_Kr = new_C
                elif C_var == 'C_CaL': C_CaL = new_C
                elif C_var == 'C_Ks': C_Ks = new_C
                elif C_var == 'C_to': C_to = new_C
                elif C_var == 'C_K1': C_K1 = new_C

    if perturb_multipliers is not None:
        if 'INa' in perturb_multipliers: C_Na *= perturb_multipliers['INa']
        if 'IKr' in perturb_multipliers: C_Kr *= perturb_multipliers['IKr']
        if 'ICaL' in perturb_multipliers: C_CaL *= perturb_multipliers['ICaL']
        if 'IKs' in perturb_multipliers: C_Ks *= perturb_multipliers['IKs']
        if 'Ito' in perturb_multipliers: C_to *= perturb_multipliers['Ito']
        if 'IK1' in perturb_multipliers: C_K1 *= perturb_multipliers['IK1']
        if 'IpK' in perturb_multipliers: C_pK *= perturb_multipliers['IpK']
        if 'INaK' in perturb_multipliers: C_NaK *= perturb_multipliers['INaK']
        if 'INaCa' in perturb_multipliers: C_NaCa *= perturb_multipliers['INaCa']
        if 'IbCa' in perturb_multipliers: C_bCa *= perturb_multipliers['IbCa']
        if 'IpCa' in perturb_multipliers: C_pCa *= perturb_multipliers['IpCa']
        if 'IbNa' in perturb_multipliers: C_bNa *= perturb_multipliers['IbNa']

    # Initial state from 2D-TNNP-pacing-general-5xdrug/app/shaders/initShader.frag
    # for env.cellType = EPI.
    V = -85.46
    sRR = 0.9891; Nai = 9.293; Ki = 136.2
    sm = 0.001633; sh = 0.7512; sj = 0.7508; sxs = 0.003214
    sd = 3.270e-5; sf = 0.9767; sf2 = 0.9995; sfcass = 1.0
    sr = 0.0; ss = 1.0; sxr1 = 0.0; sxr2 = 1.0
    Cai = 0.0001156; CaSS = 0.0002331; CaSR = 3.432

    dt = 0.1
    time_steps = int(total_time / dt)
    record_stride = max(1, int(round(record_interval / dt)))
    
    # Store history if we need it (the last 500ms like JS, but the problem wants just relative identifiability index)
    # The sensitivity requires full state at different points or currents during the last pacing cycle
    
    recorded_time = []
    recorded_voltage = []
    recorded_currents = {'INa':[], 'Ito':[], 'ICaL':[], 'IKs':[], 'IpK':[], 'INaK':[], 'IKr':[], 'INaCa':[], 'IK1':[], 'IbCa':[], 'IpCa':[], 'IbNa':[]}

    next_stim_time = pacing_period
    
    for step in range(time_steps):
        t = step * dt
        
        # Pacing logic
        I_init = 0.0
        if t >= next_stim_time and t <= next_stim_time + 1.0:
            I_init = -40.0
            
        if t > next_stim_time + 1.0:
            next_stim_time += pacing_period
            
        # Reversal potentials
        Ek = rtof * math.log(Ko / Ki)
        Ena = rtof * math.log(Nao / Nai)
        Eks = rtof * math.log((Ko + pKNa * Nao) / (Ki + pKNa * Nai))
        Eca = 0.5 * rtof * math.log(Cao / Cai)

        # m, h, j
        vv = V
        AM = 1. / (1. + math.exp((-60. - vv) / 5.))
        BM = 0.1 / (1. + math.exp((vv + 35.) / 5.)) + 0.10 / (1. + math.exp((vv - 50.) / 200.))
        minft = 1. / ((1. + math.exp((-56.86 - vv) / 9.03))**2)
        TAU_M = AM * BM
        sm = minft - (minft - sm) * math.exp(-dt / TAU_M)

        hinft = 1. / ((1. + math.exp((vv + 71.55) / 7.43))**2)
        if vv >= -40.:
            AH = 0.
            BH = (0.77 / (0.13 * (1. + math.exp(-(vv + 10.66) / 11.1))))
            AJ = 0.
            BJ = (0.6 * math.exp((0.057) * vv) / (1. + math.exp(-0.1 * (vv + 32.))))
        else:
            AH = (0.057 * math.exp(-(vv + 80.) / 6.8))
            BH = (2.7 * math.exp(0.079 * vv) + (3.1e5) * math.exp(0.3485 * vv))
            AJ = (((-2.5428e4) * math.exp(0.2444 * vv) - (6.948e-6) * math.exp(-0.04391 * vv)) * (vv + 37.78) / (1. + math.exp(0.311 * (vv + 79.23))))
            BJ = (0.02424 * math.exp(-0.01052 * vv) / (1. + math.exp(-0.1378 * (vv + 40.14))))
        
        TAU_H = 1.0 / (AH + BH)
        sh = hinft - (hinft - sh) * math.exp(-dt / TAU_H)
        
        jinft = hinft
        TAU_J = 1.0 / (AJ + BJ)
        sj = jinft - (jinft - sj) * math.exp(-dt / TAU_J)

        # xs
        xsinft = 1. / (1. + math.exp((-5. - vv) / 14.))
        Axs = (1400. / (math.sqrt(1. + math.exp((5. - vv) / 6.))))
        Bxs = (1. / (1. + math.exp((vv - 35.) / 15.)))
        TAU_Xs = Axs * Bxs + 80.
        sxs = xsinft - (xsinft - sxs) * math.exp(-dt / TAU_Xs)

        # d, f, f2, fcass
        dinft = 1. / (1. + math.exp((-8. - vv) / 7.5))
        Ad = 1.4 / (1. + math.exp((-35. - vv) / 13.)) + 0.25
        Bd = 1.4 / (1. + math.exp((vv + 5.) / 5.))
        Cd = 1. / (1. + math.exp((50. - vv) / 20.))
        TAU_D = Ad * Bd + Cd
        sd = dinft - (dinft - sd) * math.exp(-dt / TAU_D)

        finft = 1. / (1. + math.exp((vv + 20.) / 7.))
        Af = 1102.5 * math.exp(-(vv + 27.) * (vv + 27.) / 225.)
        Bf = 200. / (1. + math.exp((13. - vv) / 10.))
        Cf = (180. / (1. + math.exp((vv + 30.) / 10.))) + 20.
        TAU_F = Af + Bf + Cf
        sf = finft - (finft - sf) * math.exp(-dt / TAU_F)

        f2inft = 0.67 / (1. + math.exp((vv + 35.) / 7.)) + 0.33
        Af2 = 600. * math.exp(-(vv + 25.) * (vv + 25.) / 49.)
        Bf2 = 31. / (1. + math.exp((25. - vv) / 10.))
        Cf2 = 16. / (1. + math.exp((vv + 30.) / 10.))
        TAU_F2 = Af2 + Bf2 + Cf2
        sf2 = f2inft - (f2inft - sf2) * math.exp(-dt / TAU_F2)

        ccass = CaSS
        fcassinft = 0.6 / (1. + (ccass / 0.05)**2) + 0.4
        taufcass = 80. / (1. + (ccass / 0.05)**2) + 2.
        
        casshi = 1.0
        if CaSS >= casshi:
            FCaSS_INF = 0.4
            exptaufcass = math.exp(-dt / 2.0)
        else:
            FCaSS_INF = fcassinft
            exptaufcass = math.exp(-dt / taufcass)
        sfcass = FCaSS_INF - (FCaSS_INF - sfcass) * exptaufcass

        # r, s, xr1, xr2
        # EPI
        rinft = 1. / (1. + math.exp((20. - vv) / 6.))
        sinft = 1. / (1. + math.exp((vv + 20.) / 5.))
        TAU_R = 9.5 * math.exp(-(vv + 40.)**2 / 1800.) + 0.8
        TAU_S = 85. * math.exp(-(vv + 45.)**2 / 320.) + 5. / (1. + math.exp((vv - 20.) / 5.)) + 3.

        sr = rinft - (rinft - sr) * math.exp(-dt / TAU_R)
        ss = sinft - (sinft - ss) * math.exp(-dt / TAU_S)

        xr1inft = 1. / (1. + math.exp((-26. - vv) / 7.))
        axr1 = 450. / (1. + math.exp((-45. - vv) / 10.))
        bxr1 = 6. / (1. + math.exp((vv + 30.) / 11.5))
        TAU_Xr1 = axr1 * bxr1
        sxr1 = xr1inft - (xr1inft - sxr1) * math.exp(-dt / TAU_Xr1)

        xr2inft = 1. / (1. + math.exp((vv + 88.) / 24.))
        axr2 = 3. / (1. + math.exp((-60. - vv) / 20.))
        bxr2 = 1.12 / (1. + math.exp((vv - 60.) / 20.))
        TAU_Xr2 = axr2 * bxr2
        sxr2 = xr2inft - (xr2inft - sxr2) * math.exp(-dt / TAU_Xr2)

        # Currents
        INa = GNa * sm**3 * sh * sj * (V - Ena) * C_Na
        IKr = Gkr * Gkrfactor * sxr1 * sxr2 * (V - Ek) * C_Kr
        IKs = Gks * sxs**2 * (V - Eks) * C_Ks
        Ito = Gto * sr * ss * (V - Ek) * C_to

        vmek = V - Ek
        Ak1 = 0.1 / (1. + math.exp(0.06 * (vmek - 200.)))
        Bk1 = (3. * math.exp(0.0002 * (vmek + 100.)) + math.exp(0.1 * (vmek - 10.))) / (1. + math.exp(-0.5 * vmek))
        IK1 = GK1 * Ak1 / (Ak1 + Bk1) * vmek * C_K1

        IpK = GpK / (1. + math.exp((25. - vv) / 5.98)) * (V - Ek) * C_pK
        IbNa = GbNa * (V - Ena) * C_bNa
        INaK = (1. / (1. + 0.1245 * math.exp(-0.1 * vv * fort) + 0.0353 * math.exp(-vv * fort))) * knak * (Ko / (Ko + KmK)) * (Nai / (Nai + KmNa)) * C_NaK

        temp = math.exp((n - 1.) * vv * fort)
        temp2 = knaca / ((KmNai3 + Nao3) * (KmCa + Cao) * (1. + ksat * temp))
        inaca1t = temp2 * math.exp(n * vv * fort) * Cao
        inaca2t = temp2 * temp * Nao3 * alphanaca
        INaCa = (inaca1t * Nai**3 - inaca2t * Cai) * C_NaCa

        # Calcium
        temp_cal = math.exp(2. * (vv - 15.) * fort)
        if abs(vv - 15.) < 1e-4:
            diff = 1e-4
            temp_cal = math.exp(2. * diff * fort)
            ical1t = GCaL * 4. * diff * FF * fort * 0.25 * temp_cal / (temp_cal - 1.)
            ical2t = GCaL * 4. * diff * FF * fort * Cao / (temp_cal - 1.)
        else:
            ical1t = GCaL * 4. * (vv - 15.) * FF * fort * 0.25 * temp_cal / (temp_cal - 1.)
            ical2t = GCaL * 4. * (vv - 15.) * FF * fort * Cao / (temp_cal - 1.)
            
        ICaL = sd * sf * sf2 * sfcass * (ical1t * CaSS - ical2t) * C_CaL
        IpCa = GpCa * Cai / (KpCa + Cai) * C_pCa
        IbCa = GbCa * (vv - Eca) * C_bCa

        # Concentrations
        dNai = -(INa + IbNa + 3. * INaK + 3. * INaCa) * inverseVcF * CAPACITANCE
        Nai += dt * dNai

        # Match compShader.frag: I_init affects voltage, but the K_i update uses
        # Istim = 0.0 rather than the pacing current.
        dKi = -(IK1 + Ito + IKr + IKs - 2. * INaK + IpK) * inverseVcF * CAPACITANCE
        Ki += dt * dKi

        kCaSR = maxsr - ((maxsr - minsr) / (1. + (EC / CaSR)**2))
        k1 = k1prime / kCaSR
        k2 = k2prime * kCaSR
        dRR = k4 * (1. - sRR) - k2 * CaSS * sRR
        sRR += dt * dRR
        sOO = k1 * CaSS**2 * sRR / (k3 + k1 * CaSS**2)

        Irel = C_rel * Vrel * sOO * (CaSR - CaSS)
        Ileak = C_leak * Vleak * (CaSR - Cai)
        Iup = C_up * Vmaxup / (1. + ((Kup**2) / (Cai**2)))
        Ixfer = C_xfer * Vxfer * (CaSS - Cai)

        CaCSQN = Bufsr * CaSR / (CaSR + Kbufsr)
        dCaSR = dt * (Iup - Irel - Ileak)
        bjsr = Bufsr - CaCSQN - dCaSR - CaSR + Kbufsr
        cjsr = Kbufsr * (CaCSQN + dCaSR + CaSR)
        CaSR = (math.sqrt(bjsr**2 + 4. * cjsr) - bjsr) / 2.

        CaSSBuf = Bufss * CaSS / (CaSS + Kbufss)
        dCaSS = dt * (-Ixfer * (Vc / Vss) + Irel * (Vsr / Vss) + (-ICaL * inversevssF2 * CAPACITANCE))
        bcss = Bufss - CaSSBuf - dCaSS - CaSS + Kbufss
        ccss = Kbufss * (CaSSBuf + dCaSS + CaSS)
        CaSS = (math.sqrt(bcss**2 + 4. * ccss) - bcss) / 2.

        CaBuf = Bufc * Cai / (Cai + Kbufc)
        dCai = dt * (-(IbCa + IpCa - 2. * INaCa) * inverseVcF2 * CAPACITANCE - (Iup - Ileak) * (Vsr / Vc) + Ixfer)
        bc = Bufc - CaBuf - dCai - Cai + Kbufc
        cc = Kbufc * (CaBuf + dCai + Cai)
        Cai = (math.sqrt(bc**2 + 4. * cc) - bc) / 2.

        ISumCa = ICaL + IpCa + IbCa
        ISumNaK = INa + IbNa + INaK + IK1 + IKr + IKs + IpK + Ito
        I_sum = ISumCa + ISumNaK + INaCa + I_init

        # Update Voltage
        dVlt2dt = -I_sum / C_m
        V += dVlt2dt * dt

        # Record the same probe window as the browser simulation.
        record_time = t + dt
        if (
            initial_wait - record_pre <= record_time < initial_wait + record_post
            and (step + 1) % record_stride == 0
        ):
            recorded_time.append(t)
            recorded_voltage.append(V)
            recorded_currents['INa'].append(INa)
            recorded_currents['Ito'].append(Ito)
            recorded_currents['ICaL'].append(ICaL)
            recorded_currents['IKs'].append(IKs)
            recorded_currents['IpK'].append(IpK)
            recorded_currents['INaK'].append(INaK)
            recorded_currents['IKr'].append(IKr)
            recorded_currents['INaCa'].append(INaCa)
            recorded_currents['IK1'].append(IK1)
            recorded_currents['IbCa'].append(IbCa)
            recorded_currents['IpCa'].append(IpCa)
            recorded_currents['IbNa'].append(IbNa)

    return recorded_time, recorded_voltage, recorded_currents

if __name__ == '__main__':
    t, v, c = run_tnnp_simulation('Amiodarone I', 1000, 5)
    print(f"Voltage at end: {v[-1]}")
