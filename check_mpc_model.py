import numpy as np
import matplotlib.pyplot as plt
import sys
from casadi import *
from gym_poly_reactor.envs.params import *

# Add do_mpc to path. This is not necessary if it was installed via pip

# Import do_mpc package:
import do_mpc

model_type = 'continuous'  # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# Input struct (optimization variables):
m_dot_f = model.set_variable('_u', 'm_dot_f')
T_in_M = model.set_variable('_u', 'T_in_M')
T_in_EK = model.set_variable('_u', 'T_in_EK')

delH_R = model.set_variable('_p', 'delH_R')
k_0 = model.set_variable('_p', 'k_0')

# States struct (optimization variables):
m_W = model.set_variable('_x', 'm_W')
m_A = model.set_variable('_x', 'm_A')
m_P = model.set_variable('_x', 'm_P')
T_R = model.set_variable('_x', 'T_R')
T_S = model.set_variable('_x', 'T_S')
Tout_M = model.set_variable('_x', 'Tout_M')
T_EK = model.set_variable('_x', 'T_EK')
Tout_AWT = model.set_variable('_x', 'Tout_AWT')
accum_monom = model.set_variable('_x', 'accum_monom')
T_adiab = model.set_variable('_x', 'T_adiab')

# algebraic equations
U_m = m_P / (m_A + m_P)
m_ges = m_W + m_A + m_P
k_R1 = k_0 * exp(- E_a / (R * T_R)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
k_R2 = k_0 * exp(- E_a / (R * T_EK)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
k_K = ((m_W / m_ges) * k_WS) + ((m_A / m_ges) * k_AS) + ((m_P / m_ges) * k_PS)

# Differential equations
dot_m_W = m_dot_f * w_WF
model.set_rhs('m_W', dot_m_W)
dot_m_A = (m_dot_f * w_AF) - (k_R1 * (m_A - ((m_A * m_AWT) / (m_W + m_A + m_P)))) - (p_1 * k_R2 * (m_A / m_ges) * m_AWT)
model.set_rhs('m_A', dot_m_A)
dot_m_P = (k_R1 * (m_A - ((m_A * m_AWT) / (m_W + m_A + m_P)))) + (p_1 * k_R2 * (m_A / m_ges) * m_AWT)
model.set_rhs('m_P', dot_m_P)

dot_T_R = 1. / (c_pR * m_ges) * (
        (m_dot_f * c_pF * (T_F - T_R)) - (k_K * A_tank * (T_R - T_S)) - (fm_AWT * c_pR * (T_R - T_EK)) + (
        delH_R * k_R1 * (m_A - ((m_A * m_AWT) / (m_W + m_A + m_P)))))
model.set_rhs('T_R', dot_T_R)
model.set_rhs('T_S', 1. / (c_pS * m_S) * ((k_K * A_tank * (T_R - T_S)) - (k_K * A_tank * (T_S - Tout_M))))
model.set_rhs('Tout_M', 1. / (c_pW * m_M_KW) * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K * A_tank * (T_S - Tout_M))))
model.set_rhs('T_EK', 1. / (c_pR * m_AWT) * ((fm_AWT * c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (
        p_1 * k_R2 * (m_A / m_ges) * m_AWT * delH_R)))
model.set_rhs('Tout_AWT',
              1. / (c_pW * m_AWT_KW) * ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK))))
model.set_rhs('accum_monom', m_dot_f)
model.set_rhs('T_adiab', delH_R / (m_ges * c_pR) * dot_m_A - (dot_m_A + dot_m_W + dot_m_P) * (
        m_A * delH_R / (m_ges * m_ges * c_pR)) + dot_T_R)

# Build the model
model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 50.0 / 3600.0,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)

_x = model.x
mterm = - _x['m_P']  # terminal cost
lterm = - _x['m_P']  # stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(m_dot_f=0.002, T_in_M=0.004, T_in_EK=0.002)  # penalty on control input changes

# auxiliary term
temp_range = 2.0

# lower bound states
mpc.bounds['lower', '_x', 'm_W'] = 0.0
mpc.bounds['lower', '_x', 'm_A'] = 0.0
mpc.bounds['lower', '_x', 'm_P'] = 26.0

mpc.bounds['lower', '_x', 'T_R'] = 363.15 - temp_range
mpc.bounds['lower', '_x', 'T_S'] = 298.0
mpc.bounds['lower', '_x', 'Tout_M'] = 298.0
mpc.bounds['lower', '_x', 'T_EK'] = 288.0
mpc.bounds['lower', '_x', 'Tout_AWT'] = 288.0
mpc.bounds['lower', '_x', 'accum_monom'] = 0.0

# upper bound states
mpc.bounds['upper', '_x', 'T_R'] = 363.15 + temp_range + 10.0
mpc.bounds['upper', '_x', 'T_S'] = 400.0
mpc.bounds['upper', '_x', 'Tout_M'] = 400.0
mpc.bounds['upper', '_x', 'T_EK'] = 400.0
mpc.bounds['upper', '_x', 'Tout_AWT'] = 400.0
mpc.bounds['upper', '_x', 'accum_monom'] = 30000.0
mpc.bounds['upper', '_x', 'T_adiab'] = 382.15 + 10.0

# lower bound inputs
mpc.bounds['lower', '_u', 'm_dot_f'] = 0.0
mpc.bounds['lower', '_u', 'T_in_M'] = 333.15
mpc.bounds['lower', '_u', 'T_in_EK'] = 333.15

# upper bound inputs
mpc.bounds['upper', '_u', 'm_dot_f'] = 3.0e4
mpc.bounds['upper', '_u', 'T_in_M'] = 373.15
mpc.bounds['upper', '_u', 'T_in_EK'] = 373.15

# states
mpc.scaling['_x', 'm_W'] = 10
mpc.scaling['_x', 'm_A'] = 10
mpc.scaling['_x', 'm_P'] = 10
mpc.scaling['_x', 'accum_monom'] = 10

# control inputs
mpc.scaling['_u', 'm_dot_f'] = 100

delH_R_var = np.array([950.0, 950.0 * 1.30, 950.0 * 0.70])
k_0_var = np.array([7.0 * 1.00, 7.0 * 1.30, 7.0 * 0.70])

mpc.set_uncertainty_values([delH_R_var, k_0_var])

mpc.setup()

estimator = do_mpc.estimator.StateFeedback(model)

simulator = do_mpc.simulator.Simulator(model)

params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 50.0 / 3600.0
}

simulator.set_param(**params_simulator)

p_num = simulator.get_p_template()
tvp_num = simulator.get_tvp_template()

# uncertain parameters
p_num['delH_R'] = 950 * np.random.uniform(0.75, 1.25)
p_num['k_0'] = 7 * np.random.uniform(0.75 * 1.25)


def p_fun(t_now):
    return p_num


simulator.set_p_fun(p_fun)

simulator.setup()

# Set the initial state of the controller and simulator:
# assume nominal values of uncertain parameters as initial guess
delH_R_real = 950.0
c_pR = 5.0

# x0 is a property of the simulator - we obtain it and set values.
x0 = simulator.x0

x0['m_W'] = 10000.0
x0['m_A'] = 853.0
x0['m_P'] = 26.5

x0['T_R'] = 90.0 + 273.15
x0['T_S'] = 90.0 + 273.15
x0['Tout_M'] = 90.0 + 273.15
x0['T_EK'] = 35.0 + 273.15
x0['Tout_AWT'] = 35.0 + 273.15
x0['accum_monom'] = 300.0
x0['T_adiab'] = x0['m_A'] * delH_R_real / ((x0['m_W'] + x0['m_A'] + x0['m_P']) * c_pR) + x0['T_R']

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

for k in range(100):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
