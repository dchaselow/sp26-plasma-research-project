# --------------------------------- Imports ---------------------------------
# local imports
from dist_gen import *

# external imports
import matplotlib as mpl
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Add FFmpeg as moviewriter. Install FFmpeg and uncomment the next line if matplotlib.animation throws an error about FFmpeg being missing.
# mpl.rcParams['animation.ffmpeg_path'] = "C:\\Program Files\\FFmpeg\\bin\\ffmpeg.exe" # Common path for Windows

# ----------------------------- Simulation Code -----------------------------

# PARAMETERS/CONSTANTS

# Physical Constants

e   = 1.602176634e-19  # elementary charge		(C)
m_e = 9.10938188e-31   # electron mass			(kg)
m_p = 1.67262158e-27   # proton mass			(kg)
#k	= 8.987551785972e9 # Coulomb constant		(N*m^2/C^2)
k_B = 1.380649000e-23  # Boltzmann constant 	(J/K)
e_0 = 8.8541878188e-12 # vacuum permittivity	(F/m)

# Initialize and set defaults for and type parameter variables

# Graph/window parameters
x_min:				float	=	-2e-11	# window/simulation bounds for position (m)
x_max:				float	=	 2e-11
v_min:				float	=	-0.025	# window/simulation bounds for velocity (m/s)
v_max:				float	=	 0.025
rho_lim_scale:		float	=	 15		# scale of plot relative to greatest magnitude of initial charge density
E_lim_scale:		float	=	 20		# ditto for E field
window_res:			int		=	 250	# resolution used by 2D histograms
ec_lim_margin:		float	=	 10		# change log_10 margin from initial for energy conservation plot
# Simulation parameters
steps_per_pd:		int		=	 20		# number of time steps per plasma period
#dt:				float	=	 2e-10	# time step between frames (now calculated)
n_frames:			int		=	 500	# number of frames to generate / iterations to process
f_cells_per_lam_D:	float	=	 5		# number of field computation cells per Debye Length
max_field_res:		int		=	 5000	# maximum number of field computation cells
fixed_ions:			bool	=	 False	# if true, ions will be completely stationary (v=0)
two_stream:			bool	=	 True	# if true, two opposing streams of particles will be generated
x_pert_amp_e:		float	=	 0		# amplitude of sinusoidal electron density perturbations
x_pert_amp_i:		float	=	 0		# ditto for ions
x_pert_mode:		int		=	 1		# spatial mode of the sinusoidal density perturbation
v_th_ratio_e:		float	=	 1/60	# ratio of thermal velocity of electrons to max. velocity
v_dr_ratio_e:		float	=	 1/8	# ratio of drift velocity of electrons to max. velocity
v_th_ratio_i:		float	=	 1/60	# ratio of thermal velocity of ions to max. velocity
v_dr_ratio_i:		float	=	 1/8	# ratio of drift velocity of ions to max. velocity
# Particle counts
N_e:				int		=	 20000	# number of electrons to simulate
N_i:				int		=	 20000	# number of ions to simulate
# Solving parameters
E_solver:			int		=	 1		# electric field solver to use. (0 -> cumsum, 1 -> FFT)
integration:		int		=	 0		# particle motion integration method (0 -> Velocity Verlet, 1 -> Leapfrog)

# Computed constants

dt:			float # time step
x_width:	float # width of position sample / window
v_width:	float # width of velocity sample / window
field_dx:	float # position increment used for field computations
field_res:	int   # total number of field computation cells
n_0_e:		float # initial linear electron density
n_0_i:		float # initial linear ion density
w_p_e:		float # electron plasma frequency
w_p_i:		float # ion plasma frequency
v_th_e:		float # electron thermal velocity
v_th_i:		float # ion thermal velocity
lam_D_e:	float # electron Debye length
lam_D_i:	float # ion  Debye length

# Distribution functions (auto-normalized by distgen)

def x_dist(x) -> float: # Base default position distribution
	return np.e**-((30*((x/x_max)-0.25))**2) + np.e**-((30*((x/x_max)+0.25))**2)

def v_dist(v) -> float: # Base default position distribution
	return np.e**-((30*((v/v_max)-0.25))**2) + np.e**-((30*((v/v_max)+0.25))**2)

def x_dist_e(x) -> float: # Default electron position distribution
	return 1

def v_dist_e(v) -> float: # Default electron velocity distribution
	return np.e**-((30*((v/v_max)-0.25))**2) + np.e**-((30*((v/v_max)+0.25))**2)

def x_dist_i(x) -> float: # Default ion position distribution
	return 1

def v_dist_i(v) -> float: # Default ion velocity distribution
	return np.e**-((30*((v/v_max)-0.25))**2) + np.e**-((30*((v/v_max)+0.25))**2)

def set_distributions(v_th_e: float, v_dr_e: float, v_th_i: float, v_dr_i: float): # create distributions from thermal and drift velocities
	global x_dist_e, v_dist_e, x_dist_i, v_dist_i
	if two_stream:
		v_dist_e = lambda v: np.e**-(((v-v_dr_e)/v_th_e)**2) + np.e**-(((v+v_dr_e)/v_th_e)**2)
		v_dist_i = lambda v: np.e**-(((v-v_dr_i)/v_th_i)**2) + np.e**-(((v+v_dr_i)/v_th_i)**2)
	else:
		v_dist_e = lambda v: np.e**-(((v-v_dr_e)/v_th_e)**2)
		v_dist_i = lambda v: np.e**-(((v-v_dr_i)/v_th_i)**2)
	x_dist_e = lambda x: 1 + x_pert_amp_e * np.cos(2*np.pi*x_pert_mode*x/x_width)
	x_dist_i = lambda x: 1 + x_pert_amp_i * np.cos(2*np.pi*x_pert_mode*x/x_width)

# Fxns to set parameters and reinitialize arrays

def compute_etc_consts(): # Compute dependent constants
	global x_width, v_width, field_dx, n_0_e, n_0_i, w_p_e, w_p_i, dt, v_th_e, v_th_i, lam_D_e, lam_D_i, field_res
	x_width = x_max - x_min
	v_width = v_max - v_min
	n_0_e = N_e / x_width
	n_0_i = N_i / x_width
	w_p_e = omega_p(electron, n_0_e)
	w_p_i = omega_p(ion, n_0_i)
	dt = 2 * np.pi / (steps_per_pd * w_p_e)
	v_th_e = v_th_ratio_e * v_width
	v_th_i = v_th_ratio_i * v_width
	lam_D_e = v_th_e / (np.sqrt(2) * w_p_e)
	lam_D_i = v_th_i / (np.sqrt(2) * w_p_i)
	field_res = int(f_cells_per_lam_D * x_width / lam_D_e)
	if field_res > max_field_res:
		# print(f"Debye-sensitive field resolution breaks max limit. Setting field resolution to {max_field_res}.")
		field_res = max_field_res
	field_dx = x_width / field_res

def set_params(inp_params: dict): # Set global simulation parameters from parameter dictionary
	global x_min, x_max, v_min, v_max, rho_lim_scale, E_lim_scale, window_res, ec_lim_margin, steps_per_pd, n_frames, f_cells_per_lam_D, max_field_res, \
		fixed_ions, two_stream, N_e, N_i, x_pert_amp_e, x_pert_amp_i, x_pert_mode, v_th_ratio_e, v_th_ratio_i, v_dr_ratio_e, v_dr_ratio_i, E_solver
	valid_keys = {
		"x_min", "x_max", "v_min", "v_max", "rho_lim_scale", "E_lim_scale", "window_res", "ec_lim_margin",
		"steps_per_pd", "n_frames", "f_cells_per_lam_D", "max_field_res", "fixed_ions", "two_stream", "N_e", "N_i", "x_pert_amp_e", "x_pert_amp_i",
		"x_pert_mode", "v_th_ratio_e", "v_th_ratio_i", "v_dr_ratio_e", "v_dr_ratio_i", "E_solver", "integration"
	}
	for key_str, value in inp_params.items():
		if key_str in valid_keys:
			globals()[key_str] = value
	compute_etc_consts()
	set_solvers()
	set_distributions(v_th_ratio_e * v_width, v_dr_ratio_e * v_width, v_th_ratio_i * v_width, v_dr_ratio_i * v_width)

# Default Simulation Parameters
default_params: dict = {
	"x_min":			-2e-11,
	"x_max":			 2e-11,
	"v_min":			-0.025,
	"v_max":			 0.025,
	"steps_per_pd":		 20,
	"n_frames":			 500,
	"window_res":		 250,
	"f_cells_per_lam_D": 5,
	"max_field_res":	 5000,
    "fixed_ions":		 False,
    "two_stream":		 True,
	"rho_lim_scale":	 15,
	"E_lim_scale":		 40,
	"ec_lim_margin":	 10,
	"N_e":				 20000,
	"N_i":				 20000,
	"x_pert_amp_e":		 0,
	"x_pert_amp_i":		 0,
	"x_pert_mode":		 1,
	"v_th_ratio_e":		 1/600,
    "v_th_ratio_i":		 1/600,
    "v_dr_ratio_e":		 1/8,
	"v_dr_ratio_i":		 1/8,
	"E_solver":			 1,
	"integration":		 0
}

# PARTICLE DEFINITION/DATA HANDLING

# Define default particle params for e- and i+

electron: dict = {
	"name": "e-",
	"charge": -1*e,
	"mass": m_e
}

ion: dict = {
	"name": "i+",
	"charge": 1*e,
	"mass": m_p
}

# initialize/type particle data structs

n_lst: list = []  # particle names
x_lst: list = []  # paricle positions
v_lst: list = []  # particle velocities
a_lst: list = []  # particle accelerations
m_lst: list = []  # particle masses
q_lst: list = []  # particle charges

n_arr:			np.array = 	np.zeros(0,  dtype=str  ) # particle names
x_arr:			np.array = 	np.zeros(0,  dtype=float) # paricle positions
v_arr:			np.array = 	np.zeros(0,  dtype=float) # particle velocities
a_arr:			np.array = 	np.zeros(0,  dtype=float) # particle accelerations
m_arr:			np.array = 	np.zeros(0,  dtype=float) # particle masses
q_arr:			np.array = 	np.zeros(0,  dtype=float) # particle charges

rho_arr:		np.array = 	np.zeros(0,  dtype=float) # charge density
E_arr:			np.array = 	np.zeros(0,  dtype=float) # electric field

t_arr:			np.array = 	np.zeros(n_frames, dtype=float) # time array for energy display
#K_arr:			np.array = 	np.zeros(n_frames, dtype=float) # total kinetic energy
K_arr_e:		np.array = 	np.zeros(n_frames, dtype=float) # electron kinetic energy
K_arr_i:		np.array = 	np.zeros(n_frames, dtype=float) # ion kinetic energy
U_arr:			np.array = 	np.zeros(n_frames, dtype=float) # electric field potential
E_signal_arr:	np.array =	np.zeros(n_frames, dtype=float) # electric field signal for measuring oscillation frequency

e_mask:			np.array = 	np.zeros(0, dtype=bool)
i_mask:			np.array = 	np.zeros(0, dtype=bool)

def reinit_data_arrs(): # re-initialize/zero out all data arrays
	global n_arr, x_arr, v_arr, a_arr, m_arr, q_arr
	global rho_arr, E_arr
	global t_arr, K_arr_e, K_arr_i, U_arr, E_signal_arr
	global e_mask, i_mask
	n_arr = 		np.zeros(0,  dtype=str  )  # particle names
	x_arr = 		np.zeros(0,  dtype=float)  # paricle positions
	v_arr = 		np.zeros(0,  dtype=float)  # particle velocities
	a_arr = 		np.zeros(0,  dtype=float)  # particle accelerations
	m_arr = 		np.zeros(0,  dtype=float)  # particle masses
	q_arr = 		np.zeros(0,  dtype=float)  # particle charges

	rho_arr = 		np.zeros(field_res,  dtype=float)  # charge density
	E_arr = 		np.zeros(field_res,  dtype=float)  # electric field

	t_arr = 		np.zeros(n_frames, dtype=float) # time array for energy display
	#K_arr = 		np.zeros(n_frames, dtype=float) # total kinetic energy
	K_arr_e = 		np.zeros(n_frames, dtype=float) # electron kinetic energy
	K_arr_i = 		np.zeros(n_frames, dtype=float) # ion kinetic energy
	U_arr = 		np.zeros(n_frames, dtype=float) # electric field potential
	E_signal_arr =	np.zeros(n_frames, dtype=float) # electric field signal for FFT frequency sampling

	e_mask = 	np.zeros(0, dtype=bool)
	i_mask = 	np.zeros(0, dtype=bool)

# fxn to add <n> particles of type <particle> to pre-simulation lists based on given distribution fxns and x/v bounds
def add_particles(particle: dict, n: int, xdist = x_dist, vdist = v_dist,
				  xmin: float = x_min, xmax: float = x_max, vmin: float = v_min, vmax: float = v_max,
				  n_res: int = window_res, verbose = True):
	global n_lst, x_lst, v_lst, a_lst, m_lst, q_lst
	n_lst.extend([particle["name"] for _ in range(n)])
	x_lst.extend(evd_array(xdist, n, xmin, xmax, n_res))
	if fixed_ions and particle == ion:
		v_lst.extend([0 for _ in range(n)])
	else:
		v_lst.extend(evd_array(vdist, n, vmin, vmax, n_res))
	a_lst.extend([0 for _ in range(n)])
	m_lst.extend([particle["mass"] for _ in range(n)])
	q_lst.extend([particle["charge"] for _ in range(n)])
	if verbose: print(f"Added {n} of {particle['name']}.")

def write_to_arrays(): # Write the data from pre-simulation lists to the main simulation arrays for faster computing
	global n_arr, x_arr, v_arr, a_arr, m_arr, q_arr
	n_arr = np.array(n_lst)
	x_arr = np.array(x_lst)
	v_arr = np.array(v_lst)
	a_arr = np.array(a_lst)
	m_arr = np.array(m_lst)
	q_arr = np.array(q_lst)

def reset_lists(): # Wipe the pre-simulation lists
	global n_lst, x_lst, v_lst, a_lst, m_lst, q_lst, rho_lst, E_lst
	global rho_arr, E_arr, t_arr, K_arr_e, K_arr_i, U_arr
	n_lst = []  # particle names
	x_lst = []  # paricle positions
	v_lst = []  # particle velocities
	a_lst = []  # particle accelerations
	m_lst = []  # particle masses
	q_lst = []  # particle charges

	rho_arr =	np.zeros(field_res,	dtype=float)	# charge density
	E_arr =		np.zeros(field_res,	dtype=float)	# electric field
	t_arr =		np.zeros(n_frames,	dtype=float)	# time array for energy display
	#K_arr =	np.zeros(n_frames,	dtype=float)	# total kinetic energy
	K_arr_e =	np.zeros(n_frames,	dtype=float)	# total kinetic energy
	K_arr_i =	np.zeros(n_frames,	dtype=float)	# total kinetic energy
	U_arr =		np.zeros(n_frames,	dtype=float)	# electric field potential

# SIMULATION FUNCTIONS
# Prefixes: comp -> compute, init -> initialize

def comp_a(x = x_arr, m = m_arr, q = q_arr): # Compute particle acceleration from mass and pos. in electric field
	return q * E_at_point(x) / m

def comp_rho(x = x_arr, q = q_arr): # compute linear charge density 
	ret_arr = np.zeros(field_res, dtype = float)
	lin_idxs = (x - x_min) / field_dx		# map x vals to continuous linear space of idx vals
	idx_arr = np.floor(lin_idxs).astype(int)	# discretize idx space to integers
	tween_pos = lin_idxs - idx_arr				# find offset btw continuous idx posn and discrete posn
	idx_lft = idx_arr % field_res
	idx_rgt = (idx_arr + 1) % field_res
	np.add.at(ret_arr, idx_lft, q * (1 - tween_pos))
	np.add.at(ret_arr, idx_rgt, q * tween_pos)
	ret_arr /= field_dx
	return ret_arr

def comp_E_sum(rho = rho_arr): # Numerically integrate rho to find E
	ret_arr = np.cumsum(rho) * field_dx / e_0
	ret_arr -= np.mean(ret_arr)
	return ret_arr

def comp_E_fft(rho_arr = rho_arr): # Use numpy's FFT algorithm to find E in k-space
	rho_k = np.fft.fft(rho_arr)
	k_arr = 2 * np.pi * np.fft.fftfreq(field_res, d = field_dx)
	E_k = np.zeros_like(rho_k, dtype = complex)
	nonzero_freqs = k_arr != 0
	E_k[nonzero_freqs] = rho_k[nonzero_freqs] / (1j * k_arr[nonzero_freqs] * e_0)
	E_k[~nonzero_freqs] = 0
	ret_arr = np.real(np.fft.ifft(E_k))
	return ret_arr

comp_E = comp_E_fft

def set_solvers():
	global comp_E, step
	if E_solver == 0:
		comp_E = comp_E_sum
	elif E_solver == 1:
		comp_E = comp_E_fft
	if integration == 0:
		step = step_vv
	elif integration == 1:
		step = step_lf

def E_at_point(x = x_arr): # Find field at some point via interpolation
	cell_idx_pos = (x - x_min) * field_res / x_width
	cell_idx_pos = np.mod(cell_idx_pos, field_res)
	left_side_idx = np.floor(cell_idx_pos).astype(int)
	tween_pos = cell_idx_pos - left_side_idx
	right_side_idx = (left_side_idx + 1) % field_res
	return (1 - tween_pos) * E_arr[left_side_idx] + tween_pos * E_arr[right_side_idx]

def comp_K(v = v_arr, m = m_arr): # Compute the total kinetic energy of particles in a set of arrays
	return np.sum((m * v**2)/2)

def comp_U(E = E_arr): # Compute the electric field potential of particles in a set of arrays
	return (e_0/2) * np.sum(E**2 * field_dx)

# PLOTTING FUNCTIONS

def init_plots(pix_res = window_res): # initialize all of the subplots
	global fig, axs
	fig, axs = plt.subplots(2,3)
	global plot_psh_e, plot_psh_i, plot_vdist_e, plot_vdist_i, plot_rho, plot_E, plot_K_e, plot_K_i, plot_U, plot_energy, plot_err
	fig.set_figwidth(17)
	fig.set_figheight(11)

	# e- phase space
	axs[0][0].set_box_aspect(1)
	axs[0][0].set_xlabel("Position (m)")
	axs[0][0].set_ylabel("Velocity (m/s)")
	plot_psh_e = axs[0][0].imshow(np.zeros((pix_res, pix_res)), extent=[x_min, x_max, v_min, v_max], origin='lower', aspect="auto")

	# i+ phase space
	axs[0][1].set_box_aspect(1)
	axs[0][1].set_xlabel("Position (m)")
	axs[0][1].set_ylabel("Velocity (m/s)")
	plot_psh_i = axs[0][1].imshow(np.zeros((pix_res, pix_res)), extent=[x_min, x_max, v_min, v_max], origin='lower', aspect="auto")

	# velocity distributions
	axs[0][2].set_box_aspect(1)
	axs[0][2].set_xlabel("Velocity (m/s)")
	axs[0][2].set_ylabel("Count")
	plot_vdist_e, = axs[0][2].plot(np.linspace(v_min, v_max, pix_res), np.zeros(pix_res), label = "e-", drawstyle = "steps-mid")
	plot_vdist_i, = axs[0][2].plot(np.linspace(v_min, v_max, pix_res), np.zeros(pix_res), label = "i+", drawstyle = "steps-mid")
	axs[0][2].legend()

	# charge density
	rho_lim = rho_lim_scale * max(abs(rho_arr))
	axs[1][0].set_box_aspect(1)
	axs[1][0].set_xlabel("Position (m)")
	axs[1][0].set_ylabel("Charge Density (C/m)")
	axs[1][0].set_ylim(-rho_lim, rho_lim)
	plot_rho, = axs[1][0].plot(np.linspace(x_min, x_max, field_res), np.zeros_like(rho_arr), lw = 2)

	# electric field
	axs[1][1].set_box_aspect(1)
	axs[1][1].set_xlabel("Position (m)")
	axs[1][1].set_ylabel("Electric Field (V/m)")
	axs[1][1].set_ylim(-E_lim_scale * max(abs(E_arr)), E_lim_scale * max(abs(E_arr)))
	plot_E, = axs[1][1].plot(np.linspace(x_min, x_max, field_res), E_arr, lw = 2)

	# energy conservation
	axs[1][2].set_box_aspect(1)
	axs[1][2].set_xlabel("Time (s)")
	axs[1][2].set_ylabel("Total Energy (J)")
	axs[1][2].set_xlim(0, n_frames * dt)
	#k_t = comp_K(v_arr, m_arr)
	k_e = comp_K(v_arr[e_mask], m_arr[e_mask])
	k_i = comp_K(v_arr[i_mask], m_arr[i_mask])
	u = comp_U(E_arr)
	en_list = [k_e, u, init_energy]
	if not fixed_ions:
		en_list.append(k_i)
	axs[1][2].set_ylim(0.1*np.min(en_list), 10*np.max(en_list))
	axs[1][2].set_yscale("log")
	nigh_zero = 1e-30
	plot_energy, = axs[1][2].plot(t_arr, np.full_like(t_arr, nigh_zero), label = "Total Energy", ls = "-.", lw = 3)
	#plot_K, = axs[1][2].plot(t_arr, np.full_like(t_arr, nigh_zero), label = "Kinetic Energy", ls = "-.", lw = 3)
	plot_K_e, = axs[1][2].plot(t_arr, np.full_like(t_arr, nigh_zero), label = "Kinetic Energy (e-)", ls = "-", lw = 2)
	if not fixed_ions:
		plot_K_i, = axs[1][2].plot(t_arr, np.full_like(t_arr, nigh_zero), label = "Kinetic Energy (i+)", ls = "-", lw = 2)
	plot_U, = axs[1][2].plot(t_arr, np.full_like(t_arr, nigh_zero), label = "Field Potential", ls = "-", lw = 2)
	plot_err, = axs[1][2].plot(t_arr, np.full_like(t_arr, nigh_zero), label = "Absolute Error", ls = "-", lw = 2)
	axs[1][2].legend()

def plot_frame(frame, pix_res = window_res): # plot data onto different axes contained within each frame
	global axs, fig
	global e_mask, i_mask
	global plot_psh_e, plot_psh_i, plot_vdist_e, plot_vdist_i, plot_rho, plot_E, plot_K_e, plot_K_i, plot_U, plot_energy, plot_err
	x_e = x_arr[e_mask]
	v_e = v_arr[e_mask]
	x_i = x_arr[i_mask]
	v_i = v_arr[i_mask]
	curr_time = str(float(f'{frame*dt:.9f}'))

	# electron phase space
	axs[0][0].set_title(f"Electron Phase Space (t = {curr_time})")
	data_psh_e, _, _ = np.histogram2d(x_e, v_e, bins = pix_res, range=[[x_min, x_max], [v_min, v_max]])
	plot_psh_e.set_data(data_psh_e.T)
	plot_psh_e.set_clim(0, np.max(data_psh_e))

	# ion phase space
	axs[0][1].set_title(f"Ion Phase Space (t = {curr_time})")
	data_psh_i, _, _ = np.histogram2d(x_i, v_i, bins = pix_res, range=[[x_min, x_max], [v_min, v_max]])
	plot_psh_i.set_data(data_psh_i.T)
	plot_psh_i.set_clim(0, np.max(data_psh_i))

	# particle velocity distributions
	axs[0][2].set_title(f"Particle Distribution (t = {curr_time})")
	data_vdist_e, _ = np.histogram(v_e, bins = pix_res, range=[v_min, v_max])
	data_vdist_i, _ = np.histogram(v_i, bins = pix_res, range=[v_min, v_max])
	if frame == 0:
		max_count = max(np.max(data_vdist_e), np.max(data_vdist_i))
		axs[0][2].set_ylim(0, max_count * 1.1)
	plot_vdist_e.set_ydata(data_vdist_e)
	plot_vdist_i.set_ydata(data_vdist_i)

	# charge density
	axs[1][0].set_title(f"Charge Density (t = {curr_time})")
	plot_rho.set_ydata(rho_arr)

	# electric field
	axs[1][1].set_title(f"Electric Field (t = {curr_time})")
	plot_E.set_ydata(E_arr)

	# energy conservation
	axs[1][2].set_title(f"Energy Conservation (t = {curr_time})")
	T_arr = K_arr_e + K_arr_i + U_arr # K_arr +
	#plot_K.set_data(t_arr[:frame], K_arr_e[:frame])
	plot_K_e.set_data(t_arr[:frame], K_arr_e[:frame])
	if not fixed_ions:
		plot_K_i.set_data(t_arr[:frame], K_arr_i[:frame])
	plot_U.set_data(t_arr[:frame], U_arr[:frame])
	plot_energy.set_data(t_arr[:frame], T_arr[:frame])
	plot_err.set_data(t_arr[:frame], (np.abs(T_arr - init_energy))[:frame])

def init(N_e, N_i, plots = True, verbose = True): # Initialize the simulation with the given amount of e- and i+
	global x_arr, v_arr, a_arr, rho_arr, E_arr, m_arr
	global init_energy, dt
	if verbose: print("Resetting data structures...")
	reset_lists()
	if verbose: print("Arrays reset.")
	reinit_data_arrs()
	if verbose: print("Adding particles...")
	add_particles(electron, N_e, x_dist_e, v_dist_e, x_min, x_max, v_min, v_max, field_res, verbose)
	add_particles(ion, N_i, x_dist_i, v_dist_i, x_min, x_max, v_min, v_max, field_res, verbose)
	write_to_arrays()
	if verbose: print("Particle arrays filled!")
	dt = (2*np.pi)/(steps_per_pd * omega_p(electron, n_0_e)) # constrain/set dt while we're at it
	# create masks to separate particle types using array position
	global e_mask, i_mask
	e_mask = (n_arr == "e-")
	i_mask = (n_arr == "i+")
	# compute initial charge density, electric field and acceleration
	rho_arr = comp_rho(x_arr, q_arr)
	E_arr = comp_E(rho_arr)
	a_arr = comp_a(x_arr, m_arr, q_arr)
	# calculate initial energy of the system
	k_e = comp_K(v_arr[e_mask], m_arr[e_mask])
	k_i = comp_K(v_arr[i_mask], m_arr[i_mask])
	u = comp_U(E_arr)
	init_energy = k_e + k_i + u
	if integration == 1:
		v_arr += 0.5 * a_arr * dt
	if verbose: print("Computed arrays filled!")
	global fig, axs
	if plots:
		init_plots(window_res)
		plot_frame(0)
		if verbose: print("Plots initialized!")

def step_vv(frame, plot = True): # All code required to plot and advance one frame (Velocity Verlet scheme)
	global n_arr, x_arr, v_arr, a_arr, m_arr, q_arr, E_arr, rho_arr
	global t_arr, K_arr_e, K_arr_i, U_arr, E_signal_arr
	t_arr[frame]		=	frame * dt
	#K_arr[frame]		=	comp_K(v_arr, m_arr)
	K_arr_e[frame]		= 	comp_K(v_arr[e_mask], m_arr[e_mask])
	K_arr_i[frame]		= 	comp_K(v_arr[i_mask], m_arr[i_mask])
	U_arr[frame]		= 	comp_U(E_arr)
	E_signal_arr[frame]	=	E_arr[field_res // 2]
	if plot:
		plot_frame(frame, window_res)
	if fixed_ions:
		moving_mask = e_mask
	else:
		moving_mask = e_mask | i_mask
	x_arr[moving_mask] += (v_arr[moving_mask] * dt + 0.5 * a_arr[moving_mask] * dt**2)
	x_arr[moving_mask] = ((x_arr[moving_mask] - x_min) % x_width) + x_min
	rho_arr = comp_rho(x_arr, q_arr)
	E_arr = comp_E(rho_arr)
	a_arr_new = comp_a(x_arr, m_arr, q_arr)
	v_arr[moving_mask] += (0.5 * (a_arr[moving_mask] + a_arr_new[moving_mask]) * dt)
	a_arr = a_arr_new
	print("Current frame: " + str(frame + 1), "/", n_frames, 20*" " , end="\r")

def step_lf(frame, plot = True): # All code required to plot and advance one frame (Leapfrog scheme)
	global n_arr, x_arr, v_arr, a_arr, m_arr, q_arr, E_arr, rho_arr
	global t_arr, K_arr_e, K_arr_i, U_arr, E_signal_arr
	t_arr[frame]		=	frame * dt
	#K_arr[frame]		=	comp_K(v_arr, m_arr)
	if integration == 1:
		v_c = v_arr - 0.5 * a_arr * dt
	else:
		v_c = v_arr
	K_arr_e[frame]		= 	comp_K(v_c[e_mask], m_arr[e_mask])
	K_arr_i[frame]		= 	comp_K(v_c[i_mask], m_arr[i_mask])
	U_arr[frame]		= 	comp_U(E_arr)
	E_signal_arr[frame]	=	E_arr[field_res // 2]
	if plot:
		plot_frame(frame, window_res)
	if fixed_ions:
		moving_mask = e_mask
	else:
		moving_mask = e_mask | i_mask
	v_arr[moving_mask] += (0.5 * (a_arr[moving_mask]) * dt)
	x_arr[moving_mask] += v_arr[moving_mask] * dt
	x_arr[moving_mask] = ((x_arr[moving_mask] - x_min) % x_width) + x_min
	rho_arr = comp_rho(x_arr, q_arr)
	E_arr = comp_E(rho_arr)
	a_arr = comp_a(x_arr, m_arr, q_arr)
	v_arr[moving_mask] += (0.5 * (a_arr[moving_mask]) * dt)
	print("Current frame: " + str(frame + 1), "/", n_frames, 20*" " , end="\r")

step = step_vv

# SIMULATION ANALYTICS

def omega_p(particle, n_0): # plasma frequency for particles of a given type
	m_s = particle["mass"]
	q_s = particle["charge"]
	return np.sqrt((n_0 * q_s**2) / (m_s * e_0))

def temp(v_th, particle):
	m = particle["mass"]
	return (m * v_th**2)/(2*k_B)

def print_params(): # print input parameters and initial theoretical quantities
	print(20*"-"+" Parameters " + 20*"-")
	print(f"Position Range: [{x_min}, {x_max}] m")
	print(f"Velocity Range: [{v_min}, {v_max}] m/s")
	print(f"Time Step: {dt} s")
	print(f"Number of Electrons: {N_e}")
	print(f"Number of Ions: {N_i}")
	print(f"Number of Field Cells: {field_res}")
	print(f"Two-Stream: {two_stream}")
	print(f"Fixed Ions: {fixed_ions}")
	print(f"Electric Field Solver: {['Cumulative Sum', 'FFT'][E_solver]}")
	print(f"Particle Motion: {['Velocity Verlet', 'Leapfrog'][integration]}")

	print(19*"-" + " Theoreticals " + 19*"-")
	print(f"e- Plasma Frequency: {w_p_e} rad/s ({w_p_e/(2*np.pi)} Hz)")
	print(f"e- Plasma Period: {2*np.pi/w_p_e} s")
	print(f"e- Thermal Velocity: {v_th_e} m/s")
	print(f"e- Temperature: {temp(v_th_e, electron)} K")
	print(f"i+ Thermal Velocity: {v_th_i * int(not fixed_ions)} m/s")
	print(f"i+ Temperature: {temp(v_th_i, ion) * int(not fixed_ions)} K")
	print(f"e- Debye Length: {lam_D_e} m")
	print(52*"-")

def find_fft_freq(signal: np.array = E_signal_arr):
	sig_pd = signal - np.mean(signal)
	fft = np.fft.rfft(sig_pd) 						# all FFT values from the signal
	fft_freqs = np.fft.rfftfreq(len(sig_pd), dt)	# array of discrete frequencies
	fft_abs = np.abs(fft) 							# amplitudes of all frequencies
	idx = np.argmax(fft_abs)						# index of highest amplitude frequency
	l = fft_abs[idx-1]
	m = fft_abs[idx]
	r = fft_abs[idx+1]
	if idx in [0, len(fft_abs) - 1]:				# return if edge frequency
		return fft_freqs[idx]
	denom = l - 2*m + r
	if denom == 0 or m < l or m < r:				# return if m is somehow not a maximum
		return fft_freqs[idx]
	f_offset = 0.5 * (l - r) / (l + r - 2*m)		# interpolate the quadratic peak in frequency
	df = fft_freqs[idx] - fft_freqs[idx-1]			# find change in frequency between discrete peaks
	return fft_freqs[idx] + f_offset * df			# return frequency plus interpolated component

def osc_model(t, A, omega, phi, gamma, C):
	return A * e**(-gamma*t) * np.sin(omega * t + phi) + C

def find_fit_params(signal: np.array = E_signal_arr, time=t_arr):
	y = signal - np.mean(signal)		# subtract mean offset for stability
	# initial guesses
	A_0 = np.std(y)						# amplitude ~ std. dev.
	omega_0 = w_p_e						# theoretical omega_p
	phi_0 = 0
	gamma_0 = 0
	C_0 = 0
	popt, _ = curve_fit(osc_model, time, y, p0=[A_0, omega_0, phi_0, gamma_0, C_0], maxfev=100000)
	_, omega, _, _, gamma = popt				# A, omega, phi, c, gamma
	return omega, gamma			# return frequency (Hz)

def get_results(verbose = True):
	f_p_e = w_p_e / (2*np.pi)
	f_p_i = w_p_i / (2*np.pi)
	err_arr = (K_arr_e + K_arr_i + U_arr) - init_energy
	try:
		w_p_o, gamma = find_fit_params(E_signal_arr, t_arr)
	except RuntimeError:
		print("Curve fit failed. Switching to FFT estimation...")
		w_p_o = 2*np.pi * find_fft_freq(E_signal_arr)
		gamma = np.nan
	f_p_o = w_p_o / (2*np.pi)
	net_charge = np.sum(q_arr)
	results = {
		"w_p_e": w_p_e,											# theoretical e- plasma frequency (rad/s)
		"f_p_e": f_p_e,											# theoretical e- plasma frequency (Hz)
		"w_p_i": w_p_i,											# theoretical i+ plasma frequency (rad/s)
		"f_p_i": f_p_i,											# theoretical i+ plasma frequency (Hz)
		"tau_p_e": 2*np.pi/w_p_e,								# theoretical e- plasma period (s)
		"tau_p_i": 2*np.pi/w_p_i,								# theoretical i+ plasma period (s)
		"v_th_e": v_th_e,										# electron termal velocity
		"T_e": temp(v_th_e, electron),							# electron temperature
		"v_th_i": v_th_i * int(not fixed_ions),					# ion thermal velocity
		"T_i": temp(v_th_i, ion) * int(not fixed_ions),			# ion temperature
		"lam_D_e": lam_D_e,										# electron Debye length
		"w_obs": w_p_o,											# observed plasma frequency (rad/s)
		"f_obs": f_p_o,											# observed plasma frequency (Hz)
		"obs_freq_ratio": f_p_o/f_p_e,							# ratio of observed freq to theoretical
		"obs_decay_coeff": gamma,								# rate of oscillation decay
		"init_energy": init_energy,								# initial total energy of the system
		"energy_drift": err_arr[-1]/init_energy,				# energy drift at the end of simulation
		"max_energy_dev": np.max(np.abs(err_arr))/init_energy,	# maximum energy deviation during the simulation
		"net_charge": net_charge,								# net charge of the system
		"q_imbalance_ratio": net_charge / (e * (N_i + N_e))		# ratio of charge imbalance
	}
	if verbose:
		print(16*"-" + " Simulation Results " + 16*"-")
		for key, value in results.items():
			print(f"{key}: {value}")
		print(52*"-")
	return results

# ------------------------- MAIN SIMULATION FUNCTION ------------------------

def pic_1d(inp_params: dict = default_params, export_anim_filename: str = "", show_plot = True, verbose = True): # function to run the full simulation from .py file
	set_params(inp_params)
	init(N_e, N_i, show_plot, verbose)
	if verbose:
		print_params()
		print("Initialization complete! Running simulation...")
	global fig, axs
	if export_anim_filename == "":
		for frame in range(n_frames):
			step(frame, plot = False)
		if show_plot:
			plot_frame(n_frames - 1)
			fig.show()
	else:
		print("Video export option is enabled. Rendering overhead may cause the simulation to take some time.")
		print(f"Estimated total time: {0.273*n_frames}s")
		ani = anim.FuncAnimation(fig=fig, func=step, frames=n_frames, interval=25)
		ani.save(export_anim_filename)
	if verbose: 
		print("Simulation complete!" + 10*" ")
	results = get_results(verbose)
	return results