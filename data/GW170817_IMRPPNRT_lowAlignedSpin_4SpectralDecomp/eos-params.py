#!/usr/bin/python

# from https://git.ligo.org/publications/gw170817/bns-eos/blob/master/scripts/eos-params.py
# copiied on 10 oct 2018, gitHash 46d5ff661804823898bd00b970dea5f81b60248b

# Script to take posterior samples from 4-piece polytrope and spectral runs and
# 1) compute posterior samples in other parameters
# 2) calculate radius-mass intervals
# 3) calculate pressure-density intervals
# example:
# ./eos-params.py --post-samps ./posterior_samples.dat --out-dir ./public_html --model spectral

import numpy as np
import scipy.interpolate as interp
import lalsimulation as lalsim
import lal
from optparse import OptionParser

def parse_command_line():
        parser = OptionParser()
        parser.add_option("--post-samps", help = "Path to posterior samples")
        parser.add_option("--out-dir", help = "Path to directory to put outputs")
        parser.add_option("--group", type = 'int', default = -1, help = "1 = Histograms; 2 = M,Lambda/R; 3 = log(P)/Log(rho); Default = -1 (all plots)")
        parser.add_option("--model", help = "EOS model used (4piece or spectral)")
        options, filenames = parser.parse_args()
        return options, filenames

# Inputs
options, filename = parse_command_line()

post_samps = options.post_samps
out_dir = options.out_dir
group = options.group
model = options.model

spec_size = 4

red_z = 0.0099

# Load MCMC samples
head = np.loadtxt(post_samps,dtype='str')[0]
data = np.loadtxt(post_samps,skiprows=1).T

logp1 = None
gamma1 = None
gamma2 = None
gamma3 = None
sd_gamma0 = None
sd_gamma1 = None
sd_gamma2 = None
sd_gamma3 = None
mass1 = None
mass2 = None

# Pick out parameters from MCMC samples
# FIXME: There's got to be a better way to do this...
for i in range(len(head)):
	if model == '4piece':
		if head[i] == 'logp1':
			logp1 = data[i]
	        if head[i] == 'gamma1':
	                gamma1 = data[i]
	        if head[i] == 'gamma2':
	                gamma2 = data[i]
	        if head[i] == 'gamma3':
	                gamma3 = data[i]
	elif model == 'spectral':
	        if head[i] == 'sdgamma0':
	                sd_gamma0 = data[i]
	        if head[i] == 'sdgamma1':
	                sd_gamma1 = data[i]
	        if head[i] == 'sdgamma2':
	                sd_gamma2 = data[i]
	        if head[i] == 'sdgamma3':
	                sd_gamma3 = data[i]
	else:
		print "Model", model, "is not supported"
	if head[i] == 'm1':
                mass1 = data[i]
	if head[i] == 'm2':
                mass2 = data[i]

if model == '4piece':
	assert logp1 is not None
	assert gamma1 is not None
	assert gamma2 is not None
	assert gamma3 is not None
if model == 'spectral':
	assert sd_gamma0 is not None
	assert sd_gamma1 is not None
	assert sd_gamma3 is not None
	assert sd_gamma3 is not None
assert mass1 is not None
assert mass2 is not None

# NS PROPERTIES
if group == 1 or group == -1:
	max_mass = []
	m1_d = []
	m2_d = []
	m1_s = []
	m2_s = []
	r1 = []
	r2 = []
	l1 = []
	l2 = []
	c1 = []
	c2 = []
	k1 = []
	k2 = []
	pc1 = []
	pc2 = []
	# Loop over EOS samples
	print "MCMC samples =", len(mass1)
	for i in range(len(mass1)):
		print "Sample ",i,"/",len(mass1)
                eos = None
                if model == '4piece':
                        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(logp1[i]-1,gamma1[i],gamma2[i],gamma3[i])
                if model == 'spectral':
                        eos = lalsim.SimNeutronStarEOSSpectralDecomposition_for_plot(sd_gamma0[i],sd_gamma1[i],sd_gamma2[i],sd_gamma3[i],spec_size)
                assert eos is not None
		fam = lalsim.CreateSimNeutronStarFamily(eos)
		max_m = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI # [M_sun]
		min_m = lalsim.SimNeutronStarFamMinimumMass(fam)/lal.MSUN_SI # [M_sun]
		# Find NS properties for each EOS sample
		max_mass.append(max_m) # [M_sun]
		m1_d.append(mass1[i]) # [M_sun]
		m1_s.append(m1_d[i]/(1.+red_z)) # [M_sun]
		r1.append(lalsim.SimNeutronStarRadius(m1_s[i]*lal.MSUN_SI,fam)/1000.) # [km]
		k1.append(lalsim.SimNeutronStarLoveNumberK2(m1_s[i]*lal.MSUN_SI,fam)) # [ ]
		c1.append(m1_s[i] * lal.MRSUN_SI / (r1[i]*1000.)) # [ ]
		l1.append((2. / 3.) * k1[i] / c1[i]**5.) # [ ]
		pc1.append(lalsim.SimNeutronStarCentralPressure(m1_s[i]*lal.MSUN_SI,fam)*10.) # [dyne/cm^2]
		m2_d.append(mass2[i]) # [M_sun]
		m2_s.append(m2_d[i]/(1.+red_z)) # [M_sun]
	        r2.append(lalsim.SimNeutronStarRadius(m2_s[i]*lal.MSUN_SI,fam)/1000.) # [km]
		k2.append(lalsim.SimNeutronStarLoveNumberK2(m2_s[i]*lal.MSUN_SI,fam)) # [ ]
	        c2.append(m2_s[i] * lal.MRSUN_SI / (r2[i]*1000.)) # [ ]
	        l2.append((2. / 3.) * k2[i] / c2[i]**5.) # [ ]
		pc2.append(lalsim.SimNeutronStarCentralPressure(m2_s[i]*lal.MSUN_SI,fam)*10.) # [dyne/cm^2]

	max_mass = np.array(max_mass)
	m1_d = np.array(m1_d)
	m2_d = np.array(m2_d)
        m1_s = np.array(m1_s)
        m2_s = np.array(m2_s)
	r1 = np.array(r1)
	r2 = np.array(r2)
	k1 = np.array(k1)
	k2 = np.array(k2)
	c1 = np.array(c1)
	c2 = np.array(c2)
	l1 = np.array(l1)
	l2 = np.array(l2)
	pc1 = np.array(pc1)
	pc2 = np.array(pc2)
	# Write to dictionary
	ns_params = None
	if model == '4piece':
		ns_params = {"m1_d":m1_d,"m2_d":m2_d,\
			"m1_s":m1_s,"m2_s":m2_s,\
			"r1":r1,"r2":r2,\
			"k1":k1,"k2":k2,\
			"c1":c1,"c2":c2,\
			"l1":l1,"l2":l2,\
			"pc1":pc1,"pc2":pc2,\
			"logp1":logp1,"gamma1":gamma1, "gamma2":gamma2,"gamma3":gamma3,\
			"max_mass":max_mass}
	if model == 'spectral':
                ns_params = {"m1_d":m1_d,"m2_d":m2_d,\
			"m1_s":m1_s,"m2_s":m2_s,\
                        "r1":r1,"r2":r2,\
                        "k1":k1,"k2":k2,\
                        "c1":c1,"c2":c2,\
                        "l1":l1,"l2":l2,\
                        "pc1":pc1,"pc2":pc2,\
                        "sd_gamma0":sd_gamma0,"sd_gamma1":sd_gamma1, "sd_gamma2":sd_gamma2,"sd_gamma3":sd_gamma3,\
                        "max_mass":max_mass}
	assert ns_params is not None

	# Save dictionary
	np.save('%s/ns_params.npy'%out_dir,ns_params)
	# Save to individual text files
	np.savetxt('%s/max-mass.dat'%out_dir, max_mass.T)
	np.savetxt('%s/m1_d.dat'%out_dir, m1_d.T)
	np.savetxt('%s/m2_d.dat'%out_dir, m2_d.T)
        np.savetxt('%s/m1_s.dat'%out_dir, m1_s.T)
        np.savetxt('%s/m2_s.dat'%out_dir, m2_s.T)
	np.savetxt('%s/l1.dat'%out_dir, l1.T)
	np.savetxt('%s/l2.dat'%out_dir, l2.T)
	np.savetxt('%s/k1.dat'%out_dir, k1.T)
	np.savetxt('%s/k2.dat'%out_dir, k2.T)
	np.savetxt('%s/c1.dat'%out_dir, c1.T)
	np.savetxt('%s/c2.dat'%out_dir, c2.T)
	np.savetxt('%s/r1.dat'%out_dir, r1.T)
	np.savetxt('%s/r2.dat'%out_dir, r2.T)
	np.savetxt('%s/pc1.dat'%out_dir, pc1.T)
	np.savetxt('%s/pc2.dat'%out_dir, pc2.T)
	if model == '4piece':
		np.savetxt('%s/logp1.dat'%out_dir, logp1.T)
		np.savetxt('%s/gamma1.dat'%out_dir, gamma1.T)
		np.savetxt('%s/gamma2.dat'%out_dir, gamma2.T)
		np.savetxt('%s/gamma3.dat'%out_dir, gamma3.T)
	if model == 'spectral':
                np.savetxt('%s/sd_gamma0.dat'%out_dir, sd_gamma0.T)
                np.savetxt('%s/sd_gamma1.dat'%out_dir, sd_gamma1.T)
                np.savetxt('%s/sd_gamma2.dat'%out_dir, sd_gamma2.T)
                np.savetxt('%s/sd_gamma3.dat'%out_dir, sd_gamma3.T)

# MASS, LAMBDA VS RADIUS
if group == 2 or group == -1:
	mass = np.linspace(0.0,3.0,num=100)
	# Create dictionaries
	radius = {}
	Lambda = {}
	for i in range(len(mass)):
		radius["%g"%mass[i]]=[]
		Lambda["%g"%mass[i]]=[]
	# Loop over EOS samples
	print "MCMC samples =", len(mass1)
	for i in range(len(mass1)):
		print "EOS r(m)",i,"/",len(mass1)
		eos = None
		if model == '4piece':
		        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(logp1[i]-1,gamma1[i],gamma2[i],gamma3[i])
		if model == 'spectral':
			eos = lalsim.SimNeutronStarEOSSpectralDecomposition_for_plot(sd_gamma0[i],sd_gamma1[i],sd_gamma2[i],sd_gamma3[i],spec_size)
		assert eos is not None
	        fam = lalsim.CreateSimNeutronStarFamily(eos)
		max_m = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
		min_m = lalsim.SimNeutronStarFamMinimumMass(fam)/lal.MSUN_SI
		# Loop over mass bins
		for j in range(len(mass)):
			if mass[j] > min_m and mass[j] < max_m:
				r = lalsim.SimNeutronStarRadius(mass[j]*lal.MSUN_SI,fam)/1000.
		        	radius["%g"%mass[j]].append(r)
				k = lalsim.SimNeutronStarLoveNumberK2(mass[j]*lal.MSUN_SI,fam)
				c = mass[j] * lal.MRSUN_SI / (r*1000.)
				Lambda["%g"%mass[j]].append((2. / 3.) * k / c**5.)
	# Save dictionaries
	np.save('%s/mass-radius.npy'%out_dir, radius)
	np.save('%s/mass-Lambda.npy'%out_dir, Lambda)

# PRESSURE VS DENSITY
if group == 3 or group == -1:
	logr = np.linspace(14.0,15.5,num=100)
	# Create dictionary
	logp = {}
	for i in range(len(logr)):
	        logp["%g"%logr[i]]=[]
	print "MCMC samples =", len(mass1)
	# Loop over EOS samples
	for i in range(len(mass1)):
	        print "EOS p(rho)",i,"/",len(mass1)
                eos = None
                if model == '4piece':
                        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(logp1[i]-1,gamma1[i],gamma2[i],gamma3[i])
                if model == 'spectral':
                        eos = lalsim.SimNeutronStarEOSSpectralDecomposition_for_plot(sd_gamma0[i],sd_gamma1[i],sd_gamma2[i],sd_gamma3[i],spec_size)
                assert eos is not None
	        fam = lalsim.CreateSimNeutronStarFamily(eos)
		h = np.linspace(0.0001,lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(eos),10000)
	        rho = []
	        p = []
	        # Populates p(h) and rho(h) arrays given your enthalpy array
        	for k in range(len(h)):
			# With CGS conversions
	           	rho.append(lalsim.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(h[k],eos)*.001)
			#print h[k], "here 1"
	           	p.append(lalsim.SimNeutronStarEOSPressureOfPseudoEnthalpy(h[k],eos)*10)
			#print h[k], "here 2"
	           	# Creates a callable function for logp(logrho)
	        logp_of_logrho = interp.interp1d(np.log10(rho),np.log10(p),kind='linear')      
		# Loop over density (rho) bins
		for j in range(len(logr)):
			if logr[j] <= max(np.log10(rho)):
	           		logp["%g"%logr[j]].append(logp_of_logrho(logr[j]))
			#else:
			#	logp["%g"%logr[j]].append(None)
	# Save dictionary
	np.save('%s/pressure-density.npy'%out_dir, logp)
