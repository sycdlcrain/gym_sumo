""" Sumo simulator
Initialize the sumo simulator with the necessary processing parameters.

Kaushik Subramanian @ Honda RI, USA
Created on July 21st 2016

modified by David Isele
"""

import os, sys, subprocess
import traci

# Define the port you want to use (choices: 8813, 8843, 8873, etc.)
# PORT = 8870

def initSimulator(withGUI, portnum, descriptor):

	# Path to the sumo binary
	if withGUI:
		sumoBinary = "/usr/local/bin/sumo-gui"
	else:
		sumoBinary = "/usr/local/bin/sumo"

	# Load the scenario
	descriptor = descriptor.astype(int)
	task = str(descriptor[0])+str(descriptor[1])+str(descriptor[2])+str(descriptor[3])+'.sumocfg'
	sumoConfig = "/home/disele/Documents/PROJ-DeepRL/gym-sumo/gym_sumo/envs/roadMTL/"+task

	# Call the sumo simulator
	sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(portnum), \
		"--time-to-teleport", str(-1), "--collision.check-junctions", str(True), \
		"--no-step-log", str(True), "--no-warnings", str(True)], stdout=sys.stdout, stderr=sys.stderr)

	# Initialize the simulation
	traci.init(portnum)
	return traci

def closeSimulator(traci):
	traci.close()
	sys.stdout.flush()

