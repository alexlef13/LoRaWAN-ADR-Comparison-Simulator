# LoRaWAN-ADR-Comparison-Simulator
A comprehensive LoRaWAN simulation framework comparing multiple Adaptive Data Rate (ADR) algorithms including DQN-based optimization, Original ADR, and ML-ADR with energy consumption analysis.

# LoRaSimulator – ADR Optimization with Reinforcement Learning

This repository contains the **LoRaSimulator**, a Python-based discrete-event simulator developed for the diploma thesis *"Τεχνικές βελτιστοποίησης μηχανισμού ADR σε δίκτυα LoRa με χρήση μηχανικής μάθησης"* (Optimization of ADR mechanism in LoRa networks using Machine Learning).

The simulator extends the functionality of the original **LoRaSim** by incorporating:
- A detailed energy consumption model.
- Support for directional antennas.
- An 18-dimensional state representation per node.
- Implementation of three ADR algorithms for comparison: **Original ADR**, **ML-ADR** (Q-learning), and **DQN** (Deep Q-Network).

---

## Requirements

Before running the code, ensure you have the following installed:

- **Python 3.7 or newer**
- The following Python libraries:

```bash
pip install simpy numpy torch matplotlib

How to Run the Code
1. Download the Code

Clone the repository or download the files:
bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Execution Modes

The simulator supports two modes of operation:
2.1 Comparison Mode – Recommended

This mode automatically runs a full set of experiments for all algorithms (Original ADR, ML-ADR, DQN) across five transmission periods (1000, 3000, 6000, 10000, 20000 ms). At the end, it automatically generates comparison plots.

Command:
bash

python lora_simulator.py --compare

What you will see:

    Progress updates in the console

    Summary tables (DER, energy, battery lifetime)

    Automatically generated plots in the plots/ directory

2.2 Single Experiment Mode

This mode runs one simulation with specific parameters provided via the command line.

Syntax:
bash

python lora_simulator.py <nrNodes> <avgSendTime> <experiment> <simtime> <nrBS> <full_collision> <directionality> <nrNetworks> <baseDist>

Parameters:
Parameter	Description	Values / Example
nrNodes	Number of nodes	e.g. 1000
avgSendTime	Average transmission period (ms)	e.g. 1000
experiment	Algorithm to test	8 (DQN), 9 (Original ADR), 11 (ML-ADR)
simtime	Simulation duration (ms)	e.g. 200000
nrBS	Number of base stations	e.g. 1
full_collision	Full collision model	0 (Disabled), 1 (Enabled)
directionality	Directional antennas	0 (Disabled), 1 (Enabled)
nrNetworks	Number of networks	1 (not actively used)
baseDist	Distance between base stations (m)	e.g. 1000

Example (DQN with 1000 nodes, 1000 ms period, 1 base station, directionality enabled, 200,000 ms duration):
bash

python lora_simulator.py 1000 1000 8 200000 1 0 1 1 1000

Output – What is Generated

After execution, the simulator will create the following files:
File	Description
comparison_results_*.json	Summary results of all experiments
sf_distributions_*.json	Spreading Factor distribution per algorithm
learning_curves_*.json	DER evolution over time
computational_cost_*.json	Execution time and resources per algorithm
plots/der_comparison_*.png	DER comparison between algorithms
plots/energy_comparison_*.png	Energy per successful packet comparison
plots/true_battery_lifetime_*.png	Battery lifetime comparison
plots/sf_distribution_*.png	SF distribution bar charts
Project Structure

The key classes and their responsibilities:
Class	Description
SystemConfig	Central configuration for system and energy parameters
LoRaPacket	Data packet with airtime and RSSI calculation
QuantumActionSpace	Discrete action space of 66 (SF, BW, TX Power) combinations
NodeState	Quantized 18-dimensional state for the agents
EnergyTracker	Energy monitoring with success/failure distinction
UnifiedNodeStatistics	Collects and manages node statistics
BaseNode	Abstract base class for all node types
OriginalADRNode	Classic LoRaWAN ADR (conservative)
MLADRNode	Q-learning-based ADR
DQNNode	Deep Q-Network agent
LoRaSimulator	Main coordinator for the simulation
Notes for Users

    Execution time: Comparison mode (--compare) may take a while (especially for DQN), as it runs 15 simulations (3 algorithms × 5 periods) of 200,000 ms each.

    Memory: DQN requires more memory due to the neural network and replay buffer. For 1000 nodes, RAM usage is ~15 MB per node.

    GPU: The code is set to run on CPU. To use a GPU, change self.device = torch.device("cpu") to torch.device("cuda") in ImprovedDQNAgent.

Troubleshooting
Issue	Possible Solution
ModuleNotFoundError: No module named 'torch'	Install PyTorch: pip install torch
ModuleNotFoundError: No module named 'simpy'	Install SimPy: pip install simpy
DQN does not learn / DER is flat	Check that exploration_steps is sufficient (1000 steps)
Simulation is too slow	Reduce the number of nodes or simulation duration
Citation



License

This project is provided for academic and research purposes. All rights reserved by the author and the supervising professor.
