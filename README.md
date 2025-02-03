# Generalization with quantum geometry for learning unitaries

Companion program for "Generalization with quantum geometry for learning unitaries" by T. Haug, M.S. Kim.



Program learns to represent unitaries with parameterized ansatz unitary.
Uses supervised learning with a finite training dataset. 
Study how generalization and training success depend on circuit depth and number of training data.

Normally one requires extensive numerical training studies to determine optimal circuit depth and number of training data to achieve generalization.

Program computes data Quantum Fisher information metric (DQFIM) directly from ansatz and training data.
Maximal rank of the DQFIM determines number of circuit parameters and training data needed to converge to global minimum and generalization

Program implements training with hardware efficient circuit or XY ansatz, which has particle number symmetry.
Optimization of cost function is performed with BFGS algorithm.
Training states can be either Haar random, product states, or particle number conserved states.

NOTE: Requires older version of qutip, namely <=4.7.5, which has dependencies on older versions of numyp and scipy.
To install, make clean python environment and install packages as:
- pip install numpy==1.26.4
- pip install scipy==1.12.0
- pip install qutip==4.7.5

