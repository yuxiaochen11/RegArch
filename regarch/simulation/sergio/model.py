import numpy as np


class Gene:
    """State container for one simulated gene in one cell-state bin."""

    def __init__(self, gene_id, gene_type, bin_id=-1):

        """
        ``gene_type`` is ``MR`` for a master regulator or ``T`` for a target.
        """

        self.ID = gene_id
        self.Type = gene_type
        self.binID = bin_id
        self.Conc = []
        self.Conc_S = []
        self.dConc = []
        self.k = (
            []
        )  # For dynamics simulation it stores k1 to k4 for Rung-Kutta method, list of size 4 * num_c_to_evolve
        self.k_S = (
            []
        )  # For dynamics simulation it stores k1 to k4 for Rung-Kutta method, list of size 4 * num_c_to_evolve
        self.simulatedSteps_ = 0
        self.converged_ = False
        self.converged_S_ = False
        self.ss_U_ = 0  # This is the steady state concentration of Unspliced mRNA
        self.ss_S_ = 0  # This is the steady state concentration of Spliced mRNA

    def append_concentration(self, currConc):
        if isinstance(currConc, list):
            if currConc[0] < 0:
                self.Conc.append([0])
            else:
                self.Conc.append(currConc)
        else:
            if currConc < 0:
                self.Conc.append(0)
            else:
                self.Conc.append(currConc)

    def append_spliced_concentration(self, currConc):
        if isinstance(currConc, list):
            if currConc[0] < 0:
                self.Conc_S.append([0])
            else:
                self.Conc_S.append(currConc)
        else:
            if currConc < 0:
                self.Conc_S.append(0)
            else:
                self.Conc_S.append(currConc)

    def append_concentration_change(self, currdConc):
        self.dConc.append(currdConc)

    def append_runge_kutta(self, list_currK):
        self.k.append(list_currK)

    def append_spliced_runge_kutta(self, list_currK):
        self.k_S.append(list_currK)

    def delete_last_runge_kutta_concentration(self, K):
        for k in range(K):
            self.Conc.pop(-1)

    def delete_last_spliced_runge_kutta_concentration(self, K):
        for k in range(K):
            self.Conc_S.pop(-1)

    def clear_concentration(self):
        """
        This method clears all the concentrations except the last one that may
        serve as intial condition for rest of the simulations
        """
        self.Conc = self.Conc[-1:]

    def clear_concentration_changes(self):
        self.dConc = []

    def increment_step(self):
        self.simulatedSteps_ += 1

    def set_converged(self):
        self.converged_ = True

    def set_spliced_converged(self):
        self.converged_S_ = True

    def set_single_cell_expression(self, list_indices):
        """
        selects input indices from self.Conc and form sc Expression
        """
        self.scExpression = np.array(self.Conc)[list_indices]

    def set_unspliced_steady_state(self, u_ss):
        if u_ss < 0:
            u_ss = 0

        self.ss_U_ = u_ss

    def set_spliced_steady_state(self, s_ss):
        if s_ss < 0:
            s_ss = 0

        self.ss_S_ = s_ss

    def clear_runge_kutta(self):
        self.k = []

    def clear_spliced_runge_kutta(self):
        self.k_S = []
