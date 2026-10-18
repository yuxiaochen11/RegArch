# construct fate map
T_TOTAL = 1
NODE_LEAF = "node_leaf"
NODE_ROOT = "node_root"
NODE_INTERNAL = "node_internal"


# Inference
REGULATOR_GENE = "regulator_gene"
TARGET_GENE = "target_gene"

SIGMA = 1
LAMBDA_1 = 0.6  # (P(h)=exp(-lambda_1*t),P(l)=1-exp(-lambda_1*t))
LAMBDA_LOW_BOUND = 0.1  # (P(h)=exp(-lambda_1*t),P(l)=1-exp(-lambda_1*t))
LAMBDA_HIGH_BOUND = 0.7  # (P(h)=exp(-lambda_1*t),P(l)=1-exp(-lambda_1*t))

REGULATION_STRENGTH_LOW_BOUND = -1  # (P(h)=exp(-lambda_1*t),P(l)=1-exp(-lambda_1*t))
REGULATION_STRENGTH_HIGH_BOUND = 1  # (P(h)=exp(-lambda_1*t),P(l)=1-exp(-lambda_1*t))


BETA_1 = -2  # -2
BETA_2 = 5  # 5

ATAC_PRIOR_WEIGHT = 1
LINEAGE_PRIOR_WEIGHT = 1
LIKELIHOOD_WEIGHT = 1

OPTIMIZE_METHOD = "L-BFGS-B"  # SLSQP cobyla L-BFGS-B
OPTIMIZE_MAX_FUNCTION_EVALUATIONS = 60000

# The alternative lineage prior treats edge-state changes symmetrically. Benchmarks
# can switch to the CTMC model used by the simulator without changing the
# package's default behavior.
LINEAGE_TRANSITION_MODEL = "alt"
LINEAGE_GAIN_RATE = 0.015
LINEAGE_LOSS_RATE = 0.45
LINEAGE_SIGN_SWITCH_RATE = 0.04
LINEAGE_POSITIVE_FRACTION = 0.60
LINEAGE_EDGE_THRESHOLD = 0.1
LINEAGE_EDGE_TEMPERATURE = 0.03
LINEAGE_SIGN_TEMPERATURE = 0.05
LINEAGE_ROOT_PRIOR_MODE = "atac"
LINEAGE_ROOT_PRIOR_PROBABILITY = 0.15
"""Numerical defaults and schema labels shared across RegArch modules."""

