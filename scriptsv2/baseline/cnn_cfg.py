"""
Default hyperparameters for the supervised CNN baseline.
"""

# Training
NUM_EPOCHS   = 60
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.5
AUGMENT      = True

# Train / test split (must sum to 1.0)
TRAIN_FRACTION = 0.80
TEST_FRACTION  = 0.20
