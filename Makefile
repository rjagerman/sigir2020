# Variables, customize to your environment
YAHOO_DIR ?= /Users/rolfjagerman/Datasets/Yahoo/set1
ISTELLA_DIR ?= /Users/rolfjagerman/Datasets/istella-s
BUILD ?= build
TRAIN_ARGS ?=

# Default make target runs the entire pipeline to generate plots and tables
all: plots tables

# Directories
include makescripts/directories.mk

# Baselines
include makescripts/baselines.mk

# Clicklogs
include makescripts/clicklogs.mk

# Results
include makescripts/yahoo_optimizers.mk
include makescripts/yahoo_batch_sizes.mk
include makescripts/yahoo_etas.mk
include makescripts/istella_optimizers.mk
include makescripts/istella_batch_sizes.mk
include makescripts/istella_etas.mk
include makescripts/plots.mk
include makescripts/tables.mk

# Phony target for easier running of just experiments
experiments: yahoo_batch_sizes_repeat_5 istella_batch_sizes_repeat_5 yahoo_optimizers_repeat_5 istella_optimizers_repeat_5 yahoo_etas_repeat_5 istella_etas_repeat_5 $(BUILD)/skylines/yahoo.json $(BUILD)/skylines/istella.json
.PHONY: all experiments
