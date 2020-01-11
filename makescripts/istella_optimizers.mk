# Results for optimizers experiment under istella dataset.
istella_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_sgd_none_seed_420$(i).json)
istella_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_sgd_weight_seed_420$(i).json)
istella_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_sgd_sample_seed_420$(i).json)
istella_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adam_none_seed_420$(i).json)
istella_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adam_weight_seed_420$(i).json)
istella_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adam_sample_seed_420$(i).json)
istella_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adagrad_none_seed_420$(i).json)
istella_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adagrad_weight_seed_420$(i).json)
istella_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adagrad_sample_seed_420$(i).json)
.PHONY: istella_optimizers_repeat_5

# SGD
$(BUILD)/results/optimizers/istella_sgd_none_seed_%.json : $(BUILD)/results/optimizers/ istella_clicklogs
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@ \
		--lr 1e-04 \
		--optimizer sgd \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*

$(BUILD)/results/optimizers/istella_sgd_weight_seed_%.json : $(BUILD)/results/optimizers/ istella_clicklogs
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@ \
		--lr 3e-07 \
		--optimizer sgd \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*

$(BUILD)/results/optimizers/istella_sgd_sample_seed_%.json : $(BUILD)/results/optimizers/ istella_clicklogs
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@ \
		--lr 1e-06 \
		--optimizer sgd \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*

# Adam
$(BUILD)/results/optimizers/istella_adam_none_seed_%.json : $(BUILD)/results/optimizers/ istella_clicklogs
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@ \
		--lr 3e-04 \
		--optimizer adam \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*

$(BUILD)/results/optimizers/istella_adam_weight_seed_%.json : $(BUILD)/results/optimizers/ istella_clicklogs
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@ \
		--lr 1e-04 \
		--optimizer adam \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*

$(BUILD)/results/optimizers/istella_adam_sample_seed_%.json : $(BUILD)/results/optimizers/ istella_clicklogs
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@ \
		--lr 3e-04 \
		--optimizer adam \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*

# Adagrad
$(BUILD)/results/optimizers/istella_adagrad_none_seed_%.json : $(BUILD)/results/optimizers/ istella_clicklogs
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@ \
		--lr 1e-01 \
		--optimizer adagrad \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*

$(BUILD)/results/optimizers/istella_adagrad_weight_seed_%.json : $(BUILD)/results/optimizers/ istella_clicklogs
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@ \
		--lr 1e-01 \
		--optimizer adagrad \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*

$(BUILD)/results/optimizers/istella_adagrad_sample_seed_%.json : $(BUILD)/results/optimizers/ istella_clicklogs
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@ \
		--lr 1e-01 \
		--optimizer adagrad \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*