# Results for optimizers experiment under istella dataset.
istella_optimizers_sgd_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_sgd_none_seed_42$(i).json)
istella_optimizers_sgd_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_sgd_weight_seed_42$(i).json)
istella_optimizers_sgd_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_sgd_sample_seed_42$(i).json)
istella_optimizers_adam_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adam_none_seed_42$(i).json)
istella_optimizers_adam_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adam_weight_seed_42$(i).json)
istella_optimizers_adam_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adam_sample_seed_42$(i).json)
istella_optimizers_adagrad_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adagrad_none_seed_42$(i).json)
istella_optimizers_adagrad_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adagrad_weight_seed_42$(i).json)
istella_optimizers_adagrad_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/istella_adagrad_sample_seed_42$(i).json)
istella_optimizers_repeat_5 : istella_optimizers_sgd_repeat_5 istella_optimizers_adam_repeat_5 istella_optimizers_adagrad_repeat_5
.PHONY: istella_optimizers_repeat_5 istella_optimizers_sgd_repeat_5 istella_optimizers_adam_repeat_5 istella_optimizers_adagrad_repeat_5

# SGD
$(BUILD)/results/optimizers/istella_sgd_none_seed_%.json : $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-04 \
		--optimizer sgd \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $* $(TRAIN_ARGS)
	mv $@.tmp $@

$(BUILD)/results/optimizers/istella_sgd_weight_seed_%.json : $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-07 \
		--optimizer sgd \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $* $(TRAIN_ARGS)
	mv $@.tmp $@

$(BUILD)/results/optimizers/istella_sgd_sample_seed_%.json : $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-06 \
		--optimizer sgd \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $* $(TRAIN_ARGS)
	mv $@.tmp $@

# Adam
$(BUILD)/results/optimizers/istella_adam_none_seed_%.json : $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-03 \
		--optimizer adam \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $* $(TRAIN_ARGS)
	mv $@.tmp $@

$(BUILD)/results/optimizers/istella_adam_weight_seed_%.json : $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-04 \
		--optimizer adam \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $* $(TRAIN_ARGS)
	mv $@.tmp $@

$(BUILD)/results/optimizers/istella_adam_sample_seed_%.json : $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-04 \
		--optimizer adam \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $* $(TRAIN_ARGS)
	mv $@.tmp $@

# Adagrad
$(BUILD)/results/optimizers/istella_adagrad_none_seed_%.json : $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-01 \
		--optimizer adagrad \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $* $(TRAIN_ARGS)
	mv $@.tmp $@

$(BUILD)/results/optimizers/istella_adagrad_weight_seed_%.json : $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-01 \
		--optimizer adagrad \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $* $(TRAIN_ARGS)
	mv $@.tmp $@

$(BUILD)/results/optimizers/istella_adagrad_sample_seed_%.json : $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/istella_1m_position_eta_1.0.clog \
		--train_data $(ISTELLA_DIR)/train.txt \
		--test_data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-01 \
		--optimizer adagrad \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $* $(TRAIN_ARGS)
	mv $@.tmp $@
