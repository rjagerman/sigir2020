# Results for optimizers experiment under yahoo dataset.
yahoo_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/yahoo_sgd_none_seed_420$(i).json)
yahoo_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/yahoo_sgd_weight_seed_420$(i).json)
yahoo_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/yahoo_sgd_sample_seed_420$(i).json)
yahoo_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/yahoo_adam_none_seed_420$(i).json)
yahoo_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/yahoo_adam_weight_seed_420$(i).json)
yahoo_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/yahoo_adam_sample_seed_420$(i).json)
yahoo_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/yahoo_adagrad_none_seed_420$(i).json)
yahoo_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/yahoo_adagrad_weight_seed_420$(i).json)
yahoo_optimizers_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/optimizers/yahoo_adagrad_sample_seed_420$(i).json)
.PHONY: yahoo_optimizers_repeat_5

# SGD
$(BUILD)/results/optimizers/yahoo_sgd_none_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-02 \
		--optimizer sgd \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/optimizers/yahoo_sgd_weight_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-06 \
		--optimizer sgd \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/optimizers/yahoo_sgd_sample_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-05 \
		--optimizer sgd \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

# Adam
$(BUILD)/results/optimizers/yahoo_adam_none_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-02 \
		--optimizer adam \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/optimizers/yahoo_adam_weight_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-05 \
		--optimizer adam \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/optimizers/yahoo_adam_sample_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-02 \
		--optimizer adam \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

# Adagrad
$(BUILD)/results/optimizers/yahoo_adagrad_none_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e+00 \
		--optimizer adagrad \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/optimizers/yahoo_adagrad_weight_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-01 \
		--optimizer adagrad \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/optimizers/yahoo_adagrad_sample_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/optimizers/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e+00 \
		--optimizer adagrad \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10_000 \
		--eval_every 10_000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@
