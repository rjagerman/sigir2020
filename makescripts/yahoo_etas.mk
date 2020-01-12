# Results for batch sizes experiment under yahoo dataset.
yahoo_none_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_0.5_none_seed_420$(i).json)
yahoo_none_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_0.75_none_seed_420$(i).json)
yahoo_none_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_1.0_none_seed_420$(i).json)
yahoo_none_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_1.25_none_seed_420$(i).json)
yahoo_none_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_1.5_none_seed_420$(i).json)

yahoo_weight_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_0.5_weight_seed_420$(i).json)
yahoo_weight_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_0.75_weight_seed_420$(i).json)
yahoo_weight_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_1.0_weight_seed_420$(i).json)
yahoo_weight_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_1.25_weight_seed_420$(i).json)
yahoo_weight_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_1.5_weight_seed_420$(i).json)

yahoo_sample_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_0.5_sample_seed_420$(i).json)
yahoo_sample_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_0.75_sample_seed_420$(i).json)
yahoo_sample_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_1.0_sample_seed_420$(i).json)
yahoo_sample_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_1.25_sample_seed_420$(i).json)
yahoo_sample_etas_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/etas/yahoo_1.5_sample_seed_420$(i).json)
yahoo_etas_repeat_5: yahoo_none_etas_repeat_5 yahoo_weight_etas_repeat_5 yahoo_sample_etas_repeat_5
.PHONY: yahoo_etas_repeat_5 yahoo_none_etas_repeat_5 yahoo_weight_etas_repeat_5 yahoo_sample_etas_repeat_5

# Eta 1.0
$(BUILD)/results/etas/yahoo_1.0_none_seed_%.json : $(BUILD)/results/optimizers/yahoo_sgd_none_seed_%.json | $(BUILD)/results/etas/
	cp $(BUILD)/results/optimizers/yahoo_sgd_none_seed_$*.json $@

$(BUILD)/results/etas/yahoo_1.0_weight_seed_%.json : $(BUILD)/results/optimizers/yahoo_sgd_weight_seed_%.json | $(BUILD)/results/etas/
	cp $(BUILD)/results/optimizers/yahoo_sgd_weight_seed_$*.json $@

$(BUILD)/results/etas/yahoo_1.0_sample_seed_%.json : $(BUILD)/results/optimizers/yahoo_sgd_sample_seed_%.json | $(BUILD)/results/etas/
	cp $(BUILD)/results/optimizers/yahoo_sgd_sample_seed_$*.json $@

# Eta 0.5
$(BUILD)/results/etas/yahoo_0.5_none_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_0.5.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_0.5.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-02 \
		--optimizer sgd \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/etas/yahoo_0.5_sample_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_0.5.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_0.5.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-03 \
		--optimizer sgd \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/etas/yahoo_0.5_weight_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_0.5.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_0.5.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-04 \
		--optimizer sgd \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

# Eta 0.75
$(BUILD)/results/etas/yahoo_0.75_none_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_0.75.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_0.75.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-02 \
		--optimizer sgd \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/etas/yahoo_0.75_sample_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_0.75.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_0.75.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-03 \
		--optimizer sgd \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/etas/yahoo_0.75_weight_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_0.75.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_0.75.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-06 \
		--optimizer sgd \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@


# Eta 1.25
$(BUILD)/results/etas/yahoo_1.25_none_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.25.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.25.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-03 \
		--optimizer sgd \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/etas/yahoo_1.25_sample_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.25.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.25.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-04 \
		--optimizer sgd \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/etas/yahoo_1.25_weight_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.25.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.25.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-06 \
		--optimizer sgd \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@


# Eta 1.5
$(BUILD)/results/etas/yahoo_1.5_none_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.5.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.5.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e+00 \
		--optimizer sgd \
		--ips_strategy none \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/etas/yahoo_1.5_sample_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.5.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.5.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-04 \
		--optimizer sgd \
		--ips_strategy sample \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/etas/yahoo_1.5_weight_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.5.clog | $(BUILD)/results/etas/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.5.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-06 \
		--optimizer sgd \
		--ips_strategy weight \
		--batch_size 1 \
		--log_every 10000 \
		--eval_every 10000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@
