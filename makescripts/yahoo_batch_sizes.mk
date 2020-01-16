# Results for batch sizes experiment under yahoo dataset.
yahoo_batch_sizes_10_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/batch_sizes/yahoo_10_none_seed_42$(i).json)
yahoo_batch_sizes_10_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/batch_sizes/yahoo_10_weight_seed_42$(i).json)
yahoo_batch_sizes_10_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/batch_sizes/yahoo_10_sample_seed_42$(i).json)
yahoo_batch_sizes_20_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/batch_sizes/yahoo_20_none_seed_42$(i).json)
yahoo_batch_sizes_20_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/batch_sizes/yahoo_20_weight_seed_42$(i).json)
yahoo_batch_sizes_20_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/batch_sizes/yahoo_20_sample_seed_42$(i).json)
yahoo_batch_sizes_50_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/batch_sizes/yahoo_50_none_seed_42$(i).json)
yahoo_batch_sizes_50_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/batch_sizes/yahoo_50_weight_seed_42$(i).json)
yahoo_batch_sizes_50_repeat_5: $(foreach i,1 2 3 4 5,$(BUILD)/results/batch_sizes/yahoo_50_sample_seed_42$(i).json)
yahoo_batch_sizes_repeat_5 : yahoo_batch_sizes_10_repeat_5 yahoo_batch_sizes_20_repeat_5 yahoo_batch_sizes_50_repeat_5
.PHONY: yahoo_batch_sizes_repeat_5 yahoo_batch_sizes_10_repeat_5 yahoo_batch_sizes_20_repeat_5 yahoo_batch_sizes_50_repeat_5

# batch 10
$(BUILD)/results/batch_sizes/yahoo_10_none_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/batch_sizes/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e+00 \
		--optimizer sgd \
		--ips_strategy none \
		--batch_size 10 \
		--log_every 1000 \
		--eval_every 1000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/batch_sizes/yahoo_10_weight_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/batch_sizes/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-04 \
		--optimizer sgd \
		--ips_strategy weight \
		--batch_size 10 \
		--log_every 1000 \
		--eval_every 1000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/batch_sizes/yahoo_10_sample_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/batch_sizes/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-03 \
		--optimizer sgd \
		--ips_strategy sample \
		--batch_size 10 \
		--log_every 1000 \
		--eval_every 1000 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

# batch 20
$(BUILD)/results/batch_sizes/yahoo_20_none_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/batch_sizes/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e+00 \
		--optimizer sgd \
		--ips_strategy none \
		--batch_size 20 \
		--log_every 500 \
		--eval_every 500 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/batch_sizes/yahoo_20_weight_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/batch_sizes/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-04 \
		--optimizer sgd \
		--ips_strategy weight \
		--batch_size 20 \
		--log_every 500 \
		--eval_every 500 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/batch_sizes/yahoo_20_sample_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/batch_sizes/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-03 \
		--optimizer sgd \
		--ips_strategy sample \
		--batch_size 20 \
		--log_every 500 \
		--eval_every 500 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

# batch 50
$(BUILD)/results/batch_sizes/yahoo_50_none_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/batch_sizes/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-01 \
		--optimizer sgd \
		--ips_strategy none \
		--batch_size 50 \
		--log_every 200 \
		--eval_every 200 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/batch_sizes/yahoo_50_weight_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/batch_sizes/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 3e-04 \
		--optimizer sgd \
		--ips_strategy weight \
		--batch_size 50 \
		--log_every 200 \
		--eval_every 200 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@

$(BUILD)/results/batch_sizes/yahoo_50_sample_seed_%.json : $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog | $(BUILD)/results/batch_sizes/
	python -m experiments.train \
		--click_log $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog \
		--train_data $(YAHOO_DIR)/train.txt \
		--test_data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--lr 1e-03 \
		--optimizer sgd \
		--ips_strategy sample \
		--batch_size 50 \
		--log_every 200 \
		--eval_every 200 \
		--epochs 5 \
		--seed $*
	mv $@.tmp $@
