# Phony target for easier dependencies
baselines: $(BUILD)/baselines/yahoo.pth $(BUILD)/baselines/istella.pth
.PHONY: baselines

# Baseline rankers trained on fractions of data
$(BUILD)/baselines/yahoo.pth : | $(BUILD)/baselines/
	python -m experiments.baseline --train_data $(YAHOO_DIR)/train.txt \
		--output $@.tmp \
		--optimizer sgd \
		--lr 0.0001 \
		--fraction 0.001
	mv $@.tmp $@

$(BUILD)/baselines/yahoo.json : $(BUILD)/baselines/yahoo.pth | $(BUILD)/baselines/
	python -m experiments.eval_model --data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--model $(BUILD)/baselines/yahoo.pth
	mv $@.tmp $@

$(BUILD)/baselines/istella.pth : | $(BUILD)/baselines/
	python -m experiments.baseline --train_data $(ISTELLA_DIR)/train.txt \
		--output $@.tmp \
		--optimizer sgd \
		--lr 0.001 \
		--fraction 0.001
	mv $@.tmp $@

$(BUILD)/baselines/istella.json : $(BUILD)/baselines/istella.pth | $(BUILD)/baselines/
	python -m experiments.eval_model --data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--model $(BUILD)/baselines/istella.pth
	mv $@.tmp $@


# Skyline rankers trained on full supervision data
$(BUILD)/skylines/yahoo.pth : | $(BUILD)/skylines/
	python -m experiments.baseline --train_data $(YAHOO_DIR)/train.txt \
		--output $@.tmp \
		--optimizer sgd \
		--lr 1e-03 \
		--fraction 1.0 \
		--epochs 10 \
		--batch_size 32
	mv $@.tmp $@

$(BUILD)/skylines/yahoo.json : $(BUILD)/skylines/yahoo.pth | $(BUILD)/skylines/
	python -m experiments.eval_model --data $(YAHOO_DIR)/test.txt \
		--output $@.tmp \
		--model $(BUILD)/skylines/yahoo.pth
	mv $@.tmp $@

$(BUILD)/skylines/istella.pth : | $(BUILD)/skylines/
	python -m experiments.baseline --train_data $(ISTELLA_DIR)/train.txt \
		--output $@.tmp \
		--optimizer sgd \
		--lr 3e-04 \
		--fraction 1.0 \
		--epochs 10 \
		--batch_size 32
	mv $@.tmp $@

$(BUILD)/skylines/istella.json : $(BUILD)/skylines/istella.pth | $(BUILD)/skylines/
	python -m experiments.eval_model --data $(ISTELLA_DIR)/test.txt \
		--output $@.tmp \
		--model $(BUILD)/skylines/istella.pth
	mv $@.tmp $@
