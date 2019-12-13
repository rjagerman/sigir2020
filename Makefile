# Variables, customize to your environment
YAHOO_DIR := /Users/rolfjagerman/Datasets/Yahoo/set1
BUILD := build

# Default is to run the entire experimental pipeline
.PHONY: all baselines clicklogs
all: baselines clicklogs
baselines: $(BUILD)/baselines/yahoo.pth
clicklogs: $(BUILD)/clicklogs/yahoo_1m_perfect.clog $(BUILD)/clicklogs/yahoo_1m_position_eta_0.0.clog $(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog $(BUILD)/clicklogs/yahoo_1m_position_eta_1.5.clog $(BUILD)/clicklogs/yahoo_1m_position_eta_2.0.clog $(BUILD)/clicklogs/yahoo_1m_nearrandom_eta_1.0.clog


# Baseline rankers trained on fractions of data
$(BUILD)/baselines/yahoo.pth : | $(BUILD)/baselines/
	python -m experiments.baseline --train_data $(YAHOO_DIR)/train.txt \
		--vali_data $(YAHOO_DIR)/vali.txt \
		--output $(BUILD)/baselines/yahoo.pth \
		--optimizer sgd \
		--lr 0.0001 \
		--fraction 0.001

$(BUILD)/baselines/ :
	mkdir -p $(BUILD)/baselines/


# Click logs generated by baseline rankers
$(BUILD)/clicklogs/yahoo_1m_perfect.clog : $(BUILD)/baselines/yahoo.pth | $(BUILD)/clicklogs/
	python -m experiments.simulate_clicks --input_data $(YAHOO_DIR)/train.txt \
		--ranker $(BUILD)/baselines/yahoo.pth \
		--output_log $@ \
		--sessions 10_000_000 \
		--max_clicks 1_000_000 \
		--behavior perfect

$(BUILD)/clicklogs/yahoo_100k_position_eta_0.0.clog : $(BUILD)/baselines/yahoo.pth | $(BUILD)/clicklogs/
	python -m experiments.simulate_clicks --input_data $(YAHOO_DIR)/train.txt \
		--ranker $(BUILD)/baselines/yahoo.pth \
		--output_log $@ \
		--sessions 10_000_000 \
		--max_clicks 100_000 \
		--behavior position \
		--eta 0.0

$(BUILD)/clicklogs/yahoo_1m_position_eta_0.0.clog : $(BUILD)/baselines/yahoo.pth | $(BUILD)/clicklogs/
	python -m experiments.simulate_clicks --input_data $(YAHOO_DIR)/train.txt \
		--ranker $(BUILD)/baselines/yahoo.pth \
		--output_log $@ \
		--sessions 10_000_000 \
		--max_clicks 1_000_000 \
		--behavior position \
		--eta 0.0

$(BUILD)/clicklogs/yahoo_100k_position_eta_1.0.clog : $(BUILD)/baselines/yahoo.pth | $(BUILD)/clicklogs/
	python -m experiments.simulate_clicks --input_data $(YAHOO_DIR)/train.txt \
		--ranker $(BUILD)/baselines/yahoo.pth \
		--output_log $@ \
		--sessions 10_000_000 \
		--max_clicks 100_000 \
		--behavior position \
		--eta 1.0

$(BUILD)/clicklogs/yahoo_1m_position_eta_1.0.clog : $(BUILD)/baselines/yahoo.pth | $(BUILD)/clicklogs/
	python -m experiments.simulate_clicks --input_data $(YAHOO_DIR)/train.txt \
		--ranker $(BUILD)/baselines/yahoo.pth \
		--output_log $@ \
		--sessions 10_000_000 \
		--max_clicks 1_000_000 \
		--behavior position \
		--eta 1.0

$(BUILD)/clicklogs/yahoo_100k_position_eta_1.5.clog : $(BUILD)/baselines/yahoo.pth | $(BUILD)/clicklogs/
	python -m experiments.simulate_clicks --input_data $(YAHOO_DIR)/train.txt \
		--ranker $(BUILD)/baselines/yahoo.pth \
		--output_log $@ \
		--sessions 10_000_000 \
		--max_clicks 100_000 \
		--behavior position \
		--eta 1.5

$(BUILD)/clicklogs/yahoo_1m_position_eta_1.5.clog : $(BUILD)/baselines/yahoo.pth | $(BUILD)/clicklogs/
	python -m experiments.simulate_clicks --input_data $(YAHOO_DIR)/train.txt \
		--ranker $(BUILD)/baselines/yahoo.pth \
		--output_log $@ \
		--sessions 10_000_000 \
		--max_clicks 1_000_000 \
		--behavior position \
		--eta 1.5

$(BUILD)/clicklogs/yahoo_100k_position_eta_2.0.clog : $(BUILD)/baselines/yahoo.pth | $(BUILD)/clicklogs/
	python -m experiments.simulate_clicks --input_data $(YAHOO_DIR)/train.txt \
		--ranker $(BUILD)/baselines/yahoo.pth \
		--output_log $@ \
		--sessions 10_000_000 \
		--max_clicks 100_000 \
		--behavior position \
		--eta 2.0

$(BUILD)/clicklogs/yahoo_1m_position_eta_2.0.clog : $(BUILD)/baselines/yahoo.pth | $(BUILD)/clicklogs/
	python -m experiments.simulate_clicks --input_data $(YAHOO_DIR)/train.txt \
		--ranker $(BUILD)/baselines/yahoo.pth \
		--output_log $@ \
		--sessions 10_000_000 \
		--max_clicks 1_000_000 \
		--behavior position \
		--eta 2.0

$(BUILD)/clicklogs/yahoo_1m_nearrandom_eta_1.0.clog : $(BUILD)/baselines/yahoo.pth | $(BUILD)/clicklogs/
	python -m experiments.simulate_clicks --input_data $(YAHOO_DIR)/train.txt \
		--ranker $(BUILD)/baselines/yahoo.pth \
		--output_log $@ \
		--sessions 10_000_000 \
		--max_clicks 1_000_000 \
		--behavior nearrandom \
		--eta 1.0

$(BUILD)/clicklogs/ :
	mkdir -p $(BUILD)/clicklogs/
