plots: $(BUILD)/plots/batchsizes.pdf $(BUILD)/plots/etas.pdf $(BUILD)/plots/optimizers.pdf $(BUILD)/plots/toy-example.pdf
.PHONY: plots


$(BUILD)/plots/batchsizes.pdf : | yahoo_batch_sizes_repeat_5 istella_batch_sizes_repeat_5 $(BUILD)/plots/
	python -m experiments.plots.batch_sizes \
		--dataset test --json_files ./results/batch_sizes/* \
		--out $@.tmp.pdf \
		--format pdf \
		--legend --height 3.5
	mv $@.tmp.pdf $@

$(BUILD)/plots/optimizers.pdf : | yahoo_optimizers_repeat_5 istella_optimizers_repeat_5 $(BUILD)/plots/
	python -m experiments.plots.optimizers \
		--dataset test --json_files ./results/optimizers/* \
		--out $@.tmp.pdf \
		--format pdf \
		--legend --height 3.5
	mv $@.tmp.pdf $@

$(BUILD)/plots/etas.pdf : | yahoo_etas_repeat_5 istella_etas_repeat_5 $(BUILD)/plots/
	python -m experiments.plots.etas \
		--dataset test --json_files ./results/etas/* \
		--out $@.tmp.pdf \
		--format pdf \
		--legend --height 3.5
	mv $@.tmp.pdf $@


$(BUILD)/plots/toy-example.pdf : | $(BUILD)/plots/
	python -m experiments.plots.toy_sample --format pdf --out $@.tmp.pdf
	mv $@.tmp.pdf $@
