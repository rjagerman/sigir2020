tables: $(BUILD)/tables/batchsizes.tbl.tex $(BUILD)/tables/optimizers.tbl.tex $(BUILD)/tables/etas.tbl.tex
.PHONY: tables

$(BUILD)/tables/batchsizes.tbl.tex : $(BUILD)/skylines/yahoo.json $(BUILD)/skylines/istella.json | yahoo_batch_sizes_repeat_5 istella_batch_sizes_repeat_5 $(BUILD)/tables/
	python -m experiments.tables.batch_sizes \
		--json_files $(BUILD)/results/batch_sizes/*.json \
		--skylines $(BUILD)/skylines/*.json \
		--dataset test > $@.tmp
	mv $@.tmp $@


$(BUILD)/tables/optimizers.tbl.tex : $(BUILD)/skylines/yahoo.json $(BUILD)/skylines/istella.json | yahoo_optimizers_repeat_5 istella_optimizers_repeat_5 $(BUILD)/tables/
	python -m experiments.tables.optimizers \
		--json_files $(BUILD)/results/optimizers/*.json \
		--skylines $(BUILD)/skylines/*.json \
		--dataset test > $@.tmp
	mv $@.tmp $@

$(BUILD)/tables/etas.tbl.tex : $(BUILD)/skylines/yahoo.json $(BUILD)/skylines/istella.json | yahoo_etas_repeat_5 istella_etas_repeat_5 $(BUILD)/tables/
	python -m experiments.tables.etas \
		--json_files $(BUILD)/results/etas/*.json \
		--skylines $(BUILD)/skylines/*.json \
		--dataset test > $@.tmp
	mv $@.tmp $@


$(BUILD)/tables/ :
	mkdir -p $(BUILD)/tables/
