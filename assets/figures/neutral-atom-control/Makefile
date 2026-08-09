SOURCES := cover levels control-loop collocation hardware-pipeline
OUT := ../../img/neutral-atom-control
.SECONDARY: $(SOURCES:%=%.pdf)

.PHONY: all clean
all: $(SOURCES:%=$(OUT)/%.svg) $(SOURCES:%=$(OUT)/%.png)

%.pdf: %.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $<

$(OUT)/%.svg: %.pdf
	mkdir -p $(OUT)
	pdftocairo -svg $< $@

$(OUT)/%.png: %.pdf
	mkdir -p $(OUT)
	pdftoppm -png -r 300 -singlefile $< $(OUT)/$*

clean:
	latexmk -C
