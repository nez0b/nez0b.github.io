SOURCES := wl-limit solution-space-clustering computation-tree cover factor-graph-anatomy tree-cut-recursion worked-message-sweep bethe-counting independent-set-factor-graph matching-duality
OUT := ../../img/belief-propagation
.SECONDARY: $(SOURCES:%=%.pdf)

.PHONY: all clean
all: $(SOURCES:%=$(OUT)/%.svg) $(SOURCES:%=$(OUT)/%.png)

%.pdf: %.tex _style.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $<

$(OUT)/%.svg: %.pdf
	mkdir -p $(OUT)
	pdftocairo -svg $< $@

$(OUT)/%.png: %.pdf
	mkdir -p $(OUT)
	pdftoppm -png -r 300 -singlefile $< $(OUT)/$*

clean:
	latexmk -C
