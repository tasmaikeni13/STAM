# STAM: reproduce everything.
#
# `make all` runs the full pipeline end to end.  Individual stages are targets so a
# re-run can start from any point; each writes its artefacts under runs/<task>/.

PY := PYTHONPATH=. python3 -W ignore
TASKS := cnn gpt

.PHONY: all kernels test example train reference domain sweep landscape sharpness figures paper proofs clean

all: kernels train reference domain sweep landscape sharpness figures paper proofs

kernels:
	$(PY) bench/test_kernels.py
	$(PY) bench/bakeoff.py

test:
	$(PY) bench/test_kernels.py
	$(PY) bench/test_pipeline.py

example:
	$(PY) examples/quickstart.py

train:
	$(PY) experiments/01_train.py --task cnn --device cuda:0
	$(PY) experiments/01_train.py --task gpt --device cuda:1

reference:
	$(PY) experiments/02_reference.py --task cnn
	$(PY) experiments/02_reference.py --task gpt

domain:
	$(PY) experiments/02b_domain.py --task cnn
	$(PY) experiments/02b_domain.py --task gpt

sweep:
	$(PY) experiments/03_budget_sweep.py --task cnn --split train
	$(PY) experiments/03_budget_sweep.py --task gpt --split train

landscape:
	$(PY) experiments/04_landscape.py --task cnn --device cuda:0
	$(PY) experiments/04_landscape.py --task gpt --device cuda:1

sharpness:
	$(PY) experiments/05_sharpness.py --task cnn
	$(PY) experiments/05_sharpness.py --task gpt

figures:
	$(PY) figures/make_figures.py

paper: figures
	$(PY) paper/make_numbers.py
	cd paper && pdflatex -interaction=nonstopmode stam.tex >/dev/null \
	  && bibtex stam >/dev/null \
	  && pdflatex -interaction=nonstopmode stam.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode stam.tex >/dev/null
	@echo "paper/stam.pdf"

proofs:
	cd proofs/StamCert && lake build

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg paper/*.toc
