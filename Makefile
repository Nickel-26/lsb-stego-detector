.PHONY: help setup synthetic check-data check-signal train-classical train-cnn \
        eval-classical eval-cnn app test clean

help:
	@echo "Targets:"
	@echo "  setup            install python dependencies"
	@echo "  synthetic        generate a tiny synthetic dataset (no download needed)"
	@echo "  check-data       validate data/raw and build data/manifest.csv"
	@echo "  check-signal     show whether chi-square/RS separate cover vs stego"
	@echo "  train-classical  train SPAM+SRM -> XGBoost/SVM"
	@echo "  train-cnn        train SRNet-lite (GPU recommended)"
	@echo "  eval-classical   evaluate classical model on test split"
	@echo "  eval-cnn         evaluate CNN on test split"
	@echo "  app              launch the Streamlit web UI"
	@echo "  test             run pytest"

setup:
	python -m pip install -r requirements.txt

synthetic:
	python scripts/make_synthetic_dataset.py

check-data:
	python -m src.data.download_check

check-signal:
	python scripts/check_signal.py

train-classical:
	python -m src.train_classical

train-cnn:
	python -m src.train_cnn

eval-classical:
	python -m src.evaluate --model classical --artifacts artifacts/classical

eval-cnn:
	python -m src.evaluate --model cnn --artifacts artifacts/cnn

app:
	streamlit run app/streamlit_app.py

test:
	python -m pytest -q

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
