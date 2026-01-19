train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

gate:
	python scripts/compare.py

package:
	python scripts/package_model.py

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 7860