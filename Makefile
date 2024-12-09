pretty:
	autoflake --remove-all-unused-imports --ignore-init-module-imports -r -i src/automl/model src/automl/metrics src/automl/main.py
	codespell src/automl/model src/automl/metrics src/automl/main.py -I .codespell_ignore
	isort src/automl/model src/automl/metrics src/automl/main.py --profile black
	black src/automl/model src/automl/metrics src/automl/main.py