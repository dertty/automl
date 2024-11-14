pretty:
	autoflake --remove-all-unused-imports --ignore-init-module-imports -r -i src/automl/model src/automl/metrics
	codespell src/automl/model src/automl/metrics -I .codespell_ignore
	isort src/automl/model src/automl/metrics --profile black
	black src/automl/model src/automl/metrics