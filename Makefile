pretty:
	autoflake --remove-all-unused-imports --ignore-init-module-imports -r -i src/automl/model
	codespell src/automl/model -I .codespell_ignore
	isort src/automl/model --profile black
	black src/automl/model