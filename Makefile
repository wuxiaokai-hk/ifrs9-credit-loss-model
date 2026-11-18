.PHONY: install test lint run-sample run-public report app data

install:     ## install the package with test tools
	pip install -e ".[dev]"
test:        ## unit and end-to-end tests
	pytest -q
lint:
	ruff check .
data:        ## rebuild the public-data inputs from data/eqb_public/raw
	creditloss build-data
run-sample:  ## simulated credit-union book, with the out-of-sample backtest
	creditloss run --config config/sample.yaml --backtest --report
run-public:  ## EQB Inc. book on public data, with the out-of-sample backtest
	creditloss run --config config/eqb_public.yaml --backtest --report
app:
	creditloss app
