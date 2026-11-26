"""Model extensions built on top of the projection engine.

    canada        provincial panel data set (CBA arrears by region, StatCan unemployment) and the mortgage
                  renewal payment-shock driver
    panel         provincial panel PD satellite with the payment shock, out-of-sample comparison for Ontario,
                  and the losses it implies for the mortgage books
    uncertainty   loss distributions from parameter and residual uncertainty; reverse stress test
    bvar          model-based scenarios from a Bayesian VAR with a Minnesota prior
    history       full-chain check: model loss rates under the macro that happened vs EQB's reported PCL
    report        runs everything and writes outputs/<config>/extensions/

    creditloss extensions --config config/eqb_public.yaml
"""
