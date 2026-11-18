# Public-data sources (downloaded 2026-11-04)

Everything in this folder is public. `creditloss build-data` rebuilds the four model inputs from `raw/`.

| File (`raw/`) | Series | Source |
|---|---|---|
| `cba_arrears_ontario.csv` | Residential mortgages 90+ days in arrears, Ontario, quarter-end months 1995-03 to 2026-05 | [Canadian Bankers Association](https://votreargent.cba.ca/Assets/CanadianBankersAssociation/Documents/Articles/Statistics/stat-mortgages-arrears-may-2026-en.pdf) |
| `fred_LRHUTTTTCAQ156N.csv` | Canada unemployment rate, 15+, NSA, quarterly (OECD) | [FRED](https://fred.stlouisfed.org/data/LRHUTTTTCAQ156N) |
| `fred_QCAN628BIS.csv` | Residential property prices, Canada, 2010 = 100 (BIS) | [FRED](https://fred.stlouisfed.org/data/QCAN628BIS) |
| `fred_IR3TIB01CAQ156N.csv` | Canada 3-month interbank rate (OECD) | [FRED](https://fred.stlouisfed.org/data/IR3TIB01CAQ156N) |
| `fred_IRLTLT01CAQ156N.csv` | Canada 10-year government bond yield (OECD) | [FRED](https://fred.stlouisfed.org/data/IRLTLT01CAQ156N) |
| `fred_NAEXKP01CAQ657S.csv` | Canada real GDP, q/q growth (OECD) | [FRED](https://fred.stlouisfed.org/data/NAEXKP01CAQ657S) |
| `fred_CORCCACBS.csv` | US credit-card charge-off rate, all commercial banks, SA | [FRED](https://fred.stlouisfed.org/data/CORCCACBS) |
| `fred_CORBLACBS.csv` | US business-loan charge-off rate, all commercial banks, SA | [FRED](https://fred.stlouisfed.org/data/CORBLACBS) |
| `fred_LRUN74TTUSQ156N.csv` | US unemployment rate, 15-74, NSA (OECD) | [FRED](https://fred.stlouisfed.org/data/LRUN74TTUSQ156N) |

**Jump-off book.** EQB Inc., Report to Shareholders Q3 2026 (July 31, 2026) and Q2 2026 MD&A: balances by product,
Stage 1+2 allowance by loan type, Stage 2 shares, gross impaired loans, average uninsured LTV. The figures are typed into
`creditloss/data/build_public.py::build_snapshot`, with the source next to each; items that are not disclosed (the LTV
distribution around the 69% average, remaining amortization, prepayment, card limits, contractual terms) are marked
as assumptions there and in the config. The CBA arrears table was checked row by row (count ratio = printed %).

**Treatment.** Unemployment is seasonally adjusted (stable quarterly factors); unemployment and GDP are bridged
through 2020Q2–2021Q3, when income support and payment deferrals broke their link with credit losses. Series that end
before the jump-off quarter are carried forward and flagged in the report.

**Possible upgrades.** Ontario unemployment (StatCan 14-10-0287), the BoC 5-year mortgage rate, Ontario house prices
and OSB Ontario insolvencies would replace the national proxies; rerun `creditloss build-data`.
