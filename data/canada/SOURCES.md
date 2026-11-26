# Canadian panel data (retrieved 2026-11-26)

| File (`raw/`) | Series | Source |
|---|---|---|
| `cba_arrears_regions.csv` | Residential mortgages 90+ days in arrears, quarter-end months 1995-03 to 2026-05: Atlantic (AT), Quebec (QC), Alberta (AB), British Columbia (BC) | [Canadian Bankers Association, May 2026 table](https://votreargent.cba.ca/Assets/CanadianBankersAssociation/Documents/Articles/Statistics/stat-mortgages-arrears-may-2026-en.pdf) |
| `statcan_unemployment_provinces_q.csv` | Unemployment rate, 15+, seasonally adjusted, quarterly mean of monthly, 1990Q1–2026Q3 | Statistics Canada, table 14-10-0287-01 (Web Data Service) |

Ontario arrears come from `data/eqb_public/raw/cba_arrears_ontario.csv` (same CBA table). Manitoba and Saskatchewan
are left out: the transcription of those two tables could not be verified. The five regions cover over 90% of bank
mortgages. Atlantic unemployment is the simple mean of the four Atlantic provinces. House prices are national (BIS),
as no free provincial index covers 1995–2026.
