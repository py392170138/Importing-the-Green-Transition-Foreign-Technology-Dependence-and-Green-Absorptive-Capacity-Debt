# Public data sources and frozen local inputs

The links below point to the providers. This repository contains retrieval and processing code, not copies of provider datasets. Local acquisition took place on 21–23 August 2026; API and provider revisions can change the returned values. Keep the exact downloaded file name, query, access date, and SHA-256 when rebuilding.

| Source | Role | Frozen local version or coverage | Official access |
| --- | --- | --- | --- |
| CEPII BACI HS96 and HS07 | Bilateral product trade, green import and export measures, baseline shares | V202601; HS96 1996–2024; HS07 2007–2024 | [BACI database](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37), [January 2026 release notes](https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/release_notes_202601.pdf) |
| OECD TiVA / ICIO | Foreign and domestic value-added measures | TiVA 2025 edition; core analysis through 2022; `DFD_FVA` and `FD_VA` initialization supplements | [Trade in value added](https://www.oecd.org/en/topics/sub-issues/trade-in-value-added.html), [ICIO tables](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html) |
| OpenAlex Works API | Green and total publication counts by country and year | 59 included green topics; country-year aggregates for 1992–2024; no full snapshot | [API reference](https://help.openalex.org/api/), [paging](https://help.openalex.org/api/paging/) |
| IRENASTAT | Renewable electricity capacity, generation, and shares | Capacity through 2025; generation through 2023; exact table versions recorded in queries | [IRENA data](https://www.irena.org/Data), [PxWeb table](https://pxweb.irena.org/pxweb/en/IRENASTAT/IRENASTAT__Power%20Capacity%20and%20Generation/Country_ELECSTAT_2026_H2_PX.px/) |
| World Bank WDI | GDP, population, emissions, energy, industry, and trade controls | Annual API extracts, primarily 1996–2024 | [Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392) |
| ILOSTAT | Occupation and skill proxies | Official bulk extracts; uneven country-year coverage | [Bulk download](https://ilostat.ilo.org/data/bulk/) |
| OECD IFCMA and EPS | Policy controls and robustness | IFCMA April 2026; EPS 1990–2020 | [IFCMA database](https://www.oecd.org/en/data/datasets/ifcma-climate-policy-database.html), [EPS](https://www.oecd.org/en/topics/sub-issues/economic-policies-to-foster-green-growth/how-stringent-are-environmental-policies.html) |
| APEC/OECD/WITS/UNSD classification tables | Environmental-goods lists and HS/BEC concordance | Frozen source documents and conversion tables | [APEC Annex C](https://www.apec.org/meeting-papers/leaders-declarations/2012/2012_aelm/2012_aelm_annexc), [UNSD HS conversions](https://unstats.un.org/unsd/trade/conversions/hs%20correlation%20and%20conversion%20tables.htm) |

Two BACI ZIPs accounted for about 4.06 GB of the 4.28 GB raw-data snapshot. The local code also builds intermediate tables; the full local project used about 18 GB for that layer. These sizes describe the frozen local snapshot, not a guaranteed download size for later releases.

OpenAlex's current API documentation supports `per_page=100` and cursor paging; its rate limits and authentication may change. The code records queries and response hashes. `config/openalex_topic_decisions_v1.yaml` freezes topic selection, and `config/sources.yaml` lists source identifiers and URLs.
