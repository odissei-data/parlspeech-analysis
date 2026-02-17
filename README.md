# ParlSpeech V2 — Cross-Parliament Comparative Analysis

A demo analysis of the publicly available [ParlSpeech V2](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN) dataset, designed to run in [SURF's Blind SANE](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/96207010/Working+with+Blind+SANE+as+a+researcher) secure analysis environment.

ParlSpeech V2 contains 6.3 million parliamentary speeches from 9 democracies (Rauh & Schwalbach, 2020).

## Parliaments covered

| Country        | Corpus                    |
|----------------|---------------------------|
| Austria        | Nationalrat               |
| Czech Republic | Poslanecká sněmovna (PSP) |
| Denmark        | Folketing                 |
| Germany        | Bundestag                 |
| Netherlands    | Tweede Kamer              |
| New Zealand    | House of Representatives  |
| Spain          | Congreso de los Diputados |
| Sweden         | Riksdagen                 |
| United Kingdom | House of Commons          |

## What the script produces

- **summary.csv** — one row per parliament with aggregate statistics (total speeches, unique speakers, parties, average speech length, date range)
- **polarisation.csv** — year-by-year lexical polarisation score per parliament
- **5 charts** — total speeches, unique speakers, average speech length, number of parties per parliament, and lexical polarisation trends over time
- **report.html** — a self-contained HTML report combining all tables and charts

The polarisation analysis measures how linguistically distinct parties are from each other within each parliament per year, using mean pairwise cosine distance between party word-frequency vectors.

## How to use with Blind SANE

### Instructions

1. The data provider: Upload the 9 `.rds` corpus files to `/source/` on the Blind SANE VM.
2. The researcher: When creating the Blind SANE workspace, provide this Git repository as the `blind_python_source`:
   ```
   https://github.com/odissei-data/parlspeech-analysis.git
   ```
3. The data provider: After the analysis completes, the results can be extracted from `/results/` on the Blind SANE VM for review and release.

### Blind SANE configuration

| Field                  | Value                                                        |
|------------------------|--------------------------------------------------------------|
| `blind_python_source`  | `https://github.com/odissei-data/parlspeech-analysis.git`   |
| Recommended VM         | 32 GB RAM, 4 vCPUs                                          |
| Expected runtime       | ~20 minutes                                                  |

### Running manually

```bash
python3 script.py -i /source/ -o /results/ -t /tmp/
```

### Dependencies

Listed in [`requirements.txt`](requirements.txt): `rdata`, `pandas`, `matplotlib`, `numpy`.

## Data source

Rauh, C. & Schwalbach, J. (2020). *The ParlSpeech V2 data set: Full-text corpora of 6.3 million parliamentary speeches in the key legislative chambers of nine representative democracies.* Harvard Dataverse. [doi:10.7910/DVN/L4OAKN](https://doi.org/10.7910/DVN/L4OAKN)
