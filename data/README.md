# Dataset placement

ENDOCARE expects this file:

```
data/structured_endometriosis_data.csv
```

Required columns:

- Age
- Menstrual_Irregularity
- Chronic_Pain_Level
- Hormone_Level_Abnormality
- Infertility
- BMI
- Diagnosis

The CSV present during development contained 10,000 rows and those exact headers. Missing-value counts, duplicates, ranges and label rates are computed from the file on disk at training time; they are not hard-coded as clinical facts.

## Licence and ZIP packaging

The repository does not document a redistribution licence for the CSV. `ENDOCARE.zip` therefore omits `structured_endometriosis_data.csv`. Copy the dataset back into `data/` after unzipping, then run training if `models/` artefacts were also omitted from a given package.

If you obtained the table from Kaggle or another host, follow that host's licence. This project does not assert ownership of the data.
