# Dataset

Source: **Etienne GCC — Spotify Unpopular Songs**, hosted on Kaggle.

https://www.kaggle.com/datasets/estienneggx/spotify-unpopular-songs/data

Download the CSV from the source and save it as `data/raw/unpopular_songs.csv`.

The raw CSV is excluded from this repository. Its redistribution license has not been independently verified. The MIT license applies to project code and documentation, not to the source data. Follow the provider's terms when downloading or using it.

The reference run used **4,073 rows and 17 columns**, with popularity labels 0–9. The file contains track metadata and audio-feature values, not audio recordings. A different source version can produce different results.

SHA-256 of the reference CSV:

```text
7c70b72e86c7dd97aa3d0656f26f4670979efa6c9b81d4ab77a1ac2cfa027fc9
```

Every new run records its checksum, row count, class counts, seed and library versions in `reports/run_metadata.json`. Track names, artist names and IDs are excluded from model features. Popularity is removed before clustering and used afterwards for evaluation only.
