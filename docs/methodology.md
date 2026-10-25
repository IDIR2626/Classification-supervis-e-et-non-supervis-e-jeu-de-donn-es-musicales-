# Methodology and limits

## Question and data checks

How much do the available audio features distinguish low popularity levels, and do distance-based groups agree with those levels? The scope is this low-popularity collection, not all Spotify tracks or future hits.

The loader checks required columns, numeric descriptors, integer popularity labels and boolean/0–1 explicit values. Missing values cause an error rather than silent imputation. Features and labels retain matching row indices.

Reference file: 4,073 rows, 17 columns, no pandas-detected missing values and no fully duplicated rows. Thirteen tempos are zero and the longest duration is approximately 56.8 minutes. These observations are retained. Their presence does not establish whether they are valid beat rates or recording errors.

## Classification

1. Stratified 70/30 split, seed 42.
2. Three-fold stratified CV inside the training partition. The rarest class has three original training tracks, so each fold holds one out.
3. Fit StandardScaler on original fold-training rows for ten real-valued features, then transform validation with the same parameters.
4. Randomly oversample training classes to the majority count while retaining all original rows. Validation and test are never resampled.
5. Evaluate a depth-5 tree on all 13 descriptors and collect out-of-fold confusion counts.
6. In each fold, select at most six nonzero feature importances with a depth-5 tree trained on that fold only. Compare depths 1–5 and k-NN with 2–20 neighbors on those selected features.
7. Evaluate a most-frequent-class baseline, fitted on the original training labels, on the same validation rows.
8. Choose the tree/k-NN candidate by mean CV accuracy. Refit preprocessing, selection and candidate on the 70% partition, then evaluate once on the held-out 30%. Evaluate the baseline on that same test.

Feature names can vary between folds: CV evaluates the selection procedure, not six globally fixed columns. An all-data descriptive tree is created after held-out evaluation and does not select the evaluated candidate.

Scores are unweighted means across folds. Macro-F1 uses all observed labels, assigning zero to undefined per-class scores. Balanced accuracy averages recalls for true classes present in the evaluated split. Fit times exclude preprocessing, selection and prediction; they depend on the machine.

Accuracy is the selection metric throughout. Balanced accuracy and macro-F1 are additional diagnostics, not alternative selection criteria chosen after looking at the test.

## Oversampling limitations

Copies add weight but no diversity. The rarest class has only two unique training songs in each CV fold. With k-NN, one original song can fill several neighbor positions. A larger resampled table does not repair this lack of information.

## Clustering

Start from original rows without balancing or supervised selection. Remove popularity and retain 13 descriptors. For seeds 42–51, create different random 70/30 splits, fit scaling on the training rows, learn 10 k-means centers with 10 initializations and assign test songs to those centers.

Ten centers correspond to ten observed popularity labels; this does not prove ten natural musical groups. Compute test Davies–Bouldin, silhouette and unadjusted Rand for each split, then report mean and range. Splits overlap, so the range is not a confidence interval.

Silhouette and Davies–Bouldin measure geometry. Rand compares partitions of the same observations, counting jointly grouped and jointly separated pairs. It is not popularity prediction accuracy; many separated pairs can dominate it.

## Representation and generalization limits

Mode and explicit remain 0/1. Key remains a numeric category code to preserve the documented reference representation. Euclidean distances therefore impose an arbitrary linear geometry on tonalities. One-hot or circular encoding would be a future sensitivity check, not a result already demonstrated here.

Random splits can contain the same artist on both sides. Labels are ordinal, but classification does not exploit their order. No temporal validation is performed. The rarest held-out class has one song. K-means favors compact groups and is sensitive to outliers and feature scales.

The selected candidate does not beat the majority baseline on accuracy. Its balanced accuracy and macro-F1 are slightly higher but remain low. These results concern the tested data, models and representation. They do not prove that audio features can never be useful, nor do they justify calling this a production predictor or recommendation engine.
