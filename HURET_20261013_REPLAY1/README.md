# HuReT N014 — recovered computational evidence

The previous 12 October execution saved the manuscript to Drive, but its evidence bundle was not saved before interruption. The original execution script was recovered byte for byte (SHA-256 in the receipt). It was run again on 13 October with unchanged methods, seeds, and data. The key numerical outputs exactly match the existing revised manuscript.

This is a recovery/reproduction of the same analysis, not an independent validation, new dataset, July source reproduction, or biological experiment. The original July true-label-centroid result remains an ineligible oracle comparison.

`RESULT.json`, held-out image IDs, input perturbations, fifteen saved network weight files, grouped out-of-fold predictions, per-image losses, and source code are provided. Run the source with a NEW output directory; it refuses to overwrite prior evidence. Protocol text has its original 12 October date; `RECOVERY_RECEIPT.json` records the actual re-execution date.

One train/validation/test split; 540 held-out base-classifier images; five perturbations per image must not be treated as independent subjects. Meta-model cross-validation has overlapping fitted training folds. Bootstrap intervals are conditional on the fitted models and predictions, not full-pipeline population inference. Noise amplitude is known in this artificial benchmark. No untouched additional meta-test cohort exists.

Data come from sklearn.datasets.load_digits; the dependency supplies the dataset. Review the upstream dataset attribution and redistribution conditions before public release. This bundle is PRIVATE; no public upload or journal submission has occurred.
