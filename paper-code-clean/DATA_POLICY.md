# Data and privacy policy

This repository is source-code only. It must not contain raw images, clinical tables, sample-level feature
bags, model checkpoints, split manifests, or prediction exports.

Before publishing any additional artifact:

- confirm that it is permitted by the study consent, ethics approval, data-use agreements, and institutional policy;
- remove direct identifiers and review quasi-identifiers, filenames, free text, image metadata, and embedded file paths;
- check small cells and rare combinations for re-identification risk;
- prefer aggregate tables and synthetic fixtures over sample-level outputs;
- scan both the current Git tree and Git history, because deleting a file in a later commit does not remove it from history;
- document the lawful route by which eligible researchers can request controlled data access, if applicable.

Generated exclusion lists and split manifests may contain pseudonymous sample names. Treat them as research
data even when they contain no direct identifiers.
