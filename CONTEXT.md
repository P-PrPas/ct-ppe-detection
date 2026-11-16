# PPE detection

Detection of personal protective equipment in frozen image releases.

## Language

**Dataset release**: A frozen collection of images, labels, class order and split membership with provenance hashes.

**Experiment**: One model and training recipe applied to one dataset release, with its checkpoints and evaluation results.

**Core evaluation**: The original validation and test images and labels that remain unchanged when a release grows.

**Extra heldout**: Retained validation or test images outside core evaluation; excluded from primary evaluation and early stopping.

**Family**: Images connected by source identity, augmentation ancestry or protected video identity. Exact duplicates also connect families for split assignment.

**Semantic quarantine**: Families with uncertain labels retained in heldout data rather than used as training truth.

**Benchmark submission**: Predictions for a frozen Andromeda exam, identified by checkpoint hash and confirmed by a server run.
