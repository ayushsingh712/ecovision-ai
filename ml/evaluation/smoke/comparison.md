# Model comparison (smoke test split)

| Model | Parameters | Checkpoint | Accuracy | Macro F1 | Weighted F1 | CPU latency | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 1,175,015 | 4.5 MB | 0.488 | 0.457 | 0.457 | 29.4 ms | 2.6 min |
| MobileNetV2 | 2,232,839 | 8.7 MB | 0.700 | 0.695 | 0.695 | 23.8 ms | 1.8 min |

> **Development smoke run.** These numbers come from models trained on a small stratified subset for pipeline validation. They do not represent final model quality. Run the Colab notebook and re-run evaluation with `--stage final` after installing the artifacts.
