# Evaluation Report: PIX2VOX++ (3 Views)

- **Test objects**: 900 across 215 categories
- **Inter-object prediction diversity (CD-L1)**: 0.1360

## Overall Benchmark Summary

| Metric | Overall (Micro) | Per-Class (Macro) | Home Decor Subset | Everyday Objects |
| :--- | :--- | :--- | :--- | :--- |
| **iou@0.4** | 0.6446 | 0.6410 | 0.5944 | 0.6480 |
| **iou@0.5** | 0.6483 | 0.6437 | 0.5949 | 0.6520 |
| **cd_l2** | 0.0028 | 0.0027 | 0.0034 | 0.0028 |
| **cd_l1** | 0.0566 | 0.0563 | 0.0607 | 0.0563 |
| **f_score@0.01** | 0.0940 | 0.0948 | 0.0913 | 0.0942 |
| **f_score@0.02** | 0.4138 | 0.4177 | 0.4096 | 0.4141 |
