# Evaluation Report: PIX2VOX++ (3 Views)

- **Test objects**: 900 across 215 categories
- **Inter-object prediction diversity (CD-L1)**: 0.1361

## Overall Benchmark Summary

| Metric | Overall (Micro) | Per-Class (Macro) | Home Decor Subset | Everyday Objects |
| :--- | :--- | :--- | :--- | :--- |
| **iou@0.4** | 0.6471 | 0.6439 | 0.5941 | 0.6508 |
| **iou@0.5** | 0.6518 | 0.6474 | 0.5973 | 0.6556 |
| **cd_l2** | 0.0028 | 0.0028 | 0.0035 | 0.0027 |
| **cd_l1** | 0.0566 | 0.0566 | 0.0617 | 0.0562 |
| **f_score@0.01** | 0.0937 | 0.0941 | 0.0904 | 0.0939 |
| **f_score@0.02** | 0.4118 | 0.4142 | 0.4034 | 0.4124 |
