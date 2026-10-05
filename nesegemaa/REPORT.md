# nesegemaa track — final report (auto-generated tables + fixed protocol notes)

Scope: AtlasNet + Point-E (+ TripoSR track), branch `nesegemaa`.
Protocol (all numbers): splits.json test lists, view `00.png` unless noted, UnitBall (subtract mean, divide by max radius), squared-distance units, F threshold 0.001 (≈0.0316 Euclidean). `<3`-object categories are noise-flagged.

## AtlasNet ablation

| run | K_views | pool | patches | template | bottleneck | encoder_train | aug | loss | lr | batch | seed | best_chamfer | best_fscore | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a1_k3_max | 3 | max | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.04651217204001215 | 0.09291193634271622 | OK |
| a1_k3_attn | 3 | attn | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.06574219548039967 | 0.09488141702281104 | OK |
| a1_k5_max | 5 | max | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.053737382094065346 | 0.09317540046241549 | OK |
| a1_k5_attn | 5 | attn | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.056586981233623296 | 0.09636813484960133 | OK |
| a1_k8_max | 8 | max | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.06599624206622441 | 0.08251947744025125 | OK |
| a1_k8_attn | 8 | attn | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.060704423735539116 | 0.08377090841531754 | OK |
| a2_p1_sq | 1 | none | 1 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.04561556213431888 | 0.08204326530297597 | OK |
| a2_p10_sq | 1 | none | 10 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.1267646989888615 | 0.07854702572027843 | OK |
| a2_p25_sphere | 1 | none | 25 | SPHERE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.036068759858608246 | 0.13152664485904905 | OK |
| a2_p25_4096 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.04688040167093277 | 0.08909631354941262 | OK |
| a2_p25_bn512 | 1 | none | 25 | SQUARE | 512 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.0831894502043724 | 0.07906718138191435 | OK |
| a3_lr3e4 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.0003 | 8 | -1 | 0.16403056184450784 | 0.08060628506872389 | OK |
| a3_bs16 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 16 | -1 | 0.05010243091318342 | 0.08558411813444561 | OK |
| a3_bs32 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 32 | -1 | 0.05285551274816195 | 0.09421329448620479 | OK |
| a3_aug_rot | 1 | none | 25 | SQUARE | 1024 | finetuned | random_rotation | chamfer | 0.001 | 8 | -1 | 0.05411182799273067 | 0.08615856245160103 | OK |
| a3_aug_flip | 1 | none | 25 | SQUARE | 1024 | finetuned | data_augmentation_random_flips | chamfer | 0.001 | 8 | -1 | 0.07557891764574581 | 0.08232646435499191 | OK |
| a3_aug_aniso | 1 | none | 25 | SQUARE | 1024 | finetuned | anisotropic_scaling | chamfer | 0.001 | 8 | -1 | 0.0444109116991361 | 0.09440010123782688 | OK |
| a3_frozen | 1 | none | 25 | SQUARE | 1024 | frozen | none | chamfer | 0.001 | 8 | -1 | 0.04212144927846061 | 0.1064375498228603 | OK |
| a2_p25_sphere_s7 | 1 | none | 25 | SPHERE | 1024 | finetuned | none | chamfer | 0.001 | 8 | 7 | 0.03476879600849417 | 0.14631705234448114 | OK |
| a2_p25_sphere_s123 | 1 | none | 25 | SPHERE | 1024 | finetuned | none | chamfer | 0.001 | 8 | 123 | 0.035711860905090966 | 0.14896801031298107 | OK |
| a4_edge1e3 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer+edge | 0.001 | 8 | -1 | 0.0449589958621396 | 0.0915172580215666 | OK |
| a4_edge1e2 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer+edge | 0.001 | 8 | -1 | 0.06704063506589995 | 0.09287349300252067 | OK |

## Point-E sweeps

| tag | views | fusion | preprocessing | guidance | points | seeds | denoise | n_objects | micro_chamfer | micro_fscore | macro_chamfer | macro_fscore | noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | [0] | first | masked | 3.0 | 4096 | [0] | False | 5 | 0.1439457853191123 | 0.0744628 | 0.1439457853191123 | 0.0744628 | cup,hammer,medicine_bottle,shampoo,teapot |
| p1_best | [0, 8, 16] | best | masked | 3.0 | 4096 | [0] | False | 5 | 0.13173838414722905 | 0.0695824 | 0.13173838414722905 | 0.0695824 | cup,hammer,medicine_bottle,shampoo,teapot |
| p1_median | [0, 8, 16] | median | masked | 3.0 | 4096 | [0] | False | 5 | 0.13527812532689026 | 0.0670352 | 0.13527812532689026 | 0.0670352 | cup,hammer,medicine_bottle,shampoo,teapot |
| p1_v12 | [12] | first | masked | 3.0 | 4096 | [0] | False | 5 | 0.1416561929122874 | 0.083816 | 0.1416561929122874 | 0.083816 | cup,hammer,medicine_bottle,shampoo,teapot |
| p2_1k | [0] | first | masked | 3.0 | 1024 | [0] | False | 5 | 0.1440087181346126 | 0.0510518 | 0.14400871813461258 | 0.0510518 | cup,hammer,medicine_bottle,shampoo,teapot |
| p2_g1 | [0] | first | masked | 1.0 | 4096 | [0] | False | 5 | 0.14752660268463258 | 0.102074 | 0.14752660268463258 | 0.102074 | cup,hammer,medicine_bottle,shampoo,teapot |
| p2_g3.0_s012_1k | [0] | first | masked | 3.0 | 1024 | [0, 1, 2] | False | 5 | 0.144798864919689 | 0.05433186666666667 | 0.14479886491968902 | 0.05433186666666667 | cup,hammer,medicine_bottle,shampoo,teapot |
| p2_g3.0_s012_4k | [0] | first | masked | 3.0 | 4096 | [0, 1, 2] | False | 5 | 0.14373993246735786 | 0.08383533333333335 | 0.14373993246735783 | 0.08383533333333335 | cup,hammer,medicine_bottle,shampoo,teapot |
| p2_g5 | [0] | first | masked | 5.0 | 4096 | [0] | False | 5 | 0.14296729142857303 | 0.0769158 | 0.14296729142857303 | 0.0769158 | cup,hammer,medicine_bottle,shampoo,teapot |
| p3_crop | [0] | first | crop | 3.0 | 4096 | [0] | False | 5 | 0.14264470224822512 | 0.0715946 | 0.14264470224822512 | 0.07159460000000001 | cup,hammer,medicine_bottle,shampoo,teapot |
| p3_raw | [0] | first | raw | 3.0 | 4096 | [0] | False | 5 | 0.14035585705807435 | 0.07553299999999999 | 0.14035585705807435 | 0.07553299999999999 | cup,hammer,medicine_bottle,shampoo,teapot |
| p4_den | [0] | first | masked | 3.0 | 4096 | [0] | True | 5 | 0.14413339946576942 | 0.07436680000000001 | 0.14413339946576947 | 0.0743668 | cup,hammer,medicine_bottle,shampoo,teapot |

## TripoSR sweeps

| tag | view | preprocessing | points | mc_resolution | n_objects | micro_chamfer | micro_fscore | macro_chamfer | macro_fscore | noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | 0 | masked | 4096 | 128 | 5 | 0.09820158149025601 | 0.08756359999999999 | 0.09820158149025604 | 0.0875636 | cup,hammer,medicine_bottle,shampoo,teapot |
| crop | 0 | crop | 4096 | 128 | 5 | 0.1097147839204073 | 0.0779122 | 0.1097147839204073 | 0.0779122 | cup,hammer,medicine_bottle,shampoo,teapot |
| raw | 0 | raw | 4096 | 128 | 5 | 0.10314921769205462 | 0.06974440000000001 | 0.10314921769205461 | 0.0697444 | cup,hammer,medicine_bottle,shampoo,teapot |

## Cross-model (shared objects)

| object | atlas_run | atlas_ch | atlas_f | pointe_ch | pointe_f | triposr_ch | triposr_f |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cup_cup_003 | a2_p25_sphere_s7 | 0.01271540539993973 | 0.30408388184934637 | 0.12231496036544376 | 0.058212 | 0.0889766796806367 | 0.078613 |
| hammer_hammer_011 | a2_p25_sphere_s7 | 0.04039657617589024 | 0.11378251884741587 | 0.19395787852503654 | 0.072671 | 0.158284738432672 | 0.106896 |
| medicine_bottle_medicine_bottle_068 | a2_p25_sphere_s7 | 0.07484982794340346 | 0.04558537024685491 | 0.16207312387126338 | 0.05494 | 0.08331780645008555 | 0.071276 |
| shampoo_shampoo_003 | a2_p25_sphere_s7 | 0.011733729480746155 | 0.43445457075300914 | 0.18389601957023 | 0.112409 | 0.11138122294242224 | 0.078353 |
| teapot_teapot_003 | a2_p25_sphere_s7 | 0.012690148160550637 | 0.3760394522053049 | 0.05748694426358795 | 0.074082 | 0.04904745994546364 | 0.10268 |

## Interpretation

(Completed in the final handoff message after all phases land. Headline questions it must answer: what moved Chamfer-down/F-up vs the A0 baseline; weight provenance per winner; overfitting behavior (best vs final); noise-flagged categories excluded from claims.)

