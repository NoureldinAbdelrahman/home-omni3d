# nesegemaa track — final report (auto-generated tables + fixed protocol notes)

Scope: AtlasNet + Point-E (+ TripoSR track), branch `nesegemaa`.
Protocol (all numbers): splits.json test lists, view `00.png` unless noted, UnitBall (subtract mean, divide by max radius), squared-distance units, F threshold 0.001 (≈0.0316 Euclidean). `<3`-object categories are noise-flagged.

## AtlasNet ablation

| run | K_views | pool | patches | template | bottleneck | encoder_train | aug | loss | lr | batch | seed | best_chamfer | best_fscore | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a1_k3_max | 3 | max | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.0521612883441978 | 0.08339518639776441 | OK |
| a1_k3_attn | 3 | attn | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.0433131969637341 | 0.08691756427288055 | OK |
| a1_k5_max | 5 | max | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.059123086432615914 | 0.07589853803316753 | OK |
| a1_k5_attn | 5 | attn | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.04124867543578148 | 0.09545483522944981 | OK |
| a1_k8_max | 8 | max | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.052312228828668594 | 0.08982537521256341 | OK |
| a1_k8_attn | 8 | attn | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.06762129523687893 | 0.07019212635027038 | OK |
| a2_p1_sq | 1 | none | 1 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.04238425816098849 | 0.08601735863420698 | OK |
| a2_p10_sq | 1 | none | 10 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.06247394159436226 | 0.07881299406290054 | OK |
| a2_p25_sphere | 1 | none | 25 | SPHERE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.03473327370981375 | 0.12748425867822435 | OK |
| a2_p25_4096 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.03996867396765285 | 0.10749444282717174 | OK |
| a2_p25_bn512 | 1 | none | 25 | SQUARE | 512 | finetuned | none | chamfer | 0.001 | 8 | -1 | 0.04793103080656794 | 0.07829489145014021 | OK |
| a3_lr3e4 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.0003 | 8 | -1 | 0.06389947856465976 | 0.0650663288931052 | OK |
| a3_bs16 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 16 | -1 | 0.05348332391844855 | 0.10040643976794349 | OK |
| a3_bs32 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer | 0.001 | 32 | -1 | 0.049700293276045054 | 0.08975647224320306 | OK |
| a3_aug_rot | 1 | none | 25 | SQUARE | 1024 | finetuned | random_rotation | chamfer | 0.001 | 8 | -1 | 0.04201140130559603 | 0.09845845152934392 | OK |
| a3_aug_flip | 1 | none | 25 | SQUARE | 1024 | finetuned | data_augmentation_random_flips | chamfer | 0.001 | 8 | -1 | 0.08195297999514474 | 0.09162643220689562 | OK |
| a3_aug_aniso | 1 | none | 25 | SQUARE | 1024 | finetuned | anisotropic_scaling | chamfer | 0.001 | 8 | -1 | 0.04606790840625763 | 0.08075466089778477 | OK |
| a3_frozen | 1 | none | 25 | SQUARE | 1024 | frozen | none | chamfer | 0.001 | 8 | -1 | 0.0394778230951892 | 0.11765896032253902 | OK |
| a4_edge1e3 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer+edge | 0.001 | 8 | -1 | 0.06609317784508069 | 0.0860129478904936 | OK |
| a4_edge1e2 | 1 | none | 25 | SQUARE | 1024 | finetuned | none | chamfer+edge | 0.001 | 8 | -1 | 0.05640431410736508 | 0.08338744772805108 | OK |
| a2_p25_sphere_s7 | 1 | none | 25 | SPHERE | 1024 | finetuned | none | chamfer | 0.001 | 8 | 7 | 0.0336735470013486 | 0.14220897439453337 | OK |
| a2_p25_sphere_s123 | 1 | none | 25 | SPHERE | 1024 | finetuned | none | chamfer | 0.001 | 8 | 123 | 0.03260621904499001 | 0.1346580046746466 | OK |

## Point-E sweeps

| tag | views | fusion | preprocessing | guidance | points | seeds | denoise | n_objects | micro_chamfer | micro_fscore | macro_chamfer | macro_fscore | noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | [0] | first | masked | 3.0 | 4096 | [0] | False | 5 | 0.20332672438583033 | 0.07868939999999999 | 0.20332672438583033 | 0.0786894 | cup,hammer,medicine_bottle,shampoo,teapot |
| p1_best | [0, 8, 16] | best | masked | 3.0 | 4096 | [0] | False | 5 | 0.19411219153044254 | 0.07886499999999999 | 0.19411219153044257 | 0.07886499999999999 | cup,hammer,medicine_bottle,shampoo,teapot |
| p1_median | [0, 8, 16] | median | masked | 3.0 | 4096 | [0] | False | 5 | 0.19835343525010168 | 0.0787018 | 0.19835343525010168 | 0.0787018 | cup,hammer,medicine_bottle,shampoo,teapot |
| p1_v12 | [12] | first | masked | 3.0 | 4096 | [0] | False | 5 | 0.20713764280562788 | 0.08070639999999998 | 0.20713764280562788 | 0.0807064 | cup,hammer,medicine_bottle,shampoo,teapot |
| p2_1k | [0] | first | masked | 3.0 | 1024 | [0] | False | 5 | 0.20025228746243978 | 0.0492142 | 0.20025228746243978 | 0.0492142 | cup,hammer,medicine_bottle,shampoo,teapot |
| p2_g1 | [0] | first | masked | 1.0 | 4096 | [0] | False | 5 | 0.20254151167072293 | 0.0760358 | 0.20254151167072293 | 0.07603580000000001 | cup,hammer,medicine_bottle,shampoo,teapot |
| p2_g3.0_s012_1k | [0] | first | masked | 3.0 | 1024 | [0, 1, 2] | False | 5 | 0.20789092431558384 | 0.0487512 | 0.20789092431558384 | 0.0487512 | cup,hammer,medicine_bottle,shampoo,teapot |
| p2_g3.0_s012_4k | [0] | first | masked | 3.0 | 4096 | [0, 1, 2] | False | 5 | 0.20826756265235993 | 0.0761628 | 0.20826756265235993 | 0.0761628 | cup,hammer,medicine_bottle,shampoo,teapot |
| p2_g5 | [0] | first | masked | 5.0 | 4096 | [0] | False | 5 | 0.2080395030688515 | 0.068112 | 0.2080395030688515 | 0.068112 | cup,hammer,medicine_bottle,shampoo,teapot |
| p3_crop | [0] | first | crop | 3.0 | 4096 | [0] | False | 5 | 0.20065024882644417 | 0.0827204 | 0.20065024882644417 | 0.0827204 | cup,hammer,medicine_bottle,shampoo,teapot |
| p3_raw | [0] | first | raw | 3.0 | 4096 | [0] | False | 5 | 0.19820106142768 | 0.0860236 | 0.19820106142768 | 0.0860236 | cup,hammer,medicine_bottle,shampoo,teapot |
| p4_den | [0] | first | masked | 3.0 | 4096 | [0] | True | 5 | 0.20369728598431927 | 0.07805779999999998 | 0.20369728598431927 | 0.0780578 | cup,hammer,medicine_bottle,shampoo,teapot |

## TripoSR sweeps

| tag | view | preprocessing | points | mc_resolution | n_objects | micro_chamfer | micro_fscore | macro_chamfer | macro_fscore | noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | 0 | masked | 4096 | 128 | 5 | 0.09691283734649064 | 0.1015774 | 0.09691283734649063 | 0.1015774 | cup,hammer,medicine_bottle,shampoo,teapot |
| crop | 0 | crop | 4096 | 128 | 5 | 0.12632828721896522 | 0.07211700000000001 | 0.12632828721896522 | 0.072117 | cup,hammer,medicine_bottle,shampoo,teapot |
| raw | 0 | raw | 4096 | 128 | 5 | 0.1257322609877544 | 0.0697706 | 0.12573226098775442 | 0.0697706 | cup,hammer,medicine_bottle,shampoo,teapot |

## Cross-model (shared objects)

| object | atlas_run | atlas_ch | atlas_f | pointe_ch | pointe_f | triposr_ch | triposr_f |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cup_cup_003 | a2_p25_sphere_s123 | 0.0193967250016871 | 0.20162363971295663 | 0.10699594428385201 | 0.09461 | 0.09440701112212085 | 0.072458 |
| hammer_hammer_011 | a2_p25_sphere_s123 | 0.045494486665871414 | 0.09239245612961042 | 0.43845002656749804 | 0.05443 | 0.16023575031733592 | 0.133347 |
| medicine_bottle_medicine_bottle_068 | a2_p25_sphere_s123 | 0.024229899533997602 | 0.1114832435297787 | 0.1502299554952804 | 0.061902 | 0.07953003197105919 | 0.095453 |
| shampoo_shampoo_003 | a2_p25_sphere_s123 | 0.02294334608266027 | 0.26235754722895344 | 0.24924490879474037 | 0.083803 | 0.11538578981468056 | 0.077896 |
| teapot_teapot_003 | a2_p25_sphere_s123 | 0.02586141212538566 | 0.1470207202611249 | 0.0717127867877809 | 0.098702 | 0.035005603507256663 | 0.128733 |

## Interpretation

(Completed in the final handoff message after all phases land. Headline questions it must answer: what moved Chamfer-down/F-up vs the A0 baseline; weight provenance per winner; overfitting behavior (best vs final); noise-flagged categories excluded from claims.)

