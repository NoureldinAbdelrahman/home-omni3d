# Dataset notes: OmniObject3D subset (24 views + 4096-point clouds)

Generated 2026-11-09. Source: OpenDataLab `omniobject3d/OmniObject3D-New`, only the two modalities we use: `blender_renders_24_views` (24 PNGs + `transforms.json` per object) and `point_clouds/ply_files/4096_ply` (named per-object clouds).

## Completeness

| | have | in source | complete |
|---|---:|---:|---:|
| categories | 215 | 216 | 99.54% |
| objects (renders + cloud) | 5,893 | 5,911 | **99.70%** |
| images | 141,432 | 141,864 | 99.70% |
| point clouds | 5,893 | 5,911 | 99.70% |

Every object that is present has **all 24 views, a valid `transforms.json` and a point cloud**, and `renders/` and `point_clouds/` match 1:1. Missing: **18 of 5,911 objects (0.30%)**.

## What is missing (and why)

| object / category | objects | % of all objects | % of its category | reason |
|---|---:|---:|---:|---|
| `package` (whole category, never downloaded) | 16 | 0.271% | 100% | OpenDataLab firewall blocks the download (HTTP 405). Clouds quarantined. |
| `laptop_002` | 1 | 0.017% | 25.0% | source archive has only 9/24 views and no `transforms.json` (upstream defect); quarantined with its cloud |
| `anise_011` | 1 | 0.017% | 3.4% | source `.ply` archive has no cloud for it (upstream defect); quarantined |

Total missing: 18 objects = 0.30% of the source; 1 of 216 categories is absent (0.46%) and 2 more are one object short.

Quarantined files are kept (not deleted) in `_quarantine/` on the machine that built this dataset; they are **not** part of the shared copy.

## Layout

```
renders/<category>/<category>_NNN/{000..023}.png + transforms.json
point_clouds/<category>/<category>_NNN.npy      # float32, shape (4096, 3), name = object ID
```

Size: renders 112.7 GiB, point clouds 277 MiB.

## Notes for users

* **Pairing is correct.** Clouds come from the *named* per-object PLY archive, and for **all 215 categories (5,893 clouds)** each cloud was hash-verified against that archive (`fix_point_cloud_names.py --dry-run`: 0 renames needed, 0 skipped; `check_dataset.py` repeats this check). The old HDF5-based downloads had shuffled names - do not mix them with this dataset.
* The 76 original categories are the same ones the models were trained on; the other 139 add food, toys, etc. Pick categories explicitly (`manifest.json` lists per-category counts).
* Object counts per category vary a lot (2 to 156); see `manifest.json`.
* Verify a copy with `python check_dataset.py --dataset-dir <path>` (reads `manifest.json` for the documented exclusions).

## Per-category counts

| category | objects | source | |
|---|---:|---:|---|
| anise | 28 | 29 | short by 1 |
| antique | 10 | 10 |  |
| apple | 62 | 62 |  |
| asparagus | 14 | 14 |  |
| backpack | 22 | 22 |  |
| ball | 28 | 28 |  |
| bamboo_shoots | 37 | 37 |  |
| banana | 36 | 36 |  |
| battery | 3 | 3 |  |
| beauty_blender | 4 | 4 |  |
| bed | 2 | 2 |  |
| belt | 27 | 27 |  |
| biscuit | 118 | 118 |  |
| book | 24 | 24 |  |
| bottle | 26 | 26 |  |
| bowl | 12 | 12 |  |
| box | 134 | 134 |  |
| boxed_beverage | 10 | 10 |  |
| bread | 156 | 156 |  |
| broad_bean | 18 | 18 |  |
| broccoli | 29 | 29 |  |
| broccolini | 10 | 10 |  |
| brush | 28 | 28 |  |
| brussels_sprout | 17 | 17 |  |
| bucket_noodle | 29 | 29 |  |
| bumbag | 9 | 9 |  |
| bun | 37 | 37 |  |
| burrito | 17 | 17 |  |
| cabinet | 7 | 7 |  |
| cake | 23 | 23 |  |
| calculator | 21 | 21 |  |
| candle | 9 | 9 |  |
| candy | 49 | 49 |  |
| canned_beverage | 2 | 2 |  |
| carrot | 37 | 37 |  |
| cauliflower | 3 | 3 |  |
| chair | 29 | 29 |  |
| cheese | 41 | 41 |  |
| cherry | 13 | 13 |  |
| chess | 21 | 21 |  |
| chicken_leg | 19 | 19 |  |
| chili | 33 | 33 |  |
| china | 10 | 10 |  |
| chinese_chess | 13 | 13 |  |
| chocolate | 30 | 30 |  |
| clock | 59 | 59 |  |
| coconut | 30 | 30 |  |
| conch | 29 | 29 |  |
| corn | 32 | 32 |  |
| cucumber | 31 | 31 |  |
| cup | 39 | 39 |  |
| dice | 16 | 16 |  |
| dinosaur | 33 | 33 |  |
| dish | 18 | 18 |  |
| doll | 84 | 84 |  |
| donut | 33 | 33 |  |
| drawing | 28 | 28 |  |
| drum | 2 | 2 |  |
| dumbbell | 18 | 18 |  |
| dumpling | 22 | 22 |  |
| durian | 30 | 30 |  |
| dustbin | 20 | 20 |  |
| earplug | 7 | 7 |  |
| egg | 47 | 47 |  |
| egg_tart | 29 | 29 |  |
| eraser | 32 | 32 |  |
| facial_cream | 39 | 39 |  |
| fan | 16 | 16 |  |
| fig | 44 | 44 |  |
| fire_extinguisher | 19 | 19 |  |
| flash_light | 7 | 7 |  |
| flower_pot | 22 | 22 |  |
| flute | 2 | 2 |  |
| fork | 3 | 3 |  |
| frisbee | 7 | 7 |  |
| garage_kit | 81 | 81 |  |
| garlic | 29 | 29 |  |
| ginger | 36 | 36 |  |
| glasses | 5 | 5 |  |
| glasses_case | 37 | 37 |  |
| gloves | 17 | 17 |  |
| green_bean_cake | 16 | 16 |  |
| guitar | 2 | 2 |  |
| hair_dryer | 39 | 39 |  |
| hairpin | 26 | 26 |  |
| hamburger | 51 | 51 |  |
| hami_melon | 18 | 18 |  |
| hammer | 22 | 22 |  |
| hand_cream | 38 | 38 |  |
| handbag | 48 | 48 |  |
| hat | 30 | 30 |  |
| haw_thorn | 47 | 47 |  |
| hazelnut | 20 | 20 |  |
| helmet | 14 | 14 |  |
| hot_dog | 41 | 41 |  |
| house | 43 | 43 |  |
| insole | 13 | 13 |  |
| kennel | 4 | 4 |  |
| kettle | 12 | 12 |  |
| keyboard | 16 | 16 |  |
| kite | 23 | 23 |  |
| kiwifruit | 2 | 2 |  |
| knife | 29 | 29 |  |
| laptop | 3 | 4 | short by 1 |
| laundry_detergent | 12 | 12 |  |
| lemon | 32 | 32 |  |
| light | 39 | 39 |  |
| lipstick | 26 | 26 |  |
| litchi | 23 | 23 |  |
| longan | 20 | 20 |  |
| loquat | 18 | 18 |  |
| lotus_root | 43 | 43 |  |
| magnet | 4 | 4 |  |
| mango | 56 | 56 |  |
| mangosteen | 30 | 30 |  |
| medicine_bottle | 78 | 78 |  |
| microwaveoven | 8 | 8 |  |
| monitor | 4 | 4 |  |
| mooncake | 20 | 20 |  |
| mouse | 30 | 30 |  |
| mushroom | 67 | 67 |  |
| nipple | 8 | 8 |  |
| onion | 34 | 34 |  |
| orange | 37 | 37 |  |
| ornament | 6 | 6 |  |
| ornaments | 9 | 9 |  |
| oyster | 29 | 29 |  |
| package | 0 | 16 | **not present** |
| pad | 20 | 20 |  |
| pan | 4 | 4 |  |
| pancake | 23 | 23 |  |
| pastry | 25 | 25 |  |
| peach | 37 | 37 |  |
| peanut | 31 | 31 |  |
| pear | 37 | 37 |  |
| pen | 6 | 6 |  |
| picnic_basket | 11 | 11 |  |
| pie | 32 | 32 |  |
| pillow | 38 | 38 |  |
| pineapple | 29 | 29 |  |
| pinecone | 19 | 19 |  |
| pistachio | 22 | 22 |  |
| pitaya | 49 | 49 |  |
| pizza | 31 | 31 |  |
| plant | 14 | 14 |  |
| plug | 2 | 2 |  |
| pomegranate | 51 | 51 |  |
| popcorn | 41 | 41 |  |
| potato | 53 | 53 |  |
| potato_chips | 11 | 11 |  |
| power_strip | 35 | 35 |  |
| projector | 10 | 10 |  |
| puff | 22 | 22 |  |
| pumpkin | 37 | 37 |  |
| razor | 12 | 12 |  |
| red_jujube | 16 | 16 |  |
| red_wine_glass | 21 | 21 |  |
| remote_control | 34 | 34 |  |
| rice_cake | 20 | 20 |  |
| ricecooker | 9 | 9 |  |
| rubik_cube | 32 | 32 |  |
| sandwich | 24 | 24 |  |
| sausage | 57 | 57 |  |
| scissor | 19 | 19 |  |
| shampoo | 38 | 38 |  |
| shaomai | 18 | 18 |  |
| shoe | 38 | 38 |  |
| shrimp | 45 | 45 |  |
| skateboard | 2 | 2 |  |
| soap | 37 | 37 |  |
| sofa | 14 | 14 |  |
| spanner | 6 | 6 |  |
| speaker | 9 | 9 |  |
| squash | 39 | 39 |  |
| starfish | 20 | 20 |  |
| steamed_bun | 35 | 35 |  |
| stool | 11 | 11 |  |
| strawberry | 28 | 28 |  |
| suitcase | 9 | 9 |  |
| sushi | 42 | 42 |  |
| sweet_potato | 34 | 34 |  |
| sword_bean | 2 | 2 |  |
| table | 24 | 24 |  |
| table_tennis_bat | 21 | 21 |  |
| tape_measure | 23 | 23 |  |
| teapot | 38 | 38 |  |
| teddy_bear | 10 | 10 |  |
| thermos | 6 | 6 |  |
| thimble | 22 | 22 |  |
| timer | 22 | 22 |  |
| tissue | 54 | 54 |  |
| tomato | 30 | 30 |  |
| tooth_brush | 27 | 27 |  |
| tooth_paste | 40 | 40 |  |
| toothpick_box | 13 | 13 |  |
| toy_animals | 74 | 74 |  |
| toy_boat | 53 | 53 |  |
| toy_bus | 65 | 65 |  |
| toy_car | 28 | 28 |  |
| toy_motorcycle | 36 | 36 |  |
| toy_plane | 31 | 31 |  |
| toy_plant | 9 | 9 |  |
| toy_train | 55 | 55 |  |
| toy_truck | 44 | 44 |  |
| tvstand | 3 | 3 |  |
| umbrella | 20 | 20 |  |
| vase | 16 | 16 |  |
| waffle | 39 | 39 |  |
| wallet | 47 | 47 |  |
| walnut | 39 | 39 |  |
| watch | 12 | 12 |  |
| water_chestnut | 5 | 5 |  |
| watermelon | 42 | 42 |  |
| whistle | 25 | 25 |  |
| yam | 34 | 34 |  |
| zongzi | 25 | 25 |  |
