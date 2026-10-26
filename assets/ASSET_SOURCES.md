# Frontend asset sources

The canonical frontend uses three visibly separated asset classes.

| Class | Asset | Source | License / status | Use |
|---|---|---|---|---|
| Generated atmosphere | `generated/warm-conservation-desk-v1.png` | OpenAI ImageGen, generated 2026-07-29 | Project-created; not evidence | Empty conservation-desk hero underlay |
| Generated instrument | `generated/cave-library-conservation-desk.png` | OpenAI ImageGen, generated 2026-07-29 after conservation-workstation research | Project-created; not evidence | Selected A-direction transmitted-light desk; contains no manuscript evidence |
| Generated atmosphere set | `generated/v3/*.png` | OpenAI ImageGen, generated 2026-07-29; prompts recorded in `docs/frontend_redesign_v3/IMAGE_PROMPTS.md` | Project-created; not evidence | Empty archive, review, result, provenance, mobile, and blank-slip environments |
| Authored hero motion | `video/hero.webm`, `video/hero.mp4`, `video/hero_poster.webp` | Project motion prototype `docs/frontend_redesign_v2/motion_prototype/` | Project-created interface motion; not evidence | Optional opening sequence with WebM/MP4 fallback and poster |
| Open research manuscript | `manuscripts/science-db-s10480.jpg` | ScienceDB dataset, DOI `10.57760/sciencedb.j00001.01083`, item S.10480 | `CC BY-NC-SA 4.0` | Real manuscript specimen on the hero desk; attribution retained |
| Open research manuscript | `manuscripts/science-db-s1931.jpg` | ScienceDB dataset, DOI `10.57760/sciencedb.j00001.01083`, item S.1931 | `CC BY-NC-SA 4.0` | Real manuscript specimen on the hero desk; attribution retained |
| Verified replay input | `replay/input.png` | `outputs/final_integration/e2e_input_EDH-DX01962-02052.png` | Existing project test input; provenance retained in repository reports | Clearly labelled saved formal replay |
| Verified replay output | `replay/result.png` | `outputs/final_integration/8f45998b-6219-443e-a09d-af06243feecb/constrained_restored.png` | Existing Official AutoHDR result | Clearly labelled saved formal replay |
| Verified replay audit | `replay/approved_mask.png`, `comparison.png`, `outside_mask_audit.json` | Same final-integration run | Existing audited run | Review-only provenance and comparison |
| Verified local Rejoin input | `rejoin/fragment_a.png`, `rejoin/fragment_b.png` | `outputs/local_entry/browser_fixture/` | Existing project fixture derived from the real EDH-DX01962-02052 case | Local interaction replay |
| Rejoin complete-image reference | `rejoin/panorama_reference.png` | `outputs/final_integration/e2e_input_EDH-DX01962-02052_rerun_20260726.png` | Existing real project input | Clearly labelled reference, not a newly generated panorama |
| Real case montage | `cases/sciencedb_real_rejoin_group.png` | `outputs/plots/rejoin_real_homologous.png` | Existing real ScienceDB evaluation montage | Case archive display |

Generated atmosphere is never presented as a manuscript, evidence, model input, or restoration result. Replay assets are never presented as a new run.

The hero desk explicitly distinguishes its generated instrument environment from the two real, licensed manuscript specimens. The specimens illustrate the interaction language only and are not presented as outputs from the current model or as a matched pair.
