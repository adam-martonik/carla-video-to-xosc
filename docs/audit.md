# Fáza 1 – Audit existujúceho nástroja `video_to_xosc.py`

Dátum auditu: 2026-11-19  
Vetva: `feat/realistic-xosc-pipeline` (baseline commit `e6dbea4`)

---

## 1. Architektúra pôvodného nástroja

Pôvodný nástroj pozostáva z monolitického skriptu [`video_to_xosc.py`](../video_to_xosc.py) (1 055 riadkov) a sprievodných skriptov (`run_replay.py`, `record_carla_replay.py`, `drive_hero_headless.py`). Pipeline má päť sekvenčných fáz:

1. **Stage A – Sťahovanie videa (`download_youtube_clip`):** obal nad `yt-dlp` pre MP4 do 720p.
2. **Stage B/C – Detekcia, tracking a odhad ego-motion (`estimate_ego_motion_and_tracks`):**
   - Detektor: Ultralytics YOLO11n (`yolo11n.pt`, fallback z `yolo11m.pt`) s `conf=0.15` a triedami vozidiel (`{2,3,5,7}`) a VRU/zvierat (`{0,1,15..23}`).
   - Tracking: `bytetrack.yaml` (ByteTrack bez kompenzácie pohybu kamery).
   - Odhad 3D polohy: fúzia šírkového pinhole odhadu $z_w = f_x W_{\text{spec}} \cdot \text{angle\_factor} / b_w$ (váha 0.55) a roviny vozovky $z_{\text{gp}} = f_y h_{\text{cam}} / (v_b - c_y)$ (váha 0.45), s pevnými parametrami $\text{HFOV}=70^\circ$, $c_y = 0.52 H$, $h_{\text{cam}} = 1.35\text{ m}$.
   - Odhad ego-motion: riedky optický tok `cv2.calcOpticalFlowPyrLK` na statickej maske, z mediánu posunu sa priamo počíta `d_yaw = clip(-du/fx, -0.03, 0.03)` a `ego_v = 18.0 * clip(flow_mag / 4.2, 0.0, 1.35)`.
3. **Stage C2 – Vyhladenie a prevzorkovanie (`smooth_and_resample_trajectories`):**
   - Tri vetvy: (i) detekovaná kolízia s VRU/zverou → **úplne syntetická šablóna**, (ii) detekovaný cut-in → **úplne syntetická šablóna**, (iii) štandardná premávka → Savitzky-Golay filter (okno 15–21) + lineárna interpolácia na mriežku `sample_dt = 1.0 s`.
4. **Stage D – Projekcia na CARLA mapu (`project_frenet_to_carla_map`):**
   - Vyžaduje bežiaci CARLA 1.0 server na porte 2000; nájde najdlhší koridor na `Town04_Opt` s jazdnými pruhmi vľavo aj vpravo a mapuje $(s, d) \to (x, y, z, h)$.
5. **Stage E – Generovanie OpenSCENARIO 1.0 (`write_openscenario_xosc`):**
   - Generuje XML cez `xml.etree.ElementTree`, validuje cez `xmlschema` oproti `scenario_runner/srunner/openscenario/OpenSCENARIO.xsd`.
   - Používa `<TeleportAction>`, `<AssignControllerAction>` (`simple_vehicle_control`), `<FollowTrajectoryAction>` (`Polyline`) a časovo spúšťané `<SpeedAction>` (`dynamicsShape="step"`).

---

## 2. Kvantitatívne výsledky auditu na dostupných dátach

Audit bol vykonaný skriptom `reports/evaluation/audit_baseline.py` (úplný výstup v `reports/evaluation/baseline_audit.json`) na všetkých 4 existujúcich `.xosc` súboroch v `demo/` a na 4 reálnych videách (`dashcam_input.mp4`, `brake.mp4`, `jelen.mp4`, `stiahnute_video.mp4`).

### 2.1 Analýza existujúcich `.xosc` výstupov

| Súbor | XSD valid | Parametre | Entity | Max $\|a\|$ (m/s²) | Max $\|j\|$ (m/s³) | Max $\|\dot{\psi}\|$ (rad/s) | Min. vzdialenosť stredov (m) | Prekrytia (< 3.5 m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dashcam_reconstructed.xosc` | Áno | 0 | 4 | **13.79** (`npc_track_155`) | **21.15** | 0.430 | **2.86** | **2** |
| `moj_scenar.xosc` | Áno | 0 | 3 | **15.41** (`npc_track_26`) | **24.01** | 0.084 | **3.20** | **1** |
| `scenar_brake.xosc` | Áno | 0 | 4 | 3.03 | 1.52 | 0.122 | 5.88 | 0 |
| `scenar_jelen.xosc` | Áno | 0 | 3 | **13.03** (`oncoming`) | **13.03** | **3.134** | 15.82 | 0 |

### 2.2 Analýza behu pôvodnej pipeline na 4 demo videách

| Video | Rozlíšenie / FPS / Dĺžka | Surové ego $\max\|a\|$ (m/s²) | Ego $\max\|a\|$ po vyhladení | Najhorší aktér $\max\|a\|$ (m/s²) | Najhorší aktér $\max\|j\|$ (m/s³) | Min. Frenet vzdialenosť (m) | Poznámka |
|---|---|---:|---:|---:|---:|---:|---|
| `dashcam_input.mp4` | 1280×720 / 29.97 / 21.76 s | **728.27** | 6.94 | **9.01** | **16.59** | **2.37** (2 kroky < 3.5 m) | Štandardná premávka, skoky v rýchlosti a prekrytie |
| `brake.mp4` | 1280×720 / 29.97 / 18.39 s | **603.56** | 3.03 | 0.06 | 0.12 | 5.61 | Prepísané syntetickou šablónou; `npc_track_319` je **truck** (`class_id=7`), ale dostal blueprint `vehicle.dodge.charger` |
| `jelen.mp4` | 960×720 / 30.00 / 20.20 s | **729.00** | 5.11 | **12.43** (`oncoming`) | **12.43** | 6.10 | Prepísané šablónou; `oncoming_vehicle` prudko zastane a otočí heading o $\pi$ rad |
| `stiahnute_video.mp4` | 360×640 / 30.00 / 15.87 s | **460.29** | 3.37 | 4.57 | 9.14 | 9.96 | Vertikálne video (9:16), nesprávny `fov_deg=70` a pevný horizont; `npc_track_98` má `max_yaw_rate=0.895 rad/s` a $d=-8.11\text{ m}$ |

---

## 3. Zoznam konkrétnych zistených nedostatkov

### A. Fyzikálna vierohodnosť a kinematika (Kritické)
1. **Nefyzikálne zrýchlenia a jerk v trajektóriách:** Aktéri dosahujú $12.9\text{–}15.4\text{ m/s}^2$ a jerk $16.6\text{–}24.0\text{ m/s}^3$ (`dashcam_reconstructed.xosc`, `moj_scenar.xosc`, `dashcam_input.mp4`). Chýba stavový odhadca (Kalman/RTS smoother) a vynútenie kinematických obmedzení (bicycle model, limity $a_{\min}, a_{\max}, j_{\max}, \dot{\psi}_{\max}$).
2. **Nefyzikálne limity `<Performance>` v XML:** Všetky vozidlá majú `maxAcceleration="200"` ($> 20\text{ G}$!) a `maxDeceleration="15.0"` (riadok 810).
3. **Skokové `SpeedAction` a hrubé 1 Hz vzorkovanie:** `sample_dt = 1.0 s` vytvára hrubé lomené čiary; `SpeedActionDynamics` používa `dynamicsShape="step"`, čo spôsobuje trhanú zmenu rýchlosti. Polyčiara navyše vynecháva vrchol v `t = 0.0 s`.
4. **Anomália protismerného vozidla v `jelen.mp4`:** `max(5.0, s_onc_start - v_onc * t)` spôsobí okamžité zastavenie z 15 m/s na 0 m/s (13 m/s²), pričom následný výpočet `atan2` otočí heading o $3.134\text{ rad/s}$ ($\approx 180^\circ$).

### B. Kolízie, prekrývanie a konzistencia objektov (Kritické)
5. **Prechádzanie vozidiel cez seba:** V `dashcam_reconstructed.xosc` (2.86 m), `moj_scenar.xosc` (3.20 m) a `dashcam_input.mp4` (2.37 m) sa vozidlá dĺžky 4.5 m fyzicky prekrývajú.
6. **Chybný spawn posun v Stage D:** Riadky 740–748 posunú pri kolízii pri `t=0` iba `world_poses[0]` o `shift_s += 12.0`, ale zvyšok trajektórie `world_poses[1:]` nechajú na pôvodnom mieste → okamžitý 12 m skok medzi `t=0` a `t=1`!
7. **Falošná prítomnosť objektov pred/po objavení (extrapolačné „prilepenie“):** `np.interp(sample_times, t_arr, z_cam_arr)` mimo rozsahu pozorovania drží konštantnú okrajovú hodnotu. Vozidlo videné len v intervale $t \in [10, 15]\text{ s}$ sa tak umelo vyskytuje už od $t = 0\text{ s}$ v konštantnej vzdialenosti pred ego vozidlom.
8. **Chýbajúce spájanie fragmentovaných trackov (tracklet stitching) a filtrácia odľahlých detekcií:** Pri krátkej oklúzii alebo ID switchi vzniknú 2 oddelené tracky (alebo sa track zahodí kvôli `min_track_frames=35`). Náhla zmena šírky bboxu pri čiastočnej oklúzii spôsobí okamžitý skok hĺbky o 10–30 m.

### C. Rozmery, typy objektov a mierka (Vysoká závažnosť)
9. **Ignorovanie detegovanej triedy v `.xosc`:** `COCO_SPECS` definuje rozmery pre `car`, `truck`, `bus`, `motorcycle`, `cyclist`, `pedestrian`, ale `write_openscenario_xosc` (riadky 801–825) zapisuje **rovnaký pevný bounding box `4.5 × 2.1 × 1.8 m` a `vehicleCategory="car"`** pre všetky vozidlá, a `smooth_and_resample_trajectories` priraďuje všetkým vozidlám náhodné osobné autá z `CARLA_CAR_BLUEPRINTS` (napr. kamión `class_id=7` v `brake.mp4` sa zmenil na `vehicle.dodge.charger`).
10. **Pevná kalibrácia kamery a nesprávny odhad ego-motion:** Pevný horizont `cy = 0.52 * height` a `fov_deg = 70` bez rozlíšenia pomeru strán (zlyháva pri vertikálnom 9:16 videu `stiahnute_video.mp4`). Surové ego VO bez odfiltrovania rotačného toku a bez Kalman filtra vykazuje špičky `460–729 m/s²`.
11. **Prepísanie reálnych dát pevnými šablónami:** V prípade `brake.mp4` (cut-in) a `jelen.mp4` (kolízia) sa extrahované dáta zahodia a nahradia pevnými konštantami (`v_fast=22`, `v_cutin=11.5`, `v_other=10.0`), čím sa stráca vernosť voči reálnemu záznamu.

### D. Cestná sieť, OpenDRIVE, parametrizácia a architektúra (Vysoká závažnosť)
12. **Nulová parametrizácia scenára:** `<ParameterDeclarations/>` je prázdny; nejde o logický scenár (nemožno meniť napr. `EgoInitialSpeed`, `TimeScale`, `LateralOffsetScale`, `RoadFriction`, `SunElevation`).
13. **Pevná závislosť na bežiacom CARLA serveri a absencia OpenDRIVE (`.xodr`):** Nástroj sa nedá spustiť offline bez UE5 servera a nevie vygenerovať vlastnú geometriu cesty `.xodr` (počet pruhov, zakrivenie, šírka pruhu).
14. **Chybný `<Storyboard><StopTrigger>`:** Obsahuje prázdny `ParameterCondition(parameterRef="", value="")` (riadok 1017).
15. **Monolitická štruktúra bez konfigurácie a testov:** Všetky magické konštanty sú priamo v kóde, chýba konfiguračný súbor (YAML/JSON) a neexistuje žiadna testovacia sada (`tests/`).
