# Fáza 3 – Návrh úprav nástroja (`docs/plan.md`)

Na základe auditu (`docs/audit.md`) a rešerše odbornej literatúry (`references.bib`) navrhujeme modulárnu refaktorizáciu a 10 cielených úprav zoradených podľa pomeru **prínos / náročnosť**. Spätná kompatibilita CLI (`video_to_xosc.py --video ... --output-xosc ...`) zostane zachovaná.

---

## 1. Prioritizovaný zoznam úprav (zoradené podľa prínos / náročnosť)

| Priorita | Úprava | Riešený problém z auditu | Odborný zdroj (`references.bib`) | Očakávané zlepšenie | Riziko | Náročnosť |
|---:|---|---|---|---|---|---|
| **1** | **Fyzikálne rozmery, kategórie, nápravy a `<Performance>` podľa triedy detekcie** | #9, #2 (`truck` → `charger`, `maxAcceleration="200"`, jednotný bbox `4.5×2.1×1.8`) | ASAM OpenSCENARIO 1.0; Geiger et al. (2012) `geiger2012kitti`; Caesar et al. (2020) `caesar2020nuscenes` | 100 % správne priradenie tried (`car`, `truck`, `bus`, `motorcycle`, `bicycle`, `pedestrian`), realistické rozmery, hmotnosť a limity zrýchlenia (3.5–9.5 m/s²) | Nízke | Nízka |
| **2** | **Rauch-Tung-Striebel (RTS) Kalman smoother + kinematické obmedzenia (Bicycle model, limity $a$, jerk, $\dot{\psi}$)** | #1, #3, #4 (skoky 15.4 m/s², jerk 24 m/s³, $\dot{\psi}=3.13$ rad/s, surové VO 729 m/s²) | Rauch, Tung & Striebel (1965) `rauch1965rts`; Kong et al. (2015) `kong2015kinematic`; Polack et al. (2017) `polack2017bicycle`; Werling et al. (2010) `werling2010frenet` | Pokles max $\|a\|$ z >13 m/s² na $\le 6\text{–}9$ m/s², pokles max $\|j\|$ z >21 m/s³ na $\le 4\text{–}8$ m/s³, hladký heading bez skokov | Nízke | Stredná |
| **3** | **Riešenie prekrývania objektov (2D OBB / Frenet deconfliction) + oprava spawn posunu a časovania (0.2–0.25 s, `linear` SpeedAction)** | #5, #6, #3, #14 (prekrytia < 3.5 m, 12 m teleport pri `t=0`, `step` SpeedAction, prázdny `StopTrigger`) | Werling et al. (2010) `werling2010frenet`; Montali et al. (2023) `montali2023waymosimagents`; Hayward (1972) `hayward1972ttc` | **0 neúmyselných kolízií/prekrytí** v scenároch bežnej premávky, 4–5× jemnejšie vzorkovanie trajektórií, plynulý profil rýchlosti | Nízke | Stredná |
| **4** | **Konštantná kinematická extrapolácia + Tracklet Stitching & Outlier Rejection (MAD/Huber)** | #7, #8 (prilepenie na hranu času, ID switche, skoky hĺbky pri čiastočnej oklúzii) | Zhang et al. (2022) `zhang2022bytetrack`; Aharon et al. (2022) `aharon2022botsort`; Cao et al. (2023) `cao2023ocsort`; Du et al. (2023) `du2023strongsort` | Zlúčenie rozpadnutých trackov cez krátke oklúzie (< 1.5 s), eliminácia hĺbkových špičiek (> 3 MAD), realistický pohyb objektu mimo okna viditeľnosti | Stredné | Stredná |
| **5** | **BoT-SORT tracking s kompenzáciou pohybu kamery (CMC) + robustné VO (RANSAC homografia/rotácia + median-flow)** | #10, #8 (šum surového VO 460–729 m/s², drift pri otáčaní kamery) | Aharon et al. (2022) `aharon2022botsort`; Campos et al. (2021) `campos2021orbslam3`; Teed et al. (2023) `teed2023dpvo` | Výrazne nižší rozptyl odhadu rýchlosti a yaw-rate ego vozidla; menší počet ID switchov pri pohybe kamery | Stredné | Stredná |
| **6** | **Adaptívna kalibrácia horizontu (vanishing point + FOE) a neistotou vážená fúzia hĺbky (ground-plane + šírka)** | #10 (pevný horizont `0.52*H`, zlyhanie pri 9:16 videu `stiahnute_video.mp4`) | Song & Chandraker (2015) `song2015joint`; Ma et al. (2021) `ma2021monodle`; Yin et al. (2023) `yin2023metric3d` | Správna metrická mierka aj pri inom sklone kamery a pomere strán; stabilnejšia hĺbka pri vzdialených a blízkych objektoch | Stredné | Stredná |
| **7** | **Dátami riadená rekonštrukcia udalostí (cut-in / kolízia) bez zahadzovania meraní** | #11 (tvrdé šablóny prepisujúce reálne trajektórie v `brake.mp4` a `jelen.mp4`) | Bashetty et al. (2020) `bashetty2020deepcrashtest`; Guo et al. (2023) `guo2023scenario_extraction` | Zachovanie skutočného priebehu rýchlosti, času priblíženia a polohy ostatných aktérov z videa | Stredné | Stredná |
| **8** | **Parametrizácia logického scenára (`<ParameterDeclarations>`) v OpenSCENARIO 1.0** | #12 (0 parametrov — len jeden pevný konkrétny scenár) | Menzel et al. (2018) `menzel2018scenarios`; de Gelder & op den Camp (2024) `degelder2024parameterize`; Riedmaier et al. (2020) `riedmaier2020survey` | Generovanie logického scenára s parametrami (`EgoInitialSpeed`, `RoadFriction`, `SunElevation`, `SunIntensity`, `NpcSpeedScale` atď.) plne validného proti XSD | Nízke | Nízka |
| **9** | **Generovanie cestnej siete OpenDRIVE (`.xodr`) + offline mapová projekcia** | #13 (zlyhanie bez CARLA servera, absencia `.xodr` cestnej siete a odhadu pruhov) | Newson & Krumm (2009) `newson2009hmm`; Zheng et al. (2022) `zheng2022clrnet`; Liao et al. (2023) `liao2023maptr`; ASAM OpenDRIVE | Samostatné generovanie validného `.xodr` (viacpruhová cesta so zakrivením z ego yaw-rate a šírkou pruhov 3.5 m) + offline projekcia bez potreby UE5 | Stredné | Vyššia |
| **10** | **Modulárna architektúra (`src/xosc_pipeline/`), YAML konfigurácia (`config/default.yaml`) a testovacia sada (`tests/`)** | #15 (monolit 1055 riadkov, magické konštanty, 0 testov) | Softvérové inžinierstvo / TDD | Čistý, otypovaný, testovateľný kód s >80 % pokrytím kľúčových modulov a jednoduchou konfiguráciou | Nízke | Stredná |

---

## 2. Cieľová modulárna štruktúra projektu

```text
carla-video-to-xosc/
├── config/
│   └── default.yaml                 # Konfigurovateľné parametre kamery, trackingu, kinematiky, XOSC/XODR
├── docs/
│   ├── assumptions.md               # Projektové predpoklady a súradnicové rámce
│   ├── audit.md                     # Fáza 1 – audit pôvodného nástroja
│   └── plan.md                      # Fáza 3 – návrh a prioritizácia úprav
├── src/
│   └── xosc_pipeline/
│       ├── __init__.py
│       ├── config.py                # Pydantic/dataclass konfigurácia + YAML loader
│       ├── calibration.py           # Odhad horizontu (vanishing point / FOE), intrinsics, fúzia hĺbky
│       ├── tracking.py              # BoT-SORT/ByteTrack, RANSAC ego-VO, detekcia rezov/noci, tracklet stitching
│       ├── kinematics.py            # RTS Kalman smoother, outlier rejection, kinematické limity (bicycle model), deconfliction
│       ├── road_network.py          # Odhad pruhov/zakrivenia, generátor OpenDRIVE (.xodr), CARLA + offline projekcia
│       ├── xosc_writer.py           # OpenSCENARIO 1.0 generátor (parametre, triedy/rozmery, plynulé akcie, XSD validácia)
│       └── metrics.py               # Kvantitatívne metriky (zrýchlenie, jerk, yaw-rate, kolízie/OBB, TTC, PET, ADE/FDE)
├── tests/
│   ├── test_calibration.py
│   ├── test_kinematics.py
│   ├── test_road_network.py
│   ├── test_xosc_writer.py
│   └── test_integration.py
├── video_to_xosc.py                 # Zachované CLI rozhranie (spätne kompatibilné) delegujúce na src/xosc_pipeline
├── references.bib                   # Overená BibTeX bibliografia
└── reports/
    ├── papers/                      # Legálne stiahnuté open-access arXiv PDF články
    ├── evaluation/                  # Pred/po tabuľky, JSONy, grafy, BEV & video overlay ukážky
    └── research_and_changes.pdf     # Finálny PDF report (Fáza 6)
```

---

## 3. Overovacie kritériá úspechu (Definition of Done pre implementáciu)
1. Každý vygenerovaný `.xosc` prejde validáciou oproti `scenario_runner/srunner/openscenario/OpenSCENARIO.xsd`.
2. Pri všetkých 4 demo videách (`dashcam_input.mp4`, `brake.mp4`, `jelen.mp4`, `stiahnute_video.mp4`):
   - `max|a|` všetkých vozidiel $\le 9.5\text{ m/s}^2$ (v bežnej premávke $\le 4.5\text{ m/s}^2$),
   - `max|jerk|` $\le 8.0\text{ m/s}^3$ (v bežnej premávke $\le 5.0\text{ m/s}^3$),
   - `max|yaw_rate|` bez $180^\circ$ skokov ($\le 0.6\text{ rad/s}$),
   - **0 neúmyselných prekrytí** (OBB / min. odstup) v bežnej premávke,
   - správne rozmery a kategórie pre každú detegovanú triedu (napr. `truck` v `brake.mp4`),
   - nenulový počet `<ParameterDeclaration>` (logický scenár) a sprievodný validný `.xodr` súbor.
3. Jednotkové a integračné testy (`pytest`) prechádzajú na 100 %.
