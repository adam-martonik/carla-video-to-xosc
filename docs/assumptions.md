# Predpoklady a projektové rozhodnutia (`docs/assumptions.md`)

Tento dokument zaznamenáva predvolené technické predpoklady prijaté pri úprave nástroja `video_to_xosc` v súlade so zadaním.

## 1. Cieľový štandard OpenSCENARIO a validácia
- **Cieľová verzia:** ASAM OpenSCENARIO **1.0** (`revMajor="1"`, `revMinor="0"`), validovaná oproti schéme `scenario_runner/srunner/openscenario/OpenSCENARIO.xsd` pomocou `xmlschema==1.0.18`.
- **Dôvod:** Nainštalovaný CARLA 1.0 + ScenarioRunner v1.0 (`--openscenario`) priamo vykonáva OpenSCENARIO 1.0 (`.xosc`). Zachovanie verzie 1.0 garantuje 100 % spätnú kompatibilitu s `run_replay.py` a `scenario_runner.py`, pričom umožňuje pridať parametrizáciu (`<ParameterDeclarations>`), presné rozmery vozidiel, realistické `<Performance>` limity a plynulé `<FollowTrajectoryAction>` + `<SpeedAction>` (`linear`/`sinusoidal`).

## 2. Simulátor, cestná sieť a OpenDRIVE (`.xodr`)
- **Primárny simulátor:** CARLA 1.0 (lokálne nainštalovaný v `CARLA_1.0/`).
- **Offline / standalone podpora:** Pôvodný skript zlyhal bez bežiaceho CARLA servera (Stage D `project_frenet_to_carla_map()`). Pridávame deterministickú **offline projekciu a generovanie vlastnej cestnej siete ASAM OpenDRIVE 1.4/1.6 (`.xodr`)** s automaticky odhadnutou geometriou vozovky (počet pruhov, šírka pruhu 3.5 m, krajnica, zakrivenie z ego-motion yaw-rate) pre použitie v `esmini`, CARLA (`load_opendrive_world`) alebo pri offline testovaní/CI bez nutnosti spúšťať UE5 server. Pri zapnutom CARLA serveri zostáva zachovaná plná projekcia na CARLA mapy (napr. `Town04_Opt`).

## 3. Kamera a metrická mierka (Monocular Dashcam)
- **Typ kamery:** Nekalibrovaná monokulárna predná palubná kamera (dashcam), 16:9 alebo 4:3, bez IMU/GPS/CAN údajov.
- **Vnútorné parametre (Intrinsics):** Predvolené horizontálne zorné pole `HFOV = 70°` (konfigurovateľné v YAML/JSON konfigurácii a cez CLI), optický stred $c_x = W/2$, výška kamery $h_{\text{cam}} = 1.35\text{ m}$ (bežné osobné vozidlo).
- **Dynamický odhad horizontu ($c_y$):** Namiesto pevnej konštanty `0.52 * H` odhadujeme horizont z priesečníka statických pruhových čiar (Houghove priamky / RANSAC) a z focus-of-expansion (FOE) optického toku s bezpečným fallbackom na konfigurovateľnú hodnotu `0.52 * H`.
- **Metrická mierka:** Fúzia dvoch nezávislých geometrických odhadov — projekcia spodnej hrany bounding boxu na rovinu vozovky ($z_{\text{gp}} = f_y h_{\text{cam}} / (v_b - c_y)$) a pomer známej fyzickej šírky triedy ($z_w$) — vážená podľa neistoty (pri blízkosti horizontu dominuje šírka, pri dobrých podmienkach v blízkom/strednom poli dominuje rovina vozovky).

## 4. Súradnicové systémy a jednotky
Všetky výpočty používajú jednotky **SI** (metre, sekundy, radiány, m/s, m/s², m/s³):
1. **Obrazová rovina (Image frame):** $(u, v)$ v pixeloch, počiatok vľavo hore, $u$ doprava, $v$ nadol.
2. **Kamerový rámec (Camera frame):** $(x_{\text{cam}}, y_{\text{cam}}, z_{\text{cam}})$ v metroch, $z_{\text{cam}}$ dopredu, $x_{\text{cam}}$ doprava, $y_{\text{cam}}$ nadol.
3. **Frenetov rámec cesty (Road Frenet frame):** $(s, d)$ v metroch, $s$ pozdĺž referenčnej osi pruhu dopredu, $d$ priečne doprava od stredu pruhu ego vozidla ($d > 0$ doprava, $d < 0$ doľava).
4. **Svetový rámec OpenSCENARIO / OpenDRIVE / CARLA (World frame):** $(x, y, z, h)$ v metroch a radiánoch, pravotočivý pre OpenDRIVE/OpenSCENARIO (`h` = heading/yaw v radiánoch z `atan2(dy, dx)`). Pri projekcii na CARLA mapu sa používajú súradnice z CARLA Waypoint API v súlade s konvenciou `ScenarioRunner`.

## 5. Fyzikálne a kinematické limity vozidiel
- **Osobné vozidlá:** $v \in [0, 40]\text{ m/s}$, pozdĺžne zrýchlenie $a_{\text{long}} \in [-9.5, +4.5]\text{ m/s}^2$, pozdĺžny jerk $|j_{\text{long}}| \le 8.0\text{ m/s}^3$ (núdzové brzdenie do $12.0\text{ m/s}^3$), priečne zrýchlenie $|a_{\text{lat}}| \le 6.0\text{ m/s}^2$, rázvor $L = 2.8\text{ m}$, max. uhol natočenia kolies $\delta_{\max} = 0.55\text{ rad}$.
- **Nákladné vozidlá a autobusy:** $a_{\text{long}} \in [-6.5, +2.5]\text{ m/s}^2$, $|j_{\text{long}}| \le 5.0\text{ m/s}^3$, rázvor $L = 4.5\text{–}6.0\text{ m}$.
- **Chodci / zver / cyklisti:** špecifické rozmery a rýchlostné obmedzenia podľa triedy (chodec max. $4.5\text{ m/s}$, cyklista max. $12\text{ m/s}$, zver max. $15\text{ m/s}$).

## 6. Hardvér a dostupnosť simulátora / nástrojov
- **GPU:** AMD Radeon RX 9060 XT (16 GB VRAM). Keďže PyTorch na Windows nepoužíva CUDA (bez NVIDIA GPU), inferencia YOLO/tracking beží na CPU (dostatočne rýchle pre 15–22 s demo klipy).
- **PDF generátor:** `pdflatex`/`xelatex` nie sú v systéme nainštalované; na generovanie PDF reportu použijeme deterministický HTML + CSS Paged Media pipeline renderovaný cez Google Chrome (`--headless=new --print-to-pdf`) s vloženými SVG/PNG grafmi z `matplotlib`, číslovanými kapitolami, obsahom a klikateľnými DOI/arXiv odkazmi.
- **esmini:** Nie je nainštalovaný v PATH; vizuálnu a kvantitatívnu validáciu zabezpečujeme: (a) XSD validáciou cez `xmlschema`, (b) bird's-eye-view (BEV) + video overlay vykresľovaním trajektórií, (c) kompletným súborom kvantitatívnych kinematických/kolíznych metrík pred/po, a (d) pripravenou integráciou s lokálnym CARLA 1.0 serverom.
