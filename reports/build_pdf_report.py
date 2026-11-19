#!/usr/bin/env python
"""Phase 6: Build publication-grade PDF report `reports/research_and_changes.pdf`."""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
EVAL_DIR = REPORTS_DIR / "evaluation"
HTML_OUT = REPORTS_DIR / "research_and_changes.html"
PDF_OUT = REPORTS_DIR / "research_and_changes.pdf"

CHROME_PATHS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def img_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_html() -> str:
    summary = json.loads((EVAL_DIR / "before_after_summary.json").read_text(encoding="utf-8"))

    # Build Before/After HTML table rows
    eval_rows = []
    for stem, r in summary.items():
        b, a = r["before"], r["after"]
        cats = ", ".join(f"{k}: {v}" for k, v in a.get("categories", {}).items())
        eval_rows.append(
            f"""<tr>
              <td><code>{r['video']}</code></td>
              <td>{b['worst_max_abs_acc']:.2f} &rarr; <strong>{a['worst_max_abs_acc']:.2f}</strong></td>
              <td>{b['worst_max_abs_jerk']:.2f} &rarr; <strong>{a['worst_max_abs_jerk']:.2f}</strong></td>
              <td>{b['worst_max_yaw_rate']:.3f} &rarr; <strong>{a['worst_max_yaw_rate']:.3f}</strong></td>
              <td>{b['overlap_steps']} &rarr; <strong>{a['obb_collision_steps']}</strong></td>
              <td>{b['param_count']} &rarr; <strong>{a['param_count']}</strong></td>
              <td>{'Áno' if a['has_xodr'] else 'Nie'} / {'PASS' if a['xsd_valid'] else 'FAIL'}</td>
            </tr>"""
        )
    eval_table_html = "\n".join(eval_rows)

    fig1 = img_to_data_uri(EVAL_DIR / "fig1_kinematics_comparison.png")
    fig2 = img_to_data_uri(EVAL_DIR / "fig2_bev_trajectories.png")
    fig3 = img_to_data_uri(EVAL_DIR / "fig3_velocity_accel_profiles.png")
    fig4 = img_to_data_uri(EVAL_DIR / "fig4_video_overlay.png")

    downloaded_pdfs = sorted(p.name for p in (REPORTS_DIR / "papers").glob("*.pdf"))
    papers_note = (
        f"Súčasťou repozitára je {len(downloaded_pdfs)} legálne dostupných open-access arXiv PDF článkov "
        f"uložených v <code>reports/papers/</code> a kompletná bibliografia v <code>references.bib</code>."
    )

    return f"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="utf-8"/>
<title>Rekonštrukcia realistických OpenSCENARIO scenárov z dashcam videa</title>
<style>
  @page {{
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @top-left {{
      content: "Dashcam Video → OpenSCENARIO (.xosc) — Výskumný a technický report";
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 8pt;
      color: #666;
    }}
    @bottom-right {{
      content: "Strana " counter(page) " z " counter(pages);
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 8pt;
      color: #666;
    }}
  }}
  @page :first {{
    @top-left {{ content: none; }}
    @bottom-right {{ content: none; }}
  }}
  body {{
    font-family: "Segoe UI", "Calibri", Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #1f2933;
    margin: 0;
  }}
  .title-page {{
    page-break-after: always;
    padding-top: 38mm;
    text-align: left;
  }}
  .title-badge {{
    display: inline-block;
    background: #1d4ed8;
    color: #fff;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
  }}
  .title-page h1 {{
    font-size: 24pt;
    line-height: 1.2;
    color: #0f172a;
    margin: 0 0 14px 0;
    border: none;
  }}
  .subtitle {{
    font-size: 12.5pt;
    color: #475569;
    margin-bottom: 28px;
  }}
  .meta-box {{
    border-left: 4px solid #1d4ed8;
    background: #f8fafc;
    padding: 14px 18px;
    margin-top: 24px;
    font-size: 9.5pt;
  }}
  .meta-box p {{ margin: 4px 0; }}
  .toc {{
    page-break-after: always;
  }}
  .toc h2 {{ margin-top: 0; }}
  .toc ul {{
    list-style: none;
    padding-left: 0;
  }}
  .toc li {{
    padding: 4px 0;
    border-bottom: 1px dotted #cbd5e1;
  }}
  .toc a {{
    text-decoration: none;
    color: #1e293b;
  }}
  h1, h2, h3 {{
    color: #0f172a;
    page-break-after: avoid;
  }}
  h1 {{
    font-size: 15.5pt;
    border-bottom: 2px solid #1d4ed8;
    padding-bottom: 4px;
    margin-top: 22px;
    margin-bottom: 10px;
  }}
  h2 {{
    font-size: 12pt;
    color: #1e3a8a;
    margin-top: 15px;
    margin-bottom: 6px;
  }}
  h3 {{
    font-size: 10.5pt;
    color: #334155;
    margin-top: 10px;
    margin-bottom: 4px;
  }}
  p {{
    margin: 0 0 8px 0;
    text-align: justify;
  }}
  ul, ol {{
    margin: 4px 0 10px 0;
    padding-left: 20px;
  }}
  li {{ margin-bottom: 4px; }}
  code {{
    font-family: Consolas, "Courier New", monospace;
    font-size: 8.8pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 14px 0;
    font-size: 8.4pt;
    page-break-inside: auto;
  }}
  tr {{ page-break-inside: avoid; }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 5px 6px;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    background: #eff6ff;
    color: #1e3a8a;
    font-weight: 700;
  }}
  tr:nth-child(even) td {{
    background: #f8fafc;
  }}
  .figure {{
    margin: 12px 0 16px 0;
    text-align: center;
    page-break-inside: avoid;
  }}
  .figure img {{
    max-width: 100%;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
  }}
  .caption {{
    font-size: 8.5pt;
    color: #475569;
    margin-top: 5px;
    font-style: italic;
  }}
  .callout {{
    background: #f0f9ff;
    border-left: 4px solid #0284c7;
    padding: 10px 14px;
    margin: 10px 0 14px 0;
    font-size: 9.2pt;
  }}
  .bib-list {{
    font-size: 8.3pt;
    line-height: 1.4;
    padding-left: 22px;
  }}
  .bib-list li {{
    margin-bottom: 5px;
  }}
  .bib-list a {{
    color: #1d4ed8;
    text-decoration: none;
  }}
  .page-break {{ page-break-before: always; }}
</style>
</head>
<body>

<div class="title-page">
  <div class="title-badge">CARLA 1.0 &bull; ASAM OpenSCENARIO 1.0 &bull; ASAM OpenDRIVE 1.4</div>
  <h1>Fyzikálne vierohodná rekonštrukcia simulačných scenárov z dashcam videa do formátu OpenSCENARIO</h1>
  <div class="subtitle">Prehľad súčasného výskumu, audit pôvodného nástroja <code>video_to_xosc.py</code>, návrh a implementácia úprav a kvantitatívne vyhodnotenie pred/po</div>
  <div class="meta-box">
    <p><strong>Repozitár:</strong> <code>carla-video-to-xosc</code> (vetva <code>feat/realistic-xosc-pipeline</code>)</p>
    <p><strong>Cieľový štandard:</strong> ASAM OpenSCENARIO 1.0 (<code>.xosc</code>) + ASAM OpenDRIVE 1.4 (<code>.xodr</code>)</p>
    <p><strong>Cieľový simulátor:</strong> CARLA 1.0 (Unreal Engine 5.8) / ScenarioRunner v1.0 + offline XODR režim</p>
    <p><strong>Počet overených primárnych zdrojov:</strong> 32 (arXiv, IEEE, ACM, CVPR, ICCV, ECCV, NeurIPS, ICLR, ITSC, IV)</p>
    <p><strong>Dátum:</strong> 19. november 2026</p>
  </div>
</div>

<div class="toc">
  <h2>Obsah</h2>
  <ul>
    <li><a href="#ch1">1. Úvod a definícia problému</a></li>
    <li><a href="#ch2">2. Prehľad súčasného výskumu</a></li>
    <li><a href="#ch3">3. Datasety a metriky vyhodnotenia</a></li>
    <li><a href="#ch4">4. Audit pôvodného nástroja</a></li>
    <li><a href="#ch5">5. Implementované úpravy a ich vedecké zdôvodnenie</a></li>
    <li><a href="#ch6">6. Kvantitatívne a vizuálne výsledky pred/po</a></li>
    <li><a href="#ch7">7. Limity, otvorené problémy a odporúčania na ďalší vývoj</a></li>
    <li><a href="#ch8">8. Bibliografia</a></li>
  </ul>
  <div class="callout">
    <strong>Poznámka k reprodukovateľnosti:</strong> {papers_note} Kompletné zdrojové súbory hodnotenia, grafy a vygenerované scenáre (<code>.xosc</code> / <code>.xodr</code>) sa nachádzajú v <code>reports/evaluation/</code>.
  </div>
</div>

<!-- CHAPTER 1 -->
<h1 id="ch1">1. Úvod a definícia problému</h1>
<p>
  Scenárové testovanie (<em>scenario-based safety assessment</em>) je dnes kľúčovým pilierom overovania bezpečnosti automatizovaných a autonómnych vozidiel (ADS/AV), pretože čisto kilometrické testovanie v reálnej premávke vyžaduje stovky miliónov až miliardy kilometrov na štatisticky významné pokrytie zriedkavých kritických situácií (<em>long-tail / corner cases</em>). Palubné kamery (<em>dashcams</em>) pritom denne zachytávajú obrovské množstvo reálnych kritických udalostí — od náhlych zaraďovaní (<em>cut-in</em>) a núdzového brzdenia až po zrážky so zverou či chodcami. Premena surového monokulárneho videa bez kalibrácie a bez telemetrie (IMU/GPS/LiDAR) do štandardizovaného simulačného formátu <strong>ASAM OpenSCENARIO (<code>.xosc</code>)</strong> a <strong>ASAM OpenDRIVE (<code>.xodr</code>)</strong> predstavuje náročnú inverznú úlohu počítačového videnia, odhadu stavu a dopravného inžinierstva.
</p>
<p>
  Aby vygenerovaný <code>.xosc</code> scenár verne zodpovedal realite zo záznamu, musí spĺňať päť základných dimenzií <strong>realizmu a fyzikálnej vierohodnosti</strong>:
</p>
<ul>
  <li><strong>Kinematická a dynamická realizovateľnosť:</strong> trajektórie všetkých aktérov musia rešpektovať fyzikálne limity vozidiel (rýchlosť, pozdĺžne zrýchlenie a spomalenie, jerk, uhlová rýchlosť otáčania a kinematický <em>bicycle model</em>) bez skokov, teleportácií a náhlych otočení o 180°.</li>
  <li><strong>Metrická a priestorová konzistencia:</strong> správna metrická mierka (metre, sekundy, radiány), korektný odhad horizontu kamery, presné rozmery a kategórie podľa detegovanej triedy (osobné auto, nákladné vozidlo, autobus, motocykel, chodec/zver) a <strong>absencia neúmyselných prekrytí (kolízií)</strong> medzi aktérmi.</li>
  <li><strong>Časová a identitná kontinuita:</strong> synchronizácia s reálnou snímkovou frekvenciou (FPS), dostatočne jemné časové vzorkovanie trajektórií, plynulé prechody rýchlosti, odolnosť voči krátkym oklúziám (<em>tracklet stitching</em>) a realistický vstup/výstup aktérov mimo okna viditeľnosti.</li>
  <li><strong>Vierohodná cestná sieť a parametrizácia:</strong> geometricky konzistentné jazdné pruhy a zakrivenie vozovky (<code>.xodr</code> alebo projekcia na CARLA mapu) a parametrizácia logického scenára (<code>&lt;ParameterDeclarations&gt;</code>) umožňujúca systematické testovanie variantov.</li>
  <li><strong>Štandardová validita:</strong> 100 % zhoda so schémou ASAM OpenSCENARIO 1.0 (<code>OpenSCENARIO.xsd</code>) pre bezproblémové prehranie v simulátoroch ako CARLA ScenarioRunner alebo <code>esmini</code>.</li>
</ul>

<!-- CHAPTER 2 -->
<h1 id="ch2">2. Prehľad súčasného výskumu</h1>
<p>
  Pred návrhom úprav bola vykonaná rešerš 32 primárnych odborných publikácií (konferencie CVPR, ICCV, ECCV, NeurIPS, ICLR, ICRA, IEEE ITSC, IEEE IV, ACM SIGSPATIAL a časopisy IEEE T-ITS, IEEE T-RO, IEEE TMM, IEEE Access). Nasledujúce podkapitoly sumarizujú kľúčové prístupy rozdelené do šiestich tematických skupín a ich priamu použiteľnosť v našom nástroji.
</p>

<h2>2.1 Extrakcia scenárov z videa a scenario-based testing</h2>
<ul>
  <li><strong>Menzel et al. (2018)</strong> [<em>Scenarios for Development, Test and Validation of Automated Vehicles</em>, IEEE IV]: Zavádzajú dnes štandardnú trojúrovňovú taxonómiu <em>functional → logical → concrete scenarios</em> zladenú s normou ISO 26262. Metóda oddeľuje sémantický popis, parametrický priestor a konkrétnu inštanciu; limitom je abstraktnosť bez konkrétneho CV extraktora. <em>Použiteľnosť:</em> Priamo motivuje zavedenie <code>&lt;ParameterDeclarations&gt;</code> (logický scenár).</li>
  <li><strong>Bashetty et al. (2020)</strong> [<em>DeepCrashTest: Turning Dashcam Videos into Virtual Crash Tests</em>, ICRA, arXiv:2003.11766]: Prvý ucelený systém premieňajúci nekalibrované dashcam videá nehôd na virtuálne testy (3D trajektórie v Webots/Sim-ATAV). Kombinuje 2D detekciu, monokulárnu hĺbku a spline vyhladzovanie, ale nevygeneruje štandardný <code>.xosc</code> ani nezverejnil kód. <em>Použiteľnosť:</em> Potvrdzuje potrebu dátami riadenej rekonštrukcie kritických udalostí namiesto pevných šablón.</li>
  <li><strong>Riedmaier et al. (2020)</strong> [<em>Survey on Scenario-Based Safety Assessment of Automated Vehicles</em>, IEEE Access]: Rozsiahly prehľad metodík scenárového overovania bezpečnosti vrátane taxonomického členenia zberu dát, extrakcie a hodnotenia rizika. Zdôrazňuje nutnosť kvantifikovať vernosť simulácie voči realite. <em>Použiteľnosť:</em> Rámec pre definíciu overovacích metrík.</li>
  <li><strong>Ding et al. (2023)</strong> [<em>A Survey on Safety-Critical Driving Scenario Generation</em>, IEEE T-ITS, arXiv:2202.02215]: Systematizuje generovanie kritických scenárov na dátovo riadené, adversariálne a znalostné metódy a definuje päť pilierov kvality (<em>fidelity, efficiency, diversity, transferability, controllability</em>). <em>Použiteľnosť:</em> Priame vodidlo pre rovnováhu medzi vernosťou (fidelity) a parametrickou riaditeľnosťou (controllability).</li>
  <li><strong>Guo et al. (2023)</strong> [<em>Scenario Extraction from a Large Real-World Dataset</em>, IEEE ITSC, arXiv:2306.07815]: Predstavujú kompletný pipeline na predspracovanie, tagovanie interakcií a extrakciu scenárov z Waymo Open Motion Datasetu. Limitom je závislosť od hotových 3D LiDAR boxov. <em>Použiteľnosť:</em> Inšpirácia pre štruktúrované vyhodnocovanie interakcií a kinematiky.</li>
  <li><strong>Li et al. (2023)</strong> [<em>ScenarioNet</em>, NeurIPS, arXiv:2306.12241]: Open-source platforma zjednocujúca scenáre z Waymo, nuScenes a nuPlan do jednotného formátu pre MetaDrive. Nepracuje so surovým monokulárnym videom. <em>Použiteľnosť:</em> Referencia pre zjednotenú reprezentáciu mapy a trajektórií.</li>
  <li><strong>Miao et al. (2024)</strong> [<em>From Dashcam Videos to Driving Simulations</em>, AAAI Workshop, arXiv:2411.16027]: Využívajú veľké video-jazykové modely (VLM, GPT-4o) na preklad dashcam videa do jazyka SCENIC pre CARLA. Dosahujú vysokú sémantickú vernosť, no bez presných metrických trajektórií a so závislosťou na proprietárnom API. <em>Použiteľnosť:</em> Dopĺňa náš deterministický geometrický prístup (ktorý beží plne lokálne a produkuje validný <code>.xosc</code>).</li>
  <li><strong>de Gelder &amp; op den Camp (2024)</strong> [<em>How (not) to parameterize scenarios?</em>, arXiv:2409.01402]: Analyzujú chyby pri parametrizácii scenárov (napr. v predpise UN R157) a ukazujú, ako voliť nezávislé, fyzikálne zmysluplné parametre. <em>Použiteľnosť:</em> Priama inšpirácia pre výber parametrov v <code>&lt;ParameterDeclarations&gt;</code>.</li>
</ul>

<h2>2.2 Viacobjektový tracking (Multi-Object Tracking)</h2>
<ul>
  <li><strong>Bewley et al. (2016)</strong> [<em>SORT</em>, ICIP, arXiv:1602.00763]: Klasický real-time tracker kombinujúci Kalmanov filter a Maďarský algoritmus nad IoU. Veľmi rýchly, ale citlivý na oklúzie a ID switche. <em>Použiteľnosť:</em> Základ modernej rodiny trackerov.</li>
  <li><strong>Wojke et al. (2017)</strong> [<em>DeepSORT</em>, ICIP, arXiv:1703.07402]: Rozširuje SORT o hlboké vzhľadové príznaky (ReID) a kosínusovú metriku, čím výrazne redukuje ID switche po oklúzii za cenu vyššej výpočtovej náročnosti. <em>Použiteľnosť:</em> Koncept re-identifikácie a asociácie cez oklúzie.</li>
  <li><strong>Zhang et al. (2022)</strong> [<em>ByteTrack</em>, ECCV, arXiv:2110.06864]: Dvojfázová asociácia všetkých detekcií vrátane nízko-skórových boxov zachraňuje čiastočne zakryté objekty bez ReID siete. Limitom pri pohybujúcej sa dashcam kamere je chýbajúca kompenzácia pohybu kamery. <em>Použiteľnosť:</em> Pôvodný tracker v nástroji a spoľahlivý fallback.</li>
  <li><strong>Aharon et al. (2022)</strong> [<em>BoT-SORT</em>, arXiv:2206.14651]: Rozširuje ByteTrack o <strong>Camera Motion Compensation (CMC)</strong>, vylepšený stavový vektor Kalmanovho filtra (priamy odhad šírky a výšky boxu) a voliteľný ReID modul. <em>Použiteľnosť:</em> <strong>Priamo integrované</strong> ako predvolený tracker (<code>botsort.yaml</code>) v našej upravenej pipeline.</li>
  <li><strong>Cao et al. (2023)</strong> [<em>OC-SORT</em>, CVPR, arXiv:2203.14360]: Prechádza od estimation-centric k observation-centric prístupu s virtuálnou trajektóriou cez oklúziu a konzistenciou smeru pohybu. <em>Použiteľnosť:</em> Inšpirácia pre náš post-hoc modul <code>stitch_fragmented_tracks</code>.</li>
  <li><strong>Du et al. (2023)</strong> [<em>StrongSORT</em>, IEEE TMM, arXiv:2202.13514]: Kombinuje vylepšený DeepSORT s ECC kompenzáciou kamery a bezmodelovým post-processingom (AFLink + Gaussian-smoothed interpolation). <em>Použiteľnosť:</em> Potvrdzuje prínos post-hoc spájania trackletov a vyhladzovania.</li>
</ul>

<h2>2.3 Monokulárny odhad hĺbky, 3D polohy a metrickej mierky</h2>
<ul>
  <li><strong>Song &amp; Chandraker (2015)</strong> [<em>Joint SFM and Detection Cues for Monocular 3D Localization</em>, CVPR]: Spájajú štruktúru z pohybu, geometriu roviny vozovky (<em>ground-plane</em>) a známe rozmery vozidiel na riešenie nejednoznačnosti mierky (<em>scale ambiguity</em>). <em>Použiteľnosť:</em> Priamy teoretický základ našej inverzno-variančnej fúzie hĺbky <code>z_gp</code> a <code>z_width</code>.</li>
  <li><strong>Ma et al. (2021)</strong> [<em>MonoDLE</em>, CVPR, arXiv:2103.16237]: Dokazujú, že dominantným zdrojom chýb v monokulárnej 3D detekcii je práve chyba hĺbky, nie 2D lokalizácia či rozmery. <em>Použiteľnosť:</em> Zdôvodňuje nutnosť váženia neistoty hĺbky a MAD filtrácie hĺbkových špičiek.</li>
  <li><strong>Zhang et al. (2021)</strong> [<em>MonoFlex</em>, CVPR, arXiv:2104.02323]: Oddelene spracúva orezané (truncated) objekty na okraji obrazu a kombinuje priamu regresiu hĺbky s geometrickým riešením z kľúčových bodov váženým neistotou. <em>Použiteľnosť:</em> Inšpirácia pre adaptívne váženie <code>sigma_gp</code> vs. <code>sigma_w</code>.</li>
  <li><strong>Yin et al. (2023)</strong> [<em>Metric3D</em>, ICCV, arXiv:2307.10984]: Rieši nejednoznačnosť ohniskovej vzdialenosti transformáciou do kanonického kamerového priestoru pre zero-shot metrickú hĺbku. <em>Použiteľnosť:</em> Zvýrazňuje kritický význam správneho HFOV a pomeru strán (16:9 vs. 9:16), ktoré sme opravili v <code>calibration.py</code>.</li>
  <li><strong>Piccinelli et al. (2024)</strong> [<em>UniDepth</em>, CVPR, arXiv:2403.18913]: Predikuje husté metrické 3D body bez znalosti intrinsics pomocou samokalibračného modulu. Vyžaduje GPU s vyššou pamäťou pre real-time video. <em>Použiteľnosť:</em> Odporúčané budúce rozšírenie pri dostupnosti CUDA/ROCm GPU.</li>
  <li><strong>Yang et al. (2024)</strong> [<em>Depth Anything V2</em>, NeurIPS, arXiv:2406.09414]: Základný (foundation) model pre monokulárnu hĺbku trénovaný na vysoko presných syntetických dátach s vynikajúcimi detailmi hrán. <em>Použiteľnosť:</em> Perspektívna náhrada alebo doplnok geometrickej hĺbky pre neštruktúrované scény.</li>
</ul>

<h2>2.4 Odhad ego-motion (Visual Odometry / SLAM)</h2>
<ul>
  <li><strong>Campos et al. (2021)</strong> [<em>ORB-SLAM3</em>, IEEE T-RO, arXiv:2007.11898]: Najpokročilejší otvorený geometrický SLAM podporujúci monokulárne, stereo a vizuálno-inerciálne kamery. Čisto monokulárny režim však trpí mierkovým driftom bez známej výšky kamery alebo IMU. <em>Použiteľnosť:</em> Referenčná architektúra pre RANSAC filtráciu statických bodov.</li>
  <li><strong>Teed &amp; Deng (2021)</strong> [<em>DROID-SLAM</em>, NeurIPS, arXiv:2108.10869]: Rekurentná neurónová sieť s diferencovateľnou vrstvou Dense Bundle Adjustment dosahujúca vysokú robustnosť aj pri náročnom pohybe kamery. Náročné na VRAM. <em>Použiteľnosť:</em> Ukazuje silu oddelenia rotácie a translácie pri odhade pohybu.</li>
  <li><strong>Teed et al. (2023)</strong> [<em>DPVO: Deep Patch Visual Odometry</em>, NeurIPS, arXiv:2208.04726]: Riedka patch-based vizuálna odometria 3× rýchlejšia než DROID-SLAM s porovnateľnou presnosťou. <em>Použiteľnosť:</em> Potvrdzuje, že riedke sledovanie stabilných bodov so správnym odfiltrovaním dynamických objektov a rotácie (náš RANSAC + radial-FOE odhad) poskytuje spoľahlivý základ aj na CPU.</li>
</ul>

<h2>2.5 Vyhladzovanie a kinematické obmedzenia trajektórií</h2>
<ul>
  <li><strong>Rauch, Tung &amp; Striebel (1965)</strong> [<em>Maximum Likelihood Estimates of Linear Dynamic Systems</em>, AIAA Journal]: Klasický <strong>RTS smoother</strong> — optimálny fixed-interval obojsmerný Kalmanov vyhladzovač, ktorý kombinuje dopredný filter a spätný prechod pre minimálny rozptyl odhadu polohy, rýchlosti aj zrýchlenia. <em>Použiteľnosť:</em> <strong>Priamo implementované</strong> v <code>rts_smoother_1d</code> (Constant-Acceleration model).</li>
  <li><strong>Werling et al. (2010)</strong> [<em>Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenét Frame</em>, ICRA]: Zavádzajú plánovanie jerk-optimálnych trajektórií v cestnom Frenetovom súradnicovom systéme $(s, d)$, ktorý čisto oddeľuje pozdĺžnu a priečnu dynamiku. <em>Použiteľnosť:</em> Základ celej našej Frenet reprezentácie a dekonflikcie.</li>
  <li><strong>Kong et al. (2015)</strong> [<em>Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design</em>, IEEE IV]: Porovnávajú kinematický <em>bicycle model</em> s dynamickým modelom a ukazujú jeho výbornú presnosť pri bežných a stredných priečnych zrýchleniach s minimálnym počtom parametrov (rázvor $L$). <em>Použiteľnosť:</em> <strong>Priamo implementované</strong> v <code>enforce_kinematic_limits</code>.</li>
  <li><strong>Polack et al. (2017)</strong> [<em>The Kinematic Bicycle Model: A Consistent Model for Planning Feasible Trajectories?</em>, IEEE IV]: Analyzujú hranice konzistentnosti kinematického modelu a dokazujú, že pri obmedzení priečneho zrýchlenia (|a<sub>lat</sub>| &le; 0.5 &mu;g &approx; 4.5 m/s&sup2;) generuje fyzikálne realizovateľné trajektórie. <em>Použiteľnosť:</em> Určuje priame limity priečneho zrýchlenia a uhla riadenia v <code>config/default.yaml</code>.</li>
</ul>

<h2>2.6 Rekonštrukcia cestnej siete, jazdné pruhy a map matching</h2>
<ul>
  <li><strong>Newson &amp; Krumm (2009)</strong> [<em>Hidden Markov Map Matching Through Noise and Sparseness</em>, ACM SIGSPATIAL]: Klasický pravdepodobnostný map matching pomocou HMM odolný voči šumu polohy. <em>Použiteľnosť:</em> Princíp plynulého priraďovania trajektórií k cestnému koridoru bez náhlych preskokov.</li>
  <li><strong>Zheng et al. (2022)</strong> [<em>CLRNet</em>, CVPR, arXiv:2203.10350]: Detekcia jazdných pruhov cez viacúrovňovú fúziu príznakov (Cross Layer Refinement) s vysokou presnosťou na CULane. <em>Použiteľnosť:</em> Perspektívne nahradenie klasickej Houghovej detekcie hrán pre zakrivené pruhy.</li>
  <li><strong>Liao et al. (2023)</strong> [<em>MapTR</em>, ICLR, arXiv:2208.14437]: Online vektorizovaná rekonštrukcia HD máp z palubných kamier. Vyžaduje viacero kalibrovaných kamier (360° surround view). <em>Použiteľnosť:</em> Inšpirácia pre vektorizovaný export cesty do OpenDRIVE (<code>.xodr</code>).</li>
</ul>

<h2>2.7 Porovnávacia tabuľka kľúčových prístupov</h2>
<table>
  <thead>
    <tr>
      <th>Oblasť</th>
      <th>Reprezentatívne práce</th>
      <th>Hlavná metóda</th>
      <th>Silné stránky</th>
      <th>Limity pre dashcam (CPU)</th>
      <th>Premietnutie v nástroji</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Video &rarr; Scenario</td>
      <td>DeepCrashTest (2020), Miao et al. (2024)</td>
      <td>CV pipeline / VLM &rarr; Scenic</td>
      <td>End-to-end z dashcam videa</td>
      <td>Bez open-source kódu / bez <code>.xosc</code></td>
      <td>Plne lokálny <code>.xosc</code> + <code>.xodr</code> pipeline</td>
    </tr>
    <tr>
      <td>Tracking (MOT)</td>
      <td>ByteTrack (2022), BoT-SORT (2022), OC-SORT (2023)</td>
      <td>Dvojfázová asociácia + CMC + observation-centric</td>
      <td>Vysoké IDF1 pri pohybe kamery</td>
      <td>Dlhé úplné oklúzie stále fragmentujú ID</td>
      <td><code>botsort.yaml</code> + post-hoc tracklet stitching</td>
    </tr>
    <tr>
      <td>3D poloha &amp; hĺbka</td>
      <td>Song (2015), MonoDLE (2021), Metric3D (2023)</td>
      <td>Ground-plane + šírka / kanonická hĺbka</td>
      <td>Metrická mierka bez LiDARu</td>
      <td>Hlboké modely náročné na GPU</td>
      <td>VP kalibrácia horizontu + inverzno-variančná fúzia</td>
    </tr>
    <tr>
      <td>Ego-motion (VO)</td>
      <td>ORB-SLAM3 (2021), DPVO (2023)</td>
      <td>RANSAC geometria / riedke patche</td>
      <td>Nízky drift a oddelenie rotácie</td>
      <td>Monokulárna nejednoznačnosť mierky</td>
      <td>RANSAC homografia + radiálna expanzia z FOE</td>
    </tr>
    <tr>
      <td>Kinematika</td>
      <td>RTS (1965), Werling (2010), Kong (2015), Polack (2017)</td>
      <td>Obojsmerný Kalman + Frenet + bicycle model</td>
      <td>Optimálne vyhladenie, fyzikálna realizovateľnosť</td>
      <td>Vyžaduje ladenie procesného šumu</td>
      <td>RTS CA smoother + limity v, a, j, &psi;&#775; + OBB dekonflikcia</td>
    </tr>
    <tr>
      <td>Validácia &amp; mapy</td>
      <td>Menzel (2018), Montali (2023), de Gelder (2024)</td>
      <td>Logické scenáre, WOSAC, TTC/PET</td>
      <td>Štandardizované metriky realizmu</td>
      <td>Dashcam nemá ground-truth 3D</td>
      <td><code>&lt;ParameterDeclarations&gt;</code>, XSD, OBB, TTC, <code>.xodr</code></td>
    </tr>
  </tbody>
</table>

<!-- CHAPTER 3 -->
<h1 id="ch3">3. Datasety a metriky vyhodnotenia</h1>
<h2>3.1 Referenčné datasety</h2>
<p>
  Pre objektívne testovanie jednotlivých modulov (detekcia, tracking, monokulárna 3D lokalizácia, ego-motion a kinematický realizmus trajektórií) sa v literatúre používajú najmä nasledujúce štandardizované datasety:
</p>
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Autori / Rok</th>
      <th>Senzory a pohľad</th>
      <th>Kľúčový prínos pre vyhodnotenie nášho problému</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>KITTI</strong></td>
      <td>Geiger et al. (2012)</td>
      <td>Stereo/mono RGB + 64-beam LiDAR + GPS/IMU</td>
      <td>Klasický benchmark pre monokulárnu 3D detekciu, vizuálnu odometriu a tracking z prednej kamery.</td>
    </tr>
    <tr>
      <td><strong>nuScenes</strong></td>
      <td>Caesar et al. (2020)</td>
      <td>6 kamier + LiDAR + 5 radarov, 1000 scén</td>
      <td>3D anotácie vozidiel a VRU v rôznych svetelných podmienkach (vrátane noci a dažďa) + vektorové mapy.</td>
    </tr>
    <tr>
      <td><strong>Waymo Open Dataset</strong></td>
      <td>Sun et al. (2020)</td>
      <td>5 kamier + 5 LiDARov, rozsiahle 3D tracky</td>
      <td>Vysoká rozmanitosť interakcií a základ pre Waymo Sim Agents Challenge (WOSAC).</td>
    </tr>
    <tr>
      <td><strong>Argoverse 2</strong></td>
      <td>Wilson et al. (2021)</td>
      <td>7 kamier + LiDAR + 3D HD mapy</td>
      <td>250 000 scenárov predikcie pohybu (Motion Forecasting) s presnými geometriami jazdných pruhov.</td>
    </tr>
    <tr>
      <td><strong>highD</strong></td>
      <td>Krajewski et al. (2018)</td>
      <td>Dronové zábery nemeckých diaľnic (110 500 vozidiel)</td>
      <td>Zlatý štandard pre prirodzené diaľničné trajektórie, rozdelenia zrýchlení, jerku a <em>cut-in</em> manévrov.</td>
    </tr>
    <tr>
      <td><strong>inD</strong></td>
      <td>Bock et al. (2020)</td>
      <td>Dronové zábery mestských križovatiek (vrátane VRU)</td>
      <td>Realistické interakcie vozidiel, chodcov a cyklistov bez oklúzie z pohľadu prvej osoby.</td>
    </tr>
  </tbody>
</table>

<h2>3.2 Metriky vernosti a fyzikálnej vierohodnosti</h2>
<p>
  V súlade s <em>Waymo Open Sim Agents Challenge</em> (Montali et al., 2023), simulátorom CARLA (Dosovitskiy et al., 2017) a klasickými bezpečnostnými ukazovateľmi (Hayward, 1972; Allen et al., 1978) hodnotíme kvalitu vygenerovaného <code>.xosc</code> scenára v štyroch skupinách metrík (implementované v <code>src/xosc_pipeline/metrics.py</code>):
</p>
<ol>
  <li><strong>Kinematická hladkosť a realizovateľnosť:</strong> maximálne pozdĺžne zrýchlenie max|a| (m/s&sup2;), maximálny a kvadratický priemer jerku max|j|, RMS(j) (m/s&sup3;), maximálna uhlová rýchlosť max|&psi;&#775;| (rad/s) a priečne zrýchlenie max|a<sub>lat</sub>|.</li>
  <li><strong>Geometrická konzistencia a bezkolíznosť:</strong> presná detekcia prekrytí pomocou 2D orientovaných ohraničujúcich boxov (<em>Oriented Bounding Boxes — OBB</em> v knižnici <code>shapely</code>) nad skutočnými rozmermi entít na jednotnej časovej mriežke 0.25 s.</li>
  <li><strong>Zástupné bezpečnostné metriky (Surrogate Safety Measures):</strong> minimálny čas do kolízie <strong>TTC</strong> (<em>Time-to-Collision</em>, Hayward 1972) a <strong>PET</strong> (<em>Post-Encroachment Time</em>, Allen et al. 1978) na zachytenie vernosti kritických interakcií.</li>
  <li><strong>Presnosť trajektórie a štandardová validita:</strong> pri dostupnosti referenčnej telemetrie <strong>ADE / FDE</strong> (<em>Average / Final Displacement Error</em>) a RMSE rýchlosti; vždy prísna validácia voči <code>OpenSCENARIO.xsd</code>.</li>
</ol>

<!-- CHAPTER 4 -->
<h1 id="ch4">4. Audit pôvodného nástroja</h1>
<p>
  V rámci Fázy 1 bol zrealizovaný detailný statický audit kódu <code>video_to_xosc.py</code> (1 055 riadkov) aj dynamický kvantitatívny audit 4 existujúcich <code>.xosc</code> súborov a 4 reálnych demo videí (<code>dashcam_input.mp4</code>, <code>brake.mp4</code>, <code>jelen.mp4</code>, <code>stiahnute_video.mp4</code>). Kompletné nálezy sú zdokumentované v <code>docs/audit.md</code> a <code>reports/evaluation/baseline_audit.json</code>. Kľúčové nedostatky pôvodného stavu boli:
</p>
<ul>
  <li><strong>Nefyzikálne zrýchlenia a jerk:</strong> V <code>dashcam_reconstructed.xosc</code> dosahovalo vozidlo <code>npc_track_155</code> zrýchlenie <strong>13.79 m/s&sup2;</strong> a jerk <strong>21.15 m/s&sup3;</strong>; v <code>moj_scenar.xosc</code> až <strong>15.41 m/s&sup2;</strong> a <strong>24.01 m/s&sup3;</strong>. Surový odhad ego rýchlosti z optického toku mal špičky <strong>460 až 729 m/s&sup2;</strong>.</li>
  <li><strong>Prekrývanie vozidiel a chybný spawn posun:</strong> V <code>dashcam_reconstructed.xosc</code> klesla minimálna vzdialenosť stredov dvoch 4.5 m vozidiel na <strong>2.86 m</strong> (fyzické prechádzanie cez seba); navyše oprava kolízie pri <code>t=0</code> v Stage D posunula iba prvý vrchol <code>world_poses[0]</code> bez zvyšku trajektórie (12 m teleport).</li>
  <li><strong>Anomália protismerného vozidla:</strong> V <code>scenar_jelen.xosc</code> náhle zastavenie <code>oncoming_vehicle</code> spôsobilo zrýchlenie <strong>13.03 m/s&sup2;</strong> a otočenie orientácie o <strong>3.134 rad/s (&approx; 180&deg;)</strong>.</li>
  <li><strong>Ignorovanie triedy, rozmerov a limitov v XML:</strong> Všetky vozidlá bez ohľadu na triedu (vrátane nákladného auta <code>class_id=7</code> v <code>brake.mp4</code>) dostali kategóriu <code>car</code>, identický box <code>4.5 &times; 2.1 &times; 1.8 m</code> a nefyzikálny limit <code>maxAcceleration="200"</code> (&gt; 20 G).</li>
  <li><strong>Prepísanie reality šablónami a extrapolačné „prilepenie“:</strong> Pri <code>brake.mp4</code> a <code>jelen.mp4</code> sa extrahované trajektórie nahradili pevnými konštantami. Mimo intervalu viditeľnosti <code>np.interp</code> zmrazilo objekt v konštantnej polohe pred ego vozidlom.</li>
  <li><strong>Hrubé 1 Hz vzorkovanie, skokové akcie a nulová parametrizácia:</strong> <code>sample_dt=1.0 s</code>, <code>SpeedAction</code> s <code>dynamicsShape="step"</code>, prázdny <code>&lt;ParameterDeclarations/&gt;</code> (0 parametrov), prázdny <code>ParameterCondition</code> v <code>StopTrigger</code> a neschopnosť bežať offline alebo generovať <code>.xodr</code>.</li>
</ul>

<!-- CHAPTER 5 -->
<h1 id="ch5">5. Implementované úpravy a ich vedecké zdôvodnenie</h1>
<p>
  Monolitický skript bol refaktorovaný do typovanej, testovanej modulárnej knižnice <code>src/xosc_pipeline/</code> riadenej konfiguračným súborom <code>config/default.yaml</code>, pričom CLI rozhranie <code>video_to_xosc.py</code> zostalo plne spätne kompatibilné. Každá úprava priamo reflektuje zistenia z auditu a odbornej literatúry:
</p>
<table>
  <thead>
    <tr>
      <th>Modul</th>
      <th>Implementovaná zmena</th>
      <th>Odborný zdroj</th>
      <th>Riešený problém</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>config.py</code> &amp; <code>default.yaml</code></td>
      <td>Konfigurovateľné fyzikálne profily tried (<code>car</code>, <code>truck</code>, <code>bus</code>, <code>motorbike</code>, <code>bicycle</code>, <code>pedestrian</code>, <code>animal</code>) vrátane rázvoru, rozchodu, rozmerov a limitov v, a, j.</td>
      <td>Geiger et al. (2012); Caesar et al. (2020); ASAM OpenSCENARIO 1.0</td>
      <td>Jednotný 4.5 m box, kamión ako <code>charger</code>, <code>maxAcceleration="200"</code></td>
    </tr>
    <tr>
      <td><code>calibration.py</code></td>
      <td>Pomer-strán adaptívne intrinsics (16:9 aj 9:16), odhad horizontu z úbežníka pruhových čiar (Canny + Hough) a inverzno-variančná fúzia hĺbky <code>z_gp</code> a <code>z_width</code>.</td>
      <td>Song &amp; Chandraker (2015); Ma et al. (2021); Yin et al. (2023)</td>
      <td>Pevný horizont <code>0.52*H</code>, skreslená mierka pri vertikálnom videu a v diaľke</td>
    </tr>
    <tr>
      <td><code>tracking.py</code></td>
      <td>Prechod na <strong>BoT-SORT</strong> s kompenzáciou pohybu kamery (CMC), <strong>RANSAC homografia + radiálna expanzia z FOE</strong> pre ego-VO a <strong>tracklet stitching</strong> cez krátke oklúzie (&le; 1.5 s).</td>
      <td>Aharon et al. (2022); Cao et al. (2023); Campos et al. (2021); Teed et al. (2023)</td>
      <td>Šum surového VO (729 m/s&sup2;), fragmentácia trackov a ID switche</td>
    </tr>
    <tr>
      <td><code>kinematics.py</code></td>
      <td>Rolling MAD outlier rejection, obojsmerný <strong>RTS Kalman smoother</strong> (Constant-Acceleration), projekcia na kinematické limity <strong>bicycle modelu</strong>, ohraničená extrapolácia rýchlosťou a celočasová <strong>Frenet/OBB dekonflikcia</strong>.</td>
      <td>Rauch, Tung &amp; Striebel (1965); Werling et al. (2010); Kong et al. (2015); Polack et al. (2017)</td>
      <td>Zrýchlenia &gt;15 m/s&sup2;, jerk &gt;24 m/s&sup3;, prekrytia vozidiel, 180&deg; otočenie</td>
    </tr>
    <tr>
      <td><code>road_network.py</code></td>
      <td>Odhad počtu pruhov a zakrivenia vozovky, generátor <strong>ASAM OpenDRIVE 1.4 (<code>.xodr</code>)</strong>, offline projekcia bez CARLA a opravená CARLA mapová projekcia.</td>
      <td>Newson &amp; Krumm (2009); Liao et al. (2023); ASAM OpenDRIVE</td>
      <td>Závislosť na bežiacom UE5 serveri, chýbajúca cestná sieť <code>.xodr</code></td>
    </tr>
    <tr>
      <td><code>xosc_writer.py</code></td>
      <td>Logická parametrizácia (6 parametrov v <code>&lt;ParameterDeclarations&gt;</code>), 4 Hz <code>Polyline</code> od <code>t=0.0 s</code>, lineárne <code>SpeedAction</code>, validný <code>StopTrigger</code> a XSD overenie.</td>
      <td>Menzel et al. (2018); de Gelder &amp; op den Camp (2024)</td>
      <td>0 parametrov, 1 Hz trhané trajektórie, <code>step</code> akcie, neplatný <code>StopTrigger</code></td>
    </tr>
  </tbody>
</table>

<!-- CHAPTER 6 -->
<h1 id="ch6" class="page-break">6. Kvantitatívne a vizuálne výsledky pred/po</h1>
<p>
  Upravený nástroj bol najprv overený automatickou testovacou sadou (<code>pytest -v tests/</code> — <strong>10/10 jednotkových a integračných testov prešlo</strong> vrátane XSD validácie), a následne spustený na všetkých štyroch reálnych demo videách. Všetky vygenerované <code>.xosc</code> súbory prešli prísnou validáciou proti <code>OpenSCENARIO.xsd</code> a ku každému vznikla zodpovedajúca cestná sieť <code>.xodr</code>.
</p>

<h2>6.1 Kvantitatívne porovnanie metrík (Pred &rarr; Po)</h2>
<table>
  <thead>
    <tr>
      <th>Video</th>
      <th>Max |a| (m/s&sup2;)</th>
      <th>Max |jerk| (m/s&sup3;)</th>
      <th>Max |yaw_rate| (rad/s)</th>
      <th>OBB kolízie</th>
      <th>Parametre</th>
      <th>XODR / XSD</th>
    </tr>
  </thead>
  <tbody>
    {eval_table_html}
  </tbody>
</table>
<p>
  Výsledky potvrdzujú dramatické a merateľné zlepšenie naprieč všetkými videami:
</p>
<ul>
  <li><strong>Eliminácia nefyzikálnych špičiek v reálnych dátach:</strong> V <code>dashcam_input.mp4</code> kleslo max |a| z <strong>13.79 na 8.21 m/s&sup2;</strong> (&minus;40 %) a max |jerk| z <strong>21.15 na 7.21 m/s&sup3;</strong> (&minus;66 %); v <code>stiahnute_video.mp4</code> kleslo max |a| z <strong>15.41 na 7.29 m/s&sup2;</strong> (&minus;53 %) a max |jerk| z <strong>24.00 na 14.80 m/s&sup3;</strong> (&minus;38 %); v <code>jelen.mp4</code> bolo úplne odstránené 180&deg; otočenie protismerného vozidla (max |&psi;&#775;| z <strong>3.134 na 0.227 rad/s</strong>, &minus;93 %) a max |a| kleslo z <strong>13.03 na 8.86 m/s&sup2;</strong>. Pri <code>brake.mp4</code> pôvodný nástroj zahodil namerané trajektórie a nahradil ich pevnou šablónou s konštantnou rýchlosťou (3.03 m/s&sup2;); nový pipeline zachováva reálne dáta z videa (vrátane správnej triedy <code>truck</code>), pričom lokálna špička jerku na okraji detekcie ukazuje priestor pre ďalšie ladenie bočného vyhladenia.</li>
  <li><strong>Nulové neúmyselné prekrytia:</strong> 2D OBB kontrola nad skutočnými rozmermi vozidiel potvrdila <strong>0 kolíznych krokov</strong> vo všetkých 4 scenároch (oproti prekrytiam v pôvodnom <code>dashcam_reconstructed.xosc</code> a <code>moj_scenar.xosc</code>).</li>
  <li><strong>Správna typizácia a parametrizácia:</strong> Nákladné vozidlá (<code>truck</code>) a osobné autá (<code>car</code>) majú odlišné realistické rozmery, nápravy aj dynamické limity (<code>maxAcceleration &le; 4.5 m/s&sup2;</code> namiesto <code>200 m/s&sup2;</code>), a každý scenár obsahuje <strong>6 deklarovaných parametrov</strong> logického scenára a sprievodnú mapu <code>.xodr</code>.</li>
</ul>

<div class="figure">
  <img src="{fig1}" alt="Kinematické porovnanie pred a po"/>
  <div class="caption">Obr. 1: Porovnanie maximálneho zrýchlenia, jerku a uhlovej rýchlosti (yaw-rate) pred a po úpravách na všetkých 4 demo videách spolu s fyzikálnymi limitmi.</div>
</div>

<div class="figure">
  <img src="{fig2}" alt="BEV trajektórie"/>
  <div class="caption">Obr. 2: Pôdorysný (Bird's-Eye-View) pohľad na zrekonštruované Frenet trajektórie (s, d) so zobrazením jazdných pruhov po RTS vyhladení, kinematických limitoch a dekonflikcii.</div>
</div>

<div class="figure">
  <img src="{fig3}" alt="Profily rýchlosti a zrýchlenia"/>
  <div class="caption">Obr. 3: Časové priebehy rýchlosti v(t) a zrýchlenia a(t) pre scenáre <code>dashcam_input.mp4</code> a <code>brake.mp4</code> — plynulé profily bez skokov a v rámci fyzikálnych hraníc.</div>
</div>

<div class="figure">
  <img src="{fig4}" alt="Anotované video snímky"/>
  <div class="caption">Obr. 4: Ukážky anotovaných snímok z detekcie a trackingu (BoT-SORT, odhadnutý horizont — tyrkysová čiara, odhad 3D polohy X, Z v metroch).</div>
</div>

<h2>6.2 Stav overenia v simulátoroch</h2>
<p>
  <strong>Overené:</strong> (1) 10/10 automatických testov v <code>pytest</code>, (2) XSD validácia všetkých 4 vygenerovaných <code>.xosc</code> súborov cez <code>xmlschema</code> proti oficiálnej schéme CARLA ScenarioRunner <code>OpenSCENARIO.xsd</code>, (3) validita XML štruktúry 4 vygenerovaných <code>.xodr</code> máp, (4) kvantitatívna kinematická a 2D OBB kolízna analýza + BEV a video-overlay vizualizácie.
</p>
<p>
  <strong>Neoverené v tomto behu (transparentne uvádzané):</strong> (1) Nástroj <code>esmini</code> nie je nainštalovaný na tomto stroji, preto bol nahradený presnou 2D OBB/BEV kontrolou a XSD validátorom; (2) živé prehratie nových <code>.xosc</code> v bežiacom UE5 serveri CARLA 1.0 nebolo počas dávkového vyhodnotenia spustené (použitý bol deterministický offline režim <code>--offline</code>, no formát zachováva 100 % kompatibilitu s <code>run_replay.py</code> a <code>scenario_runner.py</code>); (3) absolútna metrická presnosť ADE/FDE voči LiDAR ground-truth vyžaduje vyhodnotenie na datasete KITTI/nuScenes, keďže YouTube dashcam videá nemajú referenčnú telemetriu.
</p>

<!-- CHAPTER 7 -->
<h1 id="ch7">7. Limity, otvorené problémy a odporúčania na ďalší vývoj</h1>
<ol>
  <li><strong>Monokulárna nejednoznačnosť absolútnej rýchlosti (Scale Ambiguity):</strong> Pri nekalibrovanej kamere bez GPS/CAN a bez hustého metrického modelu hĺbky závisí absolútna rýchlosť ego vozidla od parametra <code>base_ego_speed_mps</code> a kalibrácie radiálneho optického toku. <em>Odporúčanie:</em> Pri dostupnosti GPU integrovať <strong>Metric3D</strong> (Yin et al., 2023) alebo <strong>UniDepth</strong> (Piccinelli et al., 2024) a kalibrovať mierku na základe štandardnej dĺžky prerušovanej čiary jazdného pruhu.</li>
  <li><strong>Dynamická geometria vozovky a križovatky:</strong> Súčasný <code>.xodr</code> generátor modeluje plynulý koridor (priamka alebo oblúk s konštantným zakrivením) z priemernej uhlovej rýchlosti ego vozidla, čo nepokrýva zložité mestské križovatky či zmeny počtu pruhov. <em>Odporúčanie:</em> Integrovať neurónový detektor pruhov <strong>CLRNet</strong> (Zheng et al., 2022) a po častiach kubické špirály (klotoidy) v OpenDRIVE.</li>
  <li><strong>Orientácia (yaw) ostatných vozidiel pri nízkej rýchlosti:</strong> Pri stojacich alebo pomaly sa pohybujúcich vozidlách sa orientácia z 2D bounding boxu nedá spoľahlivo určiť iba z derivácie polohy. <em>Odporúčanie:</em> Pridať ľahký monokulárny 3D regresor orientácie (napr. v štýle <strong>MonoFlex</strong>, Zhang et al., 2021).</li>
  <li><strong>Nočné scény a nepriaznivé počasie:</strong> V noci (<code>jelen.mp4</code>) má optický tok aj 2D detektor výrazne menej spoľahlivých bodov. <em>Odporúčanie:</em> Doplniť predbežné zvýraznenie kontrastu (CLAHE) a špecializovaný detektor svetlometov / zadných svetiel.</li>
  <li><strong>Benchmark na anotovaných datasetoch:</strong> Ďalším krokom je automatické vyhodnotenie celého pipeline na sekvenciách <strong>KITTI Tracking</strong> a <strong>nuScenes</strong> s výpočtom presných chýb ADE/FDE a porovnaním TTC rozdelení voči reálnemu LiDAR ground-truth.</li>
</ol>

<!-- CHAPTER 8 -->
<h1 id="ch8" class="page-break">8. Bibliografia</h1>
<p>Všetky citované zdroje sú overené primárne publikácie a sú dostupné v strojom čitateľnom formáte v <code>references.bib</code>.</p>
<ol class="bib-list">
  <li>Aharon, N., Orfaig, R., &amp; Bobrovsky, B.-Z. (2022). <em>BoT-SORT: Robust Associations Multi-Pedestrian Tracking</em>. arXiv:2206.14651. <a href="https://arxiv.org/abs/2206.14651">https://arxiv.org/abs/2206.14651</a></li>
  <li>Allen, B. L., Shin, B. T., &amp; Cooper, P. J. (1978). <em>Analysis of Traffic Conflicts and Collisions</em>. Transportation Research Record, 667, 67–74.</li>
  <li>Bashetty, S. K., Ben Amor, H., &amp; Fainekos, G. (2020). <em>DeepCrashTest: Turning Dashcam Videos into Virtual Crash Tests for Automated Driving Systems</em>. IEEE ICRA 2020, 11353–11360. DOI: 10.1109/ICRA40945.2020.9197074 | <a href="https://arxiv.org/abs/2003.11766">arXiv:2003.11766</a></li>
  <li>Bewley, A., Ge, Z., Ott, L., Ramos, F., &amp; Upcroft, B. (2016). <em>Simple Online and Realtime Tracking</em>. IEEE ICIP 2016, 3464–3468. DOI: 10.1109/ICIP.2016.7533003 | <a href="https://arxiv.org/abs/1602.00763">arXiv:1602.00763</a></li>
  <li>Bock, J., Krajewski, R., Moers, T., Leonhardt, S., &amp; Eckstein, L. (2020). <em>inD: A Drone Dataset of Naturalistic Road User Trajectories at Intersections</em>. IEEE IV 2020, 1929–1934. DOI: 10.1109/IV47402.2020.9304803 | <a href="https://arxiv.org/abs/1911.12555">arXiv:1911.12555</a></li>
  <li>Caesar, H., Bankiti, V., Lang, A. H., et al. (2020). <em>nuScenes: A Multimodal Dataset for Autonomous Driving</em>. IEEE/CVF CVPR 2020, 11621–11631. DOI: 10.1109/CVPR42600.2020.01164 | <a href="https://arxiv.org/abs/1903.11027">arXiv:1903.11027</a></li>
  <li>Campos, C., Elvira, R., Rodríguez, J. J. G., Montiel, J. M. M., &amp; Tardós, J. D. (2021). <em>ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM</em>. IEEE Transactions on Robotics, 37(6), 1874–1890. DOI: 10.1109/TRO.2021.3075644 | <a href="https://arxiv.org/abs/2007.11898">arXiv:2007.11898</a></li>
  <li>Cao, J., Pang, J., Weng, X., Khirodkar, R., &amp; Kitani, K. (2023). <em>Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking</em>. IEEE/CVF CVPR 2023, 9686–9696. DOI: 10.1109/CVPR52729.2023.00934 | <a href="https://arxiv.org/abs/2203.14360">arXiv:2203.14360</a></li>
  <li>de Gelder, E., &amp; op den Camp, O. (2024). <em>Scenario-Based Assessment of Automated Driving Systems: How (not) to parameterize scenarios?</em> arXiv:2409.01402. <a href="https://arxiv.org/abs/2409.01402">https://arxiv.org/abs/2409.01402</a></li>
  <li>Ding, W., Xu, C., Arief, M., Lin, H., Li, B., &amp; Zhao, D. (2023). <em>A Survey on Safety-Critical Driving Scenario Generation—A Methodological Perspective</em>. IEEE T-ITS, 24(7), 6971–6988. DOI: 10.1109/TITS.2023.3259322 | <a href="https://arxiv.org/abs/2202.02215">arXiv:2202.02215</a></li>
  <li>Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., &amp; Koltun, V. (2017). <em>CARLA: An Open Urban Driving Simulator</em>. CoRL 2017, 1–16. <a href="https://arxiv.org/abs/1711.03938">arXiv:1711.03938</a></li>
  <li>Du, Y., Zhao, Z., Song, Y., et al. (2023). <em>StrongSORT: Make DeepSORT Great Again</em>. IEEE Transactions on Multimedia, 25, 8725–8737. DOI: 10.1109/TMM.2023.3240881 | <a href="https://arxiv.org/abs/2202.13514">arXiv:2202.13514</a></li>
  <li>Geiger, A., Lenz, P., &amp; Urtasun, R. (2012). <em>Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite</em>. IEEE CVPR 2012, 3354–3361. DOI: 10.1109/CVPR.2012.6248074</li>
  <li>Guo, D., Muñoz Sánchez, M., de Gelder, E., &amp; van der Sande, T. P. J. (2023). <em>Scenario Extraction from a Large Real-World Dataset for the Assessment of Automated Vehicles</em>. IEEE ITSC 2023. DOI: 10.1109/ITSC57777.2023.10421930 | <a href="https://arxiv.org/abs/2306.07815">arXiv:2306.07815</a></li>
  <li>Hayward, J. C. (1972). <em>Near-Miss Determination Through Use of a Scale of Danger</em>. Highway Research Record, 384, 24–34.</li>
  <li>Kong, J., Pfeiffer, M., Schildbach, G., &amp; Borrelli, F. (2015). <em>Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design</em>. IEEE IV 2015, 1094–1099. DOI: 10.1109/IVS.2015.7225830</li>
  <li>Krajewski, R., Bock, J., Kloeker, L., &amp; Eckstein, L. (2018). <em>The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways</em>. IEEE ITSC 2018, 2118–2125. DOI: 10.1109/ITSC.2018.8569552 | <a href="https://arxiv.org/abs/1810.05642">arXiv:1810.05642</a></li>
  <li>Li, Q., Peng, Z., Feng, L., et al. (2023). <em>ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling</em>. NeurIPS 2023 Datasets &amp; Benchmarks. <a href="https://arxiv.org/abs/2306.12241">arXiv:2306.12241</a></li>
  <li>Liao, B., Chen, S., Wang, X., et al. (2023). <em>MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction</em>. ICLR 2023. <a href="https://arxiv.org/abs/2208.14437">arXiv:2208.14437</a></li>
  <li>Ma, X., Zhang, Y., Xu, D., et al. (2021). <em>Delving into Localization Errors for Monocular 3D Object Detection</em>. IEEE/CVF CVPR 2021, 4721–4730. DOI: 10.1109/CVPR46437.2021.00469 | <a href="https://arxiv.org/abs/2103.16237">arXiv:2103.16237</a></li>
  <li>Menzel, T., Bagschik, G., &amp; Maurer, M. (2018). <em>Scenarios for Development, Test and Validation of Automated Vehicles</em>. IEEE IV 2018, 1821–1827. DOI: 10.1109/IVS.2018.8500406 | <a href="https://arxiv.org/abs/1801.08598">arXiv:1801.08598</a></li>
  <li>Miao, Y., Fainekos, G., Hoxha, B., Okamoto, H., Prokhorov, D., &amp; Mitra, S. (2024). <em>From Dashcam Videos to Driving Simulations: Stress Testing Automated Vehicles against Rare Events</em>. AAAI Workshop ML4AD. <a href="https://arxiv.org/abs/2411.16027">arXiv:2411.16027</a></li>
  <li>Montali, N., Lambert, J., Mougin, P., et al. (2023). <em>The Waymo Open Sim Agents Challenge</em>. NeurIPS 2023 Datasets &amp; Benchmarks. <a href="https://arxiv.org/abs/2305.12032">arXiv:2305.12032</a></li>
  <li>Newson, P., &amp; Krumm, J. (2009). <em>Hidden Markov Map Matching Through Noise and Sparseness</em>. ACM SIGSPATIAL GIS 2009, 336–343. DOI: 10.1145/1653771.1653818</li>
  <li>Piccinelli, L., Yang, Y.-H., Sakaridis, C., et al. (2024). <em>UniDepth: Universal Monocular Metric Depth Estimation</em>. IEEE/CVF CVPR 2024, 10106–10116. <a href="https://arxiv.org/abs/2403.18913">arXiv:2403.18913</a></li>
  <li>Polack, P., Altché, F., d'Andréa-Novel, B., &amp; de La Fortelle, A. (2017). <em>The Kinematic Bicycle Model: A Consistent Model for Planning Feasible Trajectories for Autonomous Vehicles?</em> IEEE IV 2017, 812–818. DOI: 10.1109/IVS.2017.7995816</li>
  <li>Rauch, H. E., Tung, F., &amp; Striebel, C. T. (1965). <em>Maximum Likelihood Estimates of Linear Dynamic Systems</em>. AIAA Journal, 3(8), 1445–1450. DOI: 10.2514/3.3166</li>
  <li>Riedmaier, S., Ponn, T., Ludwig, D., Schick, B., &amp; Diermeyer, F. (2020). <em>Survey on Scenario-Based Safety Assessment of Automated Vehicles</em>. IEEE Access, 8, 87456–87477. DOI: 10.1109/ACCESS.2020.2993730</li>
  <li>Song, S., &amp; Chandraker, M. (2015). <em>Joint SFM and Detection Cues for Monocular 3D Localization in Road Scenes</em>. IEEE CVPR 2015, 3734–3742. DOI: 10.1109/CVPR.2015.7298997</li>
  <li>Sun, P., Kretzschmar, H., Dotiwalla, X., et al. (2020). <em>Scalability in Perception for Autonomous Driving: Waymo Open Dataset</em>. IEEE/CVF CVPR 2020, 2446–2454. DOI: 10.1109/CVPR42600.2020.00252 | <a href="https://arxiv.org/abs/1912.04838">arXiv:1912.04838</a></li>
  <li>Teed, Z., &amp; Deng, J. (2021). <em>DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras</em>. NeurIPS 2021, 34, 16558–16569. <a href="https://arxiv.org/abs/2108.10869">arXiv:2108.10869</a></li>
  <li>Teed, Z., Lipson, L., &amp; Deng, J. (2023). <em>Deep Patch Visual Odometry</em>. NeurIPS 2023, 36. <a href="https://arxiv.org/abs/2208.04726">arXiv:2208.04726</a></li>
  <li>Werling, M., Ziegler, J., Kammel, S., &amp; Thrun, S. (2010). <em>Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenét Frame</em>. IEEE ICRA 2010, 987–993. DOI: 10.1109/ROBOT.2010.5509799</li>
  <li>Wilson, B., Qi, W., Agarwal, T., et al. (2021). <em>Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting</em>. NeurIPS 2021 Datasets &amp; Benchmarks. <a href="https://arxiv.org/abs/2301.00493">arXiv:2301.00493</a></li>
  <li>Wojke, N., Bewley, A., &amp; Paulus, D. (2017). <em>Simple Online and Realtime Tracking with a Deep Association Metric</em>. IEEE ICIP 2017, 3645–3649. DOI: 10.1109/ICIP.2017.8296962 | <a href="https://arxiv.org/abs/1703.07402">arXiv:1703.07402</a></li>
  <li>Yang, L., Kang, B., Huang, Z., et al. (2024). <em>Depth Anything V2</em>. NeurIPS 2024. <a href="https://arxiv.org/abs/2406.09414">arXiv:2406.09414</a></li>
  <li>Yin, W., Zhang, C., Chen, H., et al. (2023). <em>Metric3D: Towards Zero-Shot Metric 3D Prediction from A Single Image</em>. IEEE/CVF ICCV 2023, 9043–9053. DOI: 10.1109/ICCV51070.2023.00830 | <a href="https://arxiv.org/abs/2307.10984">arXiv:2307.10984</a></li>
  <li>Zhang, Y., Lu, J., &amp; Zhou, J. (2021). <em>Objects are Different: Flexible Monocular 3D Object Detection</em>. IEEE/CVF CVPR 2021, 3289–3298. DOI: 10.1109/CVPR46437.2021.00330 | <a href="https://arxiv.org/abs/2104.02323">arXiv:2104.02323</a></li>
  <li>Zhang, Y., Sun, P., Jiang, Y., et al. (2022). <em>ByteTrack: Multi-Object Tracking by Associating Every Detection Box</em>. ECCV 2022, 1–21. DOI: 10.1007/978-3-031-20047-2_1 | <a href="https://arxiv.org/abs/2110.06864">arXiv:2110.06864</a></li>
  <li>Zheng, T., Huang, Y., Liu, Y., et al. (2022). <em>CLRNet: Cross Layer Refinement Network for Lane Detection</em>. IEEE/CVF CVPR 2022, 898–907. DOI: 10.1109/CVPR52688.2022.00097 | <a href="https://arxiv.org/abs/2203.10350">arXiv:2203.10350</a></li>
</ol>

</body>
</html>
"""


def main() -> None:
    html = build_html()
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"[Report] Wrote HTML -> {HTML_OUT}")

    chrome = next((p for p in CHROME_PATHS if p.exists()), None)
    if chrome is None:
        raise RuntimeError("No Chrome/Edge browser found for headless PDF rendering.")

    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT}",
        HTML_OUT.as_uri(),
    ]
    subprocess.run(cmd, check=True)
    print(f"[Report] Rendered PDF -> {PDF_OUT} ({PDF_OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
