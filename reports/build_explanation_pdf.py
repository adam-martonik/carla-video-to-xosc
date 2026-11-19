#!/usr/bin/env python
"""
Generate publication-quality PDF explaining the principles and architecture
of the `video_to_xosc.py` pipeline.
Outputs: `Ako_a_na_akom_principe_funguje_skript.pdf` in project root.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
EVAL_DIR = REPORTS_DIR / "evaluation"
HTML_OUT = REPORTS_DIR / "princip_fungovania_skriptu.html"
PDF_OUT = ROOT / "Ako_a_na_akom_principe_funguje_skript.pdf"

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
    fig1 = img_to_data_uri(EVAL_DIR / "fig1_kinematics_comparison.png")
    fig2 = img_to_data_uri(EVAL_DIR / "fig2_bev_trajectories.png")
    fig3 = img_to_data_uri(EVAL_DIR / "fig3_velocity_accel_profiles.png")
    fig4 = img_to_data_uri(EVAL_DIR / "fig4_video_overlay.png")

    html = f"""<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="utf-8">
  <title>Princíp a Architektúra Pipeline: video_to_xosc.py</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm 18mm 16mm;
      @bottom-right {{
        content: counter(page);
      }}
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      font-size: 10.5pt;
      line-height: 1.55;
      color: #1a202c;
      background: #ffffff;
      margin: 0;
      padding: 0;
    }}
    h1, h2, h3, h4 {{
      color: #0f172a;
      font-weight: 700;
      page-break-after: avoid;
    }}
    h1 {{
      font-size: 20pt;
      line-height: 1.25;
      border-bottom: 3px solid #2563eb;
      padding-bottom: 8px;
      margin-top: 0;
      margin-bottom: 12px;
    }}
    .subtitle {{
      font-size: 11.5pt;
      color: #475569;
      margin-bottom: 22px;
      font-weight: 500;
    }}
    h2 {{
      font-size: 13pt;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 4px;
      margin-top: 20px;
      margin-bottom: 8px;
      color: #1e3a8a;
    }}
    h3 {{
      font-size: 11pt;
      margin-top: 14px;
      margin-bottom: 6px;
      color: #334155;
    }}
    p {{
      margin: 0 0 8px 0;
      text-align: justify;
    }}
    ul, ol {{
      margin: 4px 0 10px 22px;
      padding: 0;
    }}
    li {{
      margin-bottom: 4px;
    }}
    code {{
      background: #f1f5f9;
      color: #0f172a;
      padding: 2px 5px;
      border-radius: 4px;
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 9.5pt;
    }}
    pre {{
      background: #0f172a;
      color: #f8fafc;
      padding: 10px 14px;
      border-radius: 6px;
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 8.5pt;
      overflow-x: auto;
      page-break-inside: avoid;
      line-height: 1.4;
    }}
    .stage-card {{
      background: #f8fafc;
      border-left: 4px solid #2563eb;
      padding: 10px 14px;
      margin: 10px 0 14px 0;
      border-radius: 0 6px 6px 0;
      page-break-inside: avoid;
    }}
    .stage-title {{
      font-weight: bold;
      color: #1e40af;
      font-size: 11pt;
      margin-bottom: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 9pt;
      page-break-inside: avoid;
    }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 6px 8px;
      text-align: left;
    }}
    th {{
      background: #f1f5f9;
      color: #1e293b;
      font-weight: 600;
    }}
    .figure {{
      text-align: center;
      margin: 14px 0;
      page-break-inside: avoid;
    }}
    .figure img {{
      max-width: 98%;
      height: auto;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .caption {{
      font-size: 8.5pt;
      color: #64748b;
      margin-top: 5px;
      font-style: italic;
    }}
    .page-break {{
      page-break-before: always;
    }}
    .tag {{
      display: inline-block;
      background: #e0e7ff;
      color: #3730a3;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 8pt;
      font-weight: 600;
      margin-right: 4px;
    }}
  </style>
</head>
<body>

  <h1>CARLA Video-to-OpenSCENARIO Pipeline</h1>
  <div class="subtitle">
    <strong>Technický manuál a vysvetlenie princípu fungovania skriptu <code>video_to_xosc.py</code></strong><br>
    Automatická rekonštrukcia fyzikálne korektných 3D scenárov (ASAM OpenSCENARIO 1.0 &amp; OpenDRIVE) z monokulárneho dashcam videa.
  </div>

  <h2>1. Prehľad a Základná Myšlienka</h2>
  <p>
    Cieľom nástroja <code>video_to_xosc.py</code> je preklenúť priepasť medzi bežnými neštruktúrovanými videozáznamami z palubných autokamier (dashcam) a formálnymi simulačnými prostrediami pre autonómne vozidlá (napr. CARLA Simulator). 
    Z 2D pixelových dát videa dokáže skript automaticky zrekonštruovať plnohodnotný 3D scenár vo formáte <strong>ASAM OpenSCENARIO 1.0 (.xosc)</strong> a príslušnú cestnú sieť <strong>ASAM OpenDRIVE (.xodr)</strong>.
  </p>
  <p>
    Na rozdiel od naivných prístupov, ktoré vedú k fyzikálne nereálnym „skákavým“ trajektóriám, teleportáciám a telemetrickým anomáliám, tento pipeline garantuje:
  </p>
  <ul>
    <li><strong>Fyzikálnu a kinematickú vierohodnosť:</strong> Pohyb vozidiel podlieha kinematickému bicyklovému modelu (Bicycle Model) a striktným limitom zrýchlenia, jerku (trhnutia) a rýchlosti stáčania (yaw-rate).</li>
    <li><strong>Spatio-temporálnu integritu bez kolízií:</strong> Detekcia a eliminácia vzájomných prienikov objektov v čase pomocou orientovaných 2D bounding boxov (OBB) a Frenetových súradníc.</li>
    <li><strong>Štandardizovanú interoperabilitu:</strong> Validácia výstupov voči oficiálnej XSD schéme ASAM OpenSCENARIO a generovanie OpenDRIVE logickej geometrie cesty.</li>
  </ul>

  <h2>2. End-to-End Architektúra Pipeline</h2>
  <p>
    Celý proces je rozdelený do 7 modulárnych fáz implementovaných v balíku <code>src/xosc_pipeline/</code>:
  </p>

  <div class="stage-card">
    <div class="stage-title">Fáza 1: Kalibrácia kamery a vanishing pointu (VP)</div>
    <span class="tag">src/xosc_pipeline/calibration.py</span>
    <p>
      Skript analyzuje geometriu scény detekciou jazdných pruhov a hrán vozovky pomocou Houghovej transformácie. Z priesečníka línii odhaduje úbežník (Vanishing Point, VP) a horizont. Na základe VP, predpokladanej výšky kamery <code>h_cam ≈ 1.35 m</code> a ohniskovej vzdialenosti zostavuje homografickú maticu <strong>Inverznej perspektívnej transformácie (IPM)</strong>, ktorá premieta body zo spodku ohraničujúcich rámcov (kontaktné body s vozovkou) priamo do metrických 3D súradníc povrchu cesty.
    </p>
  </div>

  <div class="stage-card">
    <div class="stage-title">Fáza 2: Detekcia a sledovanie objektov (BoT-SORT + CMC)</div>
    <span class="tag">src/xosc_pipeline/tracking.py</span>
    <p>
      V každom rámci videa deteguje objekty model <strong>YOLO11</strong> (autá, nákladné autá, autobusy, motocykle, chodci, zver). Pre sledovanie v čase sa využíva algoritmus <strong>BoT-SORT</strong> doplnený o <strong>Camera Motion Compensation (CMC)</strong>. CMC kompenzuje natriasanie a náklon dashcam kamery pomocou afinného vyrovnania pozadia (GMC), čím zabraňuje strate identity objektov. Nasleduje obojstranné zošívanie (tracklet stitching) pre preklenutie krátkodobých oklúzií.
    </p>
  </div>

  <div class="figure">
    <img src="{fig4}" alt="Vizuálna anotácia">
    <div class="caption">Obrázok 1: Detekcia objektov a vizualizácia stopy trajektórie v reálnom dashcam videu.</div>
  </div>

  <div class="page-break"></div>

  <div class="stage-card">
    <div class="stage-title">Fáza 3: Odhad pohybu kamery a vlastného vozidla (Ego Visual Odometry)</div>
    <span class="tag">src/xosc_pipeline/tracking.py</span>
    <p>
      Vlastné vozidlo (ego/hero) nemá v dashcam videu vonkajší bounding box. Jeho rýchlosť a pozdĺžny pohyb sa odhadujú pomocou robustnej vizuálnej odometrie. Z optického toku (Farneback / Lucas-Kanade) v spodnom segmente vozovky (maskovanom mimo detegovaných áut) algoritmus RANSAC eliminuje šum a počíta reálnu doprednú rýchlosť <code>v_ego(t)</code> a polohu <code>x_ego(t)</code>.
    </p>
  </div>

  <div class="stage-card">
    <div class="stage-title">Fáza 4: Kinematická regularizácia (RTS Kalman Smoother &amp; Bicycle Model)</div>
    <span class="tag">src/xosc_pipeline/kinematics.py</span>
    <p>
      Surové trajektórie z monokulárneho odhadu vykazujú vysokofrekvenčný šum. Skript nasadzuje:
    </p>
    <ul>
      <li><strong>MAD Outlier Rejection:</strong> Odfiltrovanie fyzikálne nemožných skokov pomocou mediánu absolútnych odchýlok.</li>
      <li><strong>Rauch-Tung-Striebel (RTS) Kalman Smoother:</strong> Obojsmerný (forward-backward) filter stavového vektora <code>[x, y, v, theta]</code>, ktorý rešpektuje kinematiku bicyklového modelu.</li>
      <li><strong>Fyzikálne saturácie:</strong> Max. pozdĺžne zrýchlenie <code>|a| &le; 6.0 m/s&sup2;</code>, núdzové brzdenie <code>a &ge; -9.0 m/s&sup2;</code>, obmedzenie jerku <code>|j| &le; 10.0 m/s&sup3;</code> a rýchlosti stáčania <code>|&omega;| &le; 0.8 rad/s</code>.</li>
    </ul>
  </div>

  <div class="figure">
    <img src="{fig1}" alt="Kinematické porovnanie">
    <div class="caption">Obrázok 2: Porovnanie surových vs. fyzikálne vyhladených (RTS Smoothed) hodnôt rýchlosti, zrýchlenia a jerku.</div>
  </div>

  <div class="stage-card">
    <div class="stage-title">Fáza 5: Spatio-temporálna dekonflikcia (OBB &amp; Frenet Frame)</div>
    <span class="tag">src/xosc_pipeline/kinematics.py</span>
    <p>
      Transformácia súradníc do Frenetovho rámca cesty <code>(s, d)</code>, kde <code>s</code> je pozdĺžny staničenie a <code>d</code> je bočný odstup od stredu pruhu. Pre každý časový krok <code>t</code> sa kontroluje prekrytie 2D orientovaných boxov vozidiel (Oriented Bounding Boxes - OBB pomocou Separating Axis Theorem). Ak je zistený nereálny prienik dvoch vozidiel, algoritmus aplikuje lokálnu elastickú penalizáciu a posunie trajektóriu tak, aby bola zachovaná bezpečná medzera bez narušenia dynamiky.
    </p>
  </div>

  <div class="figure">
    <img src="{fig2}" alt="Bird's Eye View Trajektórie">
    <div class="caption">Obrázok 3: Bird's Eye View (BEV) rekonštruované trajektórie účastníkov premávky v rovine vozovky.</div>
  </div>

  <div class="page-break"></div>

  <div class="stage-card">
    <div class="stage-title">Fáza 6: Cestná sieť OpenDRIVE (.xodr) a Map Matching</div>
    <span class="tag">src/xosc_pipeline/road_network.py</span>
    <p>
      Aby bol scenár spustiteľný v akomkoľvek simulátore, vyžaduje asociovanú cestnú sieť. Skript podporuje dva módy:
    </p>
    <ul>
      <li><strong>Offline mód:</strong> Automaticky vygeneruje syntetický <code>.xodr</code> súbor reprezentujúci presný úsek cesty (jazdné pruhy, stredovú čiaru, šírku vozovky a referenčnú geometriu).</li>
      <li><strong>CARLA mód:</strong> Premietne trajektórie na existujúcu mapu CARLA (napr. <code>Town04_Opt</code>) pomocou map-matchingu k najbližším legitímnym waypointom a jazdným pruhom.</li>
    </ul>
  </div>

  <div class="stage-card">
    <div class="stage-title">Fáza 7: Export do ASAM OpenSCENARIO 1.0 (.xosc) a XSD Validácia</div>
    <span class="tag">src/xosc_pipeline/xosc_writer.py</span>
    <p>
      Finálny generátor vytvára XML štruktúru striktne kompatibilnú s <strong>ASAM OpenSCENARIO 1.0</strong>:
    </p>
    <ul>
      <li><code>&lt;FileHeader&gt;</code> s metadátami a časovou pečiatkou.</li>
      <li><code>&lt;ParameterDeclarations&gt;</code> pre parametrické škálovanie rýchlostí a časovania.</li>
      <li><code>&lt;Entities&gt;</code>: Definícia ego vozidla (<code>hero</code>) a NPC aktérov z katalógov vozidiel CARLA.</li>
      <li><code>&lt;Story&gt; / &lt;Act&gt; / &lt;ManeuverGroup&gt;</code>: Časovo synchronizované trajektórie <code>RoutingAction -&gt; FollowTrajectoryAction</code> s polyline waypointmi a atribútmi <code>time</code>.</li>
      <li>Formálna validácia výstupu voči oficiálnej schéme <code>OpenSCENARIO.xsd</code> pred uložením na disk.</li>
    </ul>
  </div>

  <div class="figure">
    <img src="{fig3}" alt="Profily zrýchlenia a rýchlostí">
    <div class="caption">Obrázok 4: Detailné profily rýchlostí, zrýchlení a stáčania rekonštruovaných aktérov v čase.</div>
  </div>

  <h2>3. Spustenie a Použitie v Praxi</h2>

  <h3>A. Základné spracovanie lokálneho videa</h3>
  <pre>python video_to_xosc.py --video "demo/volkswagen_dashcam.mp4" --output-xosc "demo/scenar.xosc"</pre>

  <h3>B. Stiahnutie z YouTube a spracovanie</h3>
  <pre>python video_to_xosc.py --youtube-url "https://youtu.be/3yis1lJDOtk" --video "demo/brake.mp4" --output-xosc "demo/scenar_brake.xosc"</pre>

  <h3>C. Čisto offline režim (bez nutnosti spustenia CARLA servera)</h3>
  <pre>python video_to_xosc.py --video "demo/dashcam_input.mp4" --offline --use-xodr-logic --output-xosc "demo/dashcam_reconstructed.xosc"</pre>

  <h3>D. Testovanie vlastného autonómneho modelu v CARLA (Closed-Loop)</h3>
  <p>
    Vygenerovaný scenár <code>.xosc</code> riadi všetkých okolitých účastníkov podľa reálneho videa. Ego vozidlo <code>hero</code> je riadené tvojím vlastným lokálnym modelom (PyTorch, ONNX, LLM/VLM agent), ktorý prijíma obraz z kamery a vracia riadenie (plyn, brzda, volant):
  </p>
  <pre># Spustenie CARLA simulátora:
./CARLA_1.0/CarlaUnreal.exe -quality-level=Epic -RenderOffScreen

# Spustenie scenára cez ScenarioRunner:
python scenario_runner/scenario_runner.py --openscenario demo/dashcam_reconstructed.xosc --reloadWorld

# Spustenie tvojho riadiaceho agenta pripojeného na hero:
python run_replay.py --scenario demo/scenar_volkswagen.json</pre>

</body>
</html>
"""
    return html


def main() -> None:
    print("[*] Generating HTML explanation...")
    html_content = build_html()
    HTML_OUT.write_text(html_content, encoding="utf-8")
    print(f"[+] Saved HTML: {HTML_OUT}")

    chrome_bin = None
    for p in CHROME_PATHS:
        if p.exists():
            chrome_bin = p
            break

    if not chrome_bin:
        raise SystemExit("[-] Neither Chrome nor Edge found on this machine.")

    print(f"[*] Compiling PDF via {chrome_bin.name} headless print-to-pdf...")
    cmd = [
        str(chrome_bin),
        "--headless=new",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT.resolve()}",
        str(HTML_OUT.resolve()),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[-] Headless print failed: {res.stderr}")
    else:
        print(f"[+] Successfully generated PDF: {PDF_OUT} ({PDF_OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
