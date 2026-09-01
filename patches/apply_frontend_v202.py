from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "chordscope" / "frontend" / "index.html"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Frontend patch anchor missing: {label}")
    return text.replace(old, new, 1)


def main() -> int:
    text = HTML.read_text(encoding="utf-8")
    text = text.replace("CHORD<b>SCOPE</b> DESKTOP 2.0", "CHORD<b>SCOPE</b> DESKTOP 2.0.2", 1)

    text = replace_once(
        text,
        '<button class="action primary" id="analyzeBtn" disabled>ANALIZAR</button>\n        <span class="progressPct" id="analysisPct" title="Progreso del análisis">0%</span>',
        '<button class="action primary" id="analyzeBtn" disabled>ANALIZAR</button>\n'
        '        <select class="action" id="analysisProfile" title="Perfil de procesamiento">\n'
        '          <option value="fast">RÁPIDO</option>\n'
        '          <option value="balanced" selected>EQUILIBRADO</option>\n'
        '          <option value="deep">PROFUNDO</option>\n'
        '        </select>\n'
        '        <button class="action" id="cancelAnalyzeBtn" disabled>CANCELAR</button>\n'
        '        <span class="progressPct" id="analysisPct" title="Progreso del análisis">0%</span>',
        "analysis controls",
    )

    text = replace_once(
        text,
        '    desktopBridge.analysisProgress.connect((p,m)=>setAnalysisProgress(p,m));\n    desktopBridge.analysisReady.connect(applyDesktopAnalysis);\n    desktopBridge.error.connect(handleDesktopError);',
        '    desktopBridge.analysisProgress.connect((p,m)=>setAnalysisProgress(p,m));\n'
        '    desktopBridge.analysisReady.connect(applyDesktopAnalysis);\n'
        '    if(desktopBridge.analysisCancelled) desktopBridge.analysisCancelled.connect(handleDesktopCancelled);\n'
        '    desktopBridge.error.connect(handleDesktopError);\n'
        '    try{ desktopBridge.setAnalysisProfile(els.analysisProfile?.value||"balanced"); }catch(_){ }',
        "desktop bridge signals",
    )

    text = replace_once(
        text,
        'function handleDesktopError(payload){\n  let e={message:String(payload||"Error desconocido")};',
        'function handleDesktopCancelled(payload){\n'
        '  let e={message:"Análisis cancelado"};\n'
        '  try{e=JSON.parse(payload)}catch(_){ }\n'
        '  els.analyzeBtn.disabled=false;\n'
        '  if(els.cancelAnalyzeBtn)els.cancelAnalyzeBtn.disabled=true;\n'
        '  setAnalysisProgress(0,e.message||"Análisis cancelado");\n'
        '  toast(e.message||"Análisis cancelado");\n'
        '}\n'
        'function handleDesktopError(payload){\n  let e={message:String(payload||"Error desconocido")};',
        "cancel handler",
    )

    text = text.replace(
        '  els.analyzeBtn.disabled=false;\n  setAnalysisProgress(0,"Error");',
        '  els.analyzeBtn.disabled=false;\n  if(els.cancelAnalyzeBtn)els.cancelAnalyzeBtn.disabled=true;\n  setAnalysisProgress(0,"Error");',
        1,
    )

    text = replace_once(
        text,
        'function applyDesktopAnalysis(payload){\n  els.analyzeBtn.disabled=false;',
        'function applyDesktopAnalysis(payload){\n  els.analyzeBtn.disabled=false;\n  if(els.cancelAnalyzeBtn)els.cancelAnalyzeBtn.disabled=true;',
        "analysis ready controls",
    )

    text = replace_once(
        text,
        'els.analyzeBtn.addEventListener("click",()=>{\n  if(desktopBridge&&desktopAudioPath){\n    els.analyzeBtn.disabled=true;\n    setAnalysisProgress(0,"Iniciando motores Python");\n    desktopBridge.analyzeAudio(desktopAudioPath);',
        'els.analyzeBtn.addEventListener("click",()=>{\n'
        '  if(desktopBridge&&desktopAudioPath){\n'
        '    els.analyzeBtn.disabled=true;\n'
        '    if(els.cancelAnalyzeBtn)els.cancelAnalyzeBtn.disabled=false;\n'
        '    try{desktopBridge.setAnalysisProfile(els.analysisProfile?.value||"balanced");}catch(_){ }\n'
        '    setAnalysisProgress(0,`Iniciando motores Python · ${els.analysisProfile?.selectedOptions?.[0]?.text||"EQUILIBRADO"}`);\n'
        '    desktopBridge.analyzeAudio(desktopAudioPath);',
        "analyze click",
    )

    text = replace_once(
        text,
        'els.exportMidiBtn.addEventListener("click",exportMidi);',
        'if(els.analysisProfile)els.analysisProfile.addEventListener("change",()=>{\n'
        '  try{if(desktopBridge)desktopBridge.setAnalysisProfile(els.analysisProfile.value);}catch(_){ }\n'
        '});\n'
        'if(els.cancelAnalyzeBtn)els.cancelAnalyzeBtn.addEventListener("click",()=>{\n'
        '  if(!desktopBridge)return;\n'
        '  els.cancelAnalyzeBtn.disabled=true;\n'
        '  setAnalysisProgress(0,"Cancelación solicitada · esperando cierre seguro");\n'
        '  try{desktopBridge.cancelAnalysis();}catch(_){ }\n'
        '});\n'
        'els.exportMidiBtn.addEventListener("click",exportMidi);',
        "profile/cancel listeners",
    )

    HTML.write_text(text, encoding="utf-8")
    print("Frontend v2.0.2 patch applied:", HTML)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
