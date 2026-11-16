"""Build an offline presentation from existing, checked scene outputs. No inference."""
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from focus_calibration import signature
from run_pipeline import load_images, load_manifest


def save_rgb(path, bgr, width=680):
    image=cv2.rotate(bgr,cv2.ROTATE_90_CLOCKWISE)
    if width: image=cv2.resize(image,(width,round(image.shape[0]*width/image.shape[1])),interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path),image):raise ValueError(f'Could not write {path}')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scene='scene-20-40-60';source=Path('results/multiscene_metric')/scene
    out=Path('results/demo_walkthrough');assets=out/'assets';assets.mkdir(parents=True,exist_ok=True)
    capture=Path('data/raw')/scene;meta=json.loads((source/'preprocessing.json').read_text())
    manifest,frames=load_manifest(capture,'lens_position_actual')
    modelraw=Path('results/focus_calibration/model.json').read_bytes();model=json.loads(modelraw)
    if hashlib.sha256((capture/'manifest.json').read_bytes()).hexdigest()!=meta['source_manifest_sha256']:raise ValueError('Capture changed')
    if hashlib.sha256(modelraw).hexdigest()!=meta['calibration_model_sha256']:raise ValueError('Calibration changed')
    if signature(manifest)!=model['signature'] or meta['registration_failed']:raise ValueError('Incompatible capture')
    images=load_images(capture,frames,1600);h,w=images[0].shape[:2];x,y,ex,ey=meta['crop']
    if len(images)!=len(meta['warps']):raise ValueError('Frame/warp mismatch')
    sweep=[]
    for i,(image,warp) in enumerate(zip(images,meta['warps'])):
        aligned=cv2.warpAffine(image,np.array(warp,np.float32),(w,h),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)[y:ey,x:ex]
        path=assets/f'focus_{i:02d}.jpg';save_rgb(path,aligned,440);sweep.append(Image.open(path).convert('RGB'))
    sweep[0].save(assets/'focus_sweep.gif',save_all=True,append_images=sweep[1:]+sweep[-2:0:-1],duration=180,loop=0)
    reference=cv2.imread(str(source/'reference_rgb.png'));aif=cv2.imread(str(source/'all_in_focus.png'))
    if reference is None or aif is None:raise ValueError('Missing RGB outputs')
    save_rgb(assets/'reference.jpg',reference);save_rgb(assets/'aif.jpg',aif)
    # Same crop around the measured Excedrin face, with surrounding printed detail.
    bbox=[round(1370*w/1920)-x,round(860*h/1440)-y,round(1760*w/1920)-x,round(1110*h/1440)-y]
    bx,by,bex,bey=bbox
    for name,image in [('reference',reference),('aif',aif)]:save_rgb(assets/f'{name}_detail.jpg',image[by:bey,bx:bex],700)
    methods=[('calibrated_focus','Calibrated focus'),('focus_scaled_relative_aif','Focus-scaled neural'),
             ('focus_neural_fusion','Confidence fusion'),('direct_lidar','LiDAR reference')]
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#313743')
    fig,axes=plt.subplots(1,4,figsize=(13,5),layout='constrained')
    coverage={}
    for ax,(key,title) in zip(axes,methods):
        values=np.load(source/(key+'_meters.npy'));valid=np.isfinite(values)&(values>0);coverage[key]=float(valid.mean())
        shown=ax.imshow(np.rot90(np.where(valid,values,np.nan),-1),cmap=cmap,vmin=.15,vmax=1.,interpolation='nearest')
        ax.set_title(f'{title}\n{100*valid.mean():.1f}% image coverage',fontsize=10);ax.axis('off')
    fig.colorbar(shown,ax=axes,shrink=.7,label='Meters · display clipped to 0.15–1.0; arrays unchanged',extend='both')
    fig.suptitle('Same capture and crop · gray = no valid measurement/prediction')
    fig.savefig(assets/'depth_comparison.png',dpi=170);plt.close(fig)
    # Scene-specific scores, never substitute the three-scene aggregate here.
    metrics=json.loads((source/'scene_metrics.json').read_text());score_rows=[]
    for key,title in methods:
        rows=[r for r in metrics['region_results'] if r['method']==key and r['score_eligible']]
        score_rows.append(f'<tr><td>{html.escape(title)}</td><td>{np.mean([abs(r["signed_residual_cm"]) for r in rows]):.2f} cm</td><td>{len(rows)}/3</td><td>{100*coverage[key]:.1f}%</td></tr>')
    lenses=[float(f['_lens']) for f in frames]
    document=TEMPLATE.replace('__SCORES__',''.join(score_rows)).replace('__LENSES__',json.dumps(lenses))
    (out/'index.html').write_text(document)
    provenance=dict(scene=scene,source_manifest_sha256=meta['source_manifest_sha256'],calibration_sha256=hashlib.sha256(modelraw).hexdigest(),
                    reference_index=meta['reference_index'],detail_bbox_cropped=bbox,coverage=coverage,
                    note='Existing development capture; offline Python processing; no new evaluation or model inference.')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(out/'index.html')


TEMPLATE='''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FocusDepth · visual walkthrough</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#10151d;color:#edf1f7;font:18px/1.5 system-ui,sans-serif}
main{max-width:1250px;margin:auto;padding:32px 42px}header{display:flex;justify-content:space-between;color:#a8bdd3;font-size:15px}
h1{font-size:clamp(30px,4vw,48px);line-height:1.15;margin:22px 0 12px}p{max-width:850px;color:#c3cfdd}
section{display:none}section.active{display:block}.grid{display:grid;grid-template-columns:1fr 1fr;gap:30px;align-items:center}
img{max-width:100%;max-height:60vh;object-fit:contain;display:block;margin:auto;border-radius:6px}
.caption{font-size:14px;color:#afbed0;margin:8px 0}.card{background:#1b2431;padding:22px;border-radius:12px}
button{background:#e3eaf5;color:#152031;border:0;border-radius:6px;padding:10px 20px;font:inherit;cursor:pointer}
button:disabled{opacity:.4;cursor:default}nav{display:flex;gap:14px;align-items:center;margin:24px 0}
input{width:100%;accent-color:#9ed5ff}table{border-collapse:collapse;width:100%;font-size:17px}td,th{text-align:left;padding:10px;border-bottom:1px solid #465365}
.wide{max-height:50vh;width:100%}.note{font-size:15px;color:#c9b991}.flow{display:flex;gap:12px;flex-wrap:wrap;margin:30px 0}.flow span{padding:15px;background:#24354b;border-radius:8px}
@media(max-width:700px){main{padding:20px}.grid{grid-template-columns:1fr}img{max-height:44vh}table{font-size:13px}td,th{padding:6px}}
@media print{section{display:block!important;break-after:page}nav,header,button,input{display:none}body{background:white;color:black}p,.caption,.note{color:#333}.card{background:#eee}img{max-height:50vh}}
</style><main>
<header><b>FOCUSDEPTH</b><span>Existing development capture · iPhone 15 Pro</span></header>
<section class="active"><h1>What can a focus stack add to neural depth?</h1>
<div class="grid"><div><p>One scene. Twenty-four focus settings. Two possible contributions: sharper image detail and physical distance evidence.</p>
<div class="card"><b>Watch what changes as focus moves.</b><p>This is a saved, registered capture—not a live camera feed. Lens position is a control coordinate, not meters.</p></div></div>
<div><img id="sweep" src="assets/focus_00.jpg" alt="Registered focus sweep frame"><label for="focus">Focus frame <span id="focusLabel"></span></label><input id="focus" type="range" min="0" max="23" value="0"><button id="play">Pause sweep</button></div></div></section>
<section><h1>The system we built</h1><p>The phone captures the stack. Python performs the processing offline using a pretrained Depth Anything V2 model.</p>
<div class="flow"><span>Focus stack</span><span>→ Registration + sharpness</span><span>→ All-in-focus image</span><span>→ Neural depth</span></div>
<div class="flow"><span>Sharpness curves</span><span>→ Frozen focus calibration</span><span>→ Metric optical evidence</span><span>→ Neural scale fit + fusion</span></div>
<div class="card"><b>Evaluation stays separate</b><p>Measured target distances and LiDAR evaluate the main methods. Neither supplies the focus-assisted scale fit. No new neural network was trained.</p></div></section>
<section><h1>First contribution: visible detail</h1><p>Switch between the reference frame and the all-in-focus composite. The crop and display size stay identical.</p>
<div class="grid"><div><img id="detail" src="assets/reference_detail.jpg" alt="Matched crop of reference or all-in-focus image"><p class="caption" id="detailLabel">Reference frame · actual frame 12</p></div>
<div><button id="toggle">Show all-in-focus</button><p>The composite selects locally sharp content across the stack.</p><div class="card">Sharper printed texture is an image-quality benefit. It does not establish more accurate depth.</div></div></div></section>
<section><h1>From sparse measurements to dense predictions</h1><img class="wide" src="assets/depth_comparison.png" alt="Calibrated focus, focus-scaled neural, fusion, and LiDAR maps with a shared meter color scale">
<p class="caption">Identical scene and crop; shared meter scale. Gray marks missing support. LiDAR is a reference, not an input to these focus/neural methods.</p>
<p>The neural map supplies dense structure. Calibrated focus supplies limited metric evidence. Fusion combines them, but greater coverage is not automatically greater accuracy.</p></section>
<section><h1>What do the measurements say?</h1><p>This displayed scene only: three measured front-face regions at approximately 20, 40 and 60 cm.</p>
<table><thead><tr><th>Method</th><th>Region-median MAE</th><th>Eligible regions</th><th>Whole-image coverage</th></tr></thead><tbody>__SCORES__</tbody></table>
<p class="note">Approximate tape measurements; no certified dense ground truth. Every method has ≥50% valid support in each evaluation patch, but their valid pixels differ. Whole-image coverage is not the fraction of accurate pixels.</p>
<p>The wider three-scene experiment also found mixed fusion gains and a stronger direct LiDAR regional baseline.</p></section>
<section><h1>Useful contributions—and a clear limit</h1><div class="grid"><div class="card"><b>What works</b><p>A working capture-to-analysis system.</p><p>AIF preserves visible detail.</p><p>Calibrated focus provides useful distance evidence on suitable surfaces.</p></div><div class="card"><b>What remains unresolved</b><p>Dense fusion does not consistently improve accuracy.</p><p>Focus confidence is not an accuracy guarantee.</p><p>Longer-distance and smooth-surface reliability need fresh evaluation.</p></div></div>
<p>Next: a small fresh regional check at 30, 50 and 70 cm, two captures each, with calibration and settings unchanged. These six captures are planned, not completed.</p>
<p class="caption">This walkthrough demonstrates existing outputs. It is not a live scanner or a complete 3D reconstruction.</p></section>
<nav><button id="prev">← Previous</button><span id="page" aria-live="polite"></span><button id="next">Next →</button><button id="full">Fullscreen</button></nav>
<p class="caption">Arrow keys navigate. All assets are local. Print to PDF for a static copy; focus_sweep.gif is available for video editing.</p></main>
<script>
const lenses=__LENSES__,sections=[...document.querySelectorAll('section')];let page=0,frame=0,playing=true,aif=false;
const $=id=>document.getElementById(id);
function show(n){page=Math.max(0,Math.min(sections.length-1,n));sections.forEach((s,i)=>s.classList.toggle('active',i===page));$('page').textContent=`${page+1} / ${sections.length}`;$('prev').disabled=page===0;$('next').disabled=page===sections.length-1;}
function focus(n){frame=n;$('sweep').src=`assets/focus_${String(n).padStart(2,'0')}.jpg`;$('focus').value=n;$('focusLabel').textContent=`${n+1}/24 · lens position ${lenses[n].toFixed(3)}`;}
$('prev').onclick=()=>show(page-1);$('next').onclick=()=>show(page+1);
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowRight')show(page+1);if(e.key==='ArrowLeft')show(page-1);});
$('focus').oninput=e=>{playing=false;$('play').textContent='Play sweep';focus(+e.target.value);};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause sweep':'Play sweep';};
setInterval(()=>{if(playing&&page===0)focus((frame+1)%lenses.length);},220);
$('toggle').onclick=()=>{aif=!aif;$('detail').src=`assets/${aif?'aif':'reference'}_detail.jpg`;$('detailLabel').textContent=aif?'All-in-focus composite':'Reference frame · actual frame 12';$('toggle').textContent=aif?'Show reference':'Show all-in-focus';};
$('full').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen?.().catch(()=>{});};show(0);focus(0);
</script></html>'''


if __name__=='__main__':main()
