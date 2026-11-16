"""Build the five-minute FocusDepth visual edit, slides and recording script.

Uses saved experiment outputs; does not refit, infer, or modify raw captures.
Optional --audio muxes the user's narration. FFmpeg is discovered from PATH or
imageio_ffmpeg; the latter can be supplied through PYTHONPATH.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from audit_fresh_scene_results import summarize

W,H=1920,1080
BG='#101827';FG='#f3f5fa';MUTED='#bac8d9';ACCENT='#76e0c4'
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'

SECTIONS=[
('Giving neural depth physical scale',22,"Hello everyone. My project is FocusDepth: giving neural depth physical scale from an iPhone focus sweep. I tested whether calibrated focus measurements can constrain a neural depth map without depth-sensor input to the estimator, including predictions on smooth objects where direct focus evidence is weak."),
('One scene, changing focus',28,"A focus stack contains images of the same scene at different focus settings. This animation shows a real saved capture. I used an iPhone 15 Pro and twenty-four focus steps, with exposure and white balance fixed during each sweep. The app also exports paired LiDAR depth for comparison. The processing shown here runs offline on the computer."),
('Two ways to use the stack',32,"After capture, I align the images and measure sharpness across the stack. One branch creates an all-in-focus image by combining locally sharp content. The other estimates the focus setting that best resolves each region or pixel. I use a pretrained Depth Anything V2 model for neural depth. I did not train a new network. LiDAR and measured scene distances remain separate evaluation references."),
('Sharper input is one benefit',30,"Here, the reference frame and all-in-focus composite use exactly the same crop. Printed detail becomes clearer in the composite. Across eleven measured patches in the development scenes, the median sharpness ratio was about two point one. However, sharper input did not consistently improve neural boundary placement. Image detail and depth accuracy are separate outcomes, so I evaluated them separately."),
('Focus can measure regional distance',36,"Lens position is a camera control value, not meters. I calibrated it using separate textured targets from twenty centimeters to one meter, and kept that calibration fixed. On six fresh target placements at thirty, fifty and seventy centimeters, regional focus estimation had one point one five centimeters mean absolute error. This is the error of a regional estimate. It does not establish equally accurate depth at every pixel, and measurement and placement uncertainty remain."),
('Sparse evidence, dense predictions',35,"This is the central challenge. Calibrated optical depth is sparse because many pixels have weak focus evidence or fall outside calibration support. I fit the relative neural prediction to confident optical measurements in inverse depth, then tested confidence-weighted fusion. The neural map supplies dense spatial structure. Optical evidence supplies metric constraints. Gray areas indicate missing support; all four maps share the same meter scale."),
('Fresh textured and smooth targets',35,"To check transfer, I evaluated nine fresh scenes, including printed targets and staple-pin boxes covered with matte napkin paper. Smooth interiors retained very little optical support, while neural and fused maps stayed dense. The scale fit can use optical anchors elsewhere in the image. These are different physical targets rather than a perfectly paired texture experiment, so the comparison does not isolate texture as the only cause."),
('Accuracy must include coverage',32,"Across fifteen fresh regions, focus-scaled neural depth had five point one one centimeters mean absolute error, and fusion five point zero one. A separate metric Depth Anything checkpoint had about one hundred ninety-six centimeters error here. That is a different checkpoint, not the same model before correction. Optical depth qualified on only one region, and LiDAR on fourteen. LiDAR had lower error on shared support. All predefined far patches remain included."),
('What the experiments taught us',28,"I also tested more selective fusion and sparse regional anchors. Neither improved the development baseline overall. These negative results helped distinguish useful optical measurements from successful dense depth correction. Confidence selected better pixels in our diagnostics, but high confidence was not a guarantee of metric accuracy. The remaining limits include distance sensitivity, weak texture, alignment and approximate reference measurements."),
('The contribution',22,"The strongest finding is scale transfer: calibrated focus supplies metric evidence, while neural predictions extend across smooth targets with little optical support. This worked on our close-range indoor captures, with remaining error. All-in-focus composition also improves visible detail. Fusion did not consistently improve accuracy. Thank you.")
]


def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)


def wrapped(draw,text,xy,width,size=35,fill=FG,bold=False,line_space=12):
    x,y=xy;f=font(size,bold);line=''
    for word in text.split():
        candidate=(line+' '+word).strip()
        if draw.textlength(candidate,font=f)>width and line:
            draw.text((x,y),line,font=f,fill=fill);y+=size+line_space;line=word
        else:line=candidate
    if line:draw.text((x,y),line,font=f,fill=fill);y+=size+line_space
    return y


def base(index,title,tag=''):
    image=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(image)
    d.text((82,48),'FOCUSDEPTH',font=font(27,True),fill=ACCENT)
    d.text((82,105),title,font=font(64,True),fill=FG)
    d.line((82,201,W-82,201),fill='#33435b',width=2)
    d.text((82,H-57),tag,font=font(24),fill=MUTED)
    d.text((W-180,H-57),f'{index:02d} / 10',font=font(24),fill=MUTED)
    return image


def place(image,path,box):
    with Image.open(path) as source:
        fitted=ImageOps.contain(source.convert('RGB'),(box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
        image.paste(fitted,(box[0]+(box[2]-box[0]-fitted.width)//2,box[1]+(box[3]-box[1]-fitted.height)//2))


def bullets(image,texts,xy=(100,260),width=1700,size=40,gap=42):
    d=ImageDraw.Draw(image);x,y=xy
    for text in texts:
        d.ellipse((x,y+13,x+12,y+25),fill=ACCENT)
        y=wrapped(d,text,(x+40,y),width-40,size)+gap


def collect_metrics():
    audit=json.loads((ROOT/'results/fresh_20261116_audit/audit.json').read_text())
    validation=json.loads((ROOT/'results/focus_validation_20261116/validation.json').read_text())['results']
    confirmation=json.loads((ROOT/'data/annotations/validation_20261116_confirmation.json').read_text())
    corrected=[]
    for row in validation:
        value=dict(row)
        if row['folder']==confirmation['capture_directory']:
            manifest=ROOT/row['folder']/'manifest.json'
            if hashlib.sha256(manifest.read_bytes()).hexdigest()!=confirmation['manifest_sha256']:raise ValueError('Correction source changed')
            if row['measured_m']!=confirmation['original_distance_m']:raise ValueError('Correction label mismatch')
            value.update(measured_m=confirmation['corrected_distance_m'],signed_residual_cm=(row['predicted_m']-confirmation['corrected_distance_m'])*100,
                         label_correction=confirmation)
        corrected.append(value)
    ids=['112654','112752','112959','113334','113426','113508']
    primary=[r for r in corrected if any(r['folder'].endswith(i) for i in ids)]
    if len(primary)!=6:raise ValueError('Primary placement set incomplete')
    boundary=json.loads((ROOT/'results/boundary_detail_study/results.json').read_text())
    sharpness=[r['ratio'] for r in boundary['image_detail'] if r['within_calibration_span']]
    return dict(primary_validation=primary,all_validation_corrected=corrected,
        primary_validation_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in primary])),
        all_eight_observations_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in corrected])),
        fresh_full_set=audit['full_set'],aif_median_sharpness_ratio=float(np.median(sharpness)),
        boundary_summary=boundary['summary'])


def make_figures(out,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rc_context
    with rc_context({'font.size':14,'axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
        rows=metrics['primary_validation'];actual=np.array([r['measured_m']*100 for r in rows]);predicted=np.array([r['predicted_m']*100 for r in rows])
        ax.plot([25,75],[25,75],color='gray',linestyle='--',label='Ideal measurement')
        ax.scatter(actual,predicted,s=90,color='#127d6e',label='Six fresh placements',zorder=3)
        ax.set(xlim=(25,75),ylim=(25,75),xlabel='Recorded front-face distance (cm)',ylabel='Regional focus estimate (cm)',title='Frozen calibration · fresh target placements')
        ax.legend();fig.savefig(out/'validation.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(14,6),layout='constrained')
        for row,(label,name) in enumerate([('Printed target','fresh_single_textured_20261116_125214'),('Napkin-covered target','fresh_single_smooth_20261116_124831')]):
            folder=ROOT/'results/fresh_20261116'/name;rgb=np.array(Image.open(folder/'reference_rgb.png').convert('RGB'))
            focus=np.load(folder/'calibrated_focus_meters.npy');valid=np.isfinite(focus)&(focus>0)
            region=json.loads((folder/'scene_metrics.json').read_text())['regions'][0];x,y,ex,ey=region['bbox']
            rgb=rgb[y:ey,x:ex];valid=valid[y:ey,x:ex]
            neural=np.load(folder/'focus_scaled_relative_aif_meters.npy')[y:ey,x:ex]
            show=rgb.copy();show[~valid]=(show[~valid]*.12).astype(np.uint8)
            axes[row,0].imshow(np.rot90(rgb,-1));axes[row,0].set_title(f'{label}\nRGB · nominal 50 cm',fontsize=13)
            axes[row,1].imshow(np.rot90(show,-1));axes[row,1].set_title(f'Focus support: {100*valid.mean():.1f}%\nUnsupported pixels dimmed',fontsize=13)
            shown=axes[row,2].imshow(np.rot90(neural,-1),vmin=.15,vmax=1.,cmap='viridis',interpolation='nearest')
            coverage=float(np.isfinite(neural).mean());median=float(np.nanmedian(neural))*100
            axes[row,2].set_title(f'Focus-scaled neural: {median:.1f} cm\n{100*coverage:.0f}% patch coverage',fontsize=13)
            for ax in axes[row]:ax.axis('off')
        fig.colorbar(shown,ax=axes[:,2],shrink=.7,label='Meters · shared display scale')
        fig.suptitle('Smooth target: little direct focus evidence, but a dense scaled neural prediction',fontsize=15)
        fig.savefig(out/'smooth_support.png',dpi=170);plt.close(fig)


def make_slides(out,metrics):
    assets=ROOT/'results/demo_walkthrough/assets';make_figures(out,metrics)
    slides=[]
    image=base(1,'Giving neural depth physical scale','Godwin Kangor · CS73 Computational Photography · November 2026')
    place(image,assets/'aif.jpg',(1230,245,1830,950))
    bullets(image,['Calibrated iPhone focus sweeps','Metric constraints for neural depth','Including smooth-target predictions'],xy=(110,300),width=1000,size=47,gap=58)
    wrapped(ImageDraw.Draw(image),'No depth-sensor input to the estimator; LiDAR is an evaluation reference.',(145,800),950,30,fill=MUTED);slides.append(image)
    image=base(2,'One scene. Twenty-four focus settings.','Actual saved capture · aligned offline · lens position is not meters')
    bullets(image,['iPhone 15 Pro: main RGB + paired LiDAR reference','Exposure and white balance fixed within each sweep','Python processing runs on the computer'],xy=(100,285),width=1020,size=39,gap=52)
    place(image,assets/'focus_00.jpg',(1230,235,1830,955));slides.append(image)
    image=base(3,'Two connected uses of a focus stack','Pretrained DAV2 Small · no new neural-network training')
    d=ImageDraw.Draw(image)
    boxes=[(100,270,550,390,'Capture + registration'),(700,270,1180,390,'Sharpness across focus'),(1310,270,1810,390,'AIF → neural depth'),(700,520,1180,650,'Calibrated focus'),(1310,520,1810,650,'Scale fit + fusion')]
    for x,y,ex,ey,text in boxes:
        d.rounded_rectangle((x,y,ex,ey),radius=15,fill='#24354c');wrapped(d,text,(x+25,y+32),ex-x-50,32,bold=True)
    for a,b in [((550,330),(700,330)),((1180,330),(1310,330)),((940,390),(940,520)),((1180,585),(1310,585)),((1560,390),(1560,520))]:
        d.line((*a,*b),fill=ACCENT,width=5);d.ellipse((b[0]-7,b[1]-7,b[0]+7,b[1]+7),fill=ACCENT)
    wrapped(d,'Measured distances and LiDAR evaluate the main methods; neither enters optical-assisted fitting.',(110,770),1680,37,fill=MUTED);slides.append(image)
    image=base(4,'Sharper RGB does not guarantee better geometry','Matched crop · development scenes · sharpness is a diagnostic, not depth accuracy')
    place(image,assets/'reference_detail.jpg',(100,250,880,850));place(image,assets/'aif_detail.jpg',(1040,250,1820,850))
    d=ImageDraw.Draw(image);d.text((120,865),'REFERENCE FRAME',font=font(32,True),fill=FG);d.text((1060,865),'ALL-IN-FOCUS',font=font(32,True),fill=ACCENT)
    d.text((100,935),f'Median patch sharpness ratio: {metrics["aif_median_sharpness_ratio"]:.2f}× · neural boundary improvements were mixed',font=font(32),fill=MUTED);slides.append(image)
    image=base(5,'Useful regional metric measurements','Fresh validation · original calibration preserved · approximate measured distances')
    place(image,out/'validation.png',(700,250,1840,870))
    bullets(image,[f'{metrics["primary_validation_mae_cm"]:.2f} cm MAE','Six designated placements','30, 50 and 70 cm','Regional estimates, not dense pixel accuracy'],xy=(100,280),width=570,size=37,gap=26)
    wrapped(ImageDraw.Draw(image),'Two additional captures are retained separately, including the user-confirmed 70 cm label correction.',(110,915),1690,28,fill=MUTED);slides.append(image)
    image=base(6,'Sparse optical evidence. Dense neural structure.','Development scene · common meter scale · gray = unsupported · display range 0.15–1.0 m')
    place(image,assets/'depth_comparison.png',(60,250,1860,915));slides.append(image)
    image=base(7,'Fresh smooth surfaces reveal the coverage limit','Different objects and approximate placements confound a pure texture comparison')
    place(image,out/'smooth_support.png',(90,245,1830,885))
    wrapped(ImageDraw.Draw(image),'Neural scaling uses anchors elsewhere in the image; dense output is not dense optical measurement.',(100,925),1720,30,fill=MUTED);slides.append(image)
    image=base(8,'Fresh results: count every predefined region','15 regions across nine new scenes · frozen methods · confirmed nominal box assignments')
    d=ImageDraw.Draw(image);columns=[110,805,1180,1490]
    for x,text in zip(columns,['Method','Eligible regions','Full-set MAE','Subset MAE']):d.text((x,260),text,font=font(31,True),fill=ACCENT)
    names=[('Focus-scaled neural','focus_scaled_relative_aif'),('Confidence fusion','focus_neural_fusion'),('Calibrated focus','calibrated_focus'),('LiDAR reference','direct_lidar'),('Metric DAV2 checkpoint','unadjusted_metric_aif')]
    for i,(title,key) in enumerate(names):
        row=metrics['fresh_full_set'][key];y=330+i*92
        full='Unavailable' if row['complete_set_mae_cm'] is None else f'{row["complete_set_mae_cm"]:.2f} cm'
        values=[title,f'{row["eligible_regions"]}/15',full,f'{row["eligible_subset_mae_cm"]:.2f} cm']
        for x,text in zip(columns,values):d.text((x,y),text,font=font(33),fill=FG)
    wrapped(d,'Different subset denominators cannot be ranked directly. On 14-region shared support: LiDAR 2.64 cm versus neural scaling 5.05 cm.',(110,825),1690,30,fill=MUTED)
    wrapped(d,'Metric DAV2 is a separate checkpoint: this comparison is not a before/after correction of one model.',(110,930),1690,27,fill=MUTED);slides.append(image)
    image=base(9,'More selective fusion was not an automatic fix','Development ablations · same eleven-region evaluation · no fresh-set tuning')
    bullets(image,['Original fusion: 4.86 cm → selective candidate: 5.94 cm','Pixel-anchor scaling: 5.77 cm → regional anchors: 6.30 cm','Preserving neural edges can reject useful optical corrections','Confidence measures focus evidence—not guaranteed metric accuracy'],xy=(105,270),width=1700,size=40,gap=35);slides.append(image)
    image=base(10,'Focus supplies constraints. Neural depth extends them.','Code: github.com/GodwinKangor/FocusDepth-w--NeuralNet')
    bullets(image,['Optical scale transfer on close-range indoor captures','Predictions on smooth targets with little direct focus support','AIF improves visible detail; fusion gains remain mixed'],xy=(105,265),width=1700,size=44,gap=35)
    wrapped(ImageDraw.Draw(image),'Limits: small dataset, approximate measurements, calibration support, and extrapolation. Phones without LiDAR and outdoor performance remain untested.',(110,740),1660,30,fill=MUTED)
    wrapped(ImageDraw.Draw(image),'Background: Suwajanakorn et al., CVPR 2015 · Yang et al., Depth Anything V2, NeurIPS 2024.',(110,915),1660,26,fill=MUTED);slides.append(image)
    for i,image in enumerate(slides,1):image.save(out/f'slide_{i:02d}.png')
    slides[0].save(out/'FocusDepth_slides.pdf',save_all=True,append_images=slides[1:],resolution=150)
    # Animated focus segment: actual frames placed into the same title layout.
    animated=[]
    for i in range(24):
        image=slides[1].copy();ImageDraw.Draw(image).rectangle((1230,235,1830,955),fill=BG)
        place(image,assets/f'focus_{i:02d}.jpg',(1230,235,1830,955))
        path=out/f'sweep_slide_{i:02d}.png';image.save(path);animated.append(path)
    # Alternate the same crop at full-size for clear evidence in the video.
    for name in ['reference','aif']:
        image=slides[3].copy();ImageDraw.Draw(image).rectangle((70,225,1850,975),fill=BG)
        place(image,assets/f'{name}_detail.jpg',(400,240,1520,910))
        ImageDraw.Draw(image).text((110,930),'REFERENCE FRAME' if name=='reference' else 'ALL-IN-FOCUS COMPOSITE',font=font(36,True),fill=ACCENT)
        image.save(out/f'detail_{name}.png')
    return slides,animated


def ffmpeg_binary():
    value=shutil.which('ffmpeg')
    if value:return value
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def audio_duration(ffmpeg,path):
    result=subprocess.run([ffmpeg,'-hide_banner','-i',str(path)],capture_output=True,text=True)
    match=re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)',result.stderr)
    if not match or 'Audio:' not in result.stderr:raise ValueError(f'Cannot establish audio duration: {path}')
    hours,minutes,seconds=map(float,match.groups());return hours*3600+minutes*60+seconds


def assemble_voice_clips(ffmpeg,directory,out):
    files=[]
    for i,(_,duration,_) in enumerate(SECTIONS,1):
        choices=[p for p in directory.iterdir() if p.stem==f'{i:02d}' and p.suffix.lower() in {'.m4a','.wav','.mp3','.aac','.aiff','.aif','.mp4'}]
        if len(choices)!=1:raise ValueError(f'Need exactly one voice clip named {i:02d} in {directory}')
        path=choices[0].resolve();length=audio_duration(ffmpeg,path)
        if length>duration+.05:raise ValueError(f'Clip {i:02d} lasts {length:.2f}s, exceeding its {duration}s window. Retiming or a shorter take is needed; no speech was truncated.')
        files.append((path,duration))
    padded=[]
    for i,(path,duration) in enumerate(files,1):
        target=out/f'voice_{i:02d}_padded.wav'
        subprocess.run([ffmpeg,'-y','-i',str(path),'-vn','-af',f'apad,atrim=duration={duration}',
                        '-ar','48000','-ac','1',str(target)],check=True,capture_output=True)
        padded.append(target)
    concat=out/'voice.ffconcat';concat.write_text('ffconcat version 1.0\n'+''.join(f"file '{p.name}'\n" for p in padded))
    target=out/'voice_assembled.wav'
    subprocess.run([ffmpeg,'-y','-f','concat','-safe','0','-i',str(concat),
                    '-af','loudnorm=I=-16:TP=-1.5:LRA=11',str(target)],check=True,capture_output=True)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/submission_video')
    parser.add_argument('--audio',type=Path)
    parser.add_argument('--audio-clips',type=Path)
    parser.add_argument('--skip-render',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.audio and args.audio_clips:raise ValueError('Choose continuous audio or separate clips')
    metrics=collect_metrics();(out/'verified_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    starts=[];time=0;script=['# FocusDepth — five-minute voice recording script','',
        'Record your own voice. The time windows below match the visual edit. Read naturally and pause between sections. '
        'You can record ten separate clips named 01.m4a through 10.m4a, or one continuous recording. '
        'Ten clips make alignment easier; do not speak the section headings. Aim to finish within each time window.','']
    for i,(title,duration,narration) in enumerate(SECTIONS,1):
        starts.append(dict(section=i,title=title,start_seconds=time,duration_seconds=duration,narration=narration))
        stamp=lambda t:f'{t//60}:{t%60:02d}'
        script.extend([f'## {i:02d} · {stamp(time)}–{stamp(time+duration)} · {title}','',narration,''])
        time+=duration
    assert time==300
    (out/'recording_script.md').write_text('\n'.join(script))
    (out/'timeline.json').write_text(json.dumps(starts,indent=2)+'\n')
    if not args.skip_render:make_slides(out,metrics)
    # FFmpeg concat demuxer: durations add to precisely 300 s; export at 30 fps.
    lines=['ffconcat version 1.0']
    for i,(_,duration,_) in enumerate(SECTIONS,1):
        if i==2:
            order=list(range(24))+list(range(22,0,-1));frames=(order*3)
            for n in frames:lines.extend([f"file 'sweep_slide_{n:02d}.png'",f'duration {duration/len(frames):.10f}'])
        elif i==4:
            for name in ['reference','aif']*3:lines.extend([f"file 'detail_{name}.png'",f'duration {duration/6:.10f}'])
        else:lines.extend([f"file 'slide_{i:02d}.png'",f'duration {duration}'])
    lines.append("file 'slide_10.png'");(out/'visuals.ffconcat').write_text('\n'.join(lines)+'\n')
    ffmpeg=ffmpeg_binary()
    visual=out/'FocusDepth_visual_edit_5min.mp4'
    if not args.skip_render:
        subprocess.run([ffmpeg,'-y','-f','concat','-safe','0','-i',str(out/'visuals.ffconcat'),'-t','300',
            '-vf','fps=30,format=yuv420p','-c:v','libx264','-preset','fast','-crf','20','-movflags','+faststart',str(visual)],check=True)
        (out/'visual_edit_manifest.json').write_text(json.dumps(dict(
            duration_seconds=300,width=W,height=H,frames_per_second=30,
            timeline_sha256=hashlib.sha256((out/'timeline.json').read_bytes()).hexdigest(),
            video_sha256=hashlib.sha256(visual.read_bytes()).hexdigest(),
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Visual edit only, no narration until user audio is supplied'),indent=2)+'\n')
    audio=args.audio
    if args.audio_clips:audio=assemble_voice_clips(ffmpeg,args.audio_clips.resolve(),out)
    if audio:
        if not visual.exists():raise ValueError('Render the visual edit first')
        render_manifest=json.loads((out/'visual_edit_manifest.json').read_text())
        if render_manifest['timeline_sha256']!=hashlib.sha256((out/'timeline.json').read_bytes()).hexdigest():
            raise ValueError('Visual edit and narration timeline differ; rerender before muxing')
        length=audio_duration(ffmpeg,audio.resolve())
        if length>300.05:raise ValueError(f'Narration lasts {length:.2f}s. Retiming or a shorter take is required; no speech was truncated.')
        subprocess.run([ffmpeg,'-y','-i',str(visual),'-i',str(audio.resolve()),'-map','0:v:0','-map','1:a:0',
            '-c:v','copy','-c:a','aac','-b:a','192k','-af','apad','-t','300','-movflags','+faststart',str(out/'FocusDepth_FINAL_5min.mp4')],check=True)
    print('Created',out,flush=True)


if __name__=='__main__':main()
