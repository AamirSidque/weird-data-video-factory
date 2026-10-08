import argparse,json,os,secrets,wave,subprocess
from pathlib import Path
import numpy as np
from .generators import all_generators,find_generator,make_data,values,title,frame
from .telegram import send_video
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'output'; TMP=ROOT/'tmp'; OUT.mkdir(exist_ok=True); TMP.mkdir(exist_ok=True)

def render_video(g,cfg,seed,data_vals,duration,out):
 fps=int(cfg['fps']); W=int(cfg['width']); H=int(cfg['height']); rng=np.random.default_rng(seed+77); params={'density':float(rng.uniform(.35,1)),'chaos':float(rng.uniform(.2,1)),'spin':float(rng.uniform(.2,1.8))}
 cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset',cfg.get('preset','veryfast'),'-crf',str(cfg.get('crf',23)),'-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 try:
  for i in range(int(duration*fps)):
   im=frame(W,H,i/max(1,int(duration*fps)-1),data_vals,g['data_type'],g['method'],seed,params); p.stdin.write(np.asarray(im,np.uint8).tobytes())
 finally: p.stdin.close(); rc=p.wait()
 if rc: raise RuntimeError('FFmpeg video render failed')

def audio(g,vals,duration,rate,seed,out):
    n=int(duration*rate); v=np.resize(vals,256); base={"frequency":220,"morse":120,"dna":180,"binary":90,"hex":130,"hashes":100}.get(g["data_type"],145)
    rng=np.random.default_rng(seed+991); phase0=0.0; chunk=max(1,rate*2)
    with wave.open(str(out),"wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        for start in range(0,n,chunk):
            m=min(chunk,n-start); pos=np.arange(start,start+m); c=np.interp(pos,np.linspace(0,n-1,256),v); t=pos/rate
            freq=base*(.55+1.5*c); phase=phase0+2*np.pi*np.cumsum(freq)/rate; phase0=float(phase[-1])
            a=.18*np.sin(phase)+.08*np.sin(phase*2.01+c*4)+.04*np.sin(phase*3.01)
            gate=(c>.62).astype(float); k=max(1,rate//120); gate=np.convolve(gate,np.ones(min(k,m))/min(k,m),'same'); a+=.1*gate*np.sin(2*np.pi*base*2*t)
            if start < rate*2:
                env=np.linspace(0,1,m) if start==0 else np.ones(m)
            elif start+m>=n-rate*2:
                env=np.linspace(1,0,m)
            else: env=np.ones(m)
            a=np.clip(a*env*.9,-.95,.95); w.writeframes((a*32767).astype(np.int16).tobytes())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--generator',default='');args=ap.parse_args();cfg=json.loads((ROOT/'config.json').read_text());seed=secrets.randbelow(2**31-1);rng=np.random.default_rng(seed);g=find_generator(args.generator) if args.generator else rng.choice(all_generators()); duration=int(round(int(cfg['min_duration_seconds'])+(int(cfg['max_duration_seconds'])-int(cfg['min_duration_seconds']))*rng.beta(2,2.8))); data=make_data(g['data_type'],rng); vals=values(g['data_type'],data); ttl=title(g,rng); stem=f'{g["id"]}_{seed}'; raw=TMP/f'{stem}.mp4'; wav=TMP/f'{stem}.wav'; final=OUT/f'{stem}.mp4'; render_video(g,cfg,seed,vals,duration,raw)
 if cfg.get('audio_enabled',True):
  audio(g,vals,duration,int(cfg.get('audio_sample_rate',44100)),seed,wav); subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(raw),'-i',str(wav),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',str(final)],check=True); raw.unlink(missing_ok=True);wav.unlink(missing_ok=True)
 else: raw.replace(final)
 meta={'title':ttl,'generator':g['id'],'data_type':g['data_type'],'data_label':g['data_label'],'visual_method':g['method'],'visual_method_label':g['method_label'],'seed':seed,'duration_seconds':duration,'fps':cfg['fps'],'resolution':f'{cfg["width"]}x{cfg["height"]}','audio':bool(cfg.get('audio_enabled',True))};(OUT/f'{stem}.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2)); token=os.getenv('TELEGRAM_BOT_TOKEN');chat=os.getenv('TELEGRAM_CHAT_ID');
 if cfg.get('send_to_telegram',True) and token and chat: send_video(token,chat,final,f'{ttl}\n\nGenerator: {g["id"]}\nData: {g["data_label"]}\nDuration: {duration}s\nSeed: {seed}')
if __name__=='__main__':main()
