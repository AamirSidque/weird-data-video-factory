import hashlib, json, math, uuid
from datetime import datetime, timedelta, timezone
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA = [
 ('D','decimal','Decimal'),('B','binary','Binary'),('H','hex','Hexadecimal'),('O','octal','Octal'),
 ('A','alphabet','Alphabet'),('X','alphanumeric','Alphanumeric'),('R','roman','Roman Numerals'),
 ('AS','ascii','ASCII Characters'),('M','morse','Morse Code'),('DNA','dna','DNA Sequences'),
 ('CO','coordinates','Coordinates'),('TS','timestamps','Timestamps'),('P','primes','Prime Numbers'),
 ('C','constants','Mathematical Constants'),('CC','color_codes','Color Codes'),('RW','random_walks','Random Walks'),
 ('EX','expressions','Mathematical Expressions'),('F','frequency','Frequency Data'),('HASH','hashes','Hashes'),
 ('UUID','uuid','UUIDs'),('RX','regex','Regular Expressions'),('JS','json','JSON Structures'),('G','graph','Graph Structures')]
METHODS=[('01','particle_universe','Particle Universe'),('02','flow_field','Flow Field'),('03','wave_ocean','Wave Ocean'),('04','cellular_evolution','Cellular Evolution'),('05','geometry_machine','Geometry Machine'),('06','fractal_world','Fractal World'),('07','data_sculpture','Data Sculpture'),('08','firestorm','Digital Firestorm'),('09','network_organism','Network Organism'),('10','data_program','Data Program')]
PREFIX={k:v for k,v,_ in DATA}; LABEL={k:v for _,k,v in DATA}
MORSE={c:m for c,m in zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',['.-','-...','-.-.','-..','.','..-.','--.','....','..','.---','-.-','.-..','--','-.','---','.--.','--.-','.-.','...','-','..-','...-','.--','-..-','-.--','--..'])}

def all_generators():
 return [{'id':f'{p}{n}','data_type':k,'data_label':label,'method':m,'method_label':ml} for p,k,label in DATA for n,m,ml in METHODS]

def find_generator(gid):
 for g in all_generators():
  if g['id'].upper()==gid.upper(): return g
 raise ValueError('Unknown generator: '+gid)

def title(g,rng):
 # P09 is the first educational pilot: title the idea, not just the effect.
 if g['id'].upper() == 'P09':
  return 'Why Do Prime Numbers Have Uneven Gaps?'
 patterns=['Raw {d} Data Became a {m}','I Turned {d} Into a {m}','This {d} Sequence Created a {m}','What Happens When {d} Controls a {m}?','{d} Data Generated This {m}','A Strange {m} Built From {d}','The {d} That Turned Into a {m}']
 return rng.choice(patterns).format(d=g['data_label'],m=g['method_label'])

def make_data(kind,rng):
 if kind=='decimal': return rng.normal(0,1,240).tolist()
 if kind=='binary': return ''.join(rng.choice(['0','1'],600))
 if kind=='hex': return ''.join(f'{int(x):02X}' for x in rng.integers(0,256,240))
 if kind=='octal': return ''.join(rng.choice(list('01234567'),300))
 if kind=='alphabet': return ''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),300))
 if kind=='alphanumeric': return ''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),300))
 if kind=='roman': return ' '.join(rng.choice(list('IVXLCDM'),220))
 if kind=='ascii': return ''.join(rng.choice(list(' .,:;!?+-=*/\\|_[]{}()<>#@$%&ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),400))
 if kind=='morse': return ' '.join(MORSE[c] for c in rng.choice(list(MORSE),150))
 if kind=='dna': return ''.join(rng.choice(list('ACGT'),500))
 if kind=='coordinates': return rng.normal(0,1,(160,3)).tolist()
 if kind=='timestamps':
  t=datetime(2026,1,1,tzinfo=timezone.utc); out=[]
  for _ in range(150): t+=timedelta(seconds=int(rng.integers(1,86400))); out.append(t.isoformat())
  return out
 if kind=='primes':
  # Start at 2 so the educational pilot can show a coherent sequence.
  out=[]; x=2
  while len(out)<150:
   if x>1 and all(x%q for q in range(2,int(x**.5)+1)): out.append(x)
   x+=1
  return out
 if kind=='constants': return (rng.choice([math.pi,math.e,(1+math.sqrt(5))/2,math.sqrt(2),math.sqrt(3),math.log(2),math.log(10)],180)*rng.uniform(.5,5,180)).tolist()
 if kind=='color_codes': return [f'#{int(x):06X}' for x in rng.integers(0,0x1000000,180)]
 if kind=='random_walks': return np.cumsum(rng.choice([-1,1],300)).tolist()
 if kind=='expressions': return [f'x{rng.choice(["+","-","*","/","sin","cos","^"])}{int(rng.integers(1,12))}' for _ in range(180)]
 if kind=='frequency': return rng.choice([55,110,220,261.63,329.63,440,523.25,659.25,880,1760],180).tolist()
 if kind=='hashes': return [hashlib.sha256(str(int(x)).encode()).hexdigest() for x in rng.integers(0,2**31,30)]
 if kind=='uuid': return [str(uuid.UUID(bytes=bytes(rng.integers(0,256,16,dtype=np.uint8)))) for _ in range(30)]
 if kind=='regex': return ''.join(rng.choice(['[A-Z]','[0-9]','\\d','\\w','.','*','+','?','|','()','{}','^','$'],220))
 if kind=='json': return json.dumps({f'node_{i}':{'value':int(rng.integers(0,1000)),'energy':float(rng.random()),'children':int(rng.integers(0,6))} for i in range(24)},separators=(',',':'))
 if kind=='graph': return [(int(rng.integers(0,40)),int(rng.integers(0,40)),float(rng.random())) for _ in range(80)]
 raise ValueError(kind)

def values(kind,data):
 if kind=='hex': a=np.fromiter((int(data[i:i+2],16) for i in range(0,len(data)-1,2)),float)
 elif kind in ('binary','octal','alphabet','alphanumeric','roman','ascii','morse','dna','regex','expressions'): a=np.fromiter((ord(c) for c in str(data)),float)
 elif kind=='color_codes': a=np.array([int(x[1:],16) for x in data],float)
 elif kind=='coordinates': a=np.asarray(data,float).ravel()
 elif kind=='graph': a=np.asarray(data,float).ravel()
 elif kind=='timestamps': a=np.fromiter((sum(map(ord,x[-12:])) for x in data),float)
 elif kind=='json': a=np.fromiter((ord(c) for c in data),float)
 elif kind in ('hashes','uuid'): a=np.fromiter((ord(c) for c in ''.join(data)),float)
 else: a=np.asarray(data,float).ravel()
 a=np.nan_to_num(a); lo,hi=float(a.min()),float(a.max())
 return np.full(max(128,len(a)),.5) if hi-lo<1e-9 else (a-lo)/(hi-lo)

def font(size):
 for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
  try:return ImageFont.truetype(p,size)
  except:pass
 return ImageFont.load_default()

def palette(kind):
 maps={'hex':[(255,65,30),(255,190,30),(255,245,180)],'binary':[(30,255,150),(30,180,255),(230,255,240)],'dna':[(70,255,120),(80,170,255),(255,90,160)],'morse':[(255,255,255),(80,190,255),(120,100,255)],'hashes':[(255,50,80),(255,150,50),(100,255,220)],'uuid':[(80,210,255),(190,80,255),(255,255,255)]}
 return maps.get(kind,[(90,200,255),(220,90,255),(255,210,70)])

def frame(W,H,p,v,kind,method,seed,params,raw_data=None):
 rng=np.random.default_rng(seed); pal=palette(kind); img=Image.new('RGB',(W,H),(3,5,12)); d=ImageDraw.Draw(img,'RGBA'); v=np.resize(v,512); env=max(.02,math.sin(math.pi*p)**.55); chaos=params['chaos']*(.4+1.2*env); phase=p*math.tau
 # Moving volumetric-looking background.
 yy,xx=np.mgrid[0:H,0:W]; g=np.sin(xx/W*math.tau*2+phase*.7)+np.cos(yy/H*math.tau*1.5-phase)
 bg=np.zeros((H,W,3),np.uint8); bg[:,:,0]=np.clip(3+4*(g+2),0,255); bg[:,:,1]=np.clip(5+5*(g+2),0,255); bg[:,:,2]=np.clip(12+10*(g+2),0,255); img=Image.fromarray(bg); d=ImageDraw.Draw(img,'RGBA')
 if kind=='primes' and method=='network_organism':
  # Educational P09 pilot: reveal actual primes in a 1–100 grid and show prime gaps.
  def is_prime(n):
   return n > 1 and all(n % q for q in range(2, int(n ** 0.5) + 1))
  margin=max(18,int(W*.035)); top=int(H*.205)
  title_font=font(max(22,int(W*.026))); sub_font=font(max(13,int(W*.014))); cell_font=font(max(11,int(W*.014)))
  # Grid sizing is proportional to the frame and leaves a separate explanation panel.
  grid_w=int(W*.53); cell_w=grid_w/10; cell_h=min(int(H*.064),int((H*.66)/10)); cell_h=max(25,cell_h)
  visible_limit=max(1,min(100,1+int(p*99)))
  visible_primes=[n for n in range(2,visible_limit+1) if is_prime(n)]
  d.text((margin,int(H*.055)),'WHY DO PRIME NUMBERS HAVE UNEVEN GAPS?',font=title_font,fill=(245,248,255,255))
  d.text((margin,int(H*.12)),'A visual experiment in number patterns',font=sub_font,fill=(150,180,210,255))
  for n in range(1,101):
   col=(n-1)%10; row=(n-1)//10; x=margin+col*cell_w; y=top+row*cell_h
   shown=n<=visible_limit; prime=is_prime(n)
   if shown and prime: fill=(24,150,145,235); outline=(90,255,220,255); text_color=(255,255,255,255)
   elif shown and n==1: fill=(70,75,90,150); outline=(110,120,140,160); text_color=(170,180,195,210)
   elif shown: fill=(28,37,54,220); outline=(58,74,96,220); text_color=(145,160,180,255)
   else: fill=(10,17,30,130); outline=(35,46,65,150); text_color=(90,105,125,180)
   pad=max(2,int(cell_w*.055))
   d.rounded_rectangle((x+pad,y+2,x+cell_w-pad,y+cell_h-3),radius=max(3,int(cell_w*.08)),fill=fill,outline=outline,width=1)
   label=str(n); box=d.textbbox((0,0),label,font=cell_font); tw=box[2]-box[0]; th=box[3]-box[1]
   d.text((x+(cell_w-tw)/2,y+(cell_h-th)/2-1),label,font=cell_font,fill=text_color)
  panel_x=int(W*.60); panel_w=int(W*.36); panel_y=int(H*.22)
  d.rounded_rectangle((panel_x,panel_y,W-margin,int(H*.77)),radius=16,fill=(10,18,33,225),outline=(55,83,115,210),width=2)
  d.text((panel_x+18,panel_y+18),'THE PRIME TEST',font=font(max(16,int(W*.019))),fill=(255,210,90,255))
  d.text((panel_x+18,panel_y+58),'A prime is greater than 1',font=sub_font,fill=(235,240,250,255))
  d.text((panel_x+18,panel_y+82),'and has exactly two positive',font=sub_font,fill=(235,240,250,255))
  d.text((panel_x+18,panel_y+106),'divisors: 1 and itself.',font=sub_font,fill=(235,240,250,255))
  d.text((panel_x+18,panel_y+148),'1 is NOT prime.',font=font(max(14,int(W*.016))),fill=(255,130,140,255))
  d.text((panel_x+18,panel_y+190),f'Primes revealed: {len(visible_primes)}',font=sub_font,fill=(100,255,215,255))
  recent_primes=visible_primes[-7:]
  gaps=[b-a for a,b in zip(recent_primes,recent_primes[1:])] if len(recent_primes)>1 else []
  gap_y=panel_y+235
  d.text((panel_x+18,gap_y),'Recent gaps between primes',font=sub_font,fill=(235,240,250,255))
  if gaps:
   max_gap=max(gaps)
   bar_x=panel_x+18; bar_y=gap_y+28; avail_w=panel_w-38; gap_slot=avail_w/len(gaps)
   for i,gap in enumerate(gaps):
    bh=max(5,int((gap/max_gap)*42)); bx=bar_x+i*gap_slot
    d.rounded_rectangle((bx,bar_y+42-bh,bx+gap_slot-5,bar_y+42),radius=3,fill=(*pal[i%len(pal)],225))
    d.text((bx+2,bar_y+48),str(gap),font=font(max(10,int(W*.011))),fill=(200,215,235,255))
  d.text((margin,int(H*.91)),'Observe: the gaps change; the rule stays the same.',font=sub_font,fill=(220,230,245,255))
 elif method=='particle_universe':
  N=int(500+500*params['density']); ids=rng.integers(0,len(v),N); a=rng.random(N)*math.tau+p*params['spin']*5; r=(.03+.97*rng.random(N)**1.8)*min(W,H)*.46; x=W/2+np.cos(a)*r+np.sin(a*3+phase)*35*chaos; y=H/2+np.sin(a)*r*.58
  for i in range(N):
   c=pal[i%3]; q=1+int(5*v[ids[i]]); d.ellipse((x[i]-q,y[i]-q,x[i]+q,y[i]+q),fill=(*c,int(70+170*env)))
 elif method=='flow_field':
  for y in range(-10,H,38):
   for x in range(-10,W,38):
    q=(x//38+y//38)%len(v); a=v[q]*math.tau*2+p*math.tau*params['spin']*2+math.sin(x*.01+y*.013+phase)*chaos; L=18+45*v[q]; c=pal[q%3]; d.line((x,y,x+math.cos(a)*L,y+math.sin(a)*L),fill=(*c,130),width=1+int(2*v[q]))
 elif method=='wave_ocean':
  for k in range(15):
   pts=[]; base=H*(.35+k*.045)
   for x in range(0,W,8):
    q=(x//8+k*19)%len(v); y=base+math.sin(x*.009*(1+k*.06)+phase*(1+k*.08))*35+math.sin(x*.021+v[q]*8+phase*1.7)*22*v[q]; pts.append((x,y))
   d.line(pts,fill=(*pal[k%3],150),width=2)
 elif method=='cellular_evolution':
  cols,rows=80,45; cw,ch=W/cols,H/rows
  for gy in range(rows):
   for gx in range(cols):
    q=(gx+gy*cols)%len(v); alive=((int(v[q]*255)+gx*7+gy*13+int(p*100))%19)<int(4+12*env)
    if alive:
     c=pal[(gx+gy)%3]; x=gx*cw;y=gy*ch; d.rectangle((x,y,x+cw-1,y+ch-1),fill=(*c,int(70+160*env)))
 elif method=='geometry_machine':
  cx,cy=W/2,H/2
  for k in range(22):
   sides=3+k%7; rad=(35+k*20)*(.7+.4*env); rot=phase*(1+k*.035)*params['spin']+k*.4; pts=[(cx+math.cos(rot+q*math.tau/sides)*rad,cy+math.sin(rot+q*math.tau/sides)*rad*.65) for q in range(sides)]; c=pal[k%3]; d.polygon(pts,outline=(*c,120),width=2)
 elif method=='fractal_world':
  sw,sh=max(180,W//5),max(100,H//5); arr=np.zeros((sh,sw,3),np.uint8); zoom=1.8/(.25+p); ox=-.5+.2*math.cos(phase); oy=.2*math.sin(phase)
  for py in range(sh):
   cy=(py/sh-.5)*zoom+oy
   for px in range(sw):
    cx=(px/sw-.5)*zoom+ox; z=0j; it=0
    while abs(z)<2 and it<8: z=z*z+complex(cx,cy); it+=1
    arr[py,px]=pal[it%3] if it<8 else (4,5,12)
  img=Image.blend(img,Image.fromarray(arr).resize((W,H),Image.Resampling.BILINEAR),.45+.3*env); d=ImageDraw.Draw(img,'RGBA')
 elif method=='data_sculpture':
  for k in range(110):
   q=k*7%len(v); a=k*.27+phase*.7; rr=30+k*5+80*v[q]*env; x=W/2+math.cos(a)*rr; y=H/2+math.sin(a)*rr*.52; w=10+55*v[q]; h=5+35*v[(q+31)%len(v)]; c=pal[k%3]; d.rounded_rectangle((x-w,y-h,x+w,y+h),radius=8,outline=(*c,105),width=2)
 elif method=='firestorm':
  N=int(900+650*params['density'])
  for _ in range(N):
   q=int(rng.integers(0,len(v))); age=(rng.random()+p*2)%1; x=W*(.12+.76*rng.random())+math.sin(age*12+phase*2)*75*chaos; y=H*(1-age); s=1+int(12*v[q]*env); c=pal[int(v[q]*3)%3]; d.ellipse((x-s,y-s,x+s,y+s),fill=(*c,int(60+170*(1-age))))
 elif method=='network_organism':
  n=70; pts=[]
  for i in range(n):
   a=i*math.tau/n+phase*.15; r=min(W,H)*.34*(.65+.35*v[i%len(v)]); pts.append((W/2+math.cos(a)*r,H/2+math.sin(a)*r*.58))
  for i,(x,y) in enumerate(pts):
   for j in range(i+1,min(i+7,n)):
    if rng.random()<.42:d.line((x,y,*pts[j]),fill=(*pal[(i+j)%3],70),width=1)
  for i,(x,y) in enumerate(pts):
   r=3+int(8*v[i%len(v)]); c=pal[i%3]; d.ellipse((x-r,y-r,x+r,y+r),fill=(*c,220))
 else:
  f=font(18); text=str(v[:80].round(3).tolist());
  for row in range(10): d.text((55+math.sin(phase+row)*30,120+row*40),text[row*55:row*55+55],font=f,fill=(*pal[row%3],175))
  for i in range(24):
   x=70+(i%8)*150; y=600+(i//8)*28; c=pal[i%3]; d.rounded_rectangle((x,y,x+25+100*v[i%len(v)],y+15),radius=5,fill=(*c,100))
 if not (kind=='primes' and method=='network_organism'):
  header=f'{kind.upper()}  •  {method.replace("_"," ").upper()}'
  d.text((24,20),header,font=font(20),fill=(235,240,250,195))
 d.rectangle((24,H-25,W-24,H-21),fill=(255,255,255,35)); d.rectangle((24,H-25,24+(W-48)*p,H-21),fill=(*pal[0],190))
 return img











# import hashlib, json, math, uuid
# from datetime import datetime, timedelta, timezone
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont

# DATA = [
#  ('D','decimal','Decimal'),('B','binary','Binary'),('H','hex','Hexadecimal'),('O','octal','Octal'),
#  ('A','alphabet','Alphabet'),('X','alphanumeric','Alphanumeric'),('R','roman','Roman Numerals'),
#  ('AS','ascii','ASCII Characters'),('M','morse','Morse Code'),('DNA','dna','DNA Sequences'),
#  ('CO','coordinates','Coordinates'),('TS','timestamps','Timestamps'),('P','primes','Prime Numbers'),
#  ('C','constants','Mathematical Constants'),('CC','color_codes','Color Codes'),('RW','random_walks','Random Walks'),
#  ('EX','expressions','Mathematical Expressions'),('F','frequency','Frequency Data'),('HASH','hashes','Hashes'),
#  ('UUID','uuid','UUIDs'),('RX','regex','Regular Expressions'),('JS','json','JSON Structures'),('G','graph','Graph Structures')]
# METHODS=[('01','particle_universe','Particle Universe'),('02','flow_field','Flow Field'),('03','wave_ocean','Wave Ocean'),('04','cellular_evolution','Cellular Evolution'),('05','geometry_machine','Geometry Machine'),('06','fractal_world','Fractal World'),('07','data_sculpture','Data Sculpture'),('08','firestorm','Digital Firestorm'),('09','network_organism','Network Organism'),('10','data_program','Data Program')]
# PREFIX={k:v for k,v,_ in DATA}; LABEL={k:v for _,k,v in DATA}
# MORSE={c:m for c,m in zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',['.-','-...','-.-.','-..','.','..-.','--.','....','..','.---','-.-','.-..','--','-.','---','.--.','--.-','.-.','...','-','..-','...-','.--','-..-','-.--','--..'])}

# def all_generators():
#  return [{'id':f'{p}{n}','data_type':k,'data_label':label,'method':m,'method_label':ml} for p,k,label in DATA for n,m,ml in METHODS]

# def find_generator(gid):
#  for g in all_generators():
#   if g['id'].upper()==gid.upper(): return g
#  raise ValueError('Unknown generator: '+gid)

# def title(g,rng):
#  patterns=['Raw {d} Data Became a {m}','I Turned {d} Into a {m}','This {d} Sequence Created a {m}','What Happens When {d} Controls a {m}?','{d} Data Generated This {m}','A Strange {m} Built From {d}','The {d} That Turned Into a {m}']
#  return rng.choice(patterns).format(d=g['data_label'],m=g['method_label'])

# def make_data(kind,rng):
#  if kind=='decimal': return rng.normal(0,1,240).tolist()
#  if kind=='binary': return ''.join(rng.choice(['0','1'],600))
#  if kind=='hex': return ''.join(f'{int(x):02X}' for x in rng.integers(0,256,240))
#  if kind=='octal': return ''.join(rng.choice(list('01234567'),300))
#  if kind=='alphabet': return ''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),300))
#  if kind=='alphanumeric': return ''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),300))
#  if kind=='roman': return ' '.join(rng.choice(list('IVXLCDM'),220))
#  if kind=='ascii': return ''.join(rng.choice(list(' .,:;!?+-=*/\\|_[]{}()<>#@$%&ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),400))
#  if kind=='morse': return ' '.join(MORSE[c] for c in rng.choice(list(MORSE),150))
#  if kind=='dna': return ''.join(rng.choice(list('ACGT'),500))
#  if kind=='coordinates': return rng.normal(0,1,(160,3)).tolist()
#  if kind=='timestamps':
#   t=datetime(2026,1,1,tzinfo=timezone.utc); out=[]
#   for _ in range(150): t+=timedelta(seconds=int(rng.integers(1,86400))); out.append(t.isoformat())
#   return out
#  if kind=='primes':
#   out=[]; x=2+int(rng.integers(0,50))
#   while len(out)<150:
#    if x>1 and all(x%q for q in range(2,int(x**.5)+1)): out.append(x)
#    x+=1
#   return out
#  if kind=='constants': return (rng.choice([math.pi,math.e,(1+math.sqrt(5))/2,math.sqrt(2),math.sqrt(3),math.log(2),math.log(10)],180)*rng.uniform(.5,5,180)).tolist()
#  if kind=='color_codes': return [f'#{int(x):06X}' for x in rng.integers(0,0x1000000,180)]
#  if kind=='random_walks': return np.cumsum(rng.choice([-1,1],300)).tolist()
#  if kind=='expressions': return [f'x{rng.choice(["+","-","*","/","sin","cos","^"])}{int(rng.integers(1,12))}' for _ in range(180)]
#  if kind=='frequency': return rng.choice([55,110,220,261.63,329.63,440,523.25,659.25,880,1760],180).tolist()
#  if kind=='hashes': return [hashlib.sha256(str(int(x)).encode()).hexdigest() for x in rng.integers(0,2**31,30)]
#  if kind=='uuid': return [str(uuid.UUID(bytes=bytes(rng.integers(0,256,16,dtype=np.uint8)))) for _ in range(30)]
#  if kind=='regex': return ''.join(rng.choice(['[A-Z]','[0-9]','\\d','\\w','.','*','+','?','|','()','{}','^','$'],220))
#  if kind=='json': return json.dumps({f'node_{i}':{'value':int(rng.integers(0,1000)),'energy':float(rng.random()),'children':int(rng.integers(0,6))} for i in range(24)},separators=(',',':'))
#  if kind=='graph': return [(int(rng.integers(0,40)),int(rng.integers(0,40)),float(rng.random())) for _ in range(80)]
#  raise ValueError(kind)

# def values(kind,data):
#  if kind=='hex': a=np.fromiter((int(data[i:i+2],16) for i in range(0,len(data)-1,2)),float)
#  elif kind in ('binary','octal','alphabet','alphanumeric','roman','ascii','morse','dna','regex','expressions'): a=np.fromiter((ord(c) for c in str(data)),float)
#  elif kind=='color_codes': a=np.array([int(x[1:],16) for x in data],float)
#  elif kind=='coordinates': a=np.asarray(data,float).ravel()
#  elif kind=='graph': a=np.asarray(data,float).ravel()
#  elif kind=='timestamps': a=np.fromiter((sum(map(ord,x[-12:])) for x in data),float)
#  elif kind=='json': a=np.fromiter((ord(c) for c in data),float)
#  elif kind in ('hashes','uuid'): a=np.fromiter((ord(c) for c in ''.join(data)),float)
#  else: a=np.asarray(data,float).ravel()
#  a=np.nan_to_num(a); lo,hi=float(a.min()),float(a.max())
#  return np.full(max(128,len(a)),.5) if hi-lo<1e-9 else (a-lo)/(hi-lo)

# def font(size):
#  for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
#   try:return ImageFont.truetype(p,size)
#   except:pass
#  return ImageFont.load_default()

# def palette(kind):
#  maps={'hex':[(255,65,30),(255,190,30),(255,245,180)],'binary':[(30,255,150),(30,180,255),(230,255,240)],'dna':[(70,255,120),(80,170,255),(255,90,160)],'morse':[(255,255,255),(80,190,255),(120,100,255)],'hashes':[(255,50,80),(255,150,50),(100,255,220)],'uuid':[(80,210,255),(190,80,255),(255,255,255)]}
#  return maps.get(kind,[(90,200,255),(220,90,255),(255,210,70)])

# def frame(W,H,p,v,kind,method,seed,params):
#  rng=np.random.default_rng(seed); pal=palette(kind); img=Image.new('RGB',(W,H),(3,5,12)); d=ImageDraw.Draw(img,'RGBA'); v=np.resize(v,512); env=max(.02,math.sin(math.pi*p)**.55); chaos=params['chaos']*(.4+1.2*env); phase=p*math.tau
#  # moving volumetric-looking background
#  yy,xx=np.mgrid[0:H,0:W]; g=np.sin(xx/W*math.tau*2+phase*.7)+np.cos(yy/H*math.tau*1.5-phase)
#  bg=np.zeros((H,W,3),np.uint8); bg[:,:,0]=np.clip(3+4*(g+2),0,255); bg[:,:,1]=np.clip(5+5*(g+2),0,255); bg[:,:,2]=np.clip(12+10*(g+2),0,255); img=Image.fromarray(bg); d=ImageDraw.Draw(img,'RGBA')
#  if method=='particle_universe':
#   N=int(500+500*params['density']); ids=rng.integers(0,len(v),N); a=rng.random(N)*math.tau+p*params['spin']*5; r=(.03+.97*rng.random(N)**1.8)*min(W,H)*.46; x=W/2+np.cos(a)*r+np.sin(a*3+phase)*35*chaos; y=H/2+np.sin(a)*r*.58
#   for i in range(N):
#    c=pal[i%3]; q=1+int(5*v[ids[i]]); d.ellipse((x[i]-q,y[i]-q,x[i]+q,y[i]+q),fill=(*c,int(70+170*env)))
#  elif method=='flow_field':
#   for y in range(-10,H,38):
#    for x in range(-10,W,38):
#     q=(x//38+y//38)%len(v); a=v[q]*math.tau*2+p*math.tau*params['spin']*2+math.sin(x*.01+y*.013+phase)*chaos; L=18+45*v[q]; c=pal[q%3]; d.line((x,y,x+math.cos(a)*L,y+math.sin(a)*L),fill=(*c,130),width=1+int(2*v[q]))
#  elif method=='wave_ocean':
#   for k in range(15):
#    pts=[]; base=H*(.35+k*.045)
#    for x in range(0,W,8):
#     q=(x//8+k*19)%len(v); y=base+math.sin(x*.009*(1+k*.06)+phase*(1+k*.08))*35+math.sin(x*.021+v[q]*8+phase*1.7)*22*v[q]; pts.append((x,y))
#    d.line(pts,fill=(*pal[k%3],150),width=2)
#  elif method=='cellular_evolution':
#   cols,rows=80,45; cw,ch=W/cols,H/rows
#   for gy in range(rows):
#    for gx in range(cols):
#     q=(gx+gy*cols)%len(v); alive=((int(v[q]*255)+gx*7+gy*13+int(p*100))%19)<int(4+12*env)
#     if alive:
#      c=pal[(gx+gy)%3]; x=gx*cw;y=gy*ch; d.rectangle((x,y,x+cw-1,y+ch-1),fill=(*c,int(70+160*env)))
#  elif method=='geometry_machine':
#   cx,cy=W/2,H/2
#   for k in range(22):
#    sides=3+k%7; rad=(35+k*20)*(.7+.4*env); rot=phase*(1+k*.035)*params['spin']+k*.4; pts=[(cx+math.cos(rot+q*math.tau/sides)*rad,cy+math.sin(rot+q*math.tau/sides)*rad*.65) for q in range(sides)]; c=pal[k%3]; d.polygon(pts,outline=(*c,120),width=2)
#  elif method=='fractal_world':
#   sw,sh=max(180,W//5),max(100,H//5); arr=np.zeros((sh,sw,3),np.uint8); zoom=1.8/(.25+p); ox=-.5+.2*math.cos(phase); oy=.2*math.sin(phase)
#   for py in range(sh):
#    cy=(py/sh-.5)*zoom+oy
#    for px in range(sw):
#     cx=(px/sw-.5)*zoom+ox; z=0j; it=0
#     while abs(z)<2 and it<8: z=z*z+complex(cx,cy); it+=1
#     arr[py,px]=pal[it%3] if it<8 else (4,5,12)
#   img=Image.blend(img,Image.fromarray(arr).resize((W,H),Image.Resampling.BILINEAR),.45+.3*env); d=ImageDraw.Draw(img,'RGBA')
#  elif method=='data_sculpture':
#   for k in range(110):
#    q=k*7%len(v); a=k*.27+phase*.7; rr=30+k*5+80*v[q]*env; x=W/2+math.cos(a)*rr; y=H/2+math.sin(a)*rr*.52; w=10+55*v[q]; h=5+35*v[(q+31)%len(v)]; c=pal[k%3]; d.rounded_rectangle((x-w,y-h,x+w,y+h),radius=8,outline=(*c,105),width=2)
#  elif method=='firestorm':
#   N=int(900+650*params['density'])
#   for _ in range(N):
#    q=int(rng.integers(0,len(v))); age=(rng.random()+p*2)%1; x=W*(.12+.76*rng.random())+math.sin(age*12+phase*2)*75*chaos; y=H*(1-age); s=1+int(12*v[q]*env); c=pal[int(v[q]*3)%3]; d.ellipse((x-s,y-s,x+s,y+s),fill=(*c,int(60+170*(1-age))))
#  elif method=='network_organism':
#   n=70; pts=[]
#   for i in range(n):
#    a=i*math.tau/n+phase*.15; r=min(W,H)*.34*(.65+.35*v[i%len(v)]); pts.append((W/2+math.cos(a)*r,H/2+math.sin(a)*r*.58))
#   for i,(x,y) in enumerate(pts):
#    for j in range(i+1,min(i+7,n)):
#     if rng.random()<.42:d.line((x,y,*pts[j]),fill=(*pal[(i+j)%3],70),width=1)
#   for i,(x,y) in enumerate(pts):
#    r=3+int(8*v[i%len(v)]); c=pal[i%3]; d.ellipse((x-r,y-r,x+r,y+r),fill=(*c,220))
#  else:
#   f=font(18); text=str(v[:80].round(3).tolist());
#   for row in range(10): d.text((55+math.sin(phase+row)*30,120+row*40),text[row*55:row*55+55],font=f,fill=(*pal[row%3],175))
#   for i in range(24):
#    x=70+(i%8)*150; y=600+(i//8)*28; c=pal[i%3]; d.rounded_rectangle((x,y,x+25+100*v[i%len(v)],y+15),radius=5,fill=(*c,100))
#  d.text((24,20),f'{kind.upper()}  •  {method.replace("_"," ").upper()}',font=font(20),fill=(235,240,250,195)); d.rectangle((24,H-25,W-24,H-21),fill=(255,255,255,35)); d.rectangle((24,H-25,24+(W-48)*p,H-21),fill=(*pal[0],190))
#  return img
