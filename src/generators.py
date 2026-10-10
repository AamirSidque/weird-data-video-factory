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

# Phase 1 adds semantic concepts with their own renderers. These are not arbitrary
# data/method combinations: each concept has a defined question and visual lesson.
PHASE1_CONCEPTS = [
    ('MATH01', 'fibonacci_growth', 'Fibonacci Growth', 'mathematics', 'fibonacci-growth'),
    ('MATH02', 'pascal_triangle', 'Pascal Triangle Patterns', 'mathematics', 'pascal-parity'),
    ('PHY01', 'wave_interference', 'Wave Interference', 'physics', 'wave-interference'),
    ('PHY02', 'orbital_mechanics', 'Orbital Motion', 'physics', 'orbital-motion'),
    ('CS01', 'sorting_algorithm', 'Sorting Algorithms', 'computer_science', 'insertion-sort'),
    ('CS02', 'pathfinding', 'Pathfinding Algorithms', 'computer_science', 'breadth-first-search'),
    ('NAT01', 'flocking_behavior', 'Flocking Behaviour', 'nature', 'flocking-behaviour'),
    ('NAT02', 'predator_prey', 'Predator–Prey Cycles', 'nature', 'population-cycles'),
    ('SAT01', 'game_of_life', 'Cellular Automata', 'satisfying_experiments', 'game-of-life'),
    ('SAT02', 'particle_attractor', 'Particle Flow Experiments', 'satisfying_experiments', 'particle-attractor'),
]
PHASE1_PROFILES = {
    'MATH01': {'concept_id':'fibonacci-growth','series':'Math Patterns','hook':'Why do simple growth rules create spiral patterns?','learning_objective':'A Fibonacci sequence adds the previous two terms to create the next; related growth patterns appear in spiral arrangements.','experiment_id':'phase1-math-v1','short_title':'FIBONACCI GROWTH','takeaway':'A simple recurrence can create a striking growth pattern.'},
    'MATH02': {'concept_id':'pascal-parity','series':'Math Patterns','hook':'What hidden pattern appears in Pascal’s triangle?','learning_objective':'Pascal’s triangle is built by adding adjacent values; highlighting odd and even entries reveals a repeating triangular pattern.','experiment_id':'phase1-math-v1','short_title':'PASCAL’S TRIANGLE','takeaway':'Parity reveals structure inside a number triangle.'},
    'PHY01': {'concept_id':'wave-interference','series':'Physics in Motion','hook':'What happens when two waves overlap?','learning_objective':'When waves overlap, their displacements combine; depending on phase, they can reinforce or partially cancel.','experiment_id':'phase1-physics-v1','short_title':'WAVE INTERFERENCE','takeaway':'The combined wave is the sum of the two wave displacements.'},
    'PHY02': {'concept_id':'orbital-motion','series':'Physics in Motion','hook':'How can gravity keep objects in orbit?','learning_objective':'An orbit is continuous motion under gravity; this visualization is a simplified illustrative model, not a numerical prediction.','experiment_id':'phase1-physics-v1','short_title':'ORBITAL MOTION','takeaway':'An orbit combines forward motion with gravitational acceleration.'},
    'CS01': {'concept_id':'insertion-sort','series':'Code Made Visible','hook':'How does a sorting algorithm put numbers in order?','learning_objective':'Insertion sort grows a sorted prefix by inserting each next item into its correct position.','experiment_id':'phase1-cs-v1','short_title':'INSERTION SORT','takeaway':'The sorted region grows one insertion at a time.'},
    'CS02': {'concept_id':'breadth-first-search','series':'Code Made Visible','hook':'How does a pathfinding algorithm explore a maze?','learning_objective':'Breadth-first search explores a grid in layers and finds a shortest path when each move has equal cost.','experiment_id':'phase1-cs-v1','short_title':'PATHFINDING','takeaway':'Breadth-first search explores nearby cells before farther ones.'},
    'NAT01': {'concept_id':'flocking-behaviour','series':'Patterns in Nature','hook':'How can simple rules create coordinated flocking?','learning_objective':'Flocking models often combine separation, alignment, and cohesion; this is a simplified agent-based demonstration.','experiment_id':'phase1-nature-v1','short_title':'FLOCKING BEHAVIOUR','takeaway':'Local rules can produce coordinated group motion.'},
    'NAT02': {'concept_id':'population-cycles','series':'Patterns in Nature','hook':'Why can predator and prey populations rise and fall in cycles?','learning_objective':'Simplified predator–prey models can produce delayed population cycles; real ecosystems are more complex.','experiment_id':'phase1-nature-v1','short_title':'POPULATION CYCLES','takeaway':'Changes in one population can influence the other with a delay.'},
    'SAT01': {'concept_id':'game-of-life','series':'Satisfying Experiments','hook':'How can a few rules make a pattern evolve?','learning_objective':'Conway’s Game of Life updates cells from their neighbours: live cells survive with two or three neighbours, and dead cells are born with three.','experiment_id':'phase1-satisfying-v1','short_title':'GAME OF LIFE','takeaway':'Local rules can create complex-looking patterns.'},
    'SAT02': {'concept_id':'particle-attractor','series':'Satisfying Experiments','hook':'How can a simple flow field shape hundreds of paths?','learning_objective':'A vector field assigns a direction at each location; particles following it can reveal coherent motion patterns.','experiment_id':'phase1-satisfying-v1','short_title':'PARTICLE FLOW','takeaway':'Many individual paths can reveal the structure of a field.'},
}
PHASE1_TITLES = {
    'MATH01':'Why Do Fibonacci Patterns Form Spirals?',
    'MATH02':'The Hidden Pattern in Pascal’s Triangle',
    'PHY01':'What Happens When Two Waves Overlap?',
    'PHY02':'How Orbital Motion Works',
    'CS01':'How Insertion Sort Puts Numbers in Order',
    'CS02':'How a Pathfinding Algorithm Solves a Maze',
    'NAT01':'How Simple Rules Create Flocking Behaviour',
    'NAT02':'Why Predator and Prey Populations Cycle',
    'SAT01':'How Simple Rules Create Complex Patterns',
    'SAT02':'Watch a Flow Field Shape Hundreds of Particles',
}
PREFIX={k:v for k,v,_ in DATA}; LABEL={k:v for _,k,v in DATA}
MORSE={c:m for c,m in zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',['.-','-...','-.-.','-..','.','..-.','--.','....','..','.---','-.-','.-..','--','-.','---','.--.','--.-','.-.','...','-','..-','...-','.--','-..-','-.--','--..'])}

def all_generators():
 legacy = [{'id':f'{p}{n}','data_type':k,'data_label':label,'method':m,'method_label':ml,
            'content_family':'legacy_experiment'} for p,k,label in DATA for n,m,ml in METHODS]
 phase1 = [{'id':gid, 'data_type':kind, 'data_label':label, 'method':'educational_simulation',
            'method_label':label, 'content_family':family, 'concept_id':concept}
           for gid,kind,label,family,concept in PHASE1_CONCEPTS]
 return legacy + phase1

def find_generator(gid):
 for g in all_generators():
  if g['id'].upper()==gid.upper(): return g
 raise ValueError('Unknown generator: '+gid)

CONTENT_PROFILES = {
    "P09": {
        "concept_id": "prime-number-gaps",
        "series": "Math Patterns",
        "hook": "Why are the gaps between prime numbers uneven?",
        "learning_objective": "A prime number is greater than 1 and has exactly two positive divisors; consecutive-prime gaps vary.",
        "experiment_id": "prime-gap-visual-v2",
    },
    "B04": {
        "concept_id": "binary-patterns",
        "series": "Code Made Visible",
        "hook": "How can simple binary values create complex patterns?",
        "learning_objective": "Binary data consists of two symbols, 0 and 1; mapping those values to visual states reveals patterns.",
        "experiment_id": "retention-visual-v1",
    },
    "DNA09": {
        "concept_id": "dna-sequence-network",
        "series": "Patterns in Nature",
        "hook": "What happens when a DNA sequence becomes a network?",
        "learning_objective": "DNA sequences use four nucleotide letters: A, C, G, and T; a visualization can encode sequence relationships.",
        "experiment_id": "retention-visual-v1",
    },
    "F03": {
        "concept_id": "frequency-waves",
        "series": "Signals in Motion",
        "hook": "How does frequency change the shape of a wave?",
        "learning_objective": "Frequency describes cycles per second and affects how rapidly a periodic signal oscillates.",
        "experiment_id": "retention-visual-v1",
    },
    "RW03": {
        "concept_id": "random-walk",
        "series": "Probability in Motion",
        "hook": "Can random steps create a visible pattern?",
        "learning_objective": "A random walk accumulates individual steps, producing a path whose overall shape can differ between trials.",
        "experiment_id": "retention-visual-v1",
    },
    "G09": {
        "concept_id": "graph-network",
        "series": "Connected Systems",
        "hook": "How do nodes and connections form a network?",
        "learning_objective": "A graph represents entities as nodes and relationships as edges.",
        "experiment_id": "retention-visual-v1",
    },
    "CC05": {
        "concept_id": "color-code-geometry",
        "series": "Digital Color",
        "hook": "How does a hexadecimal code represent a color?",
        "learning_objective": "Hex color notation commonly encodes red, green, and blue channel values.",
        "experiment_id": "retention-visual-v1",
    },
    "M10": {
        "concept_id": "morse-code-patterns",
        "series": "Signals in Motion",
        "hook": "How can dots and dashes encode a message?",
        "learning_objective": "Morse code represents letters and digits using sequences of short and long signals.",
        "experiment_id": "retention-visual-v1",
    },
    "JS10": {
        "concept_id": "json-structure",
        "series": "Data Made Visible",
        "hook": "What does structured data look like under the surface?",
        "learning_objective": "JSON organizes data using objects, arrays, keys, and values.",
        "experiment_id": "retention-visual-v1",
    },
    "CO09": {
        "concept_id": "coordinate-space",
        "series": "Geometry of Data",
        "hook": "How do coordinates turn numbers into a spatial pattern?",
        "learning_objective": "Coordinates encode positions in a defined space; plotting them makes spatial distributions visible.",
        "experiment_id": "retention-visual-v1",
    },
    "TS03": {
        "concept_id": "time-series-waves",
        "series": "Signals in Motion",
        "hook": "What patterns appear when time becomes a signal?",
        "learning_objective": "A time series records observations in temporal order, allowing changes and rhythms to be visualized.",
        "experiment_id": "retention-visual-v1",
    },
    "HASH09": {
        "concept_id": "hash-structure",
        "series": "Code Made Visible",
        "hook": "Why do hash strings look chaotic but have a fixed structure?",
        "learning_objective": "A cryptographic hash maps input data to a fixed-length digest; a digest is not an encrypted message.",
        "experiment_id": "retention-visual-v1",
    },
}


def get_content_profile(g):
    """Return editorial metadata used for titles, quality review and later analytics."""
    if g["id"].upper() in PHASE1_PROFILES:
        profile = dict(PHASE1_PROFILES[g["id"].upper()])
        profile['content_family'] = g.get('content_family', 'educational')
        return profile
    return dict(CONTENT_PROFILES.get(g["id"].upper(), {
        "concept_id": f'{g["data_type"]}-{g["method"]}',
        "series": "RawSignal Experiments",
        "hook": f'What can {g["data_label"].lower()} reveal when visualized as {g["method_label"].lower()}?',
        "learning_objective": None,
        "experiment_id": "exploratory-visual-v1",
    }))


def title(g,rng):
    profile = get_content_profile(g)
    # Keep the title promise aligned with the actual concept. The P09 wording
    # is a tested educational pilot; other curated concepts use concise hooks.
    title_map = {
        "P09": "Why Do Prime Numbers Have Uneven Gaps?",
        "B04": "How Binary Data Creates Patterns",
        "DNA09": "Turning DNA Letters Into a Network",
        "F03": "How Frequency Changes a Wave",
        "RW03": "Can Random Steps Create a Pattern?",
        "G09": "How Nodes and Connections Form a Network",
        "CC05": "How Hex Codes Represent Color",
        "M10": "How Dots and Dashes Encode Messages",
        "JS10": "Making Structured Data Visible",
        "CO09": "Turning Coordinates Into a Spatial Pattern",
        "TS03": "Finding Patterns in Time-Series Data",
        "HASH09": "Why Hashes Look Random but Stay Structured",
    }
    if g["id"].upper() in title_map:
        return title_map[g["id"].upper()]
    patterns=['Raw {d} Data Became a {m}','I Turned {d} Into a {m}','This {d} Sequence Created a {m}','What Happens When {d} Controls a {m}?','{d} Data Generated This {m}','A Strange {m} Built From {d}']
    return rng.choice(patterns).format(d=g['data_label'],m=g['method_label'])

def make_data(kind,rng):
 if kind == 'fibonacci_growth':
  out=[1,1]
  while len(out)<30: out.append(out[-1]+out[-2])
  return out
 if kind == 'pascal_triangle':
  return [int(rng.integers(0,2)) for _ in range(32)]
 if kind == 'wave_interference':
  return [float(rng.uniform(1.0,2.0)), float(rng.uniform(1.5,3.0)), float(rng.uniform(0,math.tau))]
 if kind == 'orbital_mechanics':
  return [0.42, 0.66, 0.88]
 if kind == 'sorting_algorithm':
  return rng.permutation(np.arange(1,65)).astype(int).tolist()
 if kind == 'pathfinding':
  return rng.random((20, 30)).tolist()
 if kind == 'flocking_behavior':
  return rng.random((28, 4)).tolist()
 if kind == 'predator_prey':
  return rng.random(256).tolist()
 if kind == 'game_of_life':
  return rng.integers(0, 2, 48*32).tolist()
 if kind == 'particle_attractor':
  return rng.random(256).tolist()
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

PHASE1_PROFILES_BY_KIND = {kind: PHASE1_PROFILES[gid] for gid, kind, *_ in PHASE1_CONCEPTS}
PHASE1_ID_BY_KIND = {kind: gid for gid, kind, *_ in PHASE1_CONCEPTS}


def _phase1_frame(img, W, H, p, v, kind, seed, params, raw_data=None):
 """Render a self-contained educational simulation in portrait or landscape."""
 d = ImageDraw.Draw(img, 'RGBA')
 portrait = H > W
 profile = PHASE1_PROFILES_BY_KIND[kind]
 margin = max(14, int(W * 0.045))
 title_size = max(17, int(min(W, H) * (0.045 if portrait else 0.035)))
 subtitle_size = max(11, int(min(W, H) * 0.021))
 body_size = max(10, int(min(W, H) * 0.019))
 title_font = font(title_size)
 sub_font = font(subtitle_size)
 body_font = font(body_size)
 accent = (84, 245, 205, 255)
 blue = (80, 170, 255, 245)
 pink = (245, 110, 190, 245)
 gold = (255, 207, 100, 245)
 white = (238, 244, 255, 255)
 muted = (155, 177, 204, 245)
 d.text((margin, int(H*.045)), profile['short_title'], font=title_font, fill=white)
 d.text((margin, int(H*.105)), profile['hook'], font=sub_font, fill=muted)
 left, right = margin, W-margin
 top, bottom = int(H*.20), int(H*.79)
 plot_w, plot_h = right-left, bottom-top
 d.rounded_rectangle((left, top, right, bottom), radius=max(8, int(min(W,H)*.018)), outline=(75,100,135,105), width=1, fill=(5,10,22,70))
 phase = float(p) * math.tau
 rng = np.random.default_rng(int(seed) % (2**32))

 if kind == 'fibonacci_growth':
  # Phyllotaxis-style growth visualization with Fibonacci sequence context.
  count = max(8, min(260, 8 + int(p*252)))
  cx, cy = (left+right)/2, (top+bottom)/2
  radius = min(plot_w, plot_h)*.43
  golden_angle = math.pi*(3-math.sqrt(5))
  for i in range(count):
   rr = radius*math.sqrt((i+1)/count)
   a = i*golden_angle + phase*.05
   x, y = cx+math.cos(a)*rr, cy+math.sin(a)*rr
   r = max(2, int(min(W,H)*(.004 + .004*(i%5==0))))
   color = (accent, blue, pink, gold)[i%4]
   d.ellipse((x-r,y-r,x+r,y+r), fill=color)
  seq = '1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89'
  d.text((left+int(plot_w*.025), bottom-int(plot_h*.11)), 'FIBONACCI SEQUENCE', font=body_font, fill=gold)
  d.text((left+int(plot_w*.025), bottom-int(plot_h*.065)), seq, font=body_font, fill=white)

 elif kind == 'pascal_triangle':
  rows = max(3, min(25, 3+int(p*22)))
  cell = min(plot_w/(rows+1), plot_h/(rows+1))*.72
  cell = max(4, cell)
  y0 = top+int(plot_h*.035)
  for row in range(rows):
   vals = [math.comb(row, col) for col in range(row+1)]
   y = y0 + row*cell
   for col, value in enumerate(vals):
    x = (left+right)/2 + (col-row/2)*cell
    rad = max(2, int(cell*.28))
    fill = accent if value % 2 else (38,58,82,230)
    d.ellipse((x-rad,y-rad,x+rad,y+rad), fill=fill, outline=(120,160,195,140), width=1)
  d.text((left+int(plot_w*.04), bottom-int(plot_h*.08)), 'Odd and even entries reveal a repeating pattern', font=body_font, fill=muted)

 elif kind == 'wave_interference':
  xvals = np.linspace(0, 1, max(100, plot_w//3))
  freq_a, freq_b, offset = (raw_data if isinstance(raw_data, (list, tuple)) and len(raw_data)>=3 else [1.3,2.1,0.0])
  bands = [(top+plot_h*.20, .14, blue, 'WAVE A'), (top+plot_h*.49, .14, pink, 'WAVE B'), (top+plot_h*.79, .22, accent, 'COMBINED WAVE')]
  for idx,(base,amp,color,label) in enumerate(bands):
   d.text((left+12, base-amp*plot_h-20), label, font=body_font, fill=color)
   points=[]
   for x in xvals:
    a = x*math.tau*float(freq_a)+phase
    b = x*math.tau*float(freq_b)+float(offset)-phase*.35
    yval = math.sin(a) if idx==0 else (math.sin(b) if idx==1 else (math.sin(a)+math.sin(b))*.5)
    points.append((left+12+x*(plot_w-24), base-yval*amp*plot_h))
   if len(points)>1: d.line(points, fill=color, width=max(2,int(min(W,H)*.004)))
   d.line((left+8,base,right-8,base), fill=(120,140,170,70), width=1)
  d.text((left+12, bottom-25), 'The combined displacement is the sum of both waves', font=body_font, fill=muted)

 elif kind == 'orbital_mechanics':
  cx, cy = (left+right)/2, top+plot_h*.49
  scale = min(plot_w,plot_h)*.43
  d.ellipse((cx-8,cy-8,cx+8,cy+8), fill=gold)
  d.ellipse((cx-22,cy-22,cx+22,cy+22), outline=(255,207,100,90), width=1)
  radii = [(0.34,0.27),(0.57,0.43),(0.82,0.62)]
  for i,(rx,ry) in enumerate(radii):
   ax, ay = scale*rx*2, scale*ry*2
   d.ellipse((cx-ax/2,cy-ay/2,cx+ax/2,cy+ay/2), outline=(100,145,190,150), width=2)
   angle = phase*(1.4-i*.22)+i*1.8
   ex, ey = cx+ax*.5*math.cos(angle), cy+ay*.5*math.sin(angle)
   pr = max(4,int(min(W,H)*(.012-i*.002)))
   d.ellipse((ex-pr,ey-pr,ex+pr,ey+pr), fill=(blue,pink,accent)[i])
  d.text((left+12,bottom-26), 'Illustrative orbits • not to scale', font=body_font, fill=muted)

 elif kind == 'sorting_algorithm':
  original = np.asarray(raw_data if raw_data is not None else np.arange(1,65), dtype=int).copy()
  if original.size < 2: original = np.arange(1,65)
  arr = original.copy()
  n = len(arr)
  step = min(n-1, int(p*(n-1)))
  # Replay insertion sort to a deterministic state for this frame.
  for i in range(1, step+1):
   key = arr[i]; j=i-1
   while j>=0 and arr[j]>key:
    arr[j+1]=arr[j]; j-=1
   arr[j+1]=key
  gap = max(1, plot_w/n)
  base_y = bottom-int(plot_h*.12)
  max_height = plot_h*.70
  for i,val in enumerate(arr):
   h = max(2, max_height*float(val)/max(1, int(original.max())))
   x = left+int(i*gap)+1
   color = accent if i<=step else blue
   d.rectangle((x,base_y-h,x+max(1,int(gap)-2),base_y), fill=color)
  d.text((left+12,top+12), f'INSERTION {min(step,n-1)}/{n-1}', font=body_font, fill=white)
  d.text((left+12,bottom-26), 'The sorted prefix grows one item at a time', font=body_font, fill=muted)

 elif kind == 'pathfinding':
  portrait = H>W
  cols, rows = ((13,20) if portrait else (25,13))
  cell = min((plot_w-18)/cols, (plot_h-42)/rows)
  ox = (left+right-cols*cell)/2
  oy = top+24
  grid_rng = np.random.default_rng(int(seed)+911)
  walls = grid_rng.random((rows,cols)) < .24
  start, goal = (0,0), (rows-1,cols-1)
  walls[start]=False; walls[goal]=False
  # Ensure a valid corridor, while leaving other obstacles for exploration.
  walls[0,:]=False; walls[:,cols-1]=False
  from collections import deque
  queue=deque([start]); parent={start:None}; visited=[]
  while queue:
   node=queue.popleft(); visited.append(node)
   if node==goal: break
   r,c=node
   for nxt in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
    nr,nc=nxt
    if 0<=nr<rows and 0<=nc<cols and not walls[nxt] and nxt not in parent:
     parent[nxt]=node; queue.append(nxt)
  path=[]; cur=goal
  while cur is not None and cur in parent:
   path.append(cur); cur=parent[cur]
  path.reverse()
  visible=max(1,int(len(visited)*p))
  visited_set=set(visited[:visible])
  path_set=set(path if p>.82 else [])
  for r in range(rows):
   for c in range(cols):
    x,y=ox+c*cell,oy+r*cell
    if (r,c)==start: color=gold
    elif (r,c)==goal: color=pink
    elif walls[r,c]: color=(32,43,62,245)
    elif (r,c) in path_set: color=accent
    elif (r,c) in visited_set: color=(40,125,180,220)
    else: color=(12,21,36,210)
    d.rounded_rectangle((x+1,y+1,x+cell-1,y+cell-1), radius=max(1,int(cell*.08)), fill=color)
  d.text((left+12,bottom-25), 'Explore in layers; reveal shortest path at the end', font=body_font, fill=muted)

 elif kind == 'flocking_behavior':
  # Lightweight deterministic boids model (separation, alignment, cohesion).
  n=28; brng=np.random.default_rng(int(seed)+31)
  pos=brng.random((n,2))*np.array([plot_w*.75,plot_h*.65])+np.array([left+plot_w*.125,top+plot_h*.16])
  vel=brng.normal(0,1,(n,2)); vel=vel/(np.linalg.norm(vel,axis=1)[:,None]+1e-6)*2.2
  steps=int(p*24)
  for _ in range(steps):
   delta=pos[:,None,:]-pos[None,:,:]
   dist=np.sqrt((delta**2).sum(axis=2))+1e-6
   near=dist<min(W,H)*.13
   np.fill_diagonal(near,False)
   separation=-(delta*(near/(dist**2))[:,:,None]).sum(axis=1)
   counts=near.sum(axis=1).clip(1)[:,None]
   alignment=(near[:,:,None]*vel[None,:,:]).sum(axis=1)/counts-vel
   cohesion=(near[:,:,None]*pos[None,:,:]).sum(axis=1)/counts-pos
   acc=.045*separation+.018*alignment+.0008*cohesion
   vel+=acc
   speed=np.linalg.norm(vel,axis=1).clip(.5,3.4)
   vel=vel/speed[:,None]*np.clip(speed,.5,3.4)[:,None]
   pos+=vel
   pos[:,0]=left+8+np.mod(pos[:,0]-left-8,plot_w-16)
   pos[:,1]=top+8+np.mod(pos[:,1]-top-8,plot_h-16)
  for i,(x,y) in enumerate(pos):
   angle=math.atan2(vel[i,1],vel[i,0]); size=max(4,int(min(W,H)*.012))
   pts=[(x+math.cos(angle)*size*1.5,y+math.sin(angle)*size*1.5),(x+math.cos(angle+2.5)*size,y+math.sin(angle+2.5)*size),(x+math.cos(angle-2.5)*size,y+math.sin(angle-2.5)*size)]
   d.polygon(pts, fill=(accent,blue,pink)[i%3])
  d.text((left+12,bottom-25), 'Simplified model: separation + alignment + cohesion', font=body_font, fill=muted)

 elif kind == 'predator_prey':
  plot_top=top+int(plot_h*.13); plot_bottom=bottom-int(plot_h*.12)
  for i in range(5):
   y=plot_top+i*(plot_bottom-plot_top)/4
   d.line((left+12,y,right-12,y), fill=(100,125,155,65), width=1)
  count=max(2,int(plot_w*.8))
  prey=[]; predator=[]
  for i in range(count):
   t=i/max(1,count-1)*math.tau*2.2
   prey.append(.52+.28*math.sin(t))
   predator.append(.48+.23*math.sin(t-1.05))
  visible=max(2,int(count*p))
  for series,color in ((prey,accent),(predator,pink)):
   pts=[]
   for i,val in enumerate(series[:visible]):
    x=left+12+i*(plot_w-24)/max(1,count-1)
    y=plot_bottom-val*(plot_bottom-plot_top)
    pts.append((x,y))
   if len(pts)>1: d.line(pts, fill=color, width=max(2,int(min(W,H)*.004)))
  d.text((left+12,top+10),'PREY',font=body_font,fill=accent)
  d.text((left+plot_w*.25,top+10),'PREDATOR',font=body_font,fill=pink)
  d.text((left+12,bottom-25), 'Illustrative cycle • real ecosystems are more complex', font=body_font, fill=muted)

 elif kind == 'game_of_life':
  rows,cols=32,48
  cell=min(plot_w/cols,plot_h/rows)
  ox=(left+right-cols*cell)/2; oy=(top+bottom-rows*cell)/2
  state=(np.random.default_rng(int(seed)+51).random((rows,cols))<.27)
  generations=int(p*24)
  for _ in range(generations):
   neigh=sum(np.roll(np.roll(state,dr,axis=0),dc,axis=1) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0))
   state=(neigh==3)|((state)&(neigh==2))
  for r,c in np.argwhere(state):
   x,y=ox+c*cell,oy+r*cell
   d.rounded_rectangle((x+1,y+1,x+cell-1,y+cell-1), radius=max(1,int(cell*.12)), fill=accent if (r+c)%5 else blue)
  d.text((left+12,bottom-25), f'GENERATION {generations}  •  B3/S23 RULE', font=body_font, fill=muted)

 elif kind == 'particle_attractor':
  cx,cy=(left+right)/2,(top+bottom)/2
  maxr=min(plot_w,plot_h)*.46
  n=130; steps=36
  for i in range(n):
   base=i*math.tau/n
   pts=[]
   for j in range(steps):
    t=j/(steps-1)
    a=base + t*(3.0+1.4*math.sin(base*3+phase)) + phase*.2
    r=maxr*(.12+.84*t)*(0.82+.18*math.sin(base*5+t*7+phase))
    pts.append((cx+math.cos(a)*r,cy+math.sin(a)*r*.72))
   if len(pts)>1: d.line(pts, fill=(accent,blue,pink,gold)[i%4], width=1)
   x,y=pts[min(steps-1,int(p*(steps-1)))]
   d.ellipse((x-2,y-2,x+2,y+2), fill=white)
  d.text((left+12,bottom-25), 'Each path follows a changing direction field', font=body_font, fill=muted)

 # Stable takeaway line; keep it concise enough for both layouts.
 takeaway = profile['takeaway']
 d.text((margin, int(H*.84)), takeaway, font=body_font, fill=(220,230,245,255))
 d.rectangle((margin, int(H*.955), W-margin, int(H*.962)), fill=(255,255,255,35))
 d.rectangle((margin, int(H*.955), margin+int((W-2*margin)*max(0,min(1,p))), int(H*.962)), fill=accent)
 return img


def frame(W,H,p,v,kind,method,seed,params,raw_data=None):
 rng=np.random.default_rng(seed); pal=palette(kind); img=Image.new('RGB',(W,H),(3,5,12)); d=ImageDraw.Draw(img,'RGBA'); v=np.resize(v,512); env=max(.02,math.sin(math.pi*p)**.55); chaos=params['chaos']*(.4+1.2*env); phase=p*math.tau
 # Moving volumetric-looking background.
 yy,xx=np.mgrid[0:H,0:W]; g=np.sin(xx/W*math.tau*2+phase*.7)+np.cos(yy/H*math.tau*1.5-phase)
 bg=np.zeros((H,W,3),np.uint8); bg[:,:,0]=np.clip(3+4*(g+2),0,255); bg[:,:,1]=np.clip(5+5*(g+2),0,255); bg[:,:,2]=np.clip(12+10*(g+2),0,255); img=Image.fromarray(bg); d=ImageDraw.Draw(img,'RGBA')
 if kind in PHASE1_PROFILES_BY_KIND:
  return _phase1_frame(img, W, H, p, v, kind, seed, params, raw_data)
 if kind=='primes' and method=='network_organism':
  # Educational P09 pilot: reveal actual primes in a 1–100 grid and show prime gaps.
  def is_prime(n):
   return n > 1 and all(n % q for q in range(2, int(n ** 0.5) + 1))
  portrait = H > W
  margin=max(18,int(W*.035)); top=int(H*(.17 if portrait else .205))
  title_font=font(max(22,int(W*(.030 if portrait else .026)))); sub_font=font(max(13,int(W*.014))); cell_font=font(max(11,int(W*(.018 if portrait else .014))))
  # Native portrait composition stacks the grid and lesson panel instead of cropping landscape.
  grid_w=int(W*(.94 if portrait else .53)); cell_w=grid_w/10
  cell_h=max(25, min(int(H*(.036 if portrait else .064)), int((H*(.39 if portrait else .66))/10))) if portrait else max(25, min(int(H*.064),int((H*.66)/10)))
  # Narrative pacing: hook, progressive reveal, then a short observation/payoff hold.
  if p < .14:
   visible_limit=max(8, min(18, 8 + int((p/.14)*10)))
  elif p < .76:
   visible_limit=max(18, min(100, 18 + int(((p-.14)/.62)*82)))
  else:
   visible_limit=100
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
  if portrait:
   panel_x=margin; panel_w=W-2*margin; panel_y=min(int(H*.72),top+10*cell_h+int(H*.025)); panel_bottom=min(H-int(H*.085),panel_y+int(H*.22))
  else:
   panel_x=int(W*.60); panel_w=int(W*.36); panel_y=int(H*.22); panel_bottom=int(H*.77)
  d.rounded_rectangle((panel_x,panel_y,W-margin,panel_bottom),radius=16,fill=(10,18,33,225),outline=(55,83,115,210),width=2)
  d.text((panel_x+18,panel_y+18),'THE PRIME TEST',font=font(max(16,int(W*.019))),fill=(255,210,90,255))
  if portrait:
   line_gap=max(27,int(H*.022)); small=sub_font
   d.text((panel_x+18,panel_y+line_gap+12),'A prime is greater than 1',font=small,fill=(235,240,250,255))
   d.text((panel_x+18,panel_y+2*line_gap+12),'and has exactly two positive divisors:',font=small,fill=(235,240,250,255))
   d.text((panel_x+18,panel_y+3*line_gap+12),'1 and itself. The number 1 is not prime.',font=small,fill=(255,150,160,255))
   d.text((panel_x+18,panel_y+4*line_gap+12),f'Primes revealed: {len(visible_primes)}',font=small,fill=(100,255,215,255))
   recent_primes=visible_primes[-7:]
   gaps=[b-a for a,b in zip(recent_primes,recent_primes[1:])] if len(recent_primes)>1 else []
   if gaps:
    chart_title_y=panel_y+5*line_gap+8
    d.text((panel_x+18,chart_title_y),'RECENT PRIME GAPS',font=font(max(11,int(W*.018))),fill=(190,210,235,255))
    max_gap=max(gaps); chart_x=panel_x+18; chart_y=chart_title_y+int(H*.018); chart_w=panel_w-40
    slot=chart_w/len(gaps); bar_h=max(5,int(H*.045))
    for j,gap in enumerate(gaps):
     bh=max(4,int((gap/max_gap)*bar_h)); bx=chart_x+j*slot
     d.rounded_rectangle((bx,chart_y+bar_h-bh,bx+max(3,slot-5),chart_y+bar_h),radius=3,fill=(*pal[j%len(pal)],225))
     d.text((bx+1,chart_y+bar_h+3),str(gap),font=font(max(9,int(W*.014))),fill=(205,220,240,255))
   caption_y=min(H-int(H*.045),panel_bottom+int(H*.025))
   caption='THE GAPS CHANGE. THE RULE STAYS THE SAME.' if p >= .76 else ('NOW LOOK AT THE GAPS BETWEEN PRIMES' if p >= .55 else 'WATCH THE PRIME NUMBERS EMERGE')
   d.text((margin,caption_y),caption,font=font(max(13,int(W*.018))),fill=(220,230,245,255))
  else:
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
    max_gap=max(gaps); bar_x=panel_x+18; bar_y=gap_y+28; avail_w=panel_w-38; gap_slot=avail_w/len(gaps)
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

# CONTENT_PROFILES = {
#     "P09": {
#         "concept_id": "prime-number-gaps",
#         "series": "Math Patterns",
#         "hook": "Why are the gaps between prime numbers uneven?",
#         "learning_objective": "A prime number is greater than 1 and has exactly two positive divisors; consecutive-prime gaps vary.",
#         "experiment_id": "prime-gap-visual-v2",
#     },
#     "B04": {
#         "concept_id": "binary-patterns",
#         "series": "Code Made Visible",
#         "hook": "How can simple binary values create complex patterns?",
#         "learning_objective": "Binary data consists of two symbols, 0 and 1; mapping those values to visual states reveals patterns.",
#         "experiment_id": "retention-visual-v1",
#     },
#     "DNA09": {
#         "concept_id": "dna-sequence-network",
#         "series": "Patterns in Nature",
#         "hook": "What happens when a DNA sequence becomes a network?",
#         "learning_objective": "DNA sequences use four nucleotide letters: A, C, G, and T; a visualization can encode sequence relationships.",
#         "experiment_id": "retention-visual-v1",
#     },
#     "F03": {
#         "concept_id": "frequency-waves",
#         "series": "Signals in Motion",
#         "hook": "How does frequency change the shape of a wave?",
#         "learning_objective": "Frequency describes cycles per second and affects how rapidly a periodic signal oscillates.",
#         "experiment_id": "retention-visual-v1",
#     },
#     "RW03": {
#         "concept_id": "random-walk",
#         "series": "Probability in Motion",
#         "hook": "Can random steps create a visible pattern?",
#         "learning_objective": "A random walk accumulates individual steps, producing a path whose overall shape can differ between trials.",
#         "experiment_id": "retention-visual-v1",
#     },
#     "G09": {
#         "concept_id": "graph-network",
#         "series": "Connected Systems",
#         "hook": "How do nodes and connections form a network?",
#         "learning_objective": "A graph represents entities as nodes and relationships as edges.",
#         "experiment_id": "retention-visual-v1",
#     },
#     "CC05": {
#         "concept_id": "color-code-geometry",
#         "series": "Digital Color",
#         "hook": "How does a hexadecimal code represent a color?",
#         "learning_objective": "Hex color notation commonly encodes red, green, and blue channel values.",
#         "experiment_id": "retention-visual-v1",
#     },
#     "M10": {
#         "concept_id": "morse-code-patterns",
#         "series": "Signals in Motion",
#         "hook": "How can dots and dashes encode a message?",
#         "learning_objective": "Morse code represents letters and digits using sequences of short and long signals.",
#         "experiment_id": "retention-visual-v1",
#     },
#     "JS10": {
#         "concept_id": "json-structure",
#         "series": "Data Made Visible",
#         "hook": "What does structured data look like under the surface?",
#         "learning_objective": "JSON organizes data using objects, arrays, keys, and values.",
#         "experiment_id": "retention-visual-v1",
#     },
#     "CO09": {
#         "concept_id": "coordinate-space",
#         "series": "Geometry of Data",
#         "hook": "How do coordinates turn numbers into a spatial pattern?",
#         "learning_objective": "Coordinates encode positions in a defined space; plotting them makes spatial distributions visible.",
#         "experiment_id": "retention-visual-v1",
#     },
#     "TS03": {
#         "concept_id": "time-series-waves",
#         "series": "Signals in Motion",
#         "hook": "What patterns appear when time becomes a signal?",
#         "learning_objective": "A time series records observations in temporal order, allowing changes and rhythms to be visualized.",
#         "experiment_id": "retention-visual-v1",
#     },
#     "HASH09": {
#         "concept_id": "hash-structure",
#         "series": "Code Made Visible",
#         "hook": "Why do hash strings look chaotic but have a fixed structure?",
#         "learning_objective": "A cryptographic hash maps input data to a fixed-length digest; a digest is not an encrypted message.",
#         "experiment_id": "retention-visual-v1",
#     },
# }


# def get_content_profile(g):
#     """Return editorial metadata used for titles, quality review and later analytics."""
#     return dict(CONTENT_PROFILES.get(g["id"].upper(), {
#         "concept_id": f'{g["data_type"]}-{g["method"]}',
#         "series": "RawSignal Experiments",
#         "hook": f'What can {g["data_label"].lower()} reveal when visualized as {g["method_label"].lower()}?',
#         "learning_objective": None,
#         "experiment_id": "exploratory-visual-v1",
#     }))


# def title(g,rng):
#     profile = get_content_profile(g)
#     # Keep the title promise aligned with the actual concept. The P09 wording
#     # is a tested educational pilot; other curated concepts use concise hooks.
#     title_map = {
#         "P09": "Why Do Prime Numbers Have Uneven Gaps?",
#         "B04": "How Binary Data Creates Patterns",
#         "DNA09": "Turning DNA Letters Into a Network",
#         "F03": "How Frequency Changes a Wave",
#         "RW03": "Can Random Steps Create a Pattern?",
#         "G09": "How Nodes and Connections Form a Network",
#         "CC05": "How Hex Codes Represent Color",
#         "M10": "How Dots and Dashes Encode Messages",
#         "JS10": "Making Structured Data Visible",
#         "CO09": "Turning Coordinates Into a Spatial Pattern",
#         "TS03": "Finding Patterns in Time-Series Data",
#         "HASH09": "Why Hashes Look Random but Stay Structured",
#     }
#     if g["id"].upper() in title_map:
#         return title_map[g["id"].upper()]
#     patterns=['Raw {d} Data Became a {m}','I Turned {d} Into a {m}','This {d} Sequence Created a {m}','What Happens When {d} Controls a {m}?','{d} Data Generated This {m}','A Strange {m} Built From {d}']
#     return rng.choice(patterns).format(d=g['data_label'],m=g['method_label'])

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
#   # Start at 2 so the educational pilot can show a coherent sequence.
#   out=[]; x=2
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

# def frame(W,H,p,v,kind,method,seed,params,raw_data=None):
#  rng=np.random.default_rng(seed); pal=palette(kind); img=Image.new('RGB',(W,H),(3,5,12)); d=ImageDraw.Draw(img,'RGBA'); v=np.resize(v,512); env=max(.02,math.sin(math.pi*p)**.55); chaos=params['chaos']*(.4+1.2*env); phase=p*math.tau
#  # Moving volumetric-looking background.
#  yy,xx=np.mgrid[0:H,0:W]; g=np.sin(xx/W*math.tau*2+phase*.7)+np.cos(yy/H*math.tau*1.5-phase)
#  bg=np.zeros((H,W,3),np.uint8); bg[:,:,0]=np.clip(3+4*(g+2),0,255); bg[:,:,1]=np.clip(5+5*(g+2),0,255); bg[:,:,2]=np.clip(12+10*(g+2),0,255); img=Image.fromarray(bg); d=ImageDraw.Draw(img,'RGBA')
#  if kind=='primes' and method=='network_organism':
#   # Educational P09 pilot: reveal actual primes in a 1–100 grid and show prime gaps.
#   def is_prime(n):
#    return n > 1 and all(n % q for q in range(2, int(n ** 0.5) + 1))
#   portrait = H > W
#   margin=max(18,int(W*.035)); top=int(H*(.17 if portrait else .205))
#   title_font=font(max(22,int(W*(.030 if portrait else .026)))); sub_font=font(max(13,int(W*.014))); cell_font=font(max(11,int(W*(.018 if portrait else .014))))
#   # Native portrait composition stacks the grid and lesson panel instead of cropping landscape.
#   grid_w=int(W*(.94 if portrait else .53)); cell_w=grid_w/10
#   cell_h=max(25, min(int(H*(.036 if portrait else .064)), int((H*(.39 if portrait else .66))/10))) if portrait else max(25, min(int(H*.064),int((H*.66)/10)))
#   # Narrative pacing: hook, progressive reveal, then a short observation/payoff hold.
#   if p < .14:
#    visible_limit=max(8, min(18, 8 + int((p/.14)*10)))
#   elif p < .76:
#    visible_limit=max(18, min(100, 18 + int(((p-.14)/.62)*82)))
#   else:
#    visible_limit=100
#   visible_primes=[n for n in range(2,visible_limit+1) if is_prime(n)]
#   d.text((margin,int(H*.055)),'WHY DO PRIME NUMBERS HAVE UNEVEN GAPS?',font=title_font,fill=(245,248,255,255))
#   d.text((margin,int(H*.12)),'A visual experiment in number patterns',font=sub_font,fill=(150,180,210,255))
#   for n in range(1,101):
#    col=(n-1)%10; row=(n-1)//10; x=margin+col*cell_w; y=top+row*cell_h
#    shown=n<=visible_limit; prime=is_prime(n)
#    if shown and prime: fill=(24,150,145,235); outline=(90,255,220,255); text_color=(255,255,255,255)
#    elif shown and n==1: fill=(70,75,90,150); outline=(110,120,140,160); text_color=(170,180,195,210)
#    elif shown: fill=(28,37,54,220); outline=(58,74,96,220); text_color=(145,160,180,255)
#    else: fill=(10,17,30,130); outline=(35,46,65,150); text_color=(90,105,125,180)
#    pad=max(2,int(cell_w*.055))
#    d.rounded_rectangle((x+pad,y+2,x+cell_w-pad,y+cell_h-3),radius=max(3,int(cell_w*.08)),fill=fill,outline=outline,width=1)
#    label=str(n); box=d.textbbox((0,0),label,font=cell_font); tw=box[2]-box[0]; th=box[3]-box[1]
#    d.text((x+(cell_w-tw)/2,y+(cell_h-th)/2-1),label,font=cell_font,fill=text_color)
#   if portrait:
#    panel_x=margin; panel_w=W-2*margin; panel_y=min(int(H*.72),top+10*cell_h+int(H*.025)); panel_bottom=min(H-int(H*.085),panel_y+int(H*.22))
#   else:
#    panel_x=int(W*.60); panel_w=int(W*.36); panel_y=int(H*.22); panel_bottom=int(H*.77)
#   d.rounded_rectangle((panel_x,panel_y,W-margin,panel_bottom),radius=16,fill=(10,18,33,225),outline=(55,83,115,210),width=2)
#   d.text((panel_x+18,panel_y+18),'THE PRIME TEST',font=font(max(16,int(W*.019))),fill=(255,210,90,255))
#   if portrait:
#    line_gap=max(27,int(H*.022)); small=sub_font
#    d.text((panel_x+18,panel_y+line_gap+12),'A prime is greater than 1',font=small,fill=(235,240,250,255))
#    d.text((panel_x+18,panel_y+2*line_gap+12),'and has exactly two positive divisors:',font=small,fill=(235,240,250,255))
#    d.text((panel_x+18,panel_y+3*line_gap+12),'1 and itself. The number 1 is not prime.',font=small,fill=(255,150,160,255))
#    d.text((panel_x+18,panel_y+4*line_gap+12),f'Primes revealed: {len(visible_primes)}',font=small,fill=(100,255,215,255))
#    recent_primes=visible_primes[-7:]
#    gaps=[b-a for a,b in zip(recent_primes,recent_primes[1:])] if len(recent_primes)>1 else []
#    if gaps:
#     chart_title_y=panel_y+5*line_gap+8
#     d.text((panel_x+18,chart_title_y),'RECENT PRIME GAPS',font=font(max(11,int(W*.018))),fill=(190,210,235,255))
#     max_gap=max(gaps); chart_x=panel_x+18; chart_y=chart_title_y+int(H*.018); chart_w=panel_w-40
#     slot=chart_w/len(gaps); bar_h=max(5,int(H*.045))
#     for j,gap in enumerate(gaps):
#      bh=max(4,int((gap/max_gap)*bar_h)); bx=chart_x+j*slot
#      d.rounded_rectangle((bx,chart_y+bar_h-bh,bx+max(3,slot-5),chart_y+bar_h),radius=3,fill=(*pal[j%len(pal)],225))
#      d.text((bx+1,chart_y+bar_h+3),str(gap),font=font(max(9,int(W*.014))),fill=(205,220,240,255))
#    caption_y=min(H-int(H*.045),panel_bottom+int(H*.025))
#    caption='THE GAPS CHANGE. THE RULE STAYS THE SAME.' if p >= .76 else ('NOW LOOK AT THE GAPS BETWEEN PRIMES' if p >= .55 else 'WATCH THE PRIME NUMBERS EMERGE')
#    d.text((margin,caption_y),caption,font=font(max(13,int(W*.018))),fill=(220,230,245,255))
#   else:
#    d.text((panel_x+18,panel_y+58),'A prime is greater than 1',font=sub_font,fill=(235,240,250,255))
#    d.text((panel_x+18,panel_y+82),'and has exactly two positive',font=sub_font,fill=(235,240,250,255))
#    d.text((panel_x+18,panel_y+106),'divisors: 1 and itself.',font=sub_font,fill=(235,240,250,255))
#    d.text((panel_x+18,panel_y+148),'1 is NOT prime.',font=font(max(14,int(W*.016))),fill=(255,130,140,255))
#    d.text((panel_x+18,panel_y+190),f'Primes revealed: {len(visible_primes)}',font=sub_font,fill=(100,255,215,255))
#    recent_primes=visible_primes[-7:]
#    gaps=[b-a for a,b in zip(recent_primes,recent_primes[1:])] if len(recent_primes)>1 else []
#    gap_y=panel_y+235
#    d.text((panel_x+18,gap_y),'Recent gaps between primes',font=sub_font,fill=(235,240,250,255))
#    if gaps:
#     max_gap=max(gaps); bar_x=panel_x+18; bar_y=gap_y+28; avail_w=panel_w-38; gap_slot=avail_w/len(gaps)
#     for i,gap in enumerate(gaps):
#      bh=max(5,int((gap/max_gap)*42)); bx=bar_x+i*gap_slot
#      d.rounded_rectangle((bx,bar_y+42-bh,bx+gap_slot-5,bar_y+42),radius=3,fill=(*pal[i%len(pal)],225))
#      d.text((bx+2,bar_y+48),str(gap),font=font(max(10,int(W*.011))),fill=(200,215,235,255))
#    d.text((margin,int(H*.91)),'Observe: the gaps change; the rule stays the same.',font=sub_font,fill=(220,230,245,255))
#  elif method=='particle_universe':
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
#  if not (kind=='primes' and method=='network_organism'):
#   header=f'{kind.upper()}  •  {method.replace("_"," ").upper()}'
#   d.text((24,20),header,font=font(20),fill=(235,240,250,195))
#  d.rectangle((24,H-25,W-24,H-21),fill=(255,255,255,35)); d.rectangle((24,H-25,24+(W-48)*p,H-21),fill=(*pal[0],190))
#  return img





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
#  # P09 is the first educational pilot: title the idea, not just the effect.
#  if g['id'].upper() == 'P09':
#   return 'Why Do Prime Numbers Have Uneven Gaps?'
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
#   # Start at 2 so the educational pilot can show a coherent sequence.
#   out=[]; x=2
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

# def frame(W,H,p,v,kind,method,seed,params,raw_data=None):
#  rng=np.random.default_rng(seed); pal=palette(kind); img=Image.new('RGB',(W,H),(3,5,12)); d=ImageDraw.Draw(img,'RGBA'); v=np.resize(v,512); env=max(.02,math.sin(math.pi*p)**.55); chaos=params['chaos']*(.4+1.2*env); phase=p*math.tau
#  # Moving volumetric-looking background.
#  yy,xx=np.mgrid[0:H,0:W]; g=np.sin(xx/W*math.tau*2+phase*.7)+np.cos(yy/H*math.tau*1.5-phase)
#  bg=np.zeros((H,W,3),np.uint8); bg[:,:,0]=np.clip(3+4*(g+2),0,255); bg[:,:,1]=np.clip(5+5*(g+2),0,255); bg[:,:,2]=np.clip(12+10*(g+2),0,255); img=Image.fromarray(bg); d=ImageDraw.Draw(img,'RGBA')
#  if kind=='primes' and method=='network_organism':
#   # Educational P09 pilot: reveal actual primes in a 1–100 grid and show prime gaps.
#   def is_prime(n):
#    return n > 1 and all(n % q for q in range(2, int(n ** 0.5) + 1))
#   margin=max(18,int(W*.035)); top=int(H*.205)
#   title_font=font(max(22,int(W*.026))); sub_font=font(max(13,int(W*.014))); cell_font=font(max(11,int(W*.014)))
#   # Grid sizing is proportional to the frame and leaves a separate explanation panel.
#   grid_w=int(W*.53); cell_w=grid_w/10; cell_h=min(int(H*.064),int((H*.66)/10)); cell_h=max(25,cell_h)
#   visible_limit=max(1,min(100,1+int(p*99)))
#   visible_primes=[n for n in range(2,visible_limit+1) if is_prime(n)]
#   d.text((margin,int(H*.055)),'WHY DO PRIME NUMBERS HAVE UNEVEN GAPS?',font=title_font,fill=(245,248,255,255))
#   d.text((margin,int(H*.12)),'A visual experiment in number patterns',font=sub_font,fill=(150,180,210,255))
#   for n in range(1,101):
#    col=(n-1)%10; row=(n-1)//10; x=margin+col*cell_w; y=top+row*cell_h
#    shown=n<=visible_limit; prime=is_prime(n)
#    if shown and prime: fill=(24,150,145,235); outline=(90,255,220,255); text_color=(255,255,255,255)
#    elif shown and n==1: fill=(70,75,90,150); outline=(110,120,140,160); text_color=(170,180,195,210)
#    elif shown: fill=(28,37,54,220); outline=(58,74,96,220); text_color=(145,160,180,255)
#    else: fill=(10,17,30,130); outline=(35,46,65,150); text_color=(90,105,125,180)
#    pad=max(2,int(cell_w*.055))
#    d.rounded_rectangle((x+pad,y+2,x+cell_w-pad,y+cell_h-3),radius=max(3,int(cell_w*.08)),fill=fill,outline=outline,width=1)
#    label=str(n); box=d.textbbox((0,0),label,font=cell_font); tw=box[2]-box[0]; th=box[3]-box[1]
#    d.text((x+(cell_w-tw)/2,y+(cell_h-th)/2-1),label,font=cell_font,fill=text_color)
#   panel_x=int(W*.60); panel_w=int(W*.36); panel_y=int(H*.22)
#   d.rounded_rectangle((panel_x,panel_y,W-margin,int(H*.77)),radius=16,fill=(10,18,33,225),outline=(55,83,115,210),width=2)
#   d.text((panel_x+18,panel_y+18),'THE PRIME TEST',font=font(max(16,int(W*.019))),fill=(255,210,90,255))
#   d.text((panel_x+18,panel_y+58),'A prime is greater than 1',font=sub_font,fill=(235,240,250,255))
#   d.text((panel_x+18,panel_y+82),'and has exactly two positive',font=sub_font,fill=(235,240,250,255))
#   d.text((panel_x+18,panel_y+106),'divisors: 1 and itself.',font=sub_font,fill=(235,240,250,255))
#   d.text((panel_x+18,panel_y+148),'1 is NOT prime.',font=font(max(14,int(W*.016))),fill=(255,130,140,255))
#   d.text((panel_x+18,panel_y+190),f'Primes revealed: {len(visible_primes)}',font=sub_font,fill=(100,255,215,255))
#   recent_primes=visible_primes[-7:]
#   gaps=[b-a for a,b in zip(recent_primes,recent_primes[1:])] if len(recent_primes)>1 else []
#   gap_y=panel_y+235
#   d.text((panel_x+18,gap_y),'Recent gaps between primes',font=sub_font,fill=(235,240,250,255))
#   if gaps:
#    max_gap=max(gaps)
#    bar_x=panel_x+18; bar_y=gap_y+28; avail_w=panel_w-38; gap_slot=avail_w/len(gaps)
#    for i,gap in enumerate(gaps):
#     bh=max(5,int((gap/max_gap)*42)); bx=bar_x+i*gap_slot
#     d.rounded_rectangle((bx,bar_y+42-bh,bx+gap_slot-5,bar_y+42),radius=3,fill=(*pal[i%len(pal)],225))
#     d.text((bx+2,bar_y+48),str(gap),font=font(max(10,int(W*.011))),fill=(200,215,235,255))
#   d.text((margin,int(H*.91)),'Observe: the gaps change; the rule stays the same.',font=sub_font,fill=(220,230,245,255))
#  elif method=='particle_universe':
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
#  if not (kind=='primes' and method=='network_organism'):
#   header=f'{kind.upper()}  •  {method.replace("_"," ").upper()}'
#   d.text((24,20),header,font=font(20),fill=(235,240,250,195))
#  d.rectangle((24,H-25,W-24,H-21),fill=(255,255,255,35)); d.rectangle((24,H-25,24+(W-48)*p,H-21),fill=(*pal[0],190))
#  return img











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
