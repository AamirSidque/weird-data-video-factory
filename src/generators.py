"""
60 production generator combinations:
6 data types × 10 visual methods.

The data type changes how symbols are interpreted; the visual method
changes the rendering model. Everything is deterministic from a seed.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


DATA_TYPES = {
    "decimal": "Decimal",
    "binary": "Binary",
    "hex": "Hexadecimal",
    "octal": "Octal",
    "alphabet": "Alphabet",
    "alphanumeric": "Alphanumeric",
}

METHODS = {
    1: "particle_galaxy",
    2: "flow_field",
    3: "wave_ocean",
    4: "cellular_life",
    5: "geometry_machine",
    6: "fractal_world",
    7: "data_sculpture",
    8: "fire_rain",
    9: "network_organism",
    10: "data_program",
}

TITLE_TEMPLATES = {
    "decimal": {
        "particle_galaxy": "I Turned Random Decimal Numbers Into a Galaxy",
        "flow_field": "What Do Random Decimal Numbers Look Like?",
        "wave_ocean": "A Mathematical Ocean Made From Decimal Data",
        "cellular_life": "Decimal Numbers Created This Living Pattern",
        "geometry_machine": "I Turned Numbers Into a Geometric Machine",
        "fractal_world": "A Fractal World Generated From Numbers",
        "data_sculpture": "Random Numbers Became a 3D Data Sculpture",
        "fire_rain": "Random Numbers Became a Digital Storm",
        "network_organism": "Random Numbers Created a Living Network",
        "data_program": "I Let Random Numbers Program a Visual World",
    },
    "binary": {
        "particle_galaxy": "I Turned 0s and 1s Into a Digital Universe",
        "flow_field": "What Does a Binary Flow Field Look Like?",
        "wave_ocean": "A Wave Made Entirely From Binary Data",
        "cellular_life": "I Turned Binary Data Into a Living Organism",
        "geometry_machine": "0s and 1s Built This Machine",
        "fractal_world": "A Fractal World Made From Binary",
        "data_sculpture": "Binary Data Became a Digital Sculpture",
        "fire_rain": "Millions of 0s and 1s Became a Storm",
        "network_organism": "Binary Data Created This Living Network",
        "data_program": "I Used Binary Data as a Visual Program",
    },
    "hex": {
        "particle_galaxy": "I Turned Hexadecimal Data Into a Color Universe",
        "flow_field": "Hexadecimal Data Became a Color Flow",
        "wave_ocean": "A Color Ocean Generated From Raw Hex",
        "cellular_life": "Hexadecimal Data Created This Living Pattern",
        "geometry_machine": "Raw Hexadecimal Data Built This Machine",
        "fractal_world": "I Turned Hex Data Into a Fractal World",
        "data_sculpture": "Hexadecimal Data Became a Color Sculpture",
        "fire_rain": "Raw Hex Data Became a Digital Firestorm",
        "network_organism": "Hexadecimal Data Created a Color Network",
        "data_program": "I Let Hexadecimal Data Program a World",
    },
    "octal": {
        "particle_galaxy": "I Turned Octal Digits Into a Mechanical Universe",
        "flow_field": "Octal Data Became a Geometric Flow",
        "wave_ocean": "An Ocean Generated From Octal Numbers",
        "cellular_life": "Octal Digits Created This Strange Organism",
        "geometry_machine": "I Turned Octal Digits Into a Machine",
        "fractal_world": "An Octal Fractal That Keeps Growing",
        "data_sculpture": "Octal Data Became a Geometric Sculpture",
        "fire_rain": "Octal Digits Created This Mechanical Storm",
        "network_organism": "Octal Data Created a Living Machine",
        "data_program": "I Used Octal Digits as Visual Instructions",
    },
    "alphabet": {
        "particle_galaxy": "I Turned the Alphabet Into a Galaxy",
        "flow_field": "Letters Became a Flowing Universe",
        "wave_ocean": "An Ocean Made From Letters",
        "cellular_life": "The Alphabet Became a Living Organism",
        "geometry_machine": "Letters Built This Impossible Machine",
        "fractal_world": "I Turned Letters Into a Fractal World",
        "data_sculpture": "The Alphabet Became a Living Sculpture",
        "fire_rain": "Thousands of Letters Became a Digital Storm",
        "network_organism": "The Alphabet Became a Living Network",
        "data_program": "I Used Letters as a Visual Programming Language",
    },
    "alphanumeric": {
        "particle_galaxy": "I Turned Random Characters Into a Universe",
        "flow_field": "Alphanumeric Data Became a Flowing World",
        "wave_ocean": "An Ocean Generated From Letters and Numbers",
        "cellular_life": "Random Characters Created a Living Organism",
        "geometry_machine": "Letters and Numbers Built This Machine",
        "fractal_world": "Alphanumeric Data Became a Fractal World",
        "data_sculpture": "Random Characters Became a Digital Sculpture",
        "fire_rain": "Letters and Numbers Became a Digital Storm",
        "network_organism": "Alphanumeric Data Created This Network",
        "data_program": "I Let Random Characters Program a World",
    },
}


@dataclass
class GeneratorSpec:
    generator_id: str
    data_type: str
    method: str
    title: str


def all_generators() -> list[GeneratorSpec]:
    out = []
    prefix = {"decimal": "D", "binary": "B", "hex": "H",
              "octal": "O", "alphabet": "A", "alphanumeric": "X"}
    for dt in DATA_TYPES:
        for n, method in METHODS.items():
            out.append(GeneratorSpec(
                f"{prefix[dt]}{n:02d}", dt, method,
                TITLE_TEMPLATES[dt][method]
            ))
    return out


def choose_generator(rng: np.random.Generator, fixed_id: str | None = None) -> GeneratorSpec:
    gens = all_generators()
    if fixed_id:
        for g in gens:
            if g.generator_id.lower() == fixed_id.lower():
                return g
        raise ValueError(f"Unknown generator ID: {fixed_id}")
    return gens[int(rng.integers(0, len(gens)))]


def make_data(data_type: str, rng: np.random.Generator, count: int = 4096):
    if data_type == "decimal":
        return rng.integers(0, 1000, count, dtype=np.int32)
    if data_type == "binary":
        return rng.integers(0, 2, count, dtype=np.uint8)
    if data_type == "hex":
        return rng.integers(0, 256, count, dtype=np.uint8)
    if data_type == "octal":
        return rng.integers(0, 8, count, dtype=np.uint8)
    if data_type == "alphabet":
        return np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))[
            rng.integers(0, 26, count)
        ]
    if data_type == "alphanumeric":
        chars = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"))
        return chars[rng.integers(0, len(chars), count)]
    raise ValueError(data_type)


def _norm(data):
    if data.dtype.kind in "OUS":
        vals = np.array([ord(str(x)[0]) for x in data], dtype=np.float32)
    else:
        vals = data.astype(np.float32)
    lo, hi = float(vals.min()), float(vals.max())
    return (vals - lo) / (hi - lo + 1e-9)


def _font(size=18):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _palette(data_type, t):
    palettes = {
        "decimal": ((40, 180, 255), (255, 70, 180)),
        "binary": ((40, 255, 150), (20, 100, 255)),
        "hex": ((255, 40, 150), (60, 210, 255)),
        "octal": ((255, 150, 35), (255, 245, 80)),
        "alphabet": ((180, 80, 255), (70, 220, 255)),
        "alphanumeric": ((20, 230, 255), (255, 80, 180)),
    }
    a, b = palettes[data_type]
    return tuple(int(a[i] * (1-t) + b[i] * t) for i in range(3))


def _base(size):
    return Image.new("RGB", size, (4, 6, 12))


def _particles(draw, W, H, data, t, data_type, rng):
    vals = _norm(data)
    n = min(700, len(vals))
    idx = np.linspace(0, len(vals)-1, n).astype(int)
    v = vals[idx]
    # deterministic pseudo-orbits
    theta = idx * 0.031 + t * (0.35 + v * 0.9)
    radius = 20 + v * min(W, H) * 0.44
    cx = W/2 + np.cos(theta * .7) * W*.07
    cy = H/2 + np.sin(theta * .9) * H*.07
    x = cx + np.cos(theta) * radius
    y = cy + np.sin(theta) * radius * .55
    for j in range(n):
        r = 1 + int(v[j] * 5)
        c = _palette(data_type, (v[j] + .5*math.sin(theta[j])) % 1)
        draw.ellipse((x[j]-r, y[j]-r, x[j]+r, y[j]+r), fill=c)


def _flow(draw, W, H, data, t, data_type):
    vals = _norm(data)
    grid = 30
    step = 28
    for gy in range(-1, H//step + 1):
        for gx in range(-1, W//step + 1):
            i = (gx + gy*grid) % len(vals)
            v = vals[i]
            ang = (v*math.tau + t*.7 + gx*.12 - gy*.08)
            x = gx*step + step/2
            y = gy*step + step/2
            length = 5 + v*18
            x2 = x + math.cos(ang)*length
            y2 = y + math.sin(ang)*length
            c = _palette(data_type, v)
            draw.line((x,y,x2,y2), fill=c, width=1)


def _waves(draw, W, H, data, t, data_type):
    vals = _norm(data)
    for row in range(8):
        base = H*(.15 + row*.1)
        pts=[]
        for x in range(0, W+1, 6):
            i=(x//6 + row*73)%len(vals)
            v=vals[i]
            y=base + math.sin(x*.025 + t*2 + row*.7)*18 + (v-.5)*45
            pts.append((x,y))
        c=_palette(data_type, row/8)
        draw.line(pts, fill=c, width=2)


def _cellular(draw, W, H, data, t, data_type):
    # Use a compact 80x45 binary-ish state derived from any data.
    gw, gh = 80, 45
    vals = _norm(data)
    seed = np.array([vals[(i*17)%len(vals)] > .5 for i in range(gw*gh)], dtype=np.uint8).reshape(gh,gw)
    steps = int(t*8)
    for _ in range(steps):
        n = sum(np.roll(np.roll(seed, dy, 0), dx, 1)
                for dy in (-1,0,1) for dx in (-1,0,1)
                if not (dx==0 and dy==0))
        seed = (((seed==1)&((n==2)|(n==3))) | ((seed==0)&(n==3))).astype(np.uint8)
    sx, sy = W/gw, H/gh
    c1=_palette(data_type,.2); c2=_palette(data_type,.8)
    for y in range(gh):
        for x in range(gw):
            if seed[y,x]:
                draw.rectangle((x*sx,y*sy,(x+1)*sx,(y+1)*sy), fill=c1 if (x+y)%2 else c2)


def _geometry(draw, W, H, data, t, data_type):
    vals = _norm(data)
    n=90
    cx,cy=W/2,H/2
    for k in range(n):
        v=vals[(k*31)%len(vals)]
        a=k*0.37+t*(.4+v)
        r=20+(k/n)*min(W,H)*.42
        x=cx+math.cos(a)*r
        y=cy+math.sin(a)*r*.65
        s=5+v*22
        sides=3+int(v*6)
        pts=[]
        for q in range(sides):
            aa=a+q*math.tau/sides
            pts.append((x+math.cos(aa)*s,y+math.sin(aa)*s))
        draw.polygon(pts, outline=_palette(data_type,v), width=2)


def _fractal(draw, W, H, data, t, data_type):
    # Low-res Julia-like iteration, intentionally CPU-light.
    vals=_norm(data)
    max_side=180
    scale=min(W,H)/max_side
    ox,oy=W/2,H/2
    c_re=-.75 + .2*math.sin(t*.2) + (vals[0]-.5)*.08
    c_im=.12*math.cos(t*.17) + (vals[1]-.5)*.08
    for py in range(0,H,3):
        for px in range(0,W,3):
            x=(px-ox)/(H*.42)
            y=(py-oy)/(H*.42)
            it=0
            while x*x+y*y<4 and it<18:
                x,y=x*x-y*y+c_re,2*x*y+c_im
                it+=1
            if it>2:
                c=_palette(data_type,(it/18 + vals[(px+py)%len(vals)])%1)
                draw.rectangle((px,py,px+3,py+3),fill=c)


def _sculpture(draw, W, H, data, t, data_type):
    vals=_norm(data)
    cx,cy=W/2,H/2
    for layer in range(28):
        v=vals[(layer*113)%len(vals)]
        radius=15+layer*7+v*40
        sides=5+int(v*7)
        rot=t*(.2+v)+layer*.17
        pts=[]
        for q in range(sides):
            a=rot+q*math.tau/sides
            pts.append((cx+math.cos(a)*radius, cy+math.sin(a)*radius*.55))
        draw.line(pts+[pts[0]],fill=_palette(data_type,layer/28),width=2)


def _fire_rain(draw, W, H, data, t, data_type):
    vals=_norm(data)
    for i in range(260):
        v=vals[(i*47)%len(vals)]
        x=(i*83)%W
        speed=30+v*150
        y=(i*17 + t*speed)%H
        length=5+v*35
        c=_palette(data_type,v)
        draw.line((x,y,x+(v-.5)*10,y+length),fill=c,width=max(1,int(1+v*2)))


def _network(draw, W, H, data, t, data_type):
    vals=_norm(data)
    n=55
    nodes=[]
    for i in range(n):
        v=vals[(i*71)%len(vals)]
        a=i*math.tau/n+t*(.05+v*.08)
        r=min(W,H)*(.12+.30*((i*37)%100)/100)
        x=W/2+math.cos(a)*r
        y=H/2+math.sin(a)*r*.7
        nodes.append((x,y,v))
    for i,(x,y,v) in enumerate(nodes):
        for j in range(i+1,min(i+5,n)):
            x2,y2,v2=nodes[j]
            if (i+j)%3==0:
                draw.line((x,y,x2,y2),fill=_palette(data_type,.25),width=1)
    for x,y,v in nodes:
        r=2+v*7
        draw.ellipse((x-r,y-r,x+r,y+r),fill=_palette(data_type,v))


def _data_program(draw, W, H, data, t, data_type):
    vals=_norm(data)
    cx,cy=W/2,H/2
    x,y=cx,cy
    angle=t*.3
    pts=[(x,y)]
    for i in range(240):
        v=vals[(i*19)%len(vals)]
        op=int(v*8)
        if op==0: angle-=.15+v*.5
        elif op==1: angle+=.15+v*.5
        elif op==2: x+=math.cos(angle)*(2+v*9)
        elif op==3: y+=math.sin(angle)*(2+v*9)
        elif op==4: x=cx+(x-cx)*.98
        elif op==5: y=cy+(y-cy)*.98
        elif op==6: angle+=math.sin(i)*.05
        else: x+=math.sin(i*.2)*v*3
        pts.append((x,y))
        if x<0 or x>W or y<0 or y>H:
            x,y=cx,cy
    if len(pts)>1:
        draw.line(pts,fill=_palette(data_type,.65),width=2)
    for i,(px,py) in enumerate(pts[::12]):
        r=2+(i%5)
        draw.ellipse((px-r,py-r,px+r,py+r),fill=_palette(data_type,(i/len(pts))%1))


METHOD_FN = {
    "particle_galaxy": _particles,
    "flow_field": _flow,
    "wave_ocean": _waves,
    "cellular_life": _cellular,
    "geometry_machine": _geometry,
    "fractal_world": _fractal,
    "data_sculpture": _sculpture,
    "fire_rain": _fire_rain,
    "network_organism": _network,
    "data_program": _data_program,
}


def render_frame(spec: GeneratorSpec, data, frame_no: int, fps: int, size, seed: int):
    W,H=size
    t=frame_no/fps
    img=_base(size)
    draw=ImageDraw.Draw(img)

    # Subtle animated background grid gives the output a consistent channel identity.
    if spec.method not in ("fractal_world", "cellular_life"):
        grid_alpha = 20
        for x in range(0,W,40):
            draw.line((x,0,x,H),fill=(grid_alpha,grid_alpha,grid_alpha))
        for y in range(0,H,40):
            draw.line((0,y,W,y),fill=(grid_alpha,grid_alpha,grid_alpha))

    METHOD_FN[spec.method](draw,W,H,data,t,spec.data_type)

    # Small generator signature. It makes every video self-identifying without
    # requiring an external title card.
    f=_font(12)
    label=f"{spec.generator_id}  •  {DATA_TYPES[spec.data_type]}"
    draw.text((10,H-20),label,font=f,fill=(120,130,145))

    return img
