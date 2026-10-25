"""Bounded spatial DSL. Coordinates are (row, column); bboxes are inclusive.

Zero is the default background. Components are monochrome. Bounding-box
containment is explicitly distinct from topological enclosure. No learned state.
"""
from dataclasses import dataclass, asdict
from collections import deque
from time import monotonic
from math import isfinite


def freeze(grid):
    if not isinstance(grid, (list, tuple)) or not 1 <= len(grid) <= 30:
        raise ValueError('grid height must be 1..30')
    width = len(grid[0])
    if not 1 <= width <= 30 or any(len(row) != width for row in grid):
        raise ValueError('rectangular grid width must be 1..30')
    if any(type(c) is not int or not 0 <= c <= 9 for row in grid for c in row):
        raise ValueError('ARC colors must be integers 0..9')
    return tuple(tuple(row) for row in grid)


@dataclass(frozen=True)
class Component:
    id: int
    color: int
    pixels: tuple
    bbox: tuple
    mask: tuple
    centroid: tuple

    @property
    def size(self):
        return len(self.pixels)


@dataclass(frozen=True)
class Relation:
    source: int
    kind: str
    target: int


@dataclass(frozen=True)
class Scene:
    grid: tuple
    objects: tuple
    relations: tuple
    connectivity: int
    background: int


def perceive(grid, connectivity=4, background=0):
    g = freeze(grid)
    if connectivity not in (4, 8) or background not in range(10):
        raise ValueError('invalid segmentation settings')
    h, w = len(g), len(g[0])
    steps = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if connectivity == 8:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    seen, objects = set(), []
    for r in range(h):
        for c in range(w):
            if g[r][c] == background or (r, c) in seen:
                continue
            todo, pixels = [(r, c)], []
            seen.add((r, c))
            while todo:
                y, x = todo.pop()
                pixels.append((y, x))
                for dy, dx in steps:
                    p = y + dy, x + dx
                    if 0 <= p[0] < h and 0 <= p[1] < w and p not in seen and g[p[0]][p[1]] == g[r][c]:
                        seen.add(p)
                        todo.append(p)
            pixels.sort()
            y0, x0 = min(y for y, x in pixels), min(x for y, x in pixels)
            y1, x1 = max(y for y, x in pixels), max(x for y, x in pixels)
            mask = tuple((y-y0, x-x0) for y, x in pixels)
            objects.append(Component(len(objects), g[r][c], tuple(pixels), (y0, x0, y1, x1), mask,
                                     (sum(y for y, x in pixels)/len(pixels), sum(x for y, x in pixels)/len(pixels))))
            if len(objects) > 128:
                raise ValueError('scene exceeds 128-component safety bound')
    relations = []
    sets = [set(o.pixels) for o in objects]
    neighbors = [{(y+dy, x+dx) for y, x in o.pixels for dy, dx in ((-1,0),(1,0),(0,-1),(0,1))} for o in objects]
    diagonal = [{(y+dy, x+dx) for y, x in o.pixels for dy in (-1,0,1) for dx in (-1,0,1)} for o in objects]
    for a in objects:
        for b in objects:
            if a.id == b.id:
                continue
            ay, ax, az, aw = a.bbox
            by, bx, bz, bw = b.bbox
            kinds = []
            if neighbors[a.id] & sets[b.id]: kinds.append('touching4')
            if diagonal[a.id] & sets[b.id]: kinds.append('touching8')
            if ay <= by and ax <= bx and az >= bz and aw >= bw and a.bbox != b.bbox: kinds.append('bbox_contains')
            if aw < bx: kinds.append('left_of')
            if az < by: kinds.append('above')
            if a.mask == b.mask: kinds.append('same_shape')
            if a.color == b.color: kinds.append('same_color')
            if a.centroid[0] == b.centroid[0]: kinds.append('row_aligned')
            if a.centroid[1] == b.centroid[1]: kinds.append('column_aligned')
            relations.extend(Relation(a.id, kind, b.id) for kind in kinds)
    return Scene(g, tuple(objects), tuple(relations), connectivity, background)


@dataclass(frozen=True)
class Selector:
    kind: str = 'largest'
    value: int = 0


@dataclass(frozen=True)
class Operation:
    kind: str
    selector: Selector = Selector()
    args: tuple = ()


@dataclass(frozen=True)
class Program:
    operations: tuple = ()
    connectivity: int = 4

    def __post_init__(self):
        if type(self.operations) is not tuple or self.connectivity not in (4, 8) or len(self.operations) > 2:
            raise ValueError('program bound: two operations, 4/8 connectivity')
        for op in self.operations:
            if not isinstance(op, Operation) or not isinstance(op.selector, Selector):
                raise ValueError('typed operations required')
            if op.selector.kind not in ('all','largest','smallest','leftmost','rightmost','topmost','bottommost','color','contained','touching','isolated','index'):
                raise ValueError('unknown selector')
            if type(op.selector.value) is not int or not 0 <= op.selector.value <= 127:
                raise ValueError('invalid selector value')
            arity = {'crop':0, 'keep':0, 'recolor':1, 'translate':2, 'reflect':1, 'rotate':1, 'fill_rectangle':1, 'fill_holes':1}
            if type(op.args) is not tuple or op.kind not in arity or len(op.args) != arity[op.kind] or any(type(v) is not int for v in op.args):
                raise ValueError('unknown operation or invalid arguments')
            if op.kind in ('recolor','fill_rectangle','fill_holes') and not 0 <= op.args[0] <= 9:
                raise ValueError('invalid color')
            if op.kind == 'translate' and any(abs(v) > 29 for v in op.args): raise ValueError('translation bound')
            if op.kind == 'reflect' and op.args[0] not in (0,1): raise ValueError('reflection axis')
            if op.kind == 'rotate' and op.args[0] not in (1,2,3): raise ValueError('quarter turns')

    def trace(self):
        return asdict(self)


def select(scene, selector):
    objs, kind = scene.objects, selector.kind
    if not objs: return ()
    if kind == 'all': return objs
    if kind == 'color': return tuple(o for o in objs if o.color == selector.value)
    if kind == 'index': return objs[selector.value:selector.value+1]
    if kind in ('contained','touching','isolated'):
        ids = {r.target for r in scene.relations if r.kind == ('bbox_contains' if kind == 'contained' else 'touching4')}
        return tuple(o for o in objs if (o.id in ids) == (kind != 'isolated'))
    keys = {'largest':lambda o:(-o.size,o.id), 'smallest':lambda o:(o.size,o.id),
            'leftmost':lambda o:(o.bbox[1],o.id), 'rightmost':lambda o:(-o.bbox[3],o.id),
            'topmost':lambda o:(o.bbox[0],o.id), 'bottommost':lambda o:(-o.bbox[2],o.id)}
    return (min(objs, key=keys[kind]),)


def execute(program, grid):
    if not isinstance(program, Program): raise ValueError('Program required')
    g = [list(row) for row in freeze(grid)]
    for op in program.operations:
        scene = perceive(g, program.connectivity)
        objs = select(scene, op.selector)
        if not objs: raise ValueError('empty selection')
        h, w = len(g), len(g[0])
        if op.kind == 'crop':
            y0, x0 = min(o.bbox[0] for o in objs), min(o.bbox[1] for o in objs)
            y1, x1 = max(o.bbox[2] for o in objs), max(o.bbox[3] for o in objs)
            g = [row[x0:x1+1] for row in g[y0:y1+1]]
            continue
        out = [[0]*w for _ in range(h)] if op.kind == 'keep' else [row[:] for row in g]
        moving = op.kind in ('translate','reflect','rotate')
        if moving:
            for o in objs:
                for y, x in o.pixels: out[y][x] = 0
        writes = {}
        for o in objs:
            y0, x0, y1, x1 = o.bbox
            pixels = o.pixels
            color = op.args[0] if op.kind in ('recolor','fill_rectangle','fill_holes') else o.color
            if op.kind == 'fill_rectangle':
                pixels = tuple((y,x) for y in range(y0,y1+1) for x in range(x0,x1+1))
            elif op.kind == 'fill_holes':
                # Flood exterior through every cell except this object's mask.
                barrier, exterior = set(o.pixels), set()
                todo = deque([(y,x) for y in range(h) for x in range(w) if y in (0,h-1) or x in (0,w-1)])
                while todo:
                    p = todo.popleft()
                    if p in exterior or p in barrier: continue
                    exterior.add(p)
                    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                        q = p[0]+dy, p[1]+dx
                        if 0 <= q[0] < h and 0 <= q[1] < w and q not in exterior: todo.append(q)
                pixels = tuple((y,x) for y in range(h) for x in range(w) if (y,x) not in exterior and g[y][x] == 0)
            for y, x in pixels:
                if op.kind == 'translate': y, x = y+op.args[0], x+op.args[1]
                elif op.kind == 'reflect':
                    y, x = (y0+y1-y,x) if op.args[0] == 0 else (y,x0+x1-x)
                elif op.kind == 'rotate':
                    ry, rx, oh, ow = y-y0, x-x0, y1-y0+1, x1-x0+1
                    for _ in range(op.args[0]): ry, rx, oh, ow = rx, oh-1-ry, ow, oh
                    y, x = y0+ry, x0+rx
                if not (0 <= y < h and 0 <= x < w): raise ValueError('out-of-bounds transform')
                if moving and (out[y][x] != 0 or ((y,x) in writes and writes[y,x] != color)):
                    raise ValueError('object collision')
                writes[y,x] = color
        for (y,x), color in writes.items(): out[y][x] = color
        g = out
    return g


def agreement(predictions, examples):
    exact, pixel = 0, 0.0
    for pred, ex in zip(predictions, examples):
        target = ex['output']
        exact += pred == target
        if pred is not None and len(pred) == len(target) and all(len(a) == len(b) for a,b in zip(pred,target)):
            total = sum(len(r) for r in target)
            pixel += sum(a == b for r,s in zip(pred,target) for a,b in zip(r,s))/total
    return exact, pixel/len(examples) if examples else 0.0


def _library(examples):
    colors = sorted({c for ex in examples for k in ('input','output') for row in ex[k] for c in row})
    selectors = [Selector(k) for k in ('largest','smallest','all','leftmost','rightmost','topmost','bottommost','contained','touching','isolated')]
    selectors += [Selector('color', c) for c in colors if c]
    # Simple structural rules precede constant-color and displacement rules.
    for kind in ('crop','keep'):
        for selector in selectors: yield Operation(kind,selector)
    for selector in selectors:
        for kind in ('recolor','fill_holes','fill_rectangle'):
            for c in colors: yield Operation(kind,selector,(c,))
        for axis in (0,1): yield Operation('reflect',selector,(axis,))
        for turns in (1,2,3): yield Operation('rotate',selector,(turns,))
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)):
            yield Operation('translate',selector,(dy,dx))


def synthesize(train, max_candidates=512, budget_ms=500, deadline=None, beam_width=8, max_depth=2, connectivity=4, progress=None):
    """Discover on train[:-1], then validate once on untouched train[-1].

    Holdout never influences candidate enumeration/ranking. Failed validation
    is reported, not used to restart discovery. Behavioral dedup uses discovery
    examples only. Candidate limits include failed and duplicate attempts.
    """
    if (type(max_candidates) is not int or max_candidates < 0 or not isfinite(budget_ms) or budget_ms < 0
            or type(beam_width) is not int or not 1 <= beam_width <= 32 or max_depth not in (1,2)):
        raise ValueError('invalid search bounds')
    end = min(deadline if deadline is not None else float('inf'), monotonic()+budget_ms/1000)
    discovery = train[:-1] if len(train) >= 2 else train
    if not discovery: raise ValueError('training examples required')
    for ex in train: freeze(ex['input']); freeze(ex['output'])
    library = tuple(_library(discovery))
    attempts, duplicates, invalid = 0, 0, 0
    seen, beam, best = set(), [], None
    stopped = 'space_exhausted'
    for depth in range(1,max_depth+1):
        parents = [Program((),connectivity)] if depth == 1 else [p for _,p in beam[:beam_width]]
        next_beam = []
        for parent in parents:
            for op in library:
                if attempts >= max_candidates or monotonic() >= end:
                    stopped = 'candidate_limit' if attempts >= max_candidates else 'deadline'
                    break
                program = Program(parent.operations+(op,),connectivity)
                attempts += 1
                if progress is not None:
                    progress.update(attempts=attempts,duplicates=duplicates,invalid=invalid,unique=len(seen))
                try: preds = [execute(program,e['input']) for e in discovery]
                except ValueError:
                    invalid += 1
                    continue
                signature = tuple(freeze(g) for g in preds)
                if signature in seen:
                    duplicates += 1
                    continue
                seen.add(signature)
                score = agreement(preds,discovery)
                rank = (score[0],score[1],-depth)
                next_beam.append((rank,program))
                if best is None or rank > best[0]: best = rank,program
                if score[0] == len(discovery):
                    stopped = 'discovery_exact'
                    break
            if stopped != 'space_exhausted': break
        beam = sorted(next_beam,key=lambda x:x[0],reverse=True)[:beam_width]
        if stopped != 'space_exhausted': break
    result = {'program':None,'attempts':attempts,'duplicates':duplicates,'invalid':invalid,
              'unique':len(seen),'stop_reason':stopped,'train_score':(0,0.0),'holdout':None}
    if best is not None:
        program = best[1]
        preds = []
        for ex in train:
            try: preds.append(execute(program,ex['input']))
            except ValueError: preds.append(None)
        result.update(program=program,train_score=agreement(preds,train),
                      holdout=(preds[-1] == train[-1]['output']) if len(train) >= 2 else None,
                      trace=program.trace())
        if result['train_score'][0] == len(train): result['stop_reason'] = 'all_train_exact'
    return result
