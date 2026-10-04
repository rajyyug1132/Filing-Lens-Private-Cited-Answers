"""Contact sheet of UI screenshots. usage: python scripts/ui-sheet.py <dir> <vp> <theme> <cols> <out.png> [state,state,...]"""
import sys, os
from PIL import Image, ImageDraw
d, vp, theme, cols, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
want = sys.argv[6].split(',') if len(sys.argv) > 6 else None
names = sorted({f.rsplit('-' + vp + '-', 1)[0] for f in os.listdir(d) if f.endswith(f'-{vp}-{theme}.png') or f.endswith(f'-{vp}-{theme}.FAIL.png')})
names = [n for n in (want or names) if n in names]
tw = 390 if vp == '390' else 720
ims = []
for n in names:
    p = f'{d}/{n}-{vp}-{theme}.png'
    if not os.path.exists(p): p = f'{d}/{n}-{vp}-{theme}.FAIL.png'
    im = Image.open(p).convert('RGB'); im = im.resize((tw, round(im.height * tw / im.width)), Image.LANCZOS); ims.append((n, im))
rows = [ims[i:i + cols] for i in range(0, len(ims), cols)]
H = [max(i.height for _, i in r) + 22 for r in rows]
sheet = Image.new('RGB', (cols * (tw + 8), sum(H)), (128, 128, 128)); dr = ImageDraw.Draw(sheet); y = 0
for r, h in zip(rows, H):
    for c, (n, im) in enumerate(r):
        sheet.paste(im, (c * (tw + 8), y + 20)); dr.text((c * (tw + 8) + 4, y + 4), n, fill=(255, 255, 255))
    y += h
sheet.save(out)
