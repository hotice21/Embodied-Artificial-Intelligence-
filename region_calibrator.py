"""屏幕区域自动标定工具。

用法示例：
  python region_calibrator.py --label player --templates "resources/player/*.png" --write-config

脚本会对给定模板在主显示器上做多尺度模板匹配，输出推荐的 [x,y,w,h]。
可选地把结果写回 `config.yml` 对应字段。
"""
import argparse
import glob
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image
    import mss
except Exception as exc:
    print('Missing dependency (cv2/numpy/PIL/mss). Install before running:', exc)
    sys.exit(1)


def grab_screen():
    with mss.mss() as sct:
        mon = sct.monitors[1]
        img = sct.grab(mon)
        im = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY), mon


def best_match_fullscreen(screen_gray, template_gray, scales=(1.0, 0.9, 0.8, 1.1)):
    best = (0.0, None)  # score, (x,y,w,h)
    th, tw = template_gray.shape[:2]
    for s in scales:
        try:
            if s != 1.0:
                tmpl = cv2.resize(template_gray, (max(2, int(tw * s)), max(2, int(th * s))), interpolation=cv2.INTER_AREA)
            else:
                tmpl = template_gray
        except Exception:
            tmpl = template_gray
        if screen_gray.shape[0] < tmpl.shape[0] or screen_gray.shape[1] < tmpl.shape[1]:
            continue
        res = cv2.matchTemplate(screen_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best[0]:
            x, y = max_loc
            h, w = tmpl.shape[:2]
            best = (float(max_val), (int(x), int(y), int(w), int(h)))
    return best


def aggregate_boxes(boxes, pad=8):
    if not boxes:
        return None
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    ws = [b[2] for b in boxes]
    hs = [b[3] for b in boxes]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(x + w for x, y, w, h in boxes)
    y2 = max(y + h for x, y, w, h in boxes)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    return [x1, y1, x2 - x1 + pad, y2 - y1 + pad]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--templates', required=True, help='glob for template images')
    p.add_argument('--label', required=True, choices=['player', 'player_health', 'enemy'], help='which config field to write')
    p.add_argument('--write-config', action='store_true', help='write results to config.yml')
    p.add_argument('--scales', default='1.0,0.9,0.8,1.1', help='comma-separated scales to try')
    p.add_argument('--top-k', type=int, default=5, help='top matches to consider for aggregation')
    args = p.parse_args()

    templates = sorted(glob.glob(args.templates))
    if not templates:
        print('No templates found for', args.templates)
        sys.exit(2)

    screen_gray, mon = grab_screen()
    scales = [float(s) for s in args.scales.split(',') if s.strip()]

    matches = []  # (score, (x,y,w,h), template_path)
    for tpath in templates:
        t = cv2.imread(tpath, cv2.IMREAD_GRAYSCALE)
        if t is None:
            print('Failed to read template', tpath)
            continue
        score, box = best_match_fullscreen(screen_gray, t, scales=scales)
        if box is not None:
            matches.append((score, box, tpath))
            print(f'Found {tpath} -> score={score:.3f} box={box}')

    if not matches:
        print('No matches found on screen')
        sys.exit(3)

    matches.sort(key=lambda x: x[0], reverse=True)
    top = [m[1] for m in matches[:args.top_k]]
    agg = aggregate_boxes(top)
    print('\nSuggested region for', args.label, ':', agg)

    if args.write_config:
        cfg_path = Path(__file__).parent / 'config.yml'
        if not cfg_path.exists():
            print('config.yml not found at', cfg_path)
            sys.exit(4)
        import yaml
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if 'screen' not in cfg:
            cfg['screen'] = {}
        if 'combat' not in cfg['screen']:
            cfg['screen']['combat'] = {}
        if args.label == 'player':
            cfg['screen']['combat']['player_region'] = agg
        elif args.label == 'player_health':
            cfg['screen']['combat']['player_health_region'] = agg
        elif args.label == 'enemy':
            # write as single-element list
            cfg['screen']['combat']['enemy_regions'] = [agg]
        with open(cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        print('Wrote suggestion to config.yml')


if __name__ == '__main__':
    main()
