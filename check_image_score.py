import sys
from PIL import Image, ImageOps
import pytesseract
import re
import os
import shutil
from pathlib import Path


def configure_tesseract():
    candidates = []
    env_cmd = os.environ.get('TESSERACT_CMD')
    if env_cmd:
        candidates.append(env_cmd)

    which_cmd = shutil.which('tesseract')
    if which_cmd:
        candidates.append(which_cmd)

    candidates.extend([
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ])

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            print('TESSERACT_CMD=', candidate)
            return candidate

    print('TESSERACT_CMD=NOT_FOUND')
    return None


configure_tesseract()

path = sys.argv[1] if len(sys.argv) > 1 else r'SerpentAI-unsupervised/SerpentAI-dev/datasets/picture/image3.png'

try:
    im = Image.open(path)
except Exception as e:
    print('ERROR_OPEN', e)
    sys.exit(2)

def ocr_score_from_pil(im):
    try:
        proc_im = ImageOps.autocontrast(im.convert('L'))
    except Exception:
        proc_im = im.convert('L')
    try:
        data = pytesseract.image_to_data(proc_im, lang='chi_sim+eng', config='--psm 6', output_type=pytesseract.Output.DICT)
        nums = []
        texts = data.get('text', [])
        for i, t in enumerate(texts):
            if t and t.strip().isdigit():
                try:
                    val = int(t.strip())
                except Exception:
                    continue
                left = data.get('left', [0]*len(texts))[i]
                width = data.get('width', [0]*len(texts))[i]
                center_x = left + width/2
                nums.append((center_x, val))
        if nums:
            nums.sort(key=lambda x: x[0], reverse=True)
            return nums[0][1]
    except Exception:
        pass

    try:
        txt = pytesseract.image_to_string(proc_im, lang='chi_sim+eng', config='--psm 6')
    except Exception:
        try:
            txt = pytesseract.image_to_string(proc_im, config='--psm 6')
        except Exception:
            return None

    m = re.search(r"分数\D*(\d+)", txt)
    if not m:
        m = re.search(r"分\D*(\d+)", txt)
    if not m:
        m = re.search(r"\b(\d{3,})\b", txt)
    if not m:
        nums = re.findall(r"\b(\d+)\b", txt)
        if nums:
            long_nums = [n for n in nums if len(n) >= 3]
            if long_nums:
                return int(long_nums[0])
            else:
                return int(max(nums, key=lambda s: int(s)))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

res = ocr_score_from_pil(im)
print('IMAGE_PATH=', path)
print('OCR_FULL_IMAGE=', res)

# 尝试使用 config.yml 中的 region，以及顶部中间和更宽顶部裁切
cfg_path = r'SerpentAI-unsupervised/config.yml'
try:
    import yaml
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    region = cfg.get('screen', {}).get('score_region', [10, 10, 300, 60])
except Exception:
    region = [10, 10, 300, 60]

W, H = im.size

def crop_and_test(box, name):
    x, y, w, h = box
    x = max(0, x)
    y = max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)
    crop = im.crop((x, y, x + w, y + h))
    print(f'REGION {name} {box}:')
    # 试多个预处理策略
    methods = []
    def m_autocontrast(img):
        return ImageOps.autocontrast(img.convert('L'))
    methods.append(('autocontrast', m_autocontrast))

    def m_resize2(img):
        im2 = img.convert('L').resize((img.width*2, img.height*2), Image.NEAREST)
        return ImageOps.autocontrast(im2)
    methods.append(('resize_x2', m_resize2))

    def m_resize3(img):
        im2 = img.convert('L').resize((img.width*3, img.height*3), Image.NEAREST)
        return ImageOps.autocontrast(im2)
    methods.append(('resize_x3', m_resize3))

    # 尝试 OpenCV 自适应阈值（如果可用）
    try:
        import cv2
        import numpy as np
        def m_adaptive(img):
            a = np.array(img.convert('L'))
            a = cv2.resize(a, (img.width*2, img.height*2), interpolation=cv2.INTER_CUBIC)
            a = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            return Image.fromarray(a)
        methods.append(('cv2_adaptive', m_adaptive))
    except Exception:
        pass

    for mname, mfn in methods:
        try:
            proc = mfn(crop)
            val = ocr_score_from_pil(proc)
            print(f'  [{mname}] ->', val)
            try:
                txt = pytesseract.image_to_string(proc, lang='chi_sim+eng', config='--psm 6')
                print('   RAW:', txt.strip().replace('\n', ' | '))
            except Exception:
                pass
        except Exception as e:
            print(f'  [{mname}] ERROR', e)

crop_and_test(region, 'config')

# top-middle: 高度取图片高度的30%，宽度居中64%
top_h = max(int(H * 0.30), region[3])
left = int(W * 0.18)
width = max(int(W * 0.64), region[2])
crop_and_test((left, 0, width, top_h), 'top_middle')

# full top
crop_and_test((0, 0, W, top_h), 'full_top')
