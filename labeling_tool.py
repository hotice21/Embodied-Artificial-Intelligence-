from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import simpledialog, messagebox

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

from vision_detector import build_vision_detector


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

ANNOTATIONS_CSV = 'annotations_manual.csv'
GENERATED_LABELS_CSV = 'resources/labels_generated.csv'
LABEL_PROPERTIES_JSON = 'resources/label_properties.json'
LABEL_BANK_DIR = 'resources/label_bank'
MAX_LABEL_BUTTONS = 12


def default_label_property(label: str):
    label = str(label).strip()
    if label == 'ground':
        return {
            'kind': 'background',
            'trainable': False,
            'match_threshold': 0.68,
            'suppresses': ['enemy', 'bullets_enemy', 'bullets_player'],
            'suppress_iou': 0.08,
        }
    return {
        'kind': 'positive',
        'trainable': True,
    }


def load_label_properties(root_dir: Path):
    path = root_dir / LABEL_PROPERTIES_JSON
    props = {}
    if path.exists():
        try:
            with path.open('r', encoding='utf-8') as fh:
                loaded = json.load(fh) or {}
            if isinstance(loaded, dict):
                props = {str(k): v for k, v in loaded.items() if isinstance(v, dict)}
        except Exception as exc:
            logging.warning(f'Failed to load label properties: {exc}')
    if 'ground' not in props:
        props['ground'] = default_label_property('ground')
    return props


def save_label_properties(root_dir: Path, props: Dict[str, Dict]):
    path = root_dir / LABEL_PROPERTIES_JSON
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(props, fh, ensure_ascii=False, indent=2, sort_keys=True)


def ensure_label_property(props: Dict[str, Dict], label: str):
    if label not in props:
        props[label] = default_label_property(label)
    if label == 'ground':
        props[label].update(default_label_property('ground'))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_photo_image(bgr, max_w: int = 520, max_h: int = 420):
    if cv2 is None or Image is None or ImageTk is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    w, h = img.size
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def crop_from_box(frame_bgr, box: Tuple[int, int, int, int]):
    x, y, w, h = box
    h_img, w_img = frame_bgr.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w_img, int(x + w))
    y2 = min(h_img, int(y + h))
    return frame_bgr[y1:y2, x1:x2].copy()


class LabelingUI:
    def __init__(self, root_dir: Path, labels: List[str], quick_mode: bool):
        self.root_dir = root_dir
        self.labels = sorted(set(labels))
        self.quick_mode = quick_mode
        self.current_item = None
        self.current_crop = None
        self._resolved = False
        self._result = None

        self.root = tk.Tk()
        self.root.title('SerpentAI Manual Labeling UI')
        self.root.geometry('900x680')
        self.root.configure(bg='#1a1d22')

        self.photo_ref = None

        title = tk.Label(self.root, text='Manual Labeling', fg='white', bg='#1a1d22', font=('Segoe UI', 18, 'bold'))
        title.pack(pady=10)

        self.info_var = tk.StringVar(value='Waiting for detections...')
        info = tk.Label(self.root, textvariable=self.info_var, fg='#d9e1ea', bg='#1a1d22', font=('Consolas', 10))
        info.pack(pady=4)

        self.image_label = tk.Label(self.root, bg='#0b0d10', width=560, height=430, bd=1, relief='solid')
        self.image_label.pack(pady=8)

        self.buttons_frame = tk.Frame(self.root, bg='#1a1d22')
        self.buttons_frame.pack(pady=8)

        self.suggest_frame = tk.Frame(self.root, bg='#1a1d22')
        self.suggest_frame.pack(pady=4)

        self.quick_frame = tk.Frame(self.root, bg='#1a1d22')
        self.quick_frame.pack(pady=6)

        self.footer_frame = tk.Frame(self.root, bg='#1a1d22')
        self.footer_frame.pack(pady=8)

        self._rebuild_label_buttons()
        self._build_quick_buttons()
        self._build_footer_buttons()

        self.root.bind('<Escape>', lambda _e: self._quit())

    def _rebuild_label_buttons(self):
        for child in self.buttons_frame.winfo_children():
            child.destroy()

        display_labels = self.labels[:MAX_LABEL_BUTTONS]
        cols = 4
        for i, label in enumerate(display_labels):
            btn = tk.Button(
                self.buttons_frame,
                text=f'{i+1}. {label}',
                width=20,
                command=lambda lb=label: self._select_label(lb),
                bg='#2b3440',
                fg='white',
                activebackground='#3c4a59',
                activeforeground='white',
            )
            btn.grid(row=i // cols, column=i % cols, padx=6, pady=6)

        other_btn = tk.Button(
            self.buttons_frame,
            text='Other (new label)',
            width=20,
            command=self._other_label,
            bg='#515b69',
            fg='white',
            activebackground='#69768a',
        )
        other_btn.grid(row=(len(display_labels) // cols) + 1, column=0, padx=6, pady=8)

        # suggested labels quick-add
        for child in self.suggest_frame.winfo_children():
            child.destroy()
        suggested = ['ground', 'coin', 'tear']
        for s in suggested:
            if s in self.labels:
                continue
            btn = tk.Button(self.suggest_frame, text=f'Add {s}', width=12, command=lambda lb=s: self._other_quick(lb), bg='#3b4750', fg='white')
            btn.pack(side='left', padx=6)

    def _other_quick(self, label_name: str):
        if label_name and label_name not in self.labels:
            self.labels.append(label_name)
            self.labels.sort()
            self._rebuild_label_buttons()
        self._set_result(('label', label_name))

    def _build_quick_buttons(self):
        for child in self.quick_frame.winfo_children():
            child.destroy()

        if not self.quick_mode:
            return

        tk.Button(
            self.quick_frame,
            text='Correct (Y)',
            width=18,
            command=lambda: self._quick_judge(True),
            bg='#2f7d4a',
            fg='white',
        ).pack(side='left', padx=8)

        tk.Button(
            self.quick_frame,
            text='Incorrect (N)',
            width=18,
            command=lambda: self._quick_judge(False),
            bg='#8a3b3b',
            fg='white',
        ).pack(side='left', padx=8)

    def _build_footer_buttons(self):
        tk.Button(
            self.footer_frame,
            text='Skip',
            width=14,
            command=self._skip,
            bg='#404854',
            fg='white',
        ).pack(side='left', padx=8)

        tk.Button(
            self.footer_frame,
            text='Reject (irrelevant)',
            width=18,
            command=self._reject,
            bg='#55575a',
            fg='white',
        ).pack(side='left', padx=8)

        tk.Button(
            self.footer_frame,
            text='Quit',
            width=14,
            command=self._quit,
            bg='#6a3d3d',
            fg='white',
        ).pack(side='left', padx=8)

    def _set_result(self, payload):
        self._result = payload
        self._resolved = True

    def _select_label(self, label: str):
        self._set_result(('label', label))

    def _other_label(self):
        text = simpledialog.askstring('Other Label', '请输入新标签名（例如: coin 或 floor）：', parent=self.root)
        if not text:
            return
        text = text.strip()
        if not text:
            return
        if text not in self.labels:
            self.labels.append(text)
            self.labels.sort()
            self._rebuild_label_buttons()
        self._set_result(('label', text))

    def _quick_judge(self, ok: bool):
        self._set_result(('quick', bool(ok)))

    def _skip(self):
        self._set_result(('skip', None))

    def _reject(self):
        self._set_result(('reject', None))

    def _quit(self):
        self._set_result(('quit', None))

    def bind_hotkeys(self):
        for i in range(1, 10):
            self.root.bind(str(i), lambda _e, idx=i: self._pick_by_index(idx))
        self.root.bind('o', lambda _e: self._other_label())
        self.root.bind('s', lambda _e: self._skip())
        self.root.bind('q', lambda _e: self._quit())
        if self.quick_mode:
            self.root.bind('y', lambda _e: self._quick_judge(True))
            self.root.bind('n', lambda _e: self._quick_judge(False))
        self.root.bind('r', lambda _e: self._reject())

    def _pick_by_index(self, idx: int):
        if idx <= 0:
            return
        visible = self.labels[:MAX_LABEL_BUTTONS]
        if idx <= len(visible):
            self._select_label(visible[idx - 1])

    def ask_for_item(self, item: Dict, crop_bgr):
        self.current_item = item
        self.current_crop = crop_bgr
        self._resolved = False
        self._result = None

        detected = item.get('detected_label', 'unknown')
        score = item.get('score', 0.0)
        box = item.get('box', (0, 0, 0, 0))
        self.info_var.set(f'detected={detected}  score={score:.3f}  box={box}')

        if crop_bgr is not None and crop_bgr.size > 0:
            photo = to_photo_image(crop_bgr)
            if photo is not None:
                self.photo_ref = photo
                self.image_label.configure(image=self.photo_ref, text='')
            else:
                self.image_label.configure(image='', text='PIL/OpenCV not available')
        else:
            self.image_label.configure(image='', text='Empty crop')

        while not self._resolved:
            try:
                self.root.update_idletasks()
                self.root.update()
            except Exception:
                # window closed or destroyed externally
                self._set_result(('quit', None))
                break
            time.sleep(0.01)

        return self._result


def read_config(config_path: Path):
    try:
        import yaml
        with config_path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def write_annotation_row(csv_path: Path, header: List[str], row: List):
    existed = csv_path.exists()
    with csv_path.open('a', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh)
        if not existed:
            writer.writerow(header)
        writer.writerow(row)


def save_labeled_crop(root_dir: Path, crop_bgr, chosen_label: str, detected_label: str, score: float, label_props: Optional[Dict[str, Dict]] = None):
    ts = int(time.time() * 1000)
    label_dir = root_dir / LABEL_BANK_DIR / chosen_label
    ensure_dir(label_dir)
    img_path = label_dir / f'{ts}_{chosen_label}.png'
    cv2.imwrite(str(img_path), crop_bgr)

    if label_props is not None:
        ensure_label_property(label_props, chosen_label)
        save_label_properties(root_dir, label_props)

    rel_image = img_path.relative_to(root_dir / 'resources')
    generated_csv = root_dir / GENERATED_LABELS_CSV
    write_annotation_row(
        generated_csv,
        ['image', 'label', 'source', 'detected_label', 'score', 'timestamp_ms'],
        [str(rel_image).replace('\\', '/'), chosen_label, 'labeling_tool', detected_label, float(score), ts],
    )

    csv_path = root_dir / ANNOTATIONS_CSV
    row = [ts, str(img_path.relative_to(root_dir)), detected_label, chosen_label, float(score), 'manual_label']
    write_annotation_row(
        csv_path,
        ['timestamp_ms', 'image_path', 'detected_label', 'chosen_label', 'score', 'mode'],
        row,
    )
    return img_path


def save_quick_judge(root_dir: Path, detected_label: str, score: float, ok: bool):
    ts = int(time.time() * 1000)
    csv_path = root_dir / ANNOTATIONS_CSV
    row = [ts, '', detected_label, '', float(score), 'quick_ok' if ok else 'quick_ng']
    write_annotation_row(
        csv_path,
        ['timestamp_ms', 'image_path', 'detected_label', 'chosen_label', 'score', 'mode'],
        row,
    )


def build_detection_queue(detector, frame_bgr):
    analysis_frame, _analysis_scale = detector._resize_for_analysis(frame_bgr)
    frame_gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
    queue = []

    labels = sorted(list(detector._template_sets.keys()))
    if hasattr(detector, 'label_properties') and isinstance(detector.label_properties, dict):
        labels = sorted(set(labels) | set(detector.label_properties.keys()))
    for label in labels:
        templates = detector._template_sets.get(label, [])
        if not templates:
            continue
        if hasattr(detector, '_label_match_threshold'):
            fallback = 0.70
            if label == detector.cfg.player_label:
                fallback = detector.cfg.player_template_threshold
            elif label == detector.cfg.enemy_label:
                fallback = detector.cfg.enemy_template_threshold
            elif label == detector.cfg.player_bullet_label:
                fallback = detector.cfg.player_bullet_template_threshold
            elif label == detector.cfg.enemy_bullet_label:
                fallback = detector.cfg.enemy_bullet_template_threshold
            thr = detector._label_match_threshold(label, fallback)
        else:
            if label == detector.cfg.player_label:
                thr = detector.cfg.player_template_threshold
            elif label == detector.cfg.enemy_label:
                thr = detector.cfg.enemy_template_threshold
            elif label == detector.cfg.player_bullet_label:
                thr = detector.cfg.player_bullet_template_threshold
            elif label == detector.cfg.enemy_bullet_label:
                thr = detector.cfg.enemy_bullet_template_threshold
            else:
                thr = 0.70

        matches = detector._match_templates(frame_gray, templates, float(thr), max_results=8)
        for m in matches:
            queue.append({
                'detected_label': label,
                'score': float(m.get('score', 0.0)),
                'box_analysis': m.get('box'),
            })

    queue.sort(key=lambda x: x['score'], reverse=True)
    return queue


def analysis_box_to_original(box, analysis_scale: float):
    x, y, w, h = box
    if analysis_scale and abs(float(analysis_scale) - 1.0) > 1e-6:
        s = float(analysis_scale)
        return (int(x / s), int(y / s), int(w / s), int(h / s))
    return (int(x), int(y), int(w), int(h))


def run_labeling_ui(quick_mode: bool = False, max_items: int = 300):
    if cv2 is None or Image is None or ImageTk is None:
        raise RuntimeError('Missing required UI dependencies: opencv-python and pillow')

    root_dir = Path(__file__).resolve().parent
    cfg = read_config(root_dir / 'config.yml')
    screen_cfg = cfg.get('screen', {})

    detector = build_vision_detector(screen_cfg) if callable(build_vision_detector) else None
    if detector is None:
        raise RuntimeError('Vision detector unavailable. Check config.yml screen.combat and dependencies.')

    label_props = load_label_properties(root_dir)
    labels = sorted(set(detector._template_sets.keys()) | set(label_props.keys()) | {'ground'})
    ui = LabelingUI(root_dir, labels, quick_mode=quick_mode)
    ui.bind_hotkeys()

    count = 0
    while count < max_items:
        frame_bgr, _origin = detector._capture_frame()
        if frame_bgr is None:
            time.sleep(0.05)
            ui.root.update_idletasks()
            ui.root.update()
            continue

        queue = build_detection_queue(detector, frame_bgr)
        if not queue:
            ui.info_var.set('No detections this frame, waiting...')
            time.sleep(0.03)
            ui.root.update_idletasks()
            ui.root.update()
            continue

        for item in queue:
            if count >= max_items:
                break

            box_analysis = item.get('box_analysis')
            if box_analysis is None:
                continue
            box = analysis_box_to_original(box_analysis, float(detector.cfg.analysis_scale))
            item['box'] = box

            crop = crop_from_box(frame_bgr, box)
            if crop is None or crop.size == 0:
                continue

            action, payload = ui.ask_for_item(item, crop)
            if action == 'quit':
                messagebox.showinfo('Labeling', f'已退出。共处理 {count} 条样本。')
                return
            if action == 'skip':
                # save ambiguous crop for later review
                ts = int(time.time() * 1000)
                amb_dir = root_dir / LABEL_BANK_DIR / '_skipped'
                ensure_dir(amb_dir)
                amb_path = amb_dir / f'{ts}_skipped.png'
                try:
                    cv2.imwrite(str(amb_path), crop)
                except Exception:
                    amb_path = None
                write_annotation_row(root_dir / ANNOTATIONS_CSV, ['timestamp_ms', 'image_path', 'detected_label', 'chosen_label', 'score', 'mode'], [ts, str(amb_path.relative_to(root_dir)) if amb_path else '', item['detected_label'], 'skipped', float(item['score']), 'skipped'])
                count += 1
                continue
            if action == 'reject':
                # save rejected crop under _rejected
                ts = int(time.time() * 1000)
                rej_dir = root_dir / LABEL_BANK_DIR / '_rejected'
                ensure_dir(rej_dir)
                rej_path = rej_dir / f'{ts}_rejected.png'
                try:
                    cv2.imwrite(str(rej_path), crop)
                except Exception:
                    rej_path = None
                write_annotation_row(root_dir / ANNOTATIONS_CSV, ['timestamp_ms', 'image_path', 'detected_label', 'chosen_label', 'score', 'mode'], [ts, str(rej_path.relative_to(root_dir)) if rej_path else '', item['detected_label'], 'rejected', float(item['score']), 'rejected'])
                count += 1
                continue
            if action == 'quick':
                save_quick_judge(root_dir, item['detected_label'], item['score'], bool(payload))
                count += 1
                continue
            if action == 'label':
                chosen = str(payload)
                saved = save_labeled_crop(root_dir, crop, chosen, item['detected_label'], item['score'], label_props)
                logging.info(f'Saved: {saved}')
                count += 1

    messagebox.showinfo('Labeling', f'达到上限，已完成 {count} 条样本。')


def main():
    parser = argparse.ArgumentParser(description='SerpentAI interactive labeling UI')
    parser.add_argument('--quick', action='store_true', help='Enable quick supervision mode (Y/N only)')
    parser.add_argument('--max-items', type=int, default=300, help='Max number of samples in one run')
    args = parser.parse_args()

    run_labeling_ui(quick_mode=bool(args.quick), max_items=int(args.max_items))


if __name__ == '__main__':
    main()
