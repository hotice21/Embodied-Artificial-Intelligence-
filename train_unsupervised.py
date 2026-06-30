import sys
import os
import time
import random
import yaml
import logging
import shutil
from pathlib import Path
from collections import deque
import platform
import pickle
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

try:
    from vision_detector import build_vision_detector
except Exception:
    build_vision_detector = None
try:
    from vision_worker import VisionWorker
except Exception:
    VisionWorker = None

# 导入增强模块
try:
    from player_detector import EnhancedPlayerDetector
    ENHANCED_PLAYER_DETECTOR_AVAILABLE = True
except Exception as e:
    logging.warning(f"Enhanced player detector not available: {e}")
    ENHANCED_PLAYER_DETECTOR_AVAILABLE = False

try:
    from rl_trainer import ImprovedRLTrainer, DEFAULT_CONFIG
    RL_TRAINER_AVAILABLE = True
except Exception as e:
    logging.warning(f"RL trainer not available: {e}")
    RL_TRAINER_AVAILABLE = False

# 导入GPU强化学习训练器（改进版）
try:
    from rl_trainer_gpu import ImprovedPPOTrainer, DEFAULT_GPU_CONFIG, TORCH_AVAILABLE, DEVICE
    RL_GPU_TRAINER_AVAILABLE = True
except Exception as e:
    logging.warning(f"GPU RL trainer not available: {e}")
    RL_GPU_TRAINER_AVAILABLE = False
    TORCH_AVAILABLE = False
    DEVICE = None

# 导入训练可视化模块
try:
    from training_visualizer import TrainingLogger, TrainingVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False
    TrainingLogger = None
    TrainingVisualizer = None


def configure_tesseract():
    try:
        import pytesseract
    except Exception:
        logging.warning("pytesseract import failed")
        return None

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
            logging.info(f'Using tesseract executable: {candidate}')
            return candidate

    logging.warning('tesseract.exe not found; OCR will fail until it is installed or configured')
    return None


configure_tesseract()


def get_cli_int(flag_name, default_value):
    """修复：同时支持 --key=val 和 --key val 两种传参格式"""
    for idx, arg in enumerate(sys.argv):
        if arg == flag_name:
            if idx + 1 < len(sys.argv):
                try:
                    val = int(sys.argv[idx + 1])
                    logging.info(f"Parsed {flag_name} = {val}")
                    return val
                except (ValueError, IndexError) as e:
                    logging.debug(f"Parse {flag_name} failed: {e}")
        elif arg.startswith(f"{flag_name}="):
            try:
                val = int(arg.split('=', 1)[1])
                logging.info(f"Parsed {flag_name} = {val}")
                return val
            except ValueError as e:
                logging.debug(f"Parse {flag_name} failed: {e}")
    logging.info(f"{flag_name} not set, use default: {default_value}")
    return default_value


def preflight_focus(screen_cfg):
    focus_point = screen_cfg.get('focus_point')
    if not focus_point:
        return

    try:
        import pyautogui
    except ImportError:
        logging.warning('pyautogui not available; skipping focus pre-step')
        return

    try:
        x, y = focus_point
        pyautogui.moveTo(int(x), int(y), duration=0.15)
        if screen_cfg.get('focus_click', False):
            pyautogui.click()
        wait_seconds = float(screen_cfg.get('focus_wait', 1.5))
        logging.info(f'Focus pre-step: moved to ({int(x)}, {int(y)}) wait={wait_seconds}s click={screen_cfg.get("focus_click", False)}')
        if wait_seconds > 0:
            time.sleep(wait_seconds)
    except Exception as exc:
        logging.warning(f'Focus pre-step failed: {exc}')


def ocr_score_and_hit_from_pil(im, min_score=1000, require_label=False):
    from collections import Counter
    import re
    try:
        from PIL import ImageOps
        import pytesseract
    except ImportError as e:
        logging.debug(f"OCR lib import error: {e}")
        return None, None, -1

    try:
        proc_im = ImageOps.autocontrast(im.convert('L'))
    except Exception as e:
        logging.debug(f"Image preprocess fail: {e}")
        proc_im = im.convert('L')

    entries = []
    try:
        data = pytesseract.image_to_data(proc_im, lang='chi_sim+eng', config='--psm 6', output_type=pytesseract.Output.DICT)
        texts = data.get('text', [])
        confs = data.get('conf', [-1] * len(texts))
        blocks = data.get('block_num', [0] * len(texts))
        paras = data.get('par_num', [0] * len(texts))
        lines = data.get('line_num', [0] * len(texts))
        lefts = data.get('left', [0] * len(texts))
        tops = data.get('top', [0] * len(texts))
        widths = data.get('width', [0] * len(texts))
        heights = data.get('height', [0] * len(texts))

        for i, t in enumerate(texts):
            if not t or not t.strip():
                continue
            text = t.strip()
            left = lefts[i]
            top = tops[i]
            width = widths[i]
            height = heights[i]
            right = left + width
            center = (left + width / 2, top + height / 2)
            line_key = (blocks[i], paras[i], lines[i])

            conf = -1
            try:
                conf = int(confs[i])
            except (ValueError, TypeError):
                try:
                    conf = float(confs[i])
                except (ValueError, TypeError):
                    conf = -1

            digits = re.sub(r'[^0-9]', '', text)
            if digits:
                try:
                    value = int(digits)
                except ValueError:
                    value = None
                if value is not None:
                    entries.append({
                        'kind': 'num',
                        'value': value,
                        'text': text,
                        'conf': conf,
                        'box': (left, top, right, top + height),
                        'center': center,
                        'line_key': line_key,
                    })

            if '分数' in text or text == '分' or text.startswith('分'):
                entries.append({
                    'kind': 'label',
                    'value': None,
                    'text': text,
                    'conf': conf,
                    'box': (left, top, right, top + height),
                    'center': center,
                    'line_key': line_key,
                })
    except Exception as e:
        logging.debug(f"OCR parse data fail: {e}")

    label_hits = [entry for entry in entries if entry['kind'] == 'label']
    num_hits = [entry for entry in entries if entry['kind'] == 'num' and entry['value'] >= min_score]

    def pick_best_candidate(candidates):
        if not candidates:
            return None
        vals = [entry['value'] for entry in candidates]
        cnt = Counter(vals)
        top_count = cnt.most_common(1)[0][1]
        top_vals = [v for v, c in cnt.most_common() if c == top_count]
        if len(top_vals) == 1:
            chosen_val = top_vals[0]
        else:
            rightmost = None
            for entry in candidates:
                if entry['value'] in top_vals:
                    x = entry['center'][0]
                    if rightmost is None or x > rightmost[0]:
                        rightmost = (x, entry)
            return rightmost[1] if rightmost is not None else candidates[0]
        return next((entry for entry in candidates if entry['value'] == chosen_val), candidates[0])

    if label_hits and num_hits:
        same_line = []
        for label in label_hits:
            lx1, ly1, lx2, ly2 = label['box']
            for num in num_hits:
                nx1, ny1, nx2, ny2 = num['box']
                y_overlap = min(ly2, ny2) - max(ly1, ny1)
                if num['line_key'] == label['line_key'] or y_overlap >= -2:
                    if nx1 >= lx2 - 6:
                        same_line.append(num)
        if same_line:
            chosen = pick_best_candidate(same_line)
            return chosen['value'], chosen['center'], chosen.get('conf', -1)

    if require_label:
        return None, None, -1

    if num_hits:
        chosen = pick_best_candidate(num_hits)
        return chosen['value'], chosen['center'], chosen.get('conf', -1)

    return None, None, -1


ROOT = Path(__file__).resolve().parent.parent
SERPENT_DEV = ROOT / 'SerpentAI-dev'
if SERPENT_DEV.exists():
    sys.path.insert(0, str(SERPENT_DEV))

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


def get_score_from_region(region, region_name='config', return_hit=False, locked=False):
    try:
        import mss
        from PIL import Image
        import pytesseract
    except ImportError as e:
        logging.debug(f"Screenshot/OCR lib missing: {e}")
        return (None, None, -1) if return_hit else None

    x, y, w, h = region

    def try_once():
        with mss.mss() as sct:
            img = sct.grab({'left': x, 'top': y, 'width': w, 'height': h})
            im = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
            score, hit, conf = ocr_score_and_hit_from_pil(im, require_label=not locked)
            if score is not None and score >= 0:
                logging.info(f'OCR 识别位置: {region_name} 区域 {region} -> {score} hit={hit} conf={conf}')
                return (score, hit, conf) if return_hit else score

            if locked:
                return None

            monitor = sct.monitors[1]
            top_w = monitor['width']
            top_h = max(int(monitor['height'] * 0.18), h, 60)
            top_left_img = sct.grab({
                'left': monitor['left'],
                'top': monitor['top'],
                'width': max(int(monitor['width'] * 0.34), w),
                'height': top_h
            })
            top_left_im = Image.frombytes('RGB', top_left_img.size, top_left_img.bgra, 'raw', 'BGRX')
            score, hit, conf = ocr_score_and_hit_from_pil(top_left_im, require_label=True)
            if score is not None and score >= 0:
                logging.info(f'OCR 识别位置: top_left 回退区域 -> {score} hit={hit} conf={conf}')
                return (score, hit, conf) if return_hit else score

            top_h = max(int(monitor['height'] * 0.30), h)
            left = int(monitor['width'] * 0.18)
            width = max(int(monitor['width'] * 0.64), w)
            top_img = sct.grab({
                'left': monitor['left'] + left,
                'top': monitor['top'],
                'width': width,
                'height': top_h
            })
            top_im = Image.frombytes('RGB', top_img.size, top_img.bgra, 'raw', 'BGRX')
            score, hit, conf = ocr_score_and_hit_from_pil(top_im, require_label=True)
            if score is not None and score >= 0:
                logging.info(f'OCR 识别位置: top_middle 回退区域 -> {score} hit={hit} conf={conf}')
                return (score, hit, conf) if return_hit else score

            full_top_img = sct.grab({
                'left': monitor['left'],
                'top': monitor['top'],
                'width': top_w,
                'height': top_h
            })
            full_top_im = Image.frombytes('RGB', full_top_img.size, full_top_img.bgra, 'raw', 'BGRX')
            score, hit, conf = ocr_score_and_hit_from_pil(full_top_im, require_label=True)
            if score is not None and score >= 0:
                logging.info(f'OCR 识别位置: full_top 回退区域 -> {score} hit={hit} conf={conf}')
                return (score, hit, conf) if return_hit else score
            return None

    for attempt in range(5):
        res = try_once()
        if res is not None:
            return res
        time.sleep(0.15)
    return (None, None, -1) if return_hit else None


class ScoreFilter:
    def __init__(self, conf_thresh=50, window=3, tolerance=6, ewma_alpha=0.4):
        self.conf_thresh = conf_thresh
        self.window_size = max(1, int(window))
        self.tolerance = tolerance
        self.ewma_alpha = ewma_alpha
        self.values = deque(maxlen=self.window_size)
        self.confs = deque(maxlen=self.window_size)
        self.ewma = None

    def update(self, value, conf):
        if value is None:
            return None
        try:
            conf_val = float(conf) if conf is not None else -1
        except (ValueError, TypeError):
            conf_val = -1

        if conf_val < self.conf_thresh:
            self.values.append(value)
            self.confs.append(conf_val)
            return None

        self.values.append(value)
        self.confs.append(conf_val)

        if len(self.values) < self.window_size:
            median = sorted(self.values)[len(self.values) // 2]
            return median

        sorted_vals = sorted(self.values)
        median = sorted_vals[len(sorted_vals) // 2]
        max_dev = max(abs(v - median) for v in self.values)
        if max_dev <= self.tolerance:
            if self.ewma is None:
                self.ewma = median
            else:
                self.ewma = int(self.ewma * (1 - self.ewma_alpha) + median * self.ewma_alpha)
            return int(self.ewma)
        return None


def get_health_metric(health_region):
    try:
        import mss
        from PIL import Image
    except ImportError as e:
        logging.debug(f"Health check lib missing: {e}")
        return None

    x, y, w, h = health_region
    try:
        with mss.mss() as sct:
            img = sct.grab({'left': x, 'top': y, 'width': w, 'height': h})
            im = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
            im_g = im.convert('L')
            arr = im_g.histogram()
            total = sum(i * cnt for i, cnt in enumerate(arr))
            count = sum(arr)
            if count == 0:
                return None
            mean = total / count
            return mean
    except Exception as e:
        logging.debug(f"Get health metric fail: {e}")
        return None


class SimpleController:
    def __init__(self):
        self.py = None
        try:
            import pyautogui
            self.py = pyautogui
        except ImportError:
            pass

        self._win32 = False
        self._ctypes = None
        self._user32 = None
        if platform.system().lower().startswith('win'):
            try:
                import ctypes
                self._ctypes = ctypes
                self._user32 = ctypes.WinDLL('user32', use_last_error=True)
                self._win32 = True
            except Exception as e:
                logging.debug(f"Win32 controller init fail: {e}")

    def press(self, key, duration=0.05):
        try:
            self.press_key(key, duration)
        except Exception as e:
            logging.debug(f"Press key fail: {e}")

    def press_key(self, key, duration=0.05):
        if isinstance(key, (list, tuple)):
            for candidate in key:
                if self.press_key(candidate, duration):
                    return True
            return False

        if self.py:
            try:
                self.py.keyDown(key)
                time.sleep(duration)
                self.py.keyUp(key)
                return True
            except Exception as e:
                logging.debug(f"pyautogui key fail: {e}")

        if self._win32 and self._ctypes and self._user32:
            vk_map = {
                'up': 0x26,
                'down': 0x28,
                'left': 0x25,
                'right': 0x27,
                'space': 0x20,
                'num8': 0x68,
                'num2': 0x62,
                'num4': 0x64,
                'num6': 0x66,
                'e': 0x45
            }
            vk = None
            key_l = str(key).lower()
            if key_l in vk_map:
                vk = vk_map[key_l]
            elif len(key_l) == 1 and 'a' <= key_l <= 'z':
                vk = ord(key_l.upper())

            if vk is not None:
                try:
                    PUL = self._ctypes.POINTER(self._ctypes.c_ulong)

                    class KEYBDINPUT(self._ctypes.Structure):
                        _fields_ = [
                            ("wVk", self._ctypes.c_ushort),
                            ("wScan", self._ctypes.c_ushort),
                            ("dwFlags", self._ctypes.c_ulong),
                            ("time", self._ctypes.c_ulong),
                            ("dwExtraInfo", PUL)
                        ]

                    class INPUT(self._ctypes.Structure):
                        _fields_ = [
                            ("type", self._ctypes.c_ulong),
                            ("ki", KEYBDINPUT)
                        ]

                    KEYEVENTF_SCANCODE = 0x0008
                    KEYEVENTF_KEYUP = 0x0002

                    map_func = self._user32.MapVirtualKeyW
                    map_func.argtypes = [self._ctypes.c_uint, self._ctypes.c_uint]
                    map_func.restype = self._ctypes.c_uint
                    scan = map_func(vk, 0)

                    inp_down = INPUT()
                    inp_down.type = 1
                    inp_down.ki = KEYBDINPUT(
                        wVk=0, wScan=scan,
                        dwFlags=KEYEVENTF_SCANCODE,
                        time=0,
                        dwExtraInfo=self._ctypes.pointer(self._ctypes.c_ulong(0))
                    )

                    inp_up = INPUT()
                    inp_up.type = 1
                    inp_up.ki = KEYBDINPUT(
                        wVk=0, wScan=scan,
                        dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP,
                        time=0,
                        dwExtraInfo=self._ctypes.pointer(self._ctypes.c_ulong(0))
                    )

                    self._user32.SendInput(1, self._ctypes.byref(inp_down), self._ctypes.sizeof(INPUT))
                    time.sleep(duration)
                    self._user32.SendInput(1, self._ctypes.byref(inp_up), self._ctypes.sizeof(INPUT))
                    return True
                except Exception as e:
                    logging.debug(f"SendInput fail: {e}")
                    try:
                        self._user32.keybd_event(self._ctypes.c_ubyte(vk), 0, 0, 0)
                        time.sleep(duration)
                        self._user32.keybd_event(self._ctypes.c_ubyte(vk), 0, 0x0002, 0)
                        return True
                    except Exception as e2:
                        logging.debug(f"keybd_event fail: {e2}")
        return False

    def press_many(self, keys, duration=0.05):
        if not self.py:
            return
        pressed = []
        try:
            for key in keys:
                if key:
                    self.py.keyDown(key)
                    pressed.append(key)
            time.sleep(duration)
        except Exception as e:
            logging.debug(f"press many down fail: {e}")
        finally:
            for key in reversed(pressed):
                try:
                    self.py.keyUp(key)
                except Exception as e:
                    logging.debug(f"press many up fail: {e}")


class DummyController:
    def __init__(self):
        self.log = []

    def press(self, key, duration=0.05):
        self.log.append((key, duration))

    def press_key(self, key, duration=0.05):
        self.log.append((key, duration))

    def press_many(self, keys, duration=0.05):
        self.log.append((tuple(keys), duration))


class TrainingControl:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.control_dir = self.root_dir / '.record_controls'
        self.pause_file = self.control_dir / 'PAUSED'
        self.stop_file = self.control_dir / 'STOPPED'
        self.paused = False
        self.console_paused = False
        self.stopped = False
        self.hurt_triggered = False  # 新增：标记是否触发了手动受伤
        self._msvcrt = None
        try:
            import msvcrt
            self._msvcrt = msvcrt
        except ImportError:
            pass
        self.control_dir.mkdir(parents=True, exist_ok=True)

    def _file_flags(self):
        file_paused = self.pause_file.exists()
        if self.stop_file.exists():
            self.stopped = True
        return file_paused

    def _console_keys(self):
        if not self._msvcrt or not self._msvcrt.kbhit():
            return
        try:
            key = self._msvcrt.getwch().lower()
        except Exception as e:
            logging.debug(f"Read console key fail: {e}")
            return
        if key == 'b':
            self.console_paused = not self.console_paused
            logging.info('Paused' if self.console_paused else 'Resumed')
        elif key == 'n':
            self.stopped = True
            logging.info('Stop requested from console; exiting')
        elif key == 'h':
            self.hurt_triggered = True
            print('\n[MANUAL HURT] Hurt triggered by user (H key)!')

    def poll(self):
        self.hurt_triggered = False  # 重置触发状态
        file_paused = self._file_flags()
        self._console_keys()
        self.paused = file_paused or self.console_paused
        return self.paused, self.stopped


def weighted_action(keys, last_action=None, recent_actions=None, idle_key=None, idle_weight=0.35, active_weight=1.5):
    if recent_actions is None:
        recent_actions = deque(maxlen=4)
    if not keys:
        return None

    active = [k for k in keys if k != idle_key]
    if not active:
        return random.choice(keys)

    weights = []
    for key in keys:
        weight = 1.0
        if idle_key is not None and key == idle_key:
            weight = idle_weight
            if last_action == idle_key:
                weight = min(weight, idle_weight * 0.3)
            if list(recent_actions).count(idle_key) >= 2:
                weight = min(weight, idle_weight * 0.15)
        else:
            weight = active_weight
            if last_action == key:
                weight = 0.9
            if key in recent_actions:
                weight *= 0.8
        weights.append(weight)

    total = sum(weights)
    pick = random.random() * total
    acc = 0.0
    for key, weight in zip(keys, weights):
        acc += weight
        if pick <= acc:
            return key
    return keys[-1]


def sample_multi_action(action_space, last_action=None, recent_actions=None):
    movement_opts = action_space.get('movement', ["NONE"])
    shooting_opts = action_space.get('shooting', ["NONE"])
    interaction_opts = action_space.get('interaction', ["NONE"])

    mov = weighted_action(
        movement_opts,
        last_action=(last_action[0] if last_action else None),
        recent_actions=deque([a[0] for a in (recent_actions or [])], maxlen=4),
        idle_key='NONE', idle_weight=0.22, active_weight=1.6
    )
    sho = weighted_action(
        shooting_opts,
        last_action=(last_action[1] if last_action else None),
        recent_actions=deque([a[1] for a in (recent_actions or [])], maxlen=4),
        idle_key='NONE', idle_weight=0.15, active_weight=1.7
    )
    inte = weighted_action(
        interaction_opts,
        last_action=(last_action[2] if last_action else None),
        recent_actions=deque([a[2] for a in (recent_actions or [])], maxlen=4),
        idle_key='NONE', idle_weight=0.12, active_weight=1.6
    )
    return (mov, sho, inte)


def action_to_keys(action, key_mapping=None):
    keys = []
    if isinstance(action, (list, tuple)):
        if key_mapping:
            movement = key_mapping.get('movement', {}).get(action[0], [])
            shooting = key_mapping.get('shooting', {}).get(action[1], [])
            interaction = key_mapping.get('interaction', {}).get(action[2], [])
            for part in (movement, shooting, interaction):
                if isinstance(part, (list, tuple)):
                    keys.extend([k for k in part if k])
                elif part:
                    keys.append(part)
        else:
            for item in action:
                if item and item != 'NONE':
                    keys.append(item)
    elif action and action != 'NONE':
        keys.append(action)
    return keys


# ===================== 重构：全部改为显式传参，移除全局变量 =====================
def run_episode(
    controller,
    region,
    seq=None,
    max_steps=400,
    step_interval=0.12,
    sample_every=3,
    time_penalty=-1,
    control=None,
    region_locked=False,
    filter_params=None,
    health_region=None,
    combat_detector=None,
    combat_weights=None,
    action_mode="flat",
    action_space=None,
    key_list=None,
    key_mapping=None,
    use_rl=False,
    rl_trainer=None
):
    executed = []
    cum_reward = 0
    hurt_count = 0  # 受伤次数统计
    combat_weights = combat_weights or {}
    prev_state = None  # 用于强化学习

    for i in range(max_steps):
        if control is not None:
            paused, stopped = control.poll()
            while paused and not stopped:
                time.sleep(0.2)
                paused, stopped = control.poll()
            if stopped:
                logging.info('Stop requested; ending current episode early')
                break

        # 获取视觉信息（用于RL决策）
        vision = {}
        if combat_detector is not None and use_rl:
            # 检查是否触发了手动受伤
            if control is not None and control.hurt_triggered:
                combat_detector.trigger_manual_hurt()
            
            try:
                vision = combat_detector.process_step(step=i + 1, action=None)
            except Exception as exc:
                logging.debug(f'combat detector failed for RL: {exc}')
                vision = {}
            
            # 检测到受伤时输出受伤总次数
            if vision.get('hurt'):
                total_hurt_count = vision.get('total_hurt_count', 0)
                print(f'\n[HURT DETECTED] Total hurts: {total_hurt_count}, Source: {vision.get("hurt_source")}')

        # 动作选择
        if seq is not None and i < len(seq):
            action = seq[i]
        elif use_rl and rl_trainer is not None:
            # 使用强化学习策略
            state = rl_trainer.extract_state(vision)
            action_idx, _ = rl_trainer.trainer.sample_action(state)
            
            # 将索引转换为动作（简化处理）
            if action_mode == 'multi' and action_space is not None:
                movements = action_space.get('movement', ['NONE'])
                shoot = action_space.get('shooting', ['NONE'])
                interact = action_space.get('interaction', ['NONE'])
                
                # 简单的索引映射
                mv_idx = action_idx % len(movements)
                sh_idx = (action_idx // len(movements)) % len(shoot)
                it_idx = (action_idx // (len(movements) * len(shoot))) % len(interact)
                
                action = (movements[mv_idx], shoot[sh_idx], interact[it_idx])
            else:
                action = key_list[action_idx % len(key_list)] if key_list else 'NONE'
            
            # RL训练步骤
            reward = rl_trainer.compute_reward(vision)
            rl_trainer.train_step(vision, prev_state)
            prev_state = state
        else:
            if action_mode == 'multi' and action_space is not None:
                action = sample_multi_action(
                    action_space,
                    last_action=executed[-1] if executed else None,
                    recent_actions=deque(executed[-4:], maxlen=4)
                )
            else:
                action = weighted_action(
                    key_list,
                    last_action=executed[-1] if executed else None,
                    recent_actions=deque(executed[-4:], maxlen=4)
                )

        mapped = action_to_keys(action, key_mapping)
        if mapped:
            if hasattr(controller, 'press_many') and len(mapped) > 1:
                try:
                    controller.press_many(mapped, 0.06)
                except Exception as e:
                    logging.debug(f"press many error: {e}")
                    for k in mapped:
                        if hasattr(controller, 'press_key'):
                            controller.press_key(k, 0.06)
                        else:
                            controller.press(k, 0.06)
            else:
                for k in mapped:
                    if hasattr(controller, 'press_key'):
                        controller.press_key(k, 0.06)
                    else:
                        controller.press(k, 0.06)
        time.sleep(step_interval)
        executed.append(action)

        if combat_detector is not None and not use_rl:
            # 检查是否触发了手动受伤
            if control is not None and control.hurt_triggered:
                combat_detector.trigger_manual_hurt()
            
            try:
                vision = combat_detector.process_step(step=i + 1, action=action)
            except Exception as exc:
                logging.debug(f'combat detector failed: {exc}')
                vision = {
                    'reward': int(time_penalty),
                    'player_hit_animation': False,
                    'hurt': False,
                    'bullet_hit': False,
                    'hit_pairs': [],
                    'player': [],
                    'enemy': [],
                    'bullets_player': [],
                    'bullets_enemy': []
                }

            cum_reward += int(vision.get('reward', time_penalty))

            if vision.get('player_hit_animation'):
                globals()['PLAYER_HIT_AT'] = i + 1
            if vision.get('player'):
                globals()['PLAYER_BOX'] = vision.get('player')[0].get('box') if vision.get('player') else None
            if vision.get('enemy'):
                globals()['ENEMY_BOXES'] = [item.get('box') for item in vision.get('enemy', [])]
            if vision.get('bullets_player'):
                globals()['PLAYER_BULLETS'] = [item.get('box') for item in vision.get('bullets_player', [])]
            if vision.get('bullets_enemy'):
                globals()['ENEMY_BULLETS'] = [item.get('box') for item in vision.get('enemy', [])]
            if vision.get('hurt'):
                total_hurt_count = vision.get('total_hurt_count', 0)
                print(f'\n[HURT DETECTED] Total hurts: {total_hurt_count}, Source: {vision.get("hurt_source")}')
                hurt_count += 1  # 统计受伤次数
            if vision.get('bullet_hit'):
                logging.info(f"Combat event bullet_hit pairs={len(vision.get('hit_pairs', []))} reward={vision.get('reward')}")
        else:
            cum_reward += int(time_penalty)

    return cum_reward, executed, hurt_count


def mutate_sequence(seq, action_mode, action_space, key_list, mutation_rate=0.08):
    new = seq.copy()
    for i in range(len(new)):
        if random.random() < mutation_rate:
            if action_mode == 'multi' and action_space is not None:
                last = new[i]
                mv, sh, it = last
                if random.random() < 0.4:
                    mv = weighted_action(
                        action_space.get('movement', [mv]),
                        last_action=mv,
                        recent_actions=deque([a[0] for a in new[max(0, i-4):i]], maxlen=4),
                        idle_key='NONE', idle_weight=0.22, active_weight=1.6
                    )
                if random.random() < 0.3:
                    sh = weighted_action(
                        action_space.get('shooting', [sh]),
                        last_action=sh,
                        recent_actions=deque([a[1] for a in new[max(0, i-4):i]], maxlen=4),
                        idle_key='NONE', idle_weight=0.15, active_weight=1.7
                    )
                if random.random() < 0.2:
                    it = weighted_action(
                        action_space.get('interaction', [it]),
                        last_action=it,
                        recent_actions=deque([a[2] for a in new[max(0, i-4):i]], maxlen=4),
                        idle_key='NONE', idle_weight=0.12, active_weight=1.6
                    )
                new[i] = (mv, sh, it)
            else:
                new[i] = weighted_action(
                    key_list,
                    last_action=new[i],
                    recent_actions=deque(new[max(0, i-4):i], maxlen=4),
                    idle_key='space', idle_weight=0.35, active_weight=1.5
                )
    if random.random() < 0.2:
        if action_mode == 'multi' and action_space is not None:
            new.append(sample_multi_action(
                action_space,
                last_action=new[-1] if new else None,
                recent_actions=deque(new[-4:], maxlen=4)
            ))
        else:
            new.append(weighted_action(
                key_list,
                last_action=new[-1] if new else None,
                recent_actions=deque(new[-4:], maxlen=4),
                idle_key='space', idle_weight=0.35, active_weight=1.5
            ))
    return new


def main():
    here = Path(__file__).parent
    config_path = here / 'config.yml'
    # 尝试用UTF-8读取，如果失败则用GBK
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(config_path, 'r', encoding='gbk') as f:
            cfg = yaml.safe_load(f)

    # 检查是否使用增强玩家检测
    use_enhanced_player = ENHANCED_PLAYER_DETECTOR_AVAILABLE and cfg.get('screen', {}).get('combat', {}).get('use_enhanced_player', True)
    
    # 检查是否使用强化学习
    use_rl = '--rl' in sys.argv or '--rl-gpu' in sys.argv
    use_gpu_rl = '--rl-gpu' in sys.argv and RL_GPU_TRAINER_AVAILABLE
    
    # 检查是否使用简化动作空间（初学者模式）
    use_simplified_actions = '--simplified-actions' in sys.argv or '--easy' in sys.argv
    
    rl_trainer = None
    if use_rl:
        if use_gpu_rl:
            logging.info("Using GPU-accelerated Reinforcement Learning mode")
            if TORCH_AVAILABLE and DEVICE is not None:
                logging.info(f"PyTorch device: {DEVICE}")
                if DEVICE.type == 'cuda':
                    logging.info(f"CUDA acceleration enabled")
            rl_config = DEFAULT_GPU_CONFIG.copy()
            rl_config.update(cfg.get('rl', {}))
            try:
                rl_trainer = ImprovedPPOTrainer(rl_config)
                logging.info("GPU RL trainer initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize GPU RL trainer: {e}, falling back to CPU")
                if RL_TRAINER_AVAILABLE:
                    rl_config = DEFAULT_CONFIG.copy()
                    rl_config.update(cfg.get('rl', {}))
                    rl_trainer = ImprovedRLTrainer(rl_config)
        elif RL_TRAINER_AVAILABLE:
            logging.info("Using CPU-based Reinforcement Learning mode")
            rl_config = DEFAULT_CONFIG.copy()
            rl_config.update(cfg.get('rl', {}))
            rl_trainer = ImprovedRLTrainer(rl_config)
        else:
            logging.warning("No RL trainer available, disabling RL mode")
            use_rl = False

    preflight_focus(cfg.get('screen', {}))

    agent_cfg = cfg.get('agent', {})
    action_mode = "flat"
    action_space = None
    key_mapping = None
    key_list = ["left", "right", "up", "down", "space"]

    if 'action_keys' in agent_cfg:
        key_list = agent_cfg['action_keys']
        action_mode = "flat"
    elif 'action_space' in agent_cfg:
        action_mode = "multi"
        # 检查是否使用简化动作空间（初学者模式）
        if use_simplified_actions and 'simplified_action_space' in agent_cfg:
            action_space = agent_cfg['simplified_action_space']
            logging.info("Using simplified action space (beginner mode)")
        else:
            action_space = agent_cfg['action_space']
        key_mapping = agent_cfg.get('key_mapping', {})
        key_list = None
    else:
        key_list = agent_cfg.get('action_keys', ["left", "right", "up", "down", "space"])
        action_mode = "flat"

    step_interval = cfg['agent'].get('step_interval', 0.12)
    episodes = get_cli_int('--episodes', cfg.get('trainer', {}).get('episodes', 200))
    episode_length = get_cli_int('--episode-length', cfg.get('trainer', {}).get('episode_length', 300))
    sample_every = get_cli_int('--sample-every', cfg.get('trainer', {}).get('sample_every', 3))
    time_penalty = cfg.get('trainer', {}).get('time_penalty', -1)
    log_every = cfg.get('trainer', {}).get('log_every', 1)
    control = TrainingControl(here)

    logging.info(f"ACTION_MODE={action_mode}")
    logging.info(f"ENHANCED_PLAYER_DETECTION={use_enhanced_player}")
    logging.info(f"RL_MODE={use_rl}")

    local_dev = here / 'SerpentAI-dev'
    fallback_dev = here.parent / 'SerpentAI-dev'
    if local_dev.exists():
        sys.path.insert(0, str(local_dev))
        logging.info(f"Using local SerpentAI copy: {local_dev}")
    elif fallback_dev.exists():
        sys.path.insert(0, str(fallback_dev))
        logging.info(f"Using workspace SerpentAI-dev: {fallback_dev}")
    else:
        logging.info("No SerpentAI-dev copy found in workspace; using fallback lightweight controller")

    dry_run = os.environ.get('DRY_RUN') or ('--dry-run' in sys.argv)
    if dry_run:
        controller = DummyController()
        logging.info('DRY RUN mode: no input will be sent')
    else:
        controller = SimpleController()

    # 初始化增强玩家检测器
    enhanced_player_detector = None
    if use_enhanced_player:
        enhanced_player_detector = EnhancedPlayerDetector()
        logging.info('Enhanced player detector initialized')

    best_score = float('-inf')
    best_seq = None
    best_path = here / 'best_sequence.pkl'

    try:
        if best_path.exists():
            with open(best_path, 'rb') as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict) and loaded.get('version') == 'vision_v1':
                    best_score = float(loaded.get('score', float('-inf')))
                    best_seq = loaded.get('seq')
                    logging.info(f"Loaded best from disk score={best_score} len={len(best_seq) if best_seq else 0}")
                else:
                    logging.info('Ignoring legacy best_sequence.pkl from score-based mode')
    except Exception as e:
        logging.warning(f"Load best sequence failed: {e}")
        best_score = float('-inf')
        best_seq = None

    # 用于存储历史视觉结果（用于增强检测）
    prev_frame_for_motion = None

    # 训练统计变量
    total_hurts = 0  # 总受伤次数
    episode_hurts = 0  # 当前episode受伤次数
    total_episodes = 0  # 已完成的episode数
    
    # 训练日志记录器（新增）
    training_logger = TrainingLogger() if HAS_VISUALIZER else None
    ga_best_score = float('-inf')
    ppo_best_score = float('-inf')

    for ep in range(1, episodes + 1):
        if best_seq is None:
            candidate = None
        else:
            # 修复：显式传参，不再传 None 进变异函数
            candidate = mutate_sequence(
                best_seq,
                action_mode=action_mode,
                action_space=action_space,
                key_list=key_list
            )

        screen_cfg = cfg.get('screen', {})
        combat_detector = build_vision_detector(screen_cfg) if callable(build_vision_detector) else None
        worker = None

        # 集成增强玩家检测器
        if use_enhanced_player and combat_detector is not None:
            from player_detector import patch_vision_detector_with_enhanced_player
            combat_detector = patch_vision_detector_with_enhanced_player(combat_detector)
            logging.info("Enhanced player detector patched successfully")

        if combat_detector is not None and VisionWorker is not None:
            try:
                fps = float(screen_cfg.get('vision_fps', 20))
            except (ValueError, TypeError):
                fps = 20.0
            try:
                worker = VisionWorker(combat_detector, poll_fps=fps)
                worker.start()
                combat_detector = worker
            except Exception as e:
                logging.debug('Failed to start VisionWorker; falling back to synchronous detector: %s', e)

        combat_weights = {
            'hit_reward': screen_cfg.get('combat', {}).get('bullet_hit_reward', screen_cfg.get('combat', {}).get('hit_reward', 18)),
            'player_hit_penalty': screen_cfg.get('combat', {}).get('player_hit_penalty', -8),
            'time_penalty': screen_cfg.get('combat', {}).get('time_penalty', time_penalty),
        }

        if combat_detector is not None:
            logging.info('Vision combat detector enabled')
        else:
            logging.info('Vision combat detector disabled (no combat config or unsupported setup)')

        try:
            score, executed, episode_hurts = run_episode(
                controller=controller,
                region=None,
                seq=candidate,
                max_steps=episode_length,
                step_interval=step_interval,
                sample_every=sample_every,
                time_penalty=time_penalty,
                control=control,
                combat_detector=combat_detector,
                combat_weights=combat_weights,
                action_mode=action_mode,
                action_space=action_space,
                key_list=key_list,
                key_mapping=key_mapping,
                use_rl=use_rl,
                rl_trainer=rl_trainer
            )
        finally:
            try:
                if worker is not None:
                    worker.stop()
            except Exception as e:
                logging.debug(f"Stop vision worker fail: {e}")

        # 更新历史帧用于运动检测
        if combat_detector is not None and hasattr(combat_detector, '_prev_frame_for_motion'):
            prev_frame_for_motion = combat_detector._prev_frame_for_motion

        if score is None:
            score = float('-inf')

        # 累加受伤次数
        total_hurts += episode_hurts
        total_episodes += 1

        if score > best_score:
            best_score = score
            best_seq = executed
            try:
                with open(best_path, 'wb') as f:
                    pickle.dump({'version': 'vision_v1', 'score': best_score, 'seq': best_seq}, f)
                logging.info(f"New best score={best_score} saved (episode {ep})")
            except Exception as e:
                logging.warning(f"Save best sequence failed: {e}")

        if ep % log_every == 0:
            logging.info(f"episode={ep} score={score} best={best_score}")
            
            # 记录训练数据并生成图表（新增）
            if training_logger:
                try:
                    if use_rl:
                        ppo_best_score = max(ppo_best_score, score)
                        training_logger.log_ppo_score(ep, score, ppo_best_score)
                        logging.info(f"Logged PPO score: episode={ep}, score={score}, best={ppo_best_score}")
                    else:
                        ga_best_score = max(ga_best_score, score)
                        training_logger.log_ga_score(ep, score, ga_best_score)
                        logging.info(f"Logged GA score: episode={ep}, score={score}, best={ga_best_score}")
                    
                    training_logger.log_episode_stats(ep, episode_hurts, episode_length * step_interval)
                    
                    # 每log_every轮生成一次图表
                    visualizer = TrainingVisualizer(training_logger)
                    visualizer.generate_all_plots()
                    training_logger.save_to_file()
                    logging.info(f"Training charts generated at episode {ep}")
                except Exception as e:
                    logging.error(f"Error generating training charts: {e}", exc_info=True)

        if control.stopped:
            logging.info('Training stopped by control')
            break

    # 停止时保存RL模型（保存到多个位置）
    if use_rl and rl_trainer:
        print("=" * 60)
        print("Saving model before exit...")
        print("=" * 60)
        
        # 保存到增量训练需要的位置
        try:
            rl_trainer.save_model('./models/ppo_model.pth')
            logging.info(f"RL model saved to: ./models/ppo_model.pth")
        except Exception as e:
            logging.warning(f"Failed to save model to ./models/ppo_model.pth: {e}")
        
        # 保存到原位置
        try:
            rl_trainer.save(str(here / 'final_rl_model.pkl'))
            logging.info(f"RL model saved to: final_rl_model.pkl")
        except Exception as e:
            logging.warning(f"Failed to save model to final_rl_model.pkl: {e}")

    # 保存最佳序列
    if best_seq:
        try:
            with open(best_path, 'wb') as f:
                pickle.dump({'version': 'vision_v1', 'score': best_score, 'seq': best_seq}, f)
            logging.info(f"Best sequence saved (score={best_score})")
        except Exception as e:
            logging.warning(f"Failed to save best sequence: {e}")
    
    # 训练结束后生成最终图表（新增）
    if training_logger and total_episodes > 0:
        logging.info("="*60)
        logging.info("[VISUALIZATION] Generating final training charts...")
        logging.info("="*60)
        try:
            visualizer = TrainingVisualizer(training_logger)
            visualizer.generate_all_plots()
            training_logger.save_to_file()
            logging.info("[VISUALIZATION] Charts saved to visualizations/ directory")
            logging.info("[VISUALIZATION] Training data saved to training_logs/training_data.json")
        except Exception as e:
            logging.error(f"[VISUALIZATION] Error generating charts: {e}", exc_info=True)

    # 清理停止标志文件，防止下次运行时立即停止
    try:
        control.stop_file.unlink(missing_ok=True)
        logging.info('Cleaned STOPPED flag file')
    except Exception as e:
        logging.debug(f"Failed to clean STOPPED file: {e}")

    print("=" * 60)
    print("Training stopped successfully!")
    print("=" * 60)
    print("Training Statistics:")
    print("-" * 60)
    print(f"  Total Episodes:      {total_episodes}")
    print(f"  Total Hurts:         {total_hurts}")
    if total_episodes > 0:
        print(f"  Average Hurts/Ep:    {total_hurts / total_episodes:.2f}")
    print(f"  Best Score:          {best_score:.2f}")
    print("-" * 60)
    print(f"Model saved to: ./models/ppo_model.pth")
    print("You can continue training later.")
    print("=" * 60)
    logging.info('Training finished')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("=" * 60)
        print("Training interrupted by user (Ctrl+C)")
        print("=" * 60)
        logging.info('Training interrupted by user (Ctrl+C)')
        
        try:
            # 保存最佳序列
            best_path = Path(__file__).parent / 'best_sequence.pkl'
            if 'best_seq' in dir() and best_seq:
                with open(best_path, 'wb') as f:
                    pickle.dump({'version': 'vision_v1', 'score': best_score, 'seq': best_seq}, f)
                logging.info(f"Best sequence saved before exit")
            
            # 保存RL模型（保存到增量训练需要的位置）
            if 'rl_trainer' in dir() and rl_trainer:
                try:
                    rl_trainer.save_model('./models/ppo_model.pth')
                    logging.info(f"RL model saved to: ./models/ppo_model.pth")
                except Exception as e:
                    logging.warning(f"Failed to save model to ./models/ppo_model.pth: {e}")
                
                try:
                    rl_trainer.save(str(Path(__file__).parent / 'interrupted_rl_model.pkl'))
                    logging.info("RL model saved to: interrupted_rl_model.pkl")
                except Exception as e:
                    logging.warning(f"Failed to save model to interrupted_rl_model.pkl: {e}")
        except Exception as e:
            logging.warning(f"Failed to save on interrupt: {e}")
        
        print("=" * 60)
        print("Training interrupted successfully!")
        print("Model saved to: ./models/ppo_model.pth")
        print("You can continue training later.")
        print("=" * 60)
        logging.info('Exiting gracefully')
    except Exception as e:
        logging.error(f"Training failed with exception: {e}", exc_info=True)