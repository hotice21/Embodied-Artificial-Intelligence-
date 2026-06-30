"""增强版控制面板 - 支持多种训练模式的启动和管理"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
import sys
import signal
import traceback
import subprocess
import threading
import os
import time

ROOT = Path(__file__).resolve().parent
CTRL_DIR = ROOT / '.record_controls'
PAUSE_FILE = CTRL_DIR / 'PAUSED'
STOP_FILE = CTRL_DIR / 'STOPPED'

CTRL_DIR.mkdir(parents=True, exist_ok=True)


class TrainingProcess:
    """训练进程管理类"""
    
    def __init__(self):
        self.process = None
        self.is_running = False
        self.output_buffer = []
        self.output_lock = threading.Lock()
    
    def start(self, command, on_output=None, on_exit=None):
        """启动训练进程"""
        if self.is_running:
            return False
        
        def run_process():
            try:
                self.process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=str(ROOT)
                )
                
                self.is_running = True
                
                for line in iter(self.process.stdout.readline, ''):
                    with self.output_lock:
                        self.output_buffer.append(line)
                        if len(self.output_buffer) > 1000:
                            self.output_buffer = self.output_buffer[-500:]
                    if on_output:
                        on_output(line)
                
                self.process.wait()
                
                if on_exit:
                    on_exit(self.process.returncode)
                
            except Exception as e:
                if on_output:
                    on_output(f"Error: {str(e)}\n")
                if on_exit:
                    on_exit(-1)
            finally:
                self.is_running = False
        
        threading.Thread(target=run_process, daemon=True).start()
        return True
    
    def stop(self):
        """停止训练进程"""
        if self.process and self.is_running:
            try:
                self.process.terminate()
                time.sleep(0.5)
                if self.process.poll() is None:
                    self.process.kill()
            except Exception:
                pass
            self.is_running = False
    
    def get_output(self):
        """获取输出缓冲区"""
        with self.output_lock:
            return ''.join(self.output_buffer)


class EnhancedControlPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SerpentAI Control Panel')
        self.geometry('700x550')
        self.resizable(True, True)
        
        # 训练进程管理
        self.training_process = TrainingProcess()
        
        # 状态变量
        self.status_var = tk.StringVar(value='Status: IDLE')
        self.mode_var = tk.StringVar(value='normal')
        self.training_strategy_var = tk.StringVar(value='fresh')
        self.model_path_var = tk.StringVar(value='./models/ppo_model.pth')
        
        # 确保Tk回调异常被正确处理
        self.report_callback_exception = self._handle_callback_exception
        
        # 主布局
        self._create_widgets()
        
        # 绑定全局快捷键
        self._bind_shortcuts()
        
        # 定时检查状态
        self.after(500, self._poll_status)
        
        # 打印快捷键帮助
        self._print_shortcut_help()
    
    def _create_widgets(self):
        """创建所有界面组件"""
        # 主面板
        main_frame = ttk.Frame(self, padding='10')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(status_frame, text='Training Control Panel', font=('Segoe UI', 14, 'bold')).pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground='blue', font=('Segoe UI', 10, 'bold'))
        self.status_label.pack(side=tk.RIGHT)
        
        # 模式选择面板
        mode_frame = ttk.LabelFrame(main_frame, text='Training Mode', padding='10')
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 模式选项
        modes = [
            ('Normal Training', 'normal', 'Run basic training'),
            ('RL Training (CPU)', 'rl', 'Reinforcement Learning on CPU'),
            ('RL Training (GPU)', 'rl-gpu', 'Reinforcement Learning on GPU')
        ]
        
        for i, (text, value, tooltip) in enumerate(modes):
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=value)
            rb.grid(row=0, column=i, padx=10, pady=5)
            rb.bind('<Enter>', lambda e, t=tooltip: self._show_tooltip(e, t))
            rb.bind('<Leave>', self._hide_tooltip)
        
        # 训练策略面板
        strategy_frame = ttk.LabelFrame(main_frame, text='Training Strategy', padding='10')
        strategy_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 策略选项
        ttk.Radiobutton(strategy_frame, text='Fresh Start (new model)', variable=self.training_strategy_var, value='fresh').grid(row=0, column=0, padx=10, pady=5)
        ttk.Radiobutton(strategy_frame, text='Continue Training (load existing model)', variable=self.training_strategy_var, value='continue').grid(row=0, column=1, padx=10, pady=5)
        
        # 模型路径选择
        ttk.Label(strategy_frame, text='Model Path:').grid(row=1, column=0, padx=10, pady=5, sticky=tk.E)
        model_path_entry = ttk.Entry(strategy_frame, textvariable=self.model_path_var, width=40)
        model_path_entry.grid(row=1, column=1, padx=10, pady=5)
        
        # 浏览按钮
        def browse_model():
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                title='Select Model File',
                filetypes=[('PyTorch Model', '*.pth'), ('All Files', '*.*')],
                initialdir='./models'
            )
            if filename:
                self.model_path_var.set(filename)
        
        ttk.Button(strategy_frame, text='Browse...', command=browse_model).grid(row=1, column=2, padx=10, pady=5)
        
        # 参数面板
        param_frame = ttk.LabelFrame(main_frame, text='Parameters', padding='10')
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # episodes
        ttk.Label(param_frame, text='Episodes:').grid(row=0, column=0, padx=5, pady=3)
        self.episodes_entry = ttk.Entry(param_frame, width=10)
        self.episodes_entry.insert(0, '200')
        self.episodes_entry.grid(row=0, column=1, padx=5, pady=3)
        
        # episode-length
        ttk.Label(param_frame, text='Episode Length:').grid(row=0, column=2, padx=5, pady=3)
        self.length_entry = ttk.Entry(param_frame, width=10)
        self.length_entry.insert(0, '300')
        self.length_entry.grid(row=0, column=3, padx=5, pady=3)
        
        # dry-run checkbox
        self.dry_run_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text='Dry Run', variable=self.dry_run_var).grid(row=0, column=4, padx=10, pady=3)
        
        # simplified actions checkbox（初学者模式）
        self.simplified_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text='Beginner Mode', variable=self.simplified_var).grid(row=0, column=5, padx=10, pady=3)
        
        # 控制按钮面板
        ctrl_frame = ttk.LabelFrame(main_frame, text='Controls', padding='10')
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))
        
        buttons = [
            ('Start Training (S)', self.start_training),
            ('Pause/Resume (B)', self.toggle_pause),
            ('Stop Training (N)', self.stop_training),
            ('Reset Stop (R)', self.reset_stop),
            ('Clear Log (C)', self.clear_log)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(ctrl_frame, text=text, command=command, width=18)
            btn.grid(row=0, column=i, padx=5, pady=5)
        
        # 快捷键提示面板
        shortcut_frame = ttk.LabelFrame(main_frame, text='Keyboard Shortcuts', padding='10')
        shortcut_frame.pack(fill=tk.X, pady=(0, 10))
        
        shortcuts = [
            ('S', 'Start training'),
            ('B', 'Pause/Resume'),
            ('N', 'Stop training'),
            ('R', 'Reset stop state'),
            ('C', 'Clear log'),
            ('ESC', 'Exit panel')
        ]
        
        for i, (key, desc) in enumerate(shortcuts):
            ttk.Label(shortcut_frame, text=f'  {key} - {desc}  ').grid(row=0, column=i, padx=5)
        
        # 日志显示区域
        log_frame = ttk.LabelFrame(main_frame, text='Training Log', padding='10')
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "Welcome to SerpentAI Control Panel!\n")
        self.log_text.insert(tk.END, "Select a training mode and press Start or press 'S'\n")
        self.log_text.config(state=tk.DISABLED)
    
    def _bind_shortcuts(self):
        """绑定全局快捷键"""
        self.bind('<Key-s>', lambda e: self.start_training())
        self.bind('<Key-S>', lambda e: self.start_training())
        self.bind('<Key-b>', lambda e: self.toggle_pause())
        self.bind('<Key-B>', lambda e: self.toggle_pause())
        self.bind('<Key-n>', lambda e: self.stop_training())
        self.bind('<Key-N>', lambda e: self.stop_training())
        self.bind('<Key-r>', lambda e: self.reset_stop())
        self.bind('<Key-R>', lambda e: self.reset_stop())
        self.bind('<Key-c>', lambda e: self.clear_log())
        self.bind('<Key-C>', lambda e: self.clear_log())
        self.bind('<Escape>', lambda e: self.quit())
    
    def _show_tooltip(self, event, text):
        """显示工具提示"""
        self.tooltip = tk.Toplevel(self)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{event.x_root+10}+{event.y_root+10}")
        label = ttk.Label(self.tooltip, text=text, background='yellow', padding=5)
        label.pack()
    
    def _hide_tooltip(self, event):
        """隐藏工具提示"""
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()
    
    def _print_shortcut_help(self):
        """打印快捷键帮助信息到日志"""
        help_text = """
Keyboard Shortcuts:
-------------------
S - Start training
B - Pause/Resume
N - Stop training
R - Reset stop state
C - Clear log
ESC - Exit panel

Training Modes:
---------------
Normal Training    - Basic training without RL
RL Training (CPU)  - Reinforcement Learning on CPU
RL Training (GPU)  - Reinforcement Learning on GPU

Beginner Mode:
--------------
When enabled, uses simplified action space:
- Movement: Only W/A/S/D (no diagonal moves)
- Shooting: UP/DOWN/LEFT/RIGHT
- No interaction actions

This makes it easier for the AI to learn basic movement and shooting.
"""
        self._log(help_text)
    
    def _log(self, message):
        """添加日志消息"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _handle_callback_exception(self, exc, val, tb):
        """处理回调异常"""
        if isinstance(val, KeyboardInterrupt):
            try:
                self.stop_training()
                self.quit()
            finally:
                return
        traceback.print_exception(exc, val, tb)
        self._log(f"Error: {str(val)}\n")
    
    def _poll_status(self):
        """定时检查训练状态"""
        if self.training_process.is_running:
            self.status_var.set('Status: RUNNING')
            self.status_label.config(foreground='green')
        elif PAUSE_FILE.exists():
            self.status_var.set('Status: PAUSED')
            self.status_label.config(foreground='orange')
        elif STOP_FILE.exists():
            self.status_var.set('Status: STOPPED')
            self.status_label.config(foreground='red')
        else:
            self.status_var.set('Status: IDLE')
            self.status_label.config(foreground='blue')
        
        self.after(500, self._poll_status)
    
    def _build_command(self):
        """构建训练命令"""
        mode = self.mode_var.get()
        strategy = self.training_strategy_var.get()
        
        # 如果是继续训练模式且是RL模式，使用continue_training.py
        if strategy == 'continue' and (mode == 'rl' or mode == 'rl-gpu'):
            command = 'python continue_training.py'
            # 添加GPU参数
            if mode == 'rl-gpu':
                command += ' --rl-gpu'
            elif mode == 'rl':
                command += ' --rl'
        else:
            command = 'python train_unsupervised.py'
            
            # 添加模式参数
            if mode == 'rl':
                command += ' --rl'
            elif mode == 'rl-gpu':
                command += ' --rl-gpu'
        
        # 添加 episodes 参数
        try:
            episodes = int(self.episodes_entry.get())
            command += f' --episodes={episodes}'
        except ValueError:
            pass
        
        # 添加 episode-length 参数
        try:
            length = int(self.length_entry.get())
            command += f' --episode-length={length}'
        except ValueError:
            pass
        
        # 添加 dry-run 参数
        if self.dry_run_var.get():
            command += ' --dry-run'
        
        # 添加简化动作空间参数（初学者模式）
        if self.simplified_var.get():
            command += ' --easy'
        
        return command
    
    def start_training(self):
        """启动训练"""
        if self.training_process.is_running:
            messagebox.showwarning('Warning', 'Training is already running!')
            return
        
        # 清除之前的状态文件
        self.reset_stop()
        if PAUSE_FILE.exists():
            PAUSE_FILE.unlink()
        
        command = self._build_command()
        
        self._log(f"\n{'='*60}\n")
        self._log(f"Starting training with command:\n{command}\n")
        self._log(f"{'='*60}\n")
        
        success = self.training_process.start(
            command,
            on_output=self._log,
            on_exit=self._on_training_exit
        )
        
        if success:
            self.status_var.set('Status: RUNNING')
            self.status_label.config(foreground='green')
        else:
            self._log("Failed to start training process\n")
    
    def _on_training_exit(self, return_code):
        """训练进程退出回调"""
        self._log(f"\n{'='*60}\n")
        self._log(f"Training process exited with code: {return_code}\n")
        if return_code == 0:
            self._log("Training completed successfully!\n")
        else:
            self._log(f"Training terminated with error (code: {return_code})\n")
        self._log(f"{'='*60}\n")
        
        self.status_var.set('Status: IDLE')
        self.status_label.config(foreground='blue')
    
    def toggle_pause(self):
        """暂停/恢复训练"""
        if not self.training_process.is_running:
            return
        
        if PAUSE_FILE.exists():
            try:
                PAUSE_FILE.unlink()
                self._log("Resumed training\n")
            except Exception as e:
                self._log(f"Error resuming: {e}\n")
        else:
            try:
                PAUSE_FILE.write_text('paused')
                self._log("Paused training\n")
            except Exception as e:
                self._log(f"Error pausing: {e}\n")
    
    def stop_training(self):
        """停止训练"""
        # 创建停止标志文件
        try:
            STOP_FILE.write_text('stop')
        except Exception as e:
            self._log(f"Error creating stop file: {e}\n")
        
        # 如果有运行中的进程，强制终止
        if self.training_process.is_running:
            self._log("Stopping training process...\n")
            self.training_process.stop()
    
    def reset_stop(self):
        """重置停止状态"""
        if STOP_FILE.exists():
            try:
                STOP_FILE.unlink()
                self._log("Stop state reset\n")
            except Exception as e:
                self._log(f"Error resetting stop: {e}\n")
    
    def clear_log(self):
        """清除日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "Log cleared. Ready for new training.\n")
        self.log_text.config(state=tk.DISABLED)
    
    def quit(self):
        """退出面板"""
        if self.training_process.is_running:
            if messagebox.askyesno('Confirm Exit', 'Training is running. Are you sure you want to exit?'):
                self.stop_training()
                time.sleep(0.5)
                super().quit()
        else:
            super().quit()


if __name__ == '__main__':
    app = EnhancedControlPanel()
    
    # 设置Ctrl+C退出
    signal.signal(signal.SIGINT, lambda sig, frame: app.quit())
    
    try:
        app.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)
