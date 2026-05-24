# Isaac SerpentAI 接口与系统测试报告

测试日期：2026-05-25

## 结论

张天朗负责的系统集成、环境配置、接口验证、Redis 抓帧链路、实验数据落库和基础输入接口已经完成并通过本机测试。

## 已通过测试

环境与依赖：

```powershell
.\tools\run_in_env.cmd python --version
.\tools\run_in_env.cmd python -m pip check
.\tools\serpent.cmd plugins
```

结果：

```text
Python 3.6.15
No broken requirements found.
ACTIVE Plugins:
SerpentIsaacSystemTestGameAgentPlugin
SerpentIsaacGamePlugin
```

接口冒烟测试：

```powershell
.\tools\run_in_env.cmd python tools\interface_smoke_test.py
```

结果：

```text
OK - SerpentIsaacGame is discoverable
OK - SerpentIsaacSystemTestGameAgent is discoverable
OK - isaac-ng.exe path exists
OK - teammate console mod is installed
OK - REGION_GAMEPLAY screen region exists
OK - MOVE_UP action exists
OK - SHOOT_RIGHT action exists
OK - agent handle_play method exists
OK - agent setup_play method exists
```

窗口与截图测试：

```powershell
.\tools\run_in_env.cmd python tools\live_window_test.py
.\tools\run_in_env.cmd python tools\capture_window_test.py
.\tools\run_in_env.cmd python tools\check_capture.py
```

结果：

```text
window_name=Binding of Isaac: Repentance
window_geometry={'width': 640, 'height': 360, 'x_offset': 7, 'y_offset': 30}
size=(640, 360) mode=RGB mean=138.71 extrema=(0, 255)
```

Redis 与 SerpentAI 抓帧队列：

```powershell
.\tools\start_redis.cmd
.\tools\run_in_env.cmd python tools\redis_ping.py
.\tools\run_in_env.cmd python tools\serpent_frame_queue_test.py
```

结果：

```text
True
redis_key=SERPENT:FRAMES
frame_shape=(360, 640, 3)
frame_dtype=uint8
frame_mean=40.25
```

实验数据库：

```powershell
.\tools\run_in_env.cmd python tools\db_smoke_test.py
```

结果：

```text
db_path=data\isaac_experiments.sqlite3
counts 包含新增的 episodes、steps、artifacts 记录
```

有限帧系统循环：

```powershell
.\tools\run_in_env.cmd python tools\finite_agent_loop_test.py
```

结果：

```text
redis_key=SERPENT:FRAMES
frames_processed=5
db_path=data\isaac_experiments.sqlite3
db_counts 包含本次 Agent 循环写入的 5 条 step 记录
```

动作输入接口：

```powershell
.\tools\run_in_env.cmd python tools\action_input_test.py --send-actions MOVE_RIGHT MOVE_LEFT SHOOT_RIGHT SHOOT_LEFT WAIT
```

结果：

```text
MOVE_RIGHT=['KEY_D']
MOVE_LEFT=['KEY_A']
SHOOT_RIGHT=['KEY_RIGHT']
SHOOT_LEFT=['KEY_LEFT']
WAIT=[]
send_actions=True
sent_count=4
window_name=Binding of Isaac: Repentance
```

## 已修复问题

- Redis 服务端未安装/未运行：已加入项目内便携 Redis，启动脚本为 `tools\start_redis.cmd`。
- PowerShell 执行策略可能拦截 `.ps1`：已加入 `.cmd` 包装脚本，日常命令不需要改系统策略。
- 游戏真实窗口标题为 `Binding of Isaac: Repentance`：插件已使用正确标题。
- 游戏窗口最小化时 SerpentAI 会读到 `0x0` 几何：`SerpentIsaacGame.after_launch` 已恢复窗口后再计算几何。
- 实验数据没有统一记录位置：已新增 SQLite 数据库 `data\isaac_experiments.sqlite3`，由脚本自动创建。

## 交付文件

- `plugins\SerpentIsaacGamePlugin`
- `plugins\SerpentIsaacSystemTestGameAgentPlugin`
- `tools\isaac_experiment_db.py`
- `tools\db_smoke_test.py`
- `tools\finite_agent_loop_test.py`
- `tools\action_input_test.py`
- `tools\redis_ping.py`
- `tools\serpent_frame_queue_test.py`
- `tools\start_redis.cmd`
- `tools\install_redis.cmd`
- `docs\environment_setup.md`
- `docs\save_and_sync.md`
- `docs\interface_test_report.md`
