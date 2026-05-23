# Isaac SerpentAI Interface Test Report

测试日期：2026-05-23

## 已通过

环境与命令：

```powershell
.\tools\run_in_env.cmd python --version
.\tools\run_in_env.cmd python -m pip check
.\tools\serpent.cmd plugins
```

结果：

- Python 虚拟环境可用：`Python 3.6.15`
- Python 依赖一致性通过：`No broken requirements found`
- SerpentAI 插件已激活：
  - `SerpentIsaacGamePlugin`
  - `SerpentIsaacSystemTestGameAgentPlugin`

离线接口测试：

```powershell
.\tools\run_in_env.cmd python tools\interface_smoke_test.py
```

结果：

- `SerpentIsaacGame` 可被 Offshoot/SerpentAI 发现。
- `SerpentIsaacSystemTestGameAgent` 可被 Offshoot/SerpentAI 发现。
- `isaac-ng.exe` 路径存在。
- 队友给的中文控制台 MOD 已安装。
- `REGION_GAMEPLAY` 截图区域存在。
- 动作映射存在：`MOVE_UP`、`SHOOT_RIGHT` 等。
- Agent 的 `setup_play` 和 `handle_play` 接口存在。

实时窗口测试：

```powershell
.\tools\run_in_env.cmd python tools\live_window_test.py
```

结果：

```text
window_name=Binding of Isaac: Repentance
window_geometry={'width': 640, 'height': 360, 'x_offset': 7, 'y_offset': 30}
```

实时截图测试：

```powershell
.\tools\run_in_env.cmd python tools\capture_window_test.py
.\tools\run_in_env.cmd python tools\check_capture.py
```

结果：

```text
capture_path=artifacts\frames\isaac_window_capture.png
size=(640, 360) mode=RGB mean=176.71 extrema=(0, 243)
```

说明截图不是空白图，窗口捕获链路可用。

## 已修复的问题

- PowerShell 执行策略会拦截 `.ps1`，因此新增 `.cmd` 包装脚本：
  - `tools\run_in_env.cmd`
  - `tools\serpent.cmd`
- 游戏真实窗口标题是 `Binding of Isaac: Repentance`，不是旧的 `The Binding of Isaac: Rebirth`，插件已修正。
- 新增最小 SerpentAI 插件：
  - `plugins\SerpentIsaacGamePlugin`
  - `plugins\SerpentIsaacSystemTestGameAgentPlugin`

## 当前阻塞

Redis 服务端没有运行，也没有安装在 PATH 中：

```powershell
.\tools\run_in_env.cmd python tools\redis_ping.py
```

结果：

```text
ConnectionRefusedError / Error 10061 connecting to 127.0.0.1:6379
```

SerpentAI 的完整 `play`、`record`、`grab_frames` 流程依赖 Redis 队列，因此下一步必须安装并启动 Redis。

## 下一步

1. 安装 Redis 服务端，确保 `127.0.0.1:6379` 可连接。
2. 再跑：

```powershell
.\tools\run_in_env.cmd python tools\redis_ping.py
```

3. Redis 通过后，继续测试：

```powershell
.\tools\serpent.cmd play Isaac IsaacSystemTest
```

或按插件实际类名运行 SerpentAI 的 `play` 流程。

4. 和算法同学对接动作空间：
   - 移动：`MOVE_UP`、`MOVE_DOWN`、`MOVE_LEFT`、`MOVE_RIGHT`
   - 射击：`SHOOT_UP`、`SHOOT_DOWN`、`SHOOT_LEFT`、`SHOOT_RIGHT`
   - 功能：`BOMB`、`ACTIVE_ITEM`、`DROP`、`WAIT`

5. 建立实验数据库，记录 episode、step、action、reward、frame_path、done。
