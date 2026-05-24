# 张天朗任务完成说明

日期：2026-05-25

## 负责内容

张天朗负责项目中的系统集成、环境配置、接口测试、数据记录和最终同步。

## 当前完成状态

- 已创建项目内独立 Python 3.6 虚拟环境，不依赖主机旧版 Python。
- 已安装 SerpentAI 2018.1.2 及 Windows 运行依赖。
- 已安装组员提供的中文控制台 MOD。
- 已加入便携 Redis，并验证 `127.0.0.1:6379` 可连接。
- 已实现 `SerpentIsaacGamePlugin`，能定位并抓取 Isaac 窗口。
- 已实现 `SerpentIsaacSystemTestGameAgentPlugin`，能消费帧并记录测试数据。
- 已建立 SQLite 实验数据库，支持 episode、step、artifact、checkpoint、evaluation 记录。
- 已完成窗口检测、截图、Redis 抓帧、有限帧 Agent 循环、动作输入接口测试。
- 已准备好同步到组员 GitHub 仓库的代码和文档。

## 日常运行命令

先启动 Redis：

```powershell
.\tools\start_redis.cmd
```

检查环境：

```powershell
.\tools\run_in_env.cmd python --version
.\tools\run_in_env.cmd python -m pip check
```

检查接口：

```powershell
.\tools\run_in_env.cmd python tools\interface_smoke_test.py
```

检查抓帧：

```powershell
.\tools\run_in_env.cmd python tools\serpent_frame_queue_test.py
```

跑有限帧系统测试：

```powershell
.\tools\run_in_env.cmd python tools\finite_agent_loop_test.py
```

验证输入接口：

```powershell
.\tools\run_in_env.cmd python tools\action_input_test.py --send-actions MOVE_RIGHT MOVE_LEFT SHOOT_RIGHT SHOOT_LEFT WAIT
```

## 存档说明

组员仓库里目前只有 MOD，没有发现 Isaac 存档文件。真正的游戏存档通常是：

- `persistentgamedata*.dat`
- `rep_persistentgamedata*.dat`
- `gamestate*.dat`
- `rep_gamestate*.dat`
- `options.ini`

如果组员以后发来这些文件，用下面脚本导入：

```powershell
.\tools\import_isaac_save.ps1 -SourcePath <组员存档文件或文件夹路径>
```

本机 Green 版常见存档目录是：

```text
C:\Users\Public\Documents\Steam\CODEX\250900\remote
```

导入前会自动备份原存档到 `backups\`。
