# Isaac SerpentAI Environment Setup

本项目不使用主机 Python 安装旧依赖。所有工具和包都放在项目目录中：

- 便携式 micromamba：`.tools\Library\bin\micromamba.exe`
- 主虚拟环境：`.envs\isaac-serpent`
- Python：`3.6.15`
- SerpentAI：`2018.1.2`
- 离线安装包缓存：`.wheelhouse`

## 日常使用

在项目根目录运行单条命令：

```powershell
.\tools\run_in_env.cmd python --version
.\tools\run_in_env.cmd serpent.exe --help
.\tools\serpent.cmd modules
```

如果想激活环境后连续运行命令，在 PowerShell 中点号加载脚本：

```powershell
. .\tools\activate_env.ps1
python --version
serpent.exe --help
```

## 已完成内容

- 创建了项目内隔离虚拟环境 `.envs\isaac-serpent`。
- 创建了项目内便携 Redis 服务端 `.tools\redis`。
- 安装了 `SerpentAI==2018.1.2`、`offshoot`、`Cython==0.26.1`。
- 安装了 Windows 基础依赖：`numpy`、`scipy`、`scikit-image`、`scikit-learn`、`h5py`、`redis`、`aioredis`、`mss`、`PyAutoGUI`、`pywin32`、`sneakysnek`、`autobahn`、`editdistance`。
- 初始化了 SerpentAI 配置：`config\`、`offshoot.yml`、`offshoot.manifest.json`、`requirements.txt`。
- 创建了项目目录：`plugins\`、`datasets\`、`artifacts\`、`data\`。

## 验证结果

已通过：

```powershell
.\tools\run_in_env.cmd python -m pip check
.\tools\run_in_env.cmd serpent.exe --help
.\tools\serpent.cmd modules
```

关键模块已验证可导入：`serpent`、`offshoot`、`numpy`、`scipy`、`skimage`、`sklearn`、`h5py`、`redis`、`aioredis`、`mss`、`pyautogui`、`win32gui`、`sneakysnek`、`autobahn`、`editdistance`。

Redis 启动与验证：

```powershell
.\tools\install_redis.cmd
.\tools\start_redis.cmd
.\tools\run_in_env.cmd python tools\redis_ping.py
```

## 注意事项

不要直接运行全局 `pip install`。如果要装包，优先使用：

```powershell
.\tools\run_in_env.cmd python -m pip install <package>
```

当前 Python 3.6 的 `pip` 访问 PyPI 会遇到 TLS/SSL 问题，因此本环境使用 `.wheelhouse` 离线包完成安装。后续如果继续安装旧包，建议先用主机 Python 下载 wheel 到 `.wheelhouse`，再用虚拟环境离线安装。

也不建议再运行 `serpent setup`，因为它会调用全局 `conda` 和在线 `pip`。本项目需要的基础配置已经手动初始化完成。
