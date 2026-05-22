# Save, Mod, and Project Sync Notes

## 当前结论

队友仓库 `hotice21/Embodied-Artificial-Intelligence-` 里目前没有以撒存档文件。仓库中的 `game/mod` 目录包含：

- `szx_chinese_console_3001774454.zip`：中文控制台 MOD。
- `wiki.pdf`：控制台 Lua API 文档。
- `README.md`：MOD 安装说明。

我已经把该 MOD 安装到：

```text
The Binding of Isaac Rebirth Repentance\mods\szx_chinese_console_3001774454
```

## 真正的存档需要什么文件

如果组员要给你“游戏存档”，需要让他发包含以下类似文件的文件夹或压缩包：

```text
persistentgamedata*.dat
rep_persistentgamedata*.dat
gamestate*.dat
rep_gamestate*.dat
options.ini
```

当前截图中的 GitHub 仓库不是存档仓库，至少当前 `main` 分支没有这些文件。

## 本绿色版存档路径

本机绿色版 `steam_emu.ini` 写明：

```text
C:\Users\Public\Documents\Steam\CODEX\250900
```

常见情况下实际存档会在：

```text
C:\Users\Public\Documents\Steam\CODEX\250900\remote
```

如果游戏还没启动过，这个目录可能不存在。先启动一次游戏，让它生成目录，再导入队友存档会更稳。

## 导入队友存档

拿到队友的存档文件夹后，把它放到任意位置，例如：

```text
E:\3DMGAME_Isaac_Rebirth_Repentance.EN.Green\incoming_save
```

然后在项目根目录执行：

```powershell
.\tools\import_isaac_save.ps1 .\incoming_save
```

脚本会先备份当前本机存档到：

```text
backups\saves
```

再把队友存档复制到 CODEX 存档目录。

## 重新安装队友 MOD

如果以后 MOD 被删了或要重装：

```powershell
.\tools\install_teammate_mod.ps1
```

旧 MOD 会自动备份到：

```text
backups\mods
```

## 项目同步原则

同步到 GitHub 时只提交项目文件，不提交这些目录：

```text
.envs
.mamba
.tools
.wheelhouse
The Binding of Isaac Rebirth Repentance
backups
artifacts/frames
artifacts/videos
```

应该提交：

```text
docs
tools
environment.yml
requirements.lock.txt
config
offshoot.yml
offshoot.manifest.json
requirements.txt
```

## 同步命令

进入队友仓库目录：

```powershell
cd .team_repo
git pull
git status
git add docs tools environment.yml requirements.lock.txt config offshoot.yml offshoot.manifest.json requirements.txt .gitignore
git commit -m "Add SerpentAI environment setup and save import notes"
git push
```

如果 `git push` 提示没有权限，需要让仓库所有者 `hotice21` 把你加为 collaborator，或者你 fork 后发 Pull Request。
