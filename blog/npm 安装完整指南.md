# npm 安装完整指南

npm（Node Package Manager）通常随Node.js一起安装。以下是不同操作系统下的安装方法：

---

## **一、推荐方式：Node.js官方安装包（最简单）**

### **Windows / macOS**
1. 访问 [nodejs.org](https://nodejs.org/)
2. 下载 **LTS版本**（长期支持，推荐）或 Current 版本
3. 运行安装程序，全部默认设置即可
4. 验证安装：
```bash
node -v   # 查看Node.js版本
npm -v    # 查看npm版本
```

### **Linux (Ubuntu/Debian)**
```bash
# 使用NodeSource官方源（推荐）
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 验证
node -v && npm -v
```

### **Linux (CentOS/RHEL)**
```bash
curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
sudo yum install -y nodejs

# 验证
node -v && npm -v
```

---

## **二、开发者推荐：使用版本管理工具**

### **方案A：nvm (Node Version Manager)**
适合需要切换多个Node.js版本的开发者

**安装 nvm：**
```bash
# Linux/macOS
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# 安装后重启终端或运行
source ~/.bashrc  # 或 source ~/.zshrc
```

**使用 nvm 安装 Node.js/npm：**
```bash
nvm install --lts      # 安装最新LTS版本
nvm use --lts          # 使用LTS版本
nvm alias default lts  # 设置默认版本

# 查看已安装版本
nvm ls
```

**Windows** 用户请使用 [nvm-windows](https://github.com/coreybutler/nvm-windows)

---

### **方案B：fnm (Fast Node Manager)**
用Rust编写，速度更快

```bash
# 安装 fnm
curl -fsSL https://fnm.vercel.app/install | bash

# 安装并使用Node.js
fnm install --lts
fnm use --lts
```

---

## **三、更新npm到最新版本**

npm随Node.js安装，但通常不是最新版，建议手动更新：

```bash
# 全局更新npm（使用npm自我更新）
npm install -g npm@latest

# 验证
npm -v
```

---

## **四、权限问题解决方案**

### **Linux/macOS 遇到 EACCES 错误**

**推荐方法：使用nvm**
nvm会自动处理权限问题，无需额外配置

**替代方法：更改npm默认目录**
```bash
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'

# 将以下行添加到 ~/.bashrc 或 ~/.zshrc
export PATH=~/.npm-global/bin:$PATH

# 然后
source ~/.bashrc
```

 **⚠️ 不推荐**  ：使用 `sudo npm install -g`（可能导致权限混乱）

---

## **五、快速验证安装**

创建测试项目确认npm工作正常：

```bash
# 1. 创建项目文件夹
mkdir test-npm && cd test-npm

# 2. 初始化 package.json
npm init -y

# 3. 安装一个测试包
npm install lodash

# 4. 查看是否成功
ls node_modules  # 应该能看到lodash文件夹
```

---

## **六、安装方式对比**

| 方式 | 优点 | 缺点 | 适合人群 |
|------|------|------|----------|
| **官方安装包** | 简单快捷，一键完成 | 不易切换版本 | 新手、普通用户 |
| **nvm/fnm** | 轻松管理多版本，权限友好 | 需要额外安装 | 开发者、需要测试多版本 |
| **系统包管理器** | 与系统集成 | 版本通常较旧 | Linux服务器 |

**建议**：开发者使用 **nvm**，普通用户直接使用 **官方安装包**。