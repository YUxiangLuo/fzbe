# Deployment Guide

This document outlines the system requirements and installation steps for deploying the Fangzhen Backend application on a Linux server (Ubuntu/Debian recommended).

## 1. System Packages

The following packages are required for the application to function correctly.

```bash
# Update package list
sudo apt-get update

# 1. Essential Tools
# curl/wget: For downloading scripts (e.g., Bun install)
# git: For version control
# zip/unzip: REQUIRED for class report archive export
# build-essential: For compiling native add-ons if needed
sudo apt-get install -y curl wget git unzip zip build-essential

# 2. Python Environment
# Required for:
# - Main AI models (py/)
# - Internal ADF tool (py/adf/) which spawns a separate python process
sudo apt-get install -y python3 python3-pip python3-venv python3-dev libatlas-base-dev

# 3. Chromium & Dependencies (CRITICAL)
# Required for: 'src/modules/reports/reports.service.ts' (Puppeteer) to generate PDF reports.
sudo apt-get install -y chromium-browser \
    fonts-noto-cjk \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2

# 4. Database (If hosting locally)
sudo apt-get install -y mysql-server libmysqlclient-dev
```

## 2. Dependency Justification

| Package / Tool | Usage Location | Reason |
| :--- | :--- | :--- |
| **chromium-browser** | `src/modules/reports/reports.service.ts` | Used by Puppeteer to generate PDF reports. Default path is `/usr/bin/chromium`, and can be overridden by `CHROMIUM_PATH`. |
| **fonts-noto-cjk** | `src/modules/reports/reports.service.ts` | Ensures Chinese characters render correctly in generated PDFs. |
| **zip** | `src/modules/class-grades/class-grades.service.ts` | Used to bundle multiple student reports into a single download. |
| **python3-venv** | `py/` & `py/adf/` | Required to create isolated virtual environments for Python dependencies. |
| **mysql-server** | `src/index.ts` etc. | (Optional if using remote DB) The primary data store for the application. |

## 3. Application Setup

### Step 1: Install Bun (JavaScript Runtime)
The application uses [Bun](https://bun.sh/).

```bash
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc # or source ~/.zshrc
```

### Step 2: Install Bun Dependencies
```bash
bun install
```

### Step 3: Set Up Python Environments (CRITICAL)
This application uses **two separate** Python environments. You must set up both.

**A. Main AI Models Environment**
```bash
# Installs dependencies for the core ML features (Torch, TensorFlow, etc.)
cd py
chmod +x install.sh
./install.sh
# This creates and installs into ./venv (or reuses ./.venv if already present)
cd ..
```

**B. ADF Tool Environment (Easy to miss!)**
Found in `src/modules/tools/tools.service.ts`, this tool runs a standalone script.
```bash
# Installs dependencies for the Time Series Analysis tool
cd py/adf
chmod +x bootstrap.sh
./bootstrap.sh
# This uses ./venv by default (and remains compatible with old ./.venv)
cd ../..
```

### Step 4: Environment Configuration
Ensure your `.env` file is configured.

```bash
cp .env.example .env
# Edit .env with your database credentials and secrets
```

Please pay special attention to `DEFAULT_PASSWORD_HASH`, database connection strings, and `CHROMIUM_PATH` (if Chromium is not in `/usr/bin/chromium`).

## 4. Troubleshooting

-   **PDF Generation Fails**: set `CHROMIUM_PATH` in `.env` to your Chromium binary path, for example:
    ```bash
    CHROMIUM_PATH=/path/to/actual/chromium
    ```
-   **ADF Tool Errors**: Ensure you ran `py/adf/bootstrap.sh`. The main project `requirements.txt` does **not** cover this tool.
