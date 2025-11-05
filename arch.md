# Git-LLM-Tool 架構文件

## 1. 介紹 (Introduction)

### 1.1 介紹內容
這份文件概述了 Git-LLM-Tool 的整體專案架構，這是一個 Python CLI 工具，旨在利用 LLM 自動化 git commit 和 changelog 的生成。
其主要目標是作為 AI 驅動開發的指導性架構藍圖，確保在實作 PRD 中定義的所有功能（如階層式設定、多 LLM 供應商支援 和 Jira 整合）時保持一致性。

**與前端架構的關係：**
此專案是一個純 CLI 工具，不包含圖形使用者介面，因此不需要獨立的前端架構文件。

### 1.2 啟動模板或現有專案 (Starter Template or Existing Project)
**N/A** — 這是一個全新的 (Greenfield) 專案，將從頭開始建置，不使用任何啟動模板。

### 1.3 變更日誌 (Change Log)
| 日期 | 版本 | 描述 | 作者 |
|------|------|------|------|
| 2025-11-05 | 1.0 | 初始架構草案 | Winston (Architect) |

---

## 2. 高層級架構 (High Level Architecture)

### 2.1 技術摘要 (Technical Summary)
本專案是一個 **單體 (Monolithic)** Python CLI 工具，使用 **Click** 框架。
其核心架構圍繞一個 **LLM 抽象層 (LlmProvider)**，允許動態切換多個 LLM 供應商（如 OpenAI, Anthropic, Gemini）。
工具透過 **階層式設定系統（YAML 檔案與環境變數）** 進行配置，並使用 **subprocess** 直接與本地 Git 儲存庫互動。

### 2.2 高層級概觀 (High Level Overview)

**架構風格 (Architectural Style)**：單體 (Monolith) 本地端 CLI 應用程式。
**儲存庫結構 (Repository Structure)**：Polyrepo（即單一、獨立的儲存庫）。
**使用poetry**：管理專案依賴與版本控制。

**核心流程 (Primary Flow)：**
1. 使用者在終端機執行 `git-llm commit`。
2. CLI 讀取階層式設定（全域、專案、環境變數）。
3. CLI 執行 `git diff --cached` 獲取暫存變更。
4. 若啟用 Jira 整合，CLI 嘗試從 branch name 擷取 ticket，或提示使用者輸入。
5. CLI 根據設定實例化對應的 LlmProvider（例如 OpenAiProvider）。
6. Diff 和 Jira 內容被傳遞給 Provider，呼叫外部 LLM API。
7. CLI 接收 LLM 產生的 message。
8. CLI 將 message 填入使用者的 Git 編輯器中以供檢視。

**關鍵架構決策 (Key Architectural Decisions)：**
- **LLM 抽象層**：不綁定單一廠商，透過介面支援多供應商。
- **階層式設定**：採用「全域 → 專案 → CLI 旗標」覆蓋邏輯。
- **無 LangChain**：避免多餘依賴，保持 CLI 輕量化。

### 2.3 高層級專案圖表 (High Level Project Diagram)

```mermaid
graph TD
    subgraph "使用者 (CLI)"
        User[👤 User]
    end

    subgraph "Git-LLM-Tool (Python App)"
        CLI[git-llm (Click/Typer)]
        Config[ConfigLoader]
        Git[GitHelper]
        LLM_Interface[LlmProvider (Interface)]

        CLI --> Config
        CLI --> Git
        CLI --> LLM_Interface
    end

    subgraph "LLM 抽象層 (Providers)"
        OpenAI[OpenAiProvider]
        Anthropic[AnthropicProvider]
        Gemini[GeminiProvider]

        LLM_Interface -- Implements --> OpenAI
        LLM_Interface -- Implements --> Anthropic
        LLM_Interface -- Implements --> Gemini
    end

    subgraph "外部系統"
        GitRepo[Local Git Repo]
        YAML[config.yaml files]
        API_OpenAI[OpenAI API]
        API_Anthropic[Anthropic API]
        API_Gemini[Google API]
    end

    User -- Runs --> CLI
    Config -- Reads --> YAML
    Git -- Executes --> GitRepo
    OpenAI --> API_OpenAI
    Anthropic --> API_Anthropic
    Gemini --> API_Gemini
```

### 2.4 架構與設計模式 (Architectural and Design Patterns)
- **策略模式 / 抽象工廠 (Strategy / Abstract Factory)**：
  `LlmProvider` 介面允許在執行時動態切換不同 LLM Provider。
- **命令模式 (Command Pattern)**：
  Click/Typer 框架本身即以命令模式封裝 CLI 子命令。
- **外觀模式 (Facade Pattern)**：
  `GitHelper` 封裝 subprocess 邏輯，提供乾淨的 API。
- **單例模式 (Singleton Pattern)**：
  `ConfigLoader` 為單例，確保設定只載入一次。

---

## 3. 核心規格 (Core Specifications)

### 3.1 技術棧 (Tech Stack)

| 類別 | 技術 | 版本 | 用途 |
|------|------|------|------|
| 語言 | Python | 3.12+ | 核心開發語言 |
| CLI 框架 | Click | ~8.1 | 處理命令與參數 |
| 設定 | PyYAML | ~6.0 | 解析 YAML |
| Git 互動 | subprocess | 內建 | 執行 git 指令 |
| LLM (OpenAI) | openai | ~1.0 | 呼叫 OpenAI API |
| LLM (Anthropic) | anthropic | ~0.20 | 呼叫 Claude API |
| LLM (Google) | google-generativeai | ~0.5 | 呼叫 Gemini API |
| 測試 | pytest | ~8.0 | 單元與整合測試 |
| Mock | pytest-mock | ~3.14 | 模擬 API 呼叫 |

### 3.2 設定結構 (Config Structure)

```yaml
# 全域 (~/.git-llm-tool/config.yaml) 或 專案 (./.git-llm-tool.yaml)
llm:
  default_model: 'gpt-4o'
  language: 'en'
  api_keys:
    openai: 'sk-...'
    anthropic: 'sk-...'
    google: '...'

jira:
  enabled: false
  branch_regex: null
```

### 3.3 元件設計 (Components Overview)

| 元件 | 職責 | 關鍵介面 | 依賴 |
|------|------|-----------|------|
| **LlmProvider** | 定義 LLM 抽象介面 | `generate_commit_message()` / `generate_changelog()` | AppConfig |
| **OpenAiProvider** | 實作 OpenAI API 呼叫 | 同上 | openai |
| **AnthropicProvider** | 實作 Claude API 呼叫 | 同上 | anthropic |
| **GeminiProvider** | 實作 Gemini API 呼叫 | 同上 | google-generativeai |
| **ConfigLoader** | 載入與合併設定 | `load_config()` | PyYAML, os |
| **GitHelper** | 封裝 git 指令互動 | `get_staged_diff()`, `apply_commit()` | subprocess |

---

## 4. 原始碼樹狀結構 (Source Tree)

```plaintext
git-llm-tool/
├── .github/
│   └── workflows/
│       └── python-ci.yml       # CI/CD (linting, testing)
├── .gitignore
├── README.md
├── pyproject.toml
├── src/
│   └── git_llm_tool/
│       ├── __main__.py
│       ├── cli.py
│       ├── commands/
│       │   ├── commit_cmd.py
│       │   └── changelog_cmd.py
│       ├── core/
│       │   ├── config.py
│       │   ├── git_helper.py
│       │   └── exceptions.py
│       └── providers/
│           ├── base.py
│           ├── openai.py
│           ├── anthropic.py
│           └── gemini.py
└── tests/
    ├── test_cli.py
    ├── test_config.py
    ├── test_git_helper.py
    └── providers/
        └── test_providers.py
```

---

## 5. 基礎設施與部署 (Infrastructure and Deployment)

### 5.1 CI / CD
- 工具：GitHub Actions (`.github/workflows/python-ci.yml`)
- 任務：安裝依賴 → Linting → pytest

### 5.2 部署策略
- 部署至 **PyPI**
- `release.yml` 觸發條件：新建 tag（例如 `v1.0.0`）
- 上傳工具：`twine`

---

## 6. 錯誤處理與日誌策略 (Error Handling & Logging)

- **基礎類別**：`GitLlmError`
- **衍生異常**：`ConfigError`, `GitError`, `ApiError`
- **全域處理**：CLI 主程式捕捉並顯示人性化錯誤訊息
- **日誌標準**：
  - INFO 為預設等級
  - `--verbose` 啟用 DEBUG 詳細模式
  - 禁止記錄 API 金鑰

---

## 7. 測試策略 (Testing Strategy)
- 僅進行單元測試，聚焦於核心邏輯（設定、regex、provider mocking）。
- 測試框架：pytest + pytest-mock
- 測試層級：
  - ✅ Unit Test（core, providers）
  - 🚫 Integration / E2E 測試暫不納入 MVP

---

## 8. 安全性 (Security)

- 所有外部 API 呼叫使用 HTTPS。
- 禁止將金鑰輸出至 stdout/stderr 或日誌。
- 僅允許 `ConfigLoader` 於執行時存取金鑰。
- 推薦使用 pip-audit / Dependabot 進行依賴掃描。

---
