# Git-LLM-Tool 產品需求文件 (PRD)

## 1. 目標與背景情境 (Goals and Background Context)

### 1.1 目標 (Goals)
- **G1**：自動化 git commit message 的生成，以節省開發時間。
- **G2**：根據 git diff 內容產生一致且高品質的 commit 訊息。
- **G3**：自動化 changelog 的生成，簡化版本發布流程。
- **G4**：提供一個易於使用、整合到開發流程的 Python CLI 介面。

### 1.2 背景情境 (Background Context)
目前開發者在撰寫 git commit message 時，常需花費額外時間思考如何精確描述變更，且品質參差不齊。
此外，在發布版本時，手動彙總 changelog 是一個繁瑣且容易出錯的過程。
此專案旨在利用 LLM 直接分析 git diff 的能力，來自動完成這兩項任務，從而提升開發效率與版本控制的標準化。

### 1.3 變更日誌 (Change Log)

| 日期 | 版本 | 描述 | 作者 |
|------|------|------|------|
| 2025-11-05 | 1.0 | 初始 PRD 草案，包含 Epics 與 Stories | John (PM) |

---

## 2. 需求 (Requirements) - v3

### 2.1 功能需求 (Functional)

- **FR1**：CLI 應提供命令（如 `git-llm commit`），能讀取當前 `git diff --cached`（staged changes）的內容。
- **FR2**：CLI 應使用 FR1 讀取到的 diff 內容，呼叫 LLM API 產生 commit message。
- **FR3**：預設情況下，CLI 應將 FR2 產生的 commit message 填入使用者的預設 Git 編輯器 (`COMMIT_EDITMSG`) 中，以供最終檢視與確認。
- **FR4**：CLI 應提供命令（如 `git-llm changelog`），能讀取特定範圍的 `git log`（例如自上一個 tag 以來）。
- **FR5**：CLI 應使用 FR4 讀取的 log 內容並結合語言設定 (NFR6)，呼叫 LLM API 產生結構化 changelog（例如依功能、修復、重大變更分類）。
- **FR6**：CLI 應能將 changelog 輸出至終端機 (stdout) 或指定檔案。
- **FR7**：`git-llm commit` 命令需提供選項（如 `--apply` 或 `--non-interactive`），以跳過編輯器模式並直接執行 commit。
- **FR8**：若設定中啟用 Jira 整合 (NFR8)：
  - a. 工具需先依 NFR9 定義的 `branch_regex` 從 branch name 擷取 Jira ticket number。
  - b. 若失敗或未設定 regex，需互動式提示使用者輸入 Jira ticket number（可留空）。
  - c. 需互動式提示使用者輸入此次 commit 的工時（例如 `輸入工時 (例如 1h 30m):`，可留空）。
- **FR9**：從 FR1、FR8 收集的上下文與語言設定 (NFR6) 均須整合至傳給 LLM 的提示詞 (prompt)。

### 2.2 非功能需求 (Non-Functional)

- **NFR1**：使用 Python 3.8+ 開發。
- **NFR2**：CLI 所有命令與選項均須提供清晰的 `--help` 說明。
- **NFR3**：設定系統支援階層式優先級：
  1. CLI 旗標 (最高)
  2. 專案層級設定檔 `.git-llm-tool.yaml` (中等)
  3. 全域設定檔 `~/.git-llm-tool/config.yaml` (最低)
- **NFR4**：設定系統可安全讀取 LLM API 金鑰（透過設定檔或環境變數）。
- **NFR5**：允許指定預設 LLM（例如 `model: gpt-4o`）。
- **NFR6**：允許設定預設輸出語言（例如 `'zh-TW'` 或 `'en'`）。
- **NFR7**：MVP 不支援自訂 prompt 模板。
- **NFR8**：專案設定可啟用或禁用 Jira 整合。
- **NFR9**：若啟用 Jira，設定中可定義 regex（例如 `jira: branch_regex: "feature/(JIRA-\d+)-.*"`）。
- **NFR10**：從命令執行到 LLM 回應平均耗時應低於 5 秒（不含 git 與使用者互動時間）。
- **NFR11**：支援 macOS、Linux、Windows。

---

## 3. 使用者介面設計目標 (User Interface Design Goals)
> 此為 CLI 工具，故無圖形介面設計需求。

---

## 4. 技術假設 (Technical Assumptions) - v2

### 4.1 儲存庫結構 (Repository Structure)
建議採用 **Polyrepo（單一儲存庫）**。

### 4.2 服務架構 (Service Architecture)
建議使用 **單體 (Monolith)** 的本地端 CLI 應用程式。

### 4.3 測試需求 (Testing Requirements)
- **Unit Tests**：針對設定、Jira regex 等核心邏輯。
- **Integration Tests**：測試與真實 Git 命令互動。
- **Mocking**：模擬所有 LLM API 呼叫。

### 4.4 額外技術假設 (Additional Technical Assumptions)
- 語言：Python 3.8+
- CLI 框架：Click 或 Typer
- 設定管理：PyYAML
- Git 互動：`subprocess` 或 `GitPython`
- LLM 互動：直接 API 呼叫（輕量化）
- 架構上須設計 `LlmProvider` 介面，允許多家實作（如 OpenAI、Anthropic、Gemini）。

---

## 5. Epic 列表 (Epic List)

### Epic 1：基礎建設與核心 Commit 功能 (Foundation & Core Commit Feature)

#### Epic 目標
建立 CLI 應用程式骨架、設定系統與 LLM 抽象層，並交付完整 `git-llm commit` 功能（含 Jira 整合）。

---

#### Story 1.1：專案初始化與 CLI 骨架
**As a developer**, 我希望有一個使用 Click/Typer 的基本 CLI 專案結構，以便開始開發。
**驗收標準：**
- 專案結構（`pyproject.toml`, `src/git_llm_tool`）已建立。
- Click/Typer 已加入依賴。
- `git-llm --help` 可執行並顯示歡迎訊息。
- 已配置 Linting 與測試 (pytest)。

---

#### Story 1.2：階層式設定系統
**As a user**, 我希望 CLI 能從全域與專案層設定檔載入設定。
**驗收標準：**
- 能讀取 `~/.git-llm-tool/config.yaml`（全域）。
- 能讀取 `.git-llm-tool.yaml`（專案）並覆蓋全域設定。
- 能從環境變數（如 `OPENAI_API_KEY`）讀取金鑰。
- CLI 旗標覆蓋檔案設定。
- 已安裝 PyYAML。

---

#### Story 1.3：LLM 抽象層 (OpenAI)
**As a developer**, 我希望有 LlmProvider 抽象介面與 OpenAiProvider 實作。
**驗收標準：**
- 定義抽象基礎類 LlmProvider，包含 `generate_commit(diff, config)`。
- 實作 `OpenAiProvider`。
- 加入 `openai` 函式庫依賴。
- 從設定系統讀取 API 金鑰與 model。
- 單元測試模擬 openai 呼叫。

---

#### Story 1.4：commit 命令核心邏輯 (編輯器模式)
**As a developer**, 我希望命令能讀取暫存變更、呼叫 LLM 並開啟編輯器檢視訊息。
**驗收標準：**
- 實作 `git-llm commit` 子命令。
- 透過 GitPython 或 subprocess 取得 `git diff --cached`。
- 從設定讀取 `language`。
- Diff + 語言傳給 LlmProvider 產生訊息。
- 將結果寫入 `COMMIT_EDITMSG`。
- 在 macOS/Linux/Windows 均可運作。

---

#### Story 1.5：Jira 互動式整合
**As a user**, 我希望當啟用 Jira 時能互動輸入工時與 ticket。
**驗收標準：**
- 若 `jira.enabled=true`，則互動提示輸入工時。
- 若 regex 未設定或匹配失敗時，提示輸入 Jira Ticket。
- Jira 資訊被傳入 LLM prompt。
- 若 `jira.enabled=false`，不顯示任何提示。

---

#### Story 1.6：Jira Branch Regex 自動擷取
**As a user**, 我希望 CLI 能自動從 branch name 擷取 Jira ticket。
**驗收標準：**
- 讀取設定中的 `jira.branch_regex`。
- 透過 `git symbolic-ref --short HEAD` 取得 branch name。
- 成功匹配後自動擷取 ticket。
- 匹配成功時跳過手動輸入提示。
- 提取的 ticket 傳給 LLM。

---

#### Story 1.7：commit --apply 選項
**As a power user**, 我希望能直接套用 commit 訊息而不開啟編輯器。
**驗收標準：**
- 新增 `--apply`（或 `--non-interactive`）旗標。
- 使用此旗標時自動執行 `git commit -m "..."`。
- 預設模式仍開啟編輯器。

---

#### Story 1.8：擴充 LLM 供應商支援 (Anthropic & Gemini)
**As a developer**, 我希望支援多家 LLM 供應商。
**驗收標準：**
- 建立 `AnthropicProvider` 與 `GeminiProvider`。
- 新增 `anthropic` 與 `google-generativeai` 依賴。
- 根據 `model` 名稱自動選擇對應 provider。
- 正確讀取各 API 金鑰。

---

### Epic 2：Changelog 功能 (Changelog Feature)

#### Epic 目標
利用 Epic 1 的設定系統與 LLM 抽象層，建立 `git-llm changelog` 功能，彙總指定範圍的 commits 並分類輸出。

---

#### Story 2.1：changelog 命令骨架與 Git Log 讀取
**As a user**, 我希望有命令能讀取特定範圍的 commits。
**驗收標準：**
- 實作 `git-llm changelog` 子命令。
- 支援 `--from <ref>`、`--to <ref>` 參數。
- 預設讀取自上一個 git tag 以來的 commits。
- 透過 subprocess 或 GitPython 取得 `git log`。
- 提供完整 `--help` 說明。

---

#### Story 2.2：LLM Changelog 彙總
**As a user**, 我希望 changelog 命令能透過 LLM 生成分類彙總。
**驗收標準：**
- 將 commit messages 傳入 LlmProvider。
- 包含 `language` 設定。
- Prompt 指示 LLM 依功能分類（✨ Features、🐛 Fixes、💥 Breaking Changes）並輸出 Markdown。
- 結果印至 stdout。

---

#### Story 2.3：Changelog 檔案輸出
**As a user**, 我希望能將 changelog 儲存為檔案。
**驗收標準：**
- 新增 `--output <filename>`（或 `-o`）選項。
- 若指定檔案，輸出寫入檔案而非 stdout。
- 若檔案存在，提示是否覆蓋（除非使用 `--force`）。
- 若路徑無效，顯示錯誤訊息。
