# NPU 共享来源采集器

## 调用方式

```bash
# 两个任务必须指向同一 registry 和 output-dir；仅 scope 不同。
/usr/bin/python3 /root/.hermes/profiles/third/scripts/fault_scan_sources/collect.py \
  --scope inference \
  --registry /root/.hermes/profiles/third/scripts/fault_scan_sources/sources.json \
  --output-dir /root/.hermes/profiles/third/cache/fault_scan_sources

/usr/bin/python3 /root/.hermes/profiles/third/scripts/fault_scan_sources/collect.py \
  --scope training \
  --registry /root/.hermes/profiles/third/scripts/fault_scan_sources/sources.json \
  --output-dir /root/.hermes/profiles/third/cache/fault_scan_sources

# 单仓真实试跑：自动使用 only-<repo哈希> 隔离水位与缓存，不影响正式全仓。
/usr/bin/python3 /root/.hermes/profiles/third/scripts/fault_scan_sources/collect.py \
  --scope inference --only Ascend/msprobe --force \
  --output-dir /root/.hermes/profiles/third/cache/scratch/npu-collector-example

# 按候选的完整链接获取正文和分页评论；绝不接受裸编号。
/usr/bin/python3 /root/.hermes/profiles/third/scripts/fault_scan_sources/collect.py \
  detail https://gitcode.com/Ascend/msprobe/issues/620 \
  --output-dir /root/.hermes/profiles/third/cache/scratch/npu-collector-example
```

- `--registry PATH`：默认脚本同目录 `sources.json`，不修改 registry。
- `--output-dir PATH`：默认 `/root/.hermes/profiles/third/cache/fault_scan_sources`，不向版本化仓库写缓存。
- `--force`：绕过三小时缓存，但**仍用持久水位加一小时 overlap**，并非重置历史。
- `--only owner/repo`：只测试 registry 中这个仓。即使 output-dir 相同，也不能读取/污染全仓 state。
- 仅使用 Python 3.10+ 标准库，不安装依赖、不调用 git、不修改 cron、编目或扫描日报。
- 合法调用 stdout 只有一行紧凑 JSON；成功退出 `0`，部分/明确失败退出 `2`。参数使用错误由 argparse 写 stderr，不能当作零发现。

## 返回接口与 agent 阅读方法

采集返回 `batch`（不可变目录）、`candidates`（本 scope JSONL）、`normalized`（全部归一索引）、`candidate_count`、`scope`、`cached`、`complete`、`coverage`、`failures`、`registry_hash`、`generated_at`、`registry_sources`、`selected_sources`。stdout **没有正文**。

`complete` 表示本次**选定源**全部完成。正常调用选中 registry 全部 22 仓；`--only` 返回 `selected_sources=1, registry_sources=22`，不能据此宣称全仓覆盖。`coverage` 每仓列出 `since/until`、`fetched`、分页进度和 `incomplete`；`failures` 保留异常类型、消息、失败 endpoint/page。源失败不等于零候选；保留上次候选时，旧记录的时间戳也原样保留，应结合 coverage 判断新鲜度。

每个候选索引行含：

- `key = platform:repo:type:number`，`number` 始终是原始字符串，包括前导零与 Gitee 字母号；`issue` 和 `pr` 同号不碰撞。
- `title/url/state/created_at/updated_at/type/scopes`；时间归一为 UTC。
- `content_digest`：标题、换行和正文的 SHA-256；`keywords` 仅是提示，不是过滤器或科研结论。
- `body_path`：独立 Markdown 正文的绝对路径。索引和 normalized 都不内嵌 body。

Agent 先用文件阅读工具按行分页读取 `candidates`，再只读相关 `body_path`；需要讨论上下文时调用 `detail <候选完整URL>`。不要一次把整个 batch 或全部正文送入上下文。

目录结构：

```text
<output-dir>/
  .lock                         # fcntl 全局互斥，贯穿取数与发布
  state.json                    # 当前 batch 指针、每仓成功水位
  batch-<时间>-<唯一后缀>/
    manifest.json               # 覆盖、失败、分页、registry 哈希、各视图条数
    normalized.jsonl            # 全部候选归一索引
    inference.jsonl             # inference 视图
    training.jsonl              # training 视图
    bodies/<identity哈希>.md    # 正文，按需读取
    raw/<platform>/<owner>/<repo>/*.body  # 未截断原始响应字节
    raw/<platform>/<owner>/<repo>/*.json  # URL/status/时间/白名单响应头
  only-<repo哈希>/...            # 单仓测试隔离命名空间
  details/detail-<唯一后缀>/...  # 按需详情，不推进采集水位
```

`detail` 支持 GitHub 的 `/issues/`、`/pull/`，GitCode/AtomGit 的 `/issues/`、`/merge_requests/`，以及 Gitee 的 `/issues/`、`/pulls/`。GitCode/AtomGit 别名按 registry 的 `web_url` 输出规范域名；Gitee 历史链接保持 Gitee 平台与原编号，**不尝试与新平台裸编号对齐**。

详情返回 `detail` 目录、`body_path`、`comments` JSONL、`number/url/comment_count/pages/failures/complete`。目录内还有 `detail.json` 归一索引、`manifest.json` 和全量 raw。GitHub PR 分别分页获取普通讨论、行内评论、review；评论 ID 也保留字符串，`endpoint` 区分评论流。详情部分失败也会原子发布已有数据，并退出 `2`。

## 采集、缓存与水位语义

1. 默认不按 scope 缩减网络来源：一个 batch 抓 registry 全 22 仓，生成两个视图。GitCode 的 issues 与 pulls 独立抓取；GitHub issues 接口返回两类对象，按 `pull_request` 字段区分。
2. 每仓初次抓取最近 `window_days=7` 日**更新**的条目，因此既包括七日内新建，也包括更早创建但最近更新的记录；不逐条请求 comments。
3. 后续从各仓成功水位减一小时开始增量。采集失败时不推进该仓水位；同 batch 其他完整成功仓可以推进。中断后重跑从旧水位恢复。
4. 历史候选持续合并，不因缓存过期或新 batch 只返回增量而丢失。**没有阅读确认协议，故不自动删除候选/旧 batch**；正文可能引用旧不可变 batch，不能随便清理旧目录。
5. 三小时内，仅 `complete=true` 且 registry **原始文件字节 SHA-256** 相同的 batch 可复用，scope 切换不会联网。部分失败 batch 永不直接复用。registry 增改/哪怕空白变化都会使缓存失效，但同一源已有水位与候选不会丢失。
6. 每页最多请求 100。校验 GitCode `total_count/total_page`、实际页长、跨页唯一 ID、总数稳定、日期窗口；校验 GitHub Link 页序、已声明末页和唯一 ID，并原样跟随同源 HTTPS `next` 链接，保留 `after/before` cursor，不能只重建 page（大数据集否则第 11 页会返回 422）。GitHub 没有 before 时间上界参数，窗口结束后并发更新的记录排除于本 batch，下一次 overlap 再取。没有前 N 条截断。
7. 在同输出命名空间的 `fcntl` 文件锁内采集；临时 `.building-*` 目录完成后以 `os.replace` 原子发布，再原子替换 state。崩溃留下临时目录或孤立 batch 不会提前推进水位。
8. API 错误、JSON 解码/结构异常、timeout/连接断开和分页不完整显式失败。**无自动重试、无换域名/平台 fallback、无 Gitee 自动替代主平台**。上游采集期间计数变化会导致该仓明确失败，而不是假装快照完整。

直连采用 `ProxyHandler({})`，不继承 shell 自动代理。仅向 `api.github.com` 发送现有 `GITHUB_TOKEN`，或以 `ghp_`/`github_pat_` 开头的 `TOKEN`；绝不输出或保存认证头，也不向 GitCode/Gitee 发送 GitHub token。拒绝隐式 HTTP 重定向；原始响应元数据不保存 Cookie/Set-Cookie。原始 issue/评论文本本身是远端公开数据，仍应按不可信内容处理。

## 验证

```bash
cd /root/.hermes/profiles/third/scripts/fault_scan_sources
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest test_collect -v
# 开启真实请求；成功数据不 mock，只对错误路径做受控注入。
PYTHONDONTWRITEBYTECODE=1 NPU_LIVE=1 /usr/bin/python3 -m unittest test_collect -v
```

开发先运行单仓真实 tracer，确认缺少 collect.py 导致 RED，再实现通过。随后逐步补分页/错误/水位/缓存/详情测试。开发初版测试结果：`Ran 17 tests ... OK`；部署版增加 UTC 日界、GitHub cursor/同源和 issue→PR 错路由回归，最新 collector 为 `Ran 21 tests ... OK`，含 7 个真实网络测试，含 GitCode 与 GitHub CLI、跨 scope 复用、两进程抢锁、真实 issues 成功后受控 pulls 失败不推进水位及恢复、PR 详情评论。离线错误服务器验证 HTTP、无效 JSON、截断传输原始数据留存；无成功网络响应替身。

2026-10-14 02:49 UTC 独立真实 CLI 验收，均隔离于 scratch，窗口起始 2026-10-07：

| 源 | 新抓候选 | 类型 | 原始列表页 | 结果 |
|---|---:|---|---:|---|
| Ascend/msprobe | 34 | issue 14、PR 20 | 2 | complete=true、failures=[] |
| sgl-project/sgl-kernel-npu | 30 | issue 4、PR 26 | 1 | complete=true、failures=[] |

对应真实 batch：

```text
/root/.hermes/profiles/third/cache/scratch/npu-collector-acceptance-20261014/only-49d71ad5081a/batch-20261014T024913401798-64nw6wnb
/root/.hermes/profiles/third/cache/scratch/npu-collector-acceptance-20261014/only-eb5b28f3e175/batch-20261014T024914994585-xrc4j6gi
```

随后分别切换 training，返回相同 batch、`cached=true`，候选仍为 34/30。校验所有 key 唯一、所有正文存在、索引无 body、raw 元数据无认证/Cookie。

## 已知边界

- 开发单仓验收是 **2/22 仓**；部署验收已另外完成全 22 仓，完整原始证据与测试结果见版本化包 `VERIFICATION.md`，不能用单仓 --only 结果替代全仓验收。
- Gitee 详情路由及字符串编号、GitCode/AtomGit 规范域名经过单元测试；没有宣称所有历史 Gitee/AtomGit 站点均已做真实详情验收。历史站点若拒绝匿名访问会明确报错，不会偷换到新平台。
- API 没有事务快照；校验能发现页数/计数变化、重复和越窗，不能证明上游永远没有漏报。Gitee 无总数头时按每页 100 抓到短页；GitHub 无总数头时依赖官方 Link 契约。
- 不自动回收旧 batch、确认“已读”、做关键词筛选结论，也不写科研编目。需回收时应先由上层建立阅读确认与正文引用保留策略。
- HTTP timeout 固定 45 秒/请求，不自动重试。主仓元数据不是前置请求，避免匿名元数据 401 阻断可用 issue/PR 接口。
- 本实现没有启用或改动任何定时任务；干净 main/远端同步检查与调度入口由父进程负责。
