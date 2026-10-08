# P2 原生与闭集解码控制：2026-10-02 前瞻扩展

本方案已经见过原补充实验结果，是其后的前瞻扩展，不能称为未触碰测试集或预注册研究。以下 `_v2` 路径用于新的准备、生成和评价。

## 问题与可支持的主张

原结果同时包含政策差异与输出格式失败。增加一个严格限定输出字母的解码系统，检验同一证据呈现对比在两个系统设置中如何改变。核心对象是**解码设置与呈现政策的交互**；不宣称去掉格式失败就恢复了真实理解，也不把约束生成称作事后解析或纯语义机制隔离。

| 主张 | 最低证据 | 不允许的推论 |
|---|---|---|
| C1：政策效应可能依赖既定解码设置 | 两固定模型各自的 R−O、B−O：closed 对比减 fresh-native 对比，共四项双侧交互、完整原生分母与 98.75% 区间 | 不预设闭集有利；交互不显著不等于两设置等效 |
| C2：格式合规、正确性和政策排序需要分别报告 | 每个呈现格子的格式失败变化、修复/破坏、净正确数、完整成本和覆盖；AV 迁移描述 | 合规率提高不能单独证明事实判断改善；两个任务不合池 |

原来两模型的排序和联合筛查结论保持原样，历史输出仅作单独对照。本轮解码对照必须使用新生成的 native A/B；不能拿旧运行替代其中任何一次。

## 冻结输入与总体

- 原模型专属输入分别为 `paper2/prepared/supplement_{qwen,granite}_v1/requests.jsonl`。保持请求顺序、ID、消息角色与 content 字符串的 UTF-8 内容一致。重新写入的 JSONL 仅增加 metadata；不重新生成 prompt、padding、复制次数或目标。
- metadata 只含 `corpus`、`condition`、`allowed_labels`；从原 reference 读取身份、corpus 与 condition。标签、rationale、目标索引不进入新请求。reference 只在离线评分时使用原标签。
- 每模型完整保留 500 个 AVeriTeC 原生记录 × 6 政策，加 209 个 SciFact 证据承载 claim–document pair × 8 政策，共 709 个记录、4,672 个逻辑请求/运行。
- AV：C、P、B、W、S-front、S-back，四类 ABCD 不变。SciFact：C、P、B、W、R-front、R-back、O-front、O-back，ABC 对应 SUPPORT、CONTRADICT、NOINFO。即使此总体的 NOINFO gold 支持数为 0，也必须保留 NOINFO。
- 保留原来源、标签和 hash 目标；无新增标注、标注员、人工判断、父帖、回复图或会话输入。不改变 split，不以标签、长度、效果或格式表现筛样。
- 使用原有 exact claim / full-abstract 关系：SciFact 164 个精确文本连通组，AV 491 个精确 claim 组；不读取历史 extension/legacy relationship 数据。
- 原生过长请求继续保留。已知 AV `averitec:dev:414` 的五个非 C 政策超过输入上限，计失败、计零正确，不截断也不补跑。

准备程序核对原 preparation 文件绑定、原始数据重建的 observation、模型权重与来源；复制请求而不重跑证据呈现构造。新 preparation 保存旧 preparation 指纹、新请求指纹和本方案/代码绑定。新旧 `messages` 逐条比较；金标签改动的构造检查必须证明 metadata 复制不依赖标签。

## 两个解码系统及首步诊断

Native：运行**不修改的** `shared/generate.py`，SHA256 为 `f81bc57760070dbef65aaf60fa71ce3c70aa71f345224b7a4932e493f170ef6c`。

Closed：独立入口 `shared/generate_closed_labels.py`，使用 Hugging Face 的 [`prefix_allowed_tokens_fn` / `PrefixConstrainedLogitsProcessor`](https://huggingface.co/docs/transformers/v4.57.2/en/internal/generation_utils#transformers.PrefixConstrainedLogitsProcessor)。在 Transformers 4.57.2 的本地 `generation/utils.py` 和 `generation/logits_process.py` 中核对调用顺序；无需安装依赖或自定义 hook。

1. 对 A/B/C/D 分别执行 `encode(letter, add_special_tokens=False)`，必须恰好一个非 special token，且完整 decode roundtrip 为裸字母；不允许空格前缀、换行、备用词或多 token 候选。保存实际 token ID。
2. 首步只开放当前 corpus 的 ABC 或 ABCD。沿用原生成处理器后执行 greedy legal-label argmax；不修改原 logits、repetition penalty、聊天模板或 prompt。
3. 第二步只开放模型 `generation_config.json` 的第一个 EOS：Qwen 151645，Granite 0。生成保留 max_new_tokens=16 上限，但成功请求严格检查为一个字母 token 加一个 EOS，原始 response 必须恰好一个字母。
4. 保持模型 BF16、SDPA、slow tokenizer、thinking off、greedy `do_sample=False`、seed17、单分片、batch8、输入 8192 与输出上限 16。每次独立进程，无 resume。旧补充实验两个模型的 manifest 已确认这些参数一致。
5. 两模型均继承原 `generation_config.json`。Qwen 的 `repetition_penalty=1.05` 在 greedy 下仍有效；Granite 默认为 1。Qwen 原文件的 temperature/top_p/top_k 只在采样时有效。本方案不把继承的 greedy 设置误称为“所有解码处理器均关闭”。

限定的 shadow 诊断预先固定为：在**同一个首步 forward** 上保留每个合法字母的 raw logit、经过原生处理器和合法集合 mask 后的 processed score、该 token 是否出现于输入。使用官方 `return_dict_in_generate=True, output_logits=True, output_scores=True` 获得两个步骤的数据，不增加 forward。记录同一首步 raw logits 在合法字母上的 argmax 是否与正式 processed argmax 不同，平分时按最低词表 token ID 处理。

shadow 仅为继承解码处理器的系统设置敏感性诊断，不是第三个生成系统，不替代正式输出，不挑选得分更好的决策。raw 与 processed 可能包括多个原生处理器的差别，不能把观察到的差别单独归因于 repetition penalty，也不据此声称已解释重复证据效应。验证 generated token 必须等于保存的 processed argmax。所有诊断跟随同一运行的 cache 原件传递；同时报告逻辑请求和物理生成的分歧数。预先固定按 corpus/arm/realization 报告 raw-logit shadow 正确数、相对正式 closed 的 repairs/breakages/net 和 accuracy delta，所有原生记录为分母，过长与构造失败仍为零。它们仅为描述性结果，不增加主要检验，也不另选 verbalizer。

缓存仅限一个进程内部完全相同的 `messages`；要求相同消息对应同一合法字母表。不同 decoder、模型、A/B 之间不共享缓存。记录原始生成 token、原始 response、逻辑/charged 输入输出 token、有效 batch size、cache 来源与批次秒数。OOM 或其它异常保留日志和失败 manifest，禁止静默改 batch、截断或重跑个别请求。

## 唯一主要检验族与条件面板

对模型 m、原生 SciFact pair i、解码设置 d，令 `c(d,r,p,i)` 为既有 gold 上正确性，失败为 0。先在同一个 i 内对固定 front/back 与 fresh A/B 等权平均：

```
Delta_R(d,i) = mean_{r in A,B; s in front,back}[c(d,r,R-s,i) - c(d,r,O-s,i)]
Delta_B(d,i) = mean_{r in A,B; s in front,back}[c(d,r,B,i)   - c(d,r,O-s,i)]
Interaction_T(m) = mean_{i=1..209}[Delta_T(closed,i) - Delta_T(native,i)]
```

B 在两个 placement 对比中复用同一已生成预测，不制造新样本；A/B 和 front/back 均不把 209 个 pair 变成 836 个独立样本。

唯一主要族为 `Qwen R−O interaction`、`Qwen B−O interaction`、`Granite R−O interaction`、`Granite B−O interaction`。四项均双侧，不预设有利方向。每项使用 10,000 次有放回抽样，seed **20261002**，每次抽取完整 164 个精确文本连通组并携带所有 decoder/run/placement 结果，按 native pair 数重新计算比率。不能平均各组均值。区间为 Bonferroni family=4 的双侧 98.75% percentile 区间，同时保存 pointwise95% 供阅读。四项以外的区间不属于该主要族，不宣称全篇所有比较共同控制错误率。

全 209 分母是主结论。另报告跨两 decoder、两 realization、相关 placement 的 common-valid 面板，以及两 decoder × 两 realization × B/R-front/R-back/O-front/O-back 的共同 ROB 面板。用于事实标签变化的条件面板解释须覆盖至少 95%（SciFact 至少 199/209）；不足即标明覆盖失败，不能把缩小样本的正结果称为主证据。保存两个面板的覆盖和条件效应，不把面板选择当成随机缺失。

## 原有联合筛查的描述性迁移

closed 的每模型 R−O 与 B−O 单列原有筛查规则：完整总体点估计至少 +3pp；该四项描述性迁移筛查的 98.75% 区间下界大于 0；front/back 均为正；该 decoder 两 realization 的 contrast common-valid 与共同 ROB 覆盖均至少 95%，各条件差值均为正。

两模型四项全部满足才标记这个**描述性** joint screen 通过。其四项闭集对比区间与四项 primary interaction 区间分开，绝不声称两个族合并后仍有 5% 全族错误率保障。若失败，结论为“本闭集设置仍未满足既定联合条件”的有界负结果；不换模型、目标、prompt、标签词、penalty 或解析规则继续追求正结果。

## 描述性报告与成本

- AV 全 500：S−B 的固定双 placement / 双 realization 平均以及两 decoder 差异，只作迁移描述；使用 491 个精确 claim 组、同一 bootstrap seed、pointwise95%，不与 SciFact 合池。
- 每模型、corpus、政策、decoder、realization：accuracy、原类别空间 macro-F1、各类 support/F1/confusion、valid fraction、invalid-format/input-too-long/construction-failure 数量。SciFact 三类 macro-F1 的零支持 NOINFO 处理沿用原规则。
- 在同一请求和 realization 上报告 native→closed repairs、breakages、net correctness、format-failure delta（closed 减 native）、invalid→valid、valid→invalid 与双方有效标签改变；另报告 A→B 变化。格式修复与正确性修复不得混同。
- 给出原历史两模型结论的独立引用和本轮 fresh native/closed 各自政策排序，不以新结果覆盖旧结论。
- 输入 token、charged 输入成本、有效 batch 和 cache 布局必须跨两 decoder/两 realization 完全一致。生成 token 与生成时间可以变化，分别报告。保留缓存命中逻辑数和实际物理生成数，不把 metadata 或 shadow 诊断算成新生成请求。
- 8 次运行共 **37,376** 个逻辑请求；charged 输入理论上限 **306,184,192** token，输出理论上限 **598,016** token。闭集成功请求实际固定 2 个输出 token。按旧完整运行约 10–16 分钟估计总成本约 2 aggregate GPU-hours，保守预留 4 小时；超过估计两倍先检查原因、记录进度，不自动扩大实验。

## 顺序、停止规则与执行凭据

M0：构造 CPU 检查；两模型 `prepare` 和 `verify` 核对每模型 4,672 请求与消息不变，再检查 GPU 占用并启动。

M1：每模型 native A 与 closed A 的首次完整运行兼作实际系统检查；最多四块已获准空闲本地 GPU 并行，各进程仅一块。仅当对应 A 完成，启动相同 model/decoder 的 fresh B；不能复用进程。无额外选样 sanity 推理，也不新增模型或 GPU 网格。

M2：每模型四个 run 必须全部完成，当前评价器进一步要求两模型八个 run 全部通过绑定、成本、batch/cache 与 fresh-process 凭据后才进行任何原生数据评分。只看运行进度与错误允许；中途不计算有利政策/标签再调设置。

M3：固定一次全量评价与误差分解，保留不利结果。检查主要四项结果、条件面板覆盖、描述性 joint screen、shadow 与完整成本；过程日志只留在论文外。

执行 receipt 为 JSON，顶层 `runs` 恰好八项，`additional_attempts` 显式为列表（无额外尝试时 `[]`）。每项包含：`model_key`, `decoder` (`native`/`closed`), `realization` (`A`/`B`), `argv`（含 Python 路径）、`output_path`, `output_existed_before_launch: false`, `resume: false`, `visible_gpu_count: 1`, `launch_utc`, `finish_utc`, `returncode: 0`, `output_sha256`, `manifest_sha256`。失败尝试另外记录 `argv`, `launch_utc`, `finish_utc`, `returncode`, `log_path`, `reason`；不得以成功重跑覆盖失败日志。冻结设置不得作为自动 OOM 策略更换。

完整 `argv` 必须等于当前运行解释器的 `Path(sys.executable).resolve()` 路径加冻结生成参数，空解释器或其它路径均拒绝；每个 additional attempt 必须满足 `finish_utc >= launch_utc`，防止负耗时抵消 GPU 成本。这只涉及执行凭据校验，不改变输入、生成、估计量或主要检验。所有新准备、运行和评价产物使用 `_v2` 路径。

## 精确命令

所有命令从本 source directory 的项目根目录（`paper2/` 与 `shared/` 为同级）运行，并在所选 Python 环境中使 `python` 指向当前解释器。receipt 中保存该解释器的解析后绝对路径。

```bash
python paper2/test_decoder_control.py
python paper2/decoder_control.py prepare --model-key qwen --output paper2/prepared/decoder_control_qwen_20261002_v2
python paper2/decoder_control.py prepare --model-key granite --output paper2/prepared/decoder_control_granite_20261002_v2
python paper2/decoder_control.py verify --prepared paper2/prepared/decoder_control_qwen_20261002_v2
python paper2/decoder_control.py verify --prepared paper2/prepared/decoder_control_granite_20261002_v2
```

下列命令以每个 model/decoder 各自 A 完成后 B 的顺序执行；GPU 环境及 stdout/stderr 独立日志由执行者记录。展示使用相对路径，receipt 中 `argv` 的脚本与文件路径必须保存为与 `expected_argv()` 完全一致的绝对路径。

```bash
# Qwen native A then B (unchanged native runner)
for decoder_repeat in A B; do
  python shared/generate.py --input paper2/prepared/decoder_control_qwen_20261002_v2/requests.jsonl --output "paper2/runs/qwen_decoder_control_20261002_v2/native_${decoder_repeat}.jsonl" --model models/qwen2_5_7b_instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --batch-size 8 --max-input-tokens 8192 --max-new-tokens 16 --num-shards 1 --shard-index 0 --seed 17 || break
done
# Qwen closed A then B
for decoder_repeat in A B; do
  python shared/generate_closed_labels.py --input paper2/prepared/decoder_control_qwen_20261002_v2/requests.jsonl --output "paper2/runs/qwen_decoder_control_20261002_v2/closed_${decoder_repeat}.jsonl" --model models/qwen2_5_7b_instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --batch-size 8 --max-input-tokens 8192 --max-new-tokens 16 --num-shards 1 --shard-index 0 --seed 17 || break
done
# Granite native A then B (unchanged native runner)
for decoder_repeat in A B; do
  python shared/generate.py --input paper2/prepared/decoder_control_granite_20261002_v2/requests.jsonl --output "paper2/runs/granite_decoder_control_20261002_v2/native_${decoder_repeat}.jsonl" --model models/granite_3_3_8b_instruct --model-revision 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b --batch-size 8 --max-input-tokens 8192 --max-new-tokens 16 --num-shards 1 --shard-index 0 --seed 17 || break
done
# Granite closed A then B
for decoder_repeat in A B; do
  python shared/generate_closed_labels.py --input paper2/prepared/decoder_control_granite_20261002_v2/requests.jsonl --output "paper2/runs/granite_decoder_control_20261002_v2/closed_${decoder_repeat}.jsonl" --model models/granite_3_3_8b_instruct --model-revision 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b --batch-size 8 --max-input-tokens 8192 --max-new-tokens 16 --num-shards 1 --shard-index 0 --seed 17 || break
done
```

评价必须在上述八次全部完成且 receipt 可验证之后：

```bash
python paper2/decoder_control.py evaluate \
  --qwen-prepared paper2/prepared/decoder_control_qwen_20261002_v2 \
  --qwen-native-a paper2/runs/qwen_decoder_control_20261002_v2/native_A.jsonl \
  --qwen-native-b paper2/runs/qwen_decoder_control_20261002_v2/native_B.jsonl \
  --qwen-closed-a paper2/runs/qwen_decoder_control_20261002_v2/closed_A.jsonl \
  --qwen-closed-b paper2/runs/qwen_decoder_control_20261002_v2/closed_B.jsonl \
  --granite-prepared paper2/prepared/decoder_control_granite_20261002_v2 \
  --granite-native-a paper2/runs/granite_decoder_control_20261002_v2/native_A.jsonl \
  --granite-native-b paper2/runs/granite_decoder_control_20261002_v2/native_B.jsonl \
  --granite-closed-a paper2/runs/granite_decoder_control_20261002_v2/closed_A.jsonl \
  --granite-closed-b paper2/runs/granite_decoder_control_20261002_v2/closed_B.jsonl \
  --execution-receipt paper2/results/decoder_control_20261002_v2_execution_receipt.json \
  --output paper2/results/decoder_control_20261002_v2_evaluation
```

若准备或生成检查发现真实阻断，先修复并记录；不能因计划文本存在而把尚未检查的实现标作已通过。
