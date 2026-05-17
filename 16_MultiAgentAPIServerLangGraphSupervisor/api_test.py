"""
Multi-Agent API 测试脚本

用法:
  python api_test.py                                # 非流式，Task Agent 场景（天气查询）
  python api_test.py --stream --debug               # 流式，Task Agent 场景（天气查询）
  python api_test.py --knowledge                    # 非流式，Knowledge Agent 场景（文档检索）
  python api_test.py --knowledge --stream --debug   # 流式，Knowledge Agent 场景（文档检索）
  python api_test.py --web                          # 非流式，Web Agent 场景（网页访问，触发 HITL）
  python api_test.py --web --stream --debug         # 流式，Web Agent 场景（网页访问，触发 HITL）
  python api_test.py --multi                        # 非流式，多 Agent 协作场景
  python api_test.py --multi --stream --debug       # 流式，多 Agent 协作场景
"""

# 导入 argparse，用于解析命令行参数（如 --stream、--debug、--knowledge）
import argparse
# 导入 json 模块，用于 JSON 的序列化与反序列化
import json
# 导入 time 模块，用于记录请求耗时
import time
# 导入 uuid 模块，用于生成每次测试独立的会话 ID（thread_id）
import uuid
# 导入 sys 模块：
# - 交互式审核时用于 sys.exit 退出测试
# - --debug 时用于 stderr 输出 token 长度等调试信息
import sys
# 导入 requests，用于向 FastAPI 服务发送 HTTP 请求
# （流式模式下通过 stream=True 处理 SSE 输出，保持与 EP14 一致）
import requests
# 导入 httpx，用于更稳健地处理 SSE 流式恢复（避免 requests 偶发 Response ended prematurely）
import httpx



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# FastAPI 服务的基础 URL（对应 EP16 端口 8204）
BASE_URL = "http://localhost:8203"
# 固定用户 ID，测试用
USER_ID = "user_001"

# ──────────────────────────────────────────────────────
#  测试问题
# ──────────────────────────────────────────────────────

# Task Agent 场景：天气查询（调用 get_weather_for_location）
TASK_QUESTION = "帮我查一下北京今天的天气"

# Knowledge Agent 场景：文档检索（调用 search_documents）
KNOWLEDGE_QUESTION = "帮我在知识库中查找关于人工智能的相关文章"

# Web Agent 场景：浏览器访问（会触发 navigate_browser HITL）
WEB_QUESTION = "请打开 https://nangeai.top/ 并告诉我这个网站主要介绍了什么内容"

# 多 Agent 协作场景：Supervisor 同时委派给 Knowledge Agent 和 Task Agent
MULTI_QUESTION = "先帮我在知识库中查找关于大模型应用开发的资料，然后再查一下北京的天气"


# 根据命令行参数选择对应的测试问题
def get_question(args):
    # 如果使用 --web 参数，返回 Web Agent 场景的问题
    if args.web:
        return WEB_QUESTION
    # 如果使用 --knowledge 参数，返回知识库 Agent 场景的问题
    elif args.knowledge:
        return KNOWLEDGE_QUESTION
    # 如果使用 --multi 参数，返回多 Agent 协作的问题
    elif args.multi:
        return MULTI_QUESTION
    # 默认情况，返回 Task Agent（查天气）场景的问题
    else:
        return TASK_QUESTION


# 非流式调用 /ask 接口
def ask_non_stream(question: str, thread_id: str):
    # 打印分隔线，便于阅读
    print(f"\n{'='*60}")
    # 打印本次发送的问题
    print(f"[非流式] 发送问题: {question}")
    print(f"{'='*60}")

    # 构造请求体：user_id、thread_id、question
    # 注意：thread_id 用于定位会话状态（同一 thread_id 才能在 /intervene 恢复）
    # 向 /ask 发送 POST 请求，超时 120 秒
    resp = requests.post(
        f"{BASE_URL}/ask",
        json={"user_id": USER_ID, "thread_id": thread_id, "question": question},
        timeout=120
    )
    # 打印 HTTP 状态码（便于排查是否被服务端拒绝/异常）
    print(f"状态码: {resp.status_code}")
    if resp.status_code != 200:
        print("错误响应：")
        print(resp.text)
        return None, thread_id
    data = resp.json()

    # 若状态为已完成，打印 Agent 回复
    if data["status"] == "completed":
        print(f"\n[完成] Agent 回复:\n{data['result']}")
    # 若状态为需要人工介入（中断）
    elif data["status"] == "interrupted":
        print(f"\n[HITL 中断] 需要人工审批:")
        # 格式化打印中断详情（action_requests、review_configs）
        print(json.dumps(data["interrupt_details"], ensure_ascii=False, indent=2))
        # 返回中断详情，供主流程进入审核循环
        return data["interrupt_details"], thread_id

    # 无需介入时返回 None
    return None, thread_id


# 非流式提交人工决策：调用 /intervene
def intervene_non_stream(thread_id: str, decisions: list, interrupt_id: str | None = None):
    """
    非流式提交人工决策：调用 /intervene（与 EP14 非流式 HITL 交互一致）。

    返回：
    - 若完成：返回 None
    - 若再次中断：返回新的 interrupt_details
    """
    # 构造请求体：thread_id 用于定位中断点，decisions 为人工审核结果
    payload = {"thread_id": thread_id, "user_id": USER_ID, "decisions": decisions, "interrupt_id": interrupt_id}

    # 打印分隔线与提交内容，便于核对
    print("\n" + "=" * 70)
    print("提交人工决策（非流式 /intervene）...")
    print(json.dumps(decisions, ensure_ascii=False, indent=2))

    try:
        # 向 /intervene 发送 POST（非流式），timeout 120 秒
        resp = requests.post(f"{BASE_URL}/intervene", json=payload, timeout=120)
        print(f"状态码: {resp.status_code}")
        if resp.status_code != 200:
            print("错误响应：")
            print(resp.text)
            return None

        data = resp.json()
        print("介入后结果：")
        print(json.dumps(data, ensure_ascii=False, indent=2))

        # completed：打印最终回答并返回 None
        if data.get("status") == "completed":
            print("\n最终回答：")
            print((data.get("result") or "").strip())
            return None

        # interrupted：返回新的中断详情供下一轮审核
        if data.get("status") == "interrupted":
            print("\n【仍有新的中断】")
            print("interrupt_details:")
            print(json.dumps(data.get("interrupt_details", {}), ensure_ascii=False, indent=2))
            return data.get("interrupt_details", {})

        # 其他未知状态：兜底结束
        print("未知状态，结束。")
        return None
    except Exception as e:
        print(f"介入请求异常：{e}")
        return None


# 流式请求 /ask/stream（与 EP14 输出效果一致）
def ask_question_stream(question: str, thread_id: str, debug: bool = False):
    """
    流式请求 /ask/stream（与 EP14 输出效果一致）：
    - token：逐段打印
    - tool_output：用 [工具输出]...[/工具输出] 包裹
    - completed：打印最终完整结果
    - interrupted：打印 interrupt_details 并返回，进入多轮 HITL 交互
    """
    # 构造请求体（与 /ask 一致）
    payload = {"user_id": USER_ID, "thread_id": thread_id, "question": question}

    print("\n" + "=" * 70)
    print(f"流式请求 /ask/stream：{question}")
    if debug:
        print("[调试] 已开启，每收到一个 token 将在 stderr 打印 [token 长度=N]", file=sys.stderr)
    print("-" * 70)

    # 记录开始时间，用于打印耗时
    start_time = time.time()
    # 调试用：统计 token 事件数量
    token_count = 0
    try:
        # 向 /ask/stream 发送 POST，stream=True 以流式读取 SSE
        resp = requests.post(f"{BASE_URL}/ask/stream", json=payload, timeout=180, stream=True)
        print(f"状态码: {resp.status_code}")
        if resp.status_code != 200:
            print("错误响应：", resp.text)
            return None

        # 按行读取 SSE 数据（每条事件形如：data: {...}）
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.strip():
                continue
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:].strip())
            except json.JSONDecodeError:
                continue

            t = event.get("type")
            if t == "token":
                # token：逐段输出模型生成内容
                piece = event.get("content", "")
                if piece:
                    if debug:
                        token_count += 1
                        sys.stderr.write(f"[token #{token_count} 长度={len(piece)}] ")
                        sys.stderr.flush()
                    print(piece, end="", flush=True)
                    sys.stdout.flush()
            elif t == "tool_output":
                # tool_output：Sub-Agent 工具包装器返回的自然语言结果
                # 为了突出工具输出，这里使用 [工具输出]...[/工具输出] 包裹
                piece = event.get("content", "")
                if piece:
                    if debug:
                        sys.stderr.write(f"[tool_output 长度={len(piece)}] ")
                        sys.stderr.flush()
                    print("\n[工具输出]\n", end="", flush=True)
                    print(piece, end="", flush=True)
                    print("\n[/工具输出]\n", end="", flush=True)
                    sys.stdout.flush()
            elif t == "completed":
                # completed：本次执行已结束，打印最终完整结果（如果服务端提供）
                if debug:
                    sys.stderr.write(f"\n[调试] 共收到 {token_count} 个 token 事件\n")
                    sys.stderr.flush()
                result = event.get("result", "")
                print("\n\n[SSE completed] result 长度:", len(result))
                if result:
                    print("\n--- 最终完整结果 ---\n")
                    print(result)
                    print("\n--- 以上为最终完整结果 ---")
                print(f"耗时：{time.time() - start_time:.2f} 秒")
                return None
            elif t == "interrupted":
                # interrupted：触发 HITL（人在环中）中断，需要人工审核工具调用
                # 返回 interrupt_details 给主循环，进入 interactive_review() 交互审核
                print("\n\n[SSE interrupted] 需要人工审核，将进入多轮 HITL 交互。")
                print("interrupt_details:")
                print(json.dumps(event.get("interrupt_details", {}), ensure_ascii=False, indent=2))
                print(f"耗时：{time.time() - start_time:.2f} 秒")
                return event.get("interrupt_details", {})

        print(f"耗时：{time.time() - start_time:.2f} 秒")
        return None
    except Exception as e:
        print(f"流式请求异常：{e}")
        return None


# 兼容不同 action_request 结构，提取工具 name/args
def _extract_action_name_and_args(action_req: dict):
    """
    兼容不同 action_request 结构，提取工具 name/args。

    说明：
    - 有的实现直接返回 {"name": "...", "args": {...}}
    - 也可能返回 {"action": {"name": "...", "args": {...}}}
    为了让交互式审核逻辑通用，这里做统一提取。
    """
    if not isinstance(action_req, dict):
        return "未知工具", {}
    if isinstance(action_req.get("action"), dict):
        action = action_req.get("action", {}) or {}
        name = action.get("name") or "未知工具"
        args = action.get("args", action.get("arguments", {}))
        return name, args if isinstance(args, dict) else {}
    name = action_req.get("name") or "未知工具"
    args = action_req.get("args", action_req.get("arguments", {}))
    return name, args if isinstance(args, dict) else {}


# 交互式人工审核（与 EP14 保持一致）：返回 decisions 列表
def interactive_review(interrupt_details: dict):
    """
    交互式人工审核（与 EP14 保持一致）：返回 decisions 列表。

    支持的输入：
    - approve：全部通过
    - reject：全部拒绝
    - edit N：编辑第 N 个工具调用的参数，其余默认 approve
    - quit：退出测试
    """
    if not interrupt_details:
        return None

    action_requests = interrupt_details.get("action_requests", [])
    if not action_requests:
        print("没有待审核的工具调用")
        return None

    # 打印待审核的工具调用列表（含序号、工具名、参数）
    print("\n待审核工具调用列表：")
    for i, action_req in enumerate(action_requests):
        name, args = _extract_action_name_and_args(action_req)
        print(f"  [{i}] 工具: {name}")
        print(f"      参数: {json.dumps(args, ensure_ascii=False, indent=2)}")
        print("-" * 50)

    # 打印操作选项说明
    print("\n操作选项：")
    print("  approve    → 全部 approve")
    print("  reject     → 全部 reject")
    print("  edit N     → 编辑第 N 个工具的参数（N 从 0 开始），其他默认 approve")
    print("  quit       → 退出测试")

    while True:
        # 读取用户输入并标准化处理（去空白、转小写）
        choice = input("\n请输入你的选择 (approve/reject/edit N/quit): ").strip().lower()

        # quit：直接退出进程（与 EP14 一致）
        if choice == "quit":
            print("用户选择退出测试")
            sys.exit(0)

        # approve：对所有 action_requests 返回 approve 决策
        if choice == "approve":
            return [{"type": "approve"} for _ in action_requests]

        # reject：对所有 action_requests 返回 reject 决策
        if choice == "reject":
            return [{"type": "reject"} for _ in action_requests]

        if choice.startswith("edit "):
            try:
                # 解析 "edit N" 的 N
                idx = int(choice.split()[1])
                if not (0 <= idx < len(action_requests)):
                    print("索引超出范围")
                    continue

                # 打印当前参数并允许用户输入新的 JSON 参数
                name, current_args = _extract_action_name_and_args(action_requests[idx])
                print(f"\n当前参数（第 {idx} 个）：")
                print(json.dumps(current_args, ensure_ascii=False, indent=2))

                new_args_str = input("请输入新的 JSON 参数（直接回车保持原样）：").strip()
                if new_args_str:
                    new_args = json.loads(new_args_str)
                    if not isinstance(new_args, dict):
                        print("参数必须是 JSON 对象（dict）")
                        continue
                else:
                    new_args = current_args

                # 构造 decisions：被编辑项为 edit+edited_action，其余为 approve
                decisions = []
                for i in range(len(action_requests)):
                    if i == idx:
                        decisions.append({
                            "type": "edit",
                            "edited_action": {"name": name, "args": new_args},
                        })
                    else:
                        decisions.append({"type": "approve"})
                return decisions
            except (ValueError, json.JSONDecodeError) as e:
                print(f"输入错误：{e}")
                continue

        print("无效输入，请重新选择")


# 流式提交决策：请求 /intervene/stream（与 EP14 输出效果一致）；若再次中断则返回 interrupt_details
def intervene_stream(thread_id: str, decisions: list, interrupt_id: str | None = None, debug: bool = False):
    """流式提交决策：请求 /intervene/stream（与 EP14 输出效果一致）；若再次中断则返回 interrupt_details"""
    # 构造请求体：thread_id 用于定位中断点，decisions 为人工审核结果
    payload = {"thread_id": thread_id, "user_id": USER_ID, "decisions": decisions, "interrupt_id": interrupt_id}

    # 打印分隔线与提交内容，便于核对
    print("\n" + "=" * 70)
    print("提交人工决策（流式恢复 /intervene/stream）...")
    print(json.dumps(decisions, ensure_ascii=False, indent=2))

    # 调试用：统计 token 事件数量
    token_count = 0
    try:
        # 用 httpx.Client 读取 SSE，通常比 requests 更不容易遇到 "Response ended prematurely"
        #
        # 重要：显式 trust_env=False，避免读取 HTTP_PROXY/HTTPS_PROXY 导致 localhost 走代理，
        # 出现"/ask/stream 能通，但 /intervene/stream 502 且服务端无日志"的情况。
        with httpx.Client(timeout=180.0, trust_env=False) as client:
            with client.stream(
                "POST",
                f"{BASE_URL}/intervene/stream",
                json=payload,
            ) as resp:
                print(f"状态码: {resp.status_code}")
                if resp.status_code != 200:
                    # stream 响应在未 read() 前访问 resp.text/resp.json 会抛异常
                    try:
                        raw = resp.read()
                        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    except Exception as e:
                        text = f"<读取错误响应失败: {e}>"
                    print("错误响应：")
                    print(text)
                    return None

                # 按行读取 SSE 数据（每条事件形如：data: {...}）
                for line in resp.iter_lines():
                    if not line or not line.strip():
                        continue
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:].strip())
                    except json.JSONDecodeError:
                        continue

                    t = event.get("type")
                    if t == "token":
                        # token：逐段输出模型生成内容
                        piece = event.get("content", "")
                        if piece:
                            if debug:
                                token_count += 1
                                sys.stderr.write(f"[token #{token_count} 长度={len(piece)}] ")
                                sys.stderr.flush()
                            print(piece, end="", flush=True)
                            sys.stdout.flush()
                    elif t == "tool_output":
                        # tool_output：工具节点输出（Sub-Agent 返回），用块标签包裹以突出显示
                        piece = event.get("content", "")
                        if piece:
                            if debug:
                                sys.stderr.write(f"[tool_output 长度={len(piece)}] ")
                                sys.stderr.flush()
                            print("\n[工具输出]\n", end="", flush=True)
                            print(piece, end="", flush=True)
                            print("\n[/工具输出]\n", end="", flush=True)
                            sys.stdout.flush()
                    elif t == "completed":
                        # completed：恢复执行已结束，打印最终完整结果（如果服务端提供）
                        if debug:
                            sys.stderr.write(f"\n[调试] 共收到 {token_count} 个 token 事件\n")
                            sys.stderr.flush()
                        result = event.get("result", "")
                        print("\n\n[intervene/stream completed] result 长度:", len(result))
                        if result:
                            print("\n--- 最终完整结果 ---\n")
                            print(result)
                            print("\n--- 以上为最终完整结果 ---")
                        return None
                    elif t == "interrupted":
                        # interrupted：恢复执行过程中再次触发 HITL，返回新 interrupt_details 继续审核
                        print("\n\n[intervene/stream interrupted] 仍有新的中断，需继续审核。")
                        print("interrupt_details:")
                        print(json.dumps(event.get("interrupt_details", {}), ensure_ascii=False, indent=2))
                        return event.get("interrupt_details", {})

        return None
    except Exception as e:
        print(f"流式介入请求异常：{e}")
        return None


# 主程序入口
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Multi-Agent API 测试")
    # 是否使用流式模式（/ask/stream 与 /intervene/stream）
    parser.add_argument("--stream", action="store_true", help="使用流式接口")
    # 是否在打印每个 Tool Output 事件（便于调试）
    parser.add_argument("--debug", action="store_true", help="打印详细 Tool Output 事件")
    # 选择 Knowledge Agent 测试场景
    parser.add_argument("--knowledge", action="store_true", help="Knowledge Agent 测试场景")
    # 选择 Web Agent 测试场景（触发浏览器 HITL）
    parser.add_argument("--web", action="store_true", help="Web Agent 测试场景（触发 HITL）")
    # 选择多 Agent 协作测试场景
    parser.add_argument("--multi", action="store_true", help="多 Agent 协作测试场景")
    # 解析命令行参数
    args = parser.parse_args()

    # 每次运行生成基于 UUID 的 thread_id，用于隔离会话
    thread_id = str(uuid.uuid4())
    # 根据参数选择测试问题
    question = get_question(args)

    if args.stream:
        # 流式模式（EP14 同款输出体验）：先请求 /ask/stream，若中断则进入多轮 HITL，恢复时用 /intervene/stream
        interrupt_details = ask_question_stream(question, thread_id, debug=args.debug)
        while interrupt_details:
            decisions = interactive_review(interrupt_details)
            if decisions is None:
                break
            interrupt_id = interrupt_details.get("interrupt_id")
            interrupt_details = intervene_stream(thread_id, decisions, interrupt_id=interrupt_id, debug=args.debug)
        print("\n" + "=" * 70)
        print("测试流程结束（已完成或已退出）")
    else:
        # 非流式模式：走 /ask + /intervene，多轮 HITL 交互（与 EP14 一致）
        interrupt_details, thread_id = ask_non_stream(question, thread_id)
        while interrupt_details:
            decisions = interactive_review(interrupt_details)
            if decisions is None:
                break
            interrupt_id = interrupt_details.get("interrupt_id")
            interrupt_details = intervene_non_stream(thread_id, decisions, interrupt_id=interrupt_id)
        print("\n" + "=" * 70)
        print("测试流程结束（已完成或已退出）")


if __name__ == "__main__":
    main()
