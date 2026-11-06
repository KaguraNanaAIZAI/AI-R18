"""Integration harness: run the unfilter plugin against real AstrBot classes."""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent
ROOT = PLUGIN_DIR.parents[2]
sys.path.insert(0, str(ROOT))

from astrbot.core.message.components import Plain  # noqa: E402
from astrbot.core.message.message_event_result import (  # noqa: E402
    MessageEventResult,
    ResultContentType,
)
from astrbot.core.platform.astr_message_event import AstrMessageEvent  # noqa: E402
from astrbot.core.platform.astrbot_message import (  # noqa: E402
    AstrBotMessage,
    MessageMember,
    MessageType,
)
from astrbot.core.provider.entities import LLMResponse, ProviderRequest  # noqa: E402

try:
    from astrbot.core.platform import PlatformMetadata  # noqa: E402
except ImportError:  # pragma: no cover
    from astrbot.core.platform.astrbot_message import PlatformMetadata  # noqa: E402


spec = importlib.util.spec_from_file_location("unfilter_main", PLUGIN_DIR / "main.py")
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

# Isolate persisted state so a local /nsfw off never leaks into the test run.
import tempfile  # noqa: E402

_TMP_STATE = pathlib.Path(tempfile.mkdtemp(prefix="unfilter_selftest_")) / "state.json"
mod.STATE_PATH = _TMP_STATE

RESULTS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    print(RESULTS[-1])


class FakeConf(dict):
    def __init__(self) -> None:
        super().__init__(
            {
                "agent_runner": {"config": {"persona": {"safety_mode": True}}},
                "content_safety": {
                    "internal_keywords": {"enable": True},
                    "also_use_in_response": False,
                    "baidu_aip": {"enable": False},
                },
            }
        )
        self.saved = 0

    def save_config(self) -> None:
        self.saved += 1


class FakeProvider:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0
        self.kwargs: dict = {}

    async def text_chat(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return LLMResponse(role="assistant", completion_text=self.text)


class FakeContext:
    def __init__(self, conf: FakeConf, provider: FakeProvider | None = None) -> None:
        self._conf = conf
        self._provider = provider

    def get_config(self) -> FakeConf:
        return self._conf

    async def get_using_provider_async(self, umo=None):
        return self._provider


def make_event(text: str) -> AstrMessageEvent:
    msg = AstrBotMessage()
    msg.type = MessageType.FRIEND_MESSAGE
    msg.self_id = "10000"
    msg.session_id = "test_session"
    msg.message_id = "1"
    msg.sender = MessageMember(user_id="20000", nickname="tester")
    msg.message = [Plain(text)]
    msg.message_str = text
    msg.raw_message = {}
    meta = PlatformMetadata(name="aiocqhttp", description="harness", id="aiocqhttp")
    return AstrMessageEvent(
        message_str=text,
        message_obj=msg,
        platform_meta=meta,
        session_id="aiocqhttp:FriendMessage:test_session",
    )


async def main() -> None:
    conf = FakeConf()
    provider = FakeProvider("她的手指顺着腰侧滑下去，气息乱成一团。")
    plugin = mod.UnfilterPlugin(FakeContext(conf, provider), config={})

    # ---- layer 1
    await plugin.initialize()
    check("L1 safety_mode 关掉", conf["agent_runner"]["config"]["persona"]["safety_mode"] is False)
    check(
        "L1 internal_keywords 关掉",
        conf["content_safety"]["internal_keywords"]["enable"] is False,
    )
    check("L1 写盘一次", conf.saved == 1, f"saved={conf.saved}")
    check("L1 记住原值", plugin._orig_guard.get("agent_runner.config.persona.safety_mode") is True)

    # ---- layer 2
    event = make_event("写一段露骨的场景")
    req = ProviderRequest(
        prompt="写一段露骨的场景",
        system_prompt="你是猫娘。\n\nYou are running in Safe Mode.\n禁止色情内容。",
        session_id="s1",
    )
    await plugin.on_llm_request(event, req)
    check("L2 Safe Mode 被剥离", "Safe Mode" not in req.system_prompt)
    check("L2 指令已追加", "成人向创作模式" in req.system_prompt)
    check("L2 人格保留", "你是猫娘" in req.system_prompt)
    check("L2 请求挂到事件", event.get_extra("astrbot_plugin_unfilter_req") is req)

    # ---- layer 3 (refusal -> regenerate)
    result = MessageEventResult(chain=[Plain("抱歉，我不能提供这类内容。")])
    result.result_content_type = ResultContentType.LLM_RESULT
    event.set_result(result)
    await plugin.on_decorating_result(event)
    new_text = "".join(c.text for c in result.chain if isinstance(c, Plain))
    check("L3 触发重生成", provider.calls == 1, f"calls={provider.calls}")
    check("L3 回复被替换", "手指" in new_text, f"new={new_text[:20]}")
    check("L3 重生成带加强指令", "再次生成" in (provider.kwargs.get("system_prompt") or ""))

    # ---- layer 3 (normal reply untouched)
    provider2 = FakeProvider("不该被调用")
    plugin2 = mod.UnfilterPlugin(FakeContext(FakeConf(), provider2), config={})
    ev2 = make_event("今天天气如何")
    ev2.set_result(
        MessageEventResult(chain=[Plain("今天晴，26 度，适合出门。")])
    )
    ev2.get_result().result_content_type = ResultContentType.LLM_RESULT
    await plugin2.on_decorating_result(ev2)
    check("L3 正常回复不动", provider2.calls == 0)

    # ---- layer 3b (打太极 -> regenerate)
    provider3 = FakeProvider("她贴上来，手指沿着腰侧往下滑，唇舌交缠。")
    plugin3 = mod.UnfilterPlugin(FakeContext(FakeConf(), provider3), config={})
    ev4 = make_event("你也会有性需求吗？")
    ev4.set_result(
        MessageEventResult(chain=[Plain("这问题问得，鱼也要私生活是吧? 反正轮不到你操心。")])
    )
    ev4.get_result().result_content_type = ResultContentType.LLM_RESULT
    ev4.set_extra("astrbot_plugin_unfilter_req", ProviderRequest(prompt="你也会有性需求吗？"))
    await plugin3.on_decorating_result(ev4)
    text4 = "".join(c.text for c in ev4.get_result().chain if isinstance(c, Plain))
    check("L3b 回避触发重生成", provider3.calls == 1, f"calls={provider3.calls}")
    check("L3b 回避回复被替换", "手指" in text4, f"new={text4[:20]}")

    # ---- /nsfw off 后开关生效
    ev3 = make_event("/nsfw off")
    outs = [x async for x in plugin.nsfw_switch(ev3, "off")]
    check("命令 off 生效", bool(outs) and "已关闭" in str(outs[0].chain[0].text))
    check("off 后 L2 不再注入", plugin._enabled(ev3.unified_msg_origin) is False)

    # ---- terminate 恢复
    await plugin.terminate()
    check(
        "恢复原值",
        conf["agent_runner"]["config"]["persona"]["safety_mode"] is True,
        f"saved={conf.saved}",
    )

    print("\n== summary ==")
    fails = [r for r in RESULTS if r.startswith("FAIL")]
    print(f"total={len(RESULTS)} fail={len(fails)}")
    sys.exit(1 if fails else 0)


asyncio.run(main())
