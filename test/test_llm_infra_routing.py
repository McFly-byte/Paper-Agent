"""llm_infra 路由与 provider 别名单测。

手动验证清单（集成，需有效 API Key）：
1. 不设 DEFAULT_LLM_PROVIDER：主流程仍走 siliconflow（models.yaml 各 *-model）。
2. 设置 DEFAULT_LLM_PROVIDER=bailian、DASHSCOPE_API_KEY：任意节点应命中 dashscope base_url，且日志含 enable_thinking。
3. 设置 DASHSCOPE_BATCH_BASE_URL：subanalyse-cluster-model / rag-eval-* 等 batch 策略应使用 batch host。
4. 设置 BAILIAN_QWEN35FLASH_RPM 等：限流桶参数应覆盖 yaml 默认值（日志「LLM 限流桶已创建」）。
5. search-model degradable=true：主模型失败时可回落 MODEL_FALLBACK_LIGHT（dashscope + flash + thinking=false）。
6. LangSmith 脚本：rag-eval-chat-model / judge 走 LangChain 封装，与 AutoGen 路由一致。
"""

from src.core.llm_infra.routing import (
    effective_chat_provider,
    effective_embedding_provider,
    normalize_provider_key,
)


def test_normalize_bailian_to_dashscope() -> None:
    assert normalize_provider_key("bailian") == "dashscope"
    assert normalize_provider_key("siliconflow") == "siliconflow"


def test_effective_chat_provider_force_siliconflow(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "siliconflow")
    assert effective_chat_provider("dashscope") == "siliconflow"


def test_effective_chat_provider_force_bailian(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "bailian")
    assert effective_chat_provider("siliconflow") == "dashscope"


def test_effective_embedding_provider_force(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_EMBEDDING_PROVIDER", "bailian")
    assert effective_embedding_provider("siliconflow") == "dashscope"
