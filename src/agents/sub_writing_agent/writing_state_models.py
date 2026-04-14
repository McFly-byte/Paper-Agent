from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Literal, Optional, Dict, Any, TypedDict

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from asyncio import Queue


@dataclass
class WritingRunContext:
    """写作子图运行时依赖（SSE 队列），不参与 checkpoint。"""

    state_queue: Queue


class WritingStage(Enum):
    INIT = auto()
    OUTLINE = auto()
    SECTION_SPLIT = auto()
    WRITING = auto()
    RESEARCH = auto()
    COMPLETED = auto()

class ReviewDecision(BaseModel):
    """审查结构化输出；termination_phrase 含 APPROVE 以满足 TextMentionTermination。"""

    verdict: Literal["pass", "revise", "fail"] = Field(description="pass=可交付该节；revise=需修改；fail=不可接受")
    summary: str = Field(default="", description="一句话结论")
    blocking_issues: List[str] = Field(default_factory=list)
    minor_suggestions: List[str] = Field(default_factory=list)
    termination_phrase: str = Field(
        default="",
        description="verdict 为 pass 时必须包含 APPROVE（供 SelectorGroupChat 终止检测）",
    )

    @model_validator(mode="after")
    def _approve_token(self) -> "ReviewDecision":
        if self.verdict == "pass":
            return self.model_copy(update={"termination_phrase": "APPROVE"})
        return self.model_copy(update={"termination_phrase": ""})


class SectionState(BaseModel):
    # title: str
    content: Optional[str] = None
    # research_materials: List[Dict[str, Any]] = []
    completed: bool = False
    review_verdict: Optional[str] = None
    review_summary: Optional[str] = None

class WritingState(TypedDict, total=False):
    user_request: str
    global_analysis: Optional[str]
    sections: List[str]
    writted_sections: List[SectionState]
    current_section_index: int
    retrieved_docs: List[Dict[str, Any]]
    recovery_feedback: str  # 编排器注入的写作阶段恢复说明（纯文本）

