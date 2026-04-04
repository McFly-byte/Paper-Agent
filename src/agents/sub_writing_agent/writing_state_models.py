from dataclasses import dataclass
from enum import Enum, auto
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, TypedDict
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

class SectionState(BaseModel):
    # title: str
    content: Optional[str] = None
    # research_materials: List[Dict[str, Any]] = []
    completed: bool = False

class WritingState(TypedDict, total=False):
    user_request: str
    global_analysis: Optional[str]
    sections: List[str]
    writted_sections: List[SectionState]
    current_section_index: int
    retrieved_docs: List[Dict[str, Any]]

