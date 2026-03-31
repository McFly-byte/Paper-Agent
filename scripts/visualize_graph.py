# ---------------------------------------------------------------------------
# 使用 pygraphviz 或 graphviz 可视化项目中的 LangGraph 图结构
# 方式一（推荐）：pip install pygraphviz，并安装系统 Graphviz（含开发头文件）
# 方式二：pip install graphviz，并确保系统 Graphviz 的 dot 在 PATH 中
# 系统 Graphviz：https://graphviz.org/download/（Windows 可 choco install graphviz）
# ---------------------------------------------------------------------------

import os
import sys

# 项目根目录加入路径，便于单独运行脚本
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUTPUT_DIR = os.path.join(_ROOT, "data", "graph_viz")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- DOT 源（供 pygraphviz 或 graphviz 包使用）----------

def get_orchestrator_dot() -> str:
    """主编排器 DAG 的 DOT 源：检索 → 阅读 → 分析 → 写作 → 报告，含条件边与错误节点。"""
    return r"""
digraph {
    rankdir=LR;
    splines=polyline;
    label="Paper-Agent 主编排 DAG (LangGraph)";
    labelloc=t;
    fontsize=14;
    fontname="Microsoft YaHei";

    START [shape=ellipse, style=filled, fillcolor="#e8f5e9", label="START"];
    search_node [shape=box, style="rounded,filled", fillcolor="#bbdefb", label="search_node\n(检索)"];
    reading_node [shape=box, style="rounded,filled", fillcolor="#bbdefb", label="reading_node\n(阅读)"];
    analyse_node [shape=box, style="rounded,filled", fillcolor="#bbdefb", label="analyse_node\n(分析)"];
    writing_node [shape=box, style="rounded,filled", fillcolor="#bbdefb", label="writing_node\n(写作)"];
    report_node [shape=box, style="rounded,filled", fillcolor="#bbdefb", label="report_node\n(报告)"];
    handle_error_node [shape=box, style="rounded,filled", fillcolor="#ffcdd2", label="handle_error_node\n(错误处理)"];
    END [shape=ellipse, style=filled, fillcolor="#fff3e0", label="END"];

    START -> search_node;
    search_node -> reading_node [label="成功"];
    search_node -> handle_error_node [label="错误", color=red];
    reading_node -> analyse_node [label="成功"];
    reading_node -> handle_error_node [label="错误", color=red];
    analyse_node -> writing_node [label="成功"];
    analyse_node -> handle_error_node [label="错误", color=red];
    writing_node -> report_node [label="成功"];
    writing_node -> handle_error_node [label="错误", color=red];
    report_node -> END [label="成功"];
    report_node -> handle_error_node [label="错误", color=red];
    handle_error_node -> END;
}
"""


def get_writing_workflow_dot() -> str:
    """写作子工作流 DOT 源：writing_director_node → parallel_writing_node → END。"""
    return r"""
digraph {
    rankdir=LR;
    splines=polyline;
    label="写作子工作流 (WritingWorkflow)";
    labelloc=t;
    fontsize=14;
    fontname="Microsoft YaHei";

    START [shape=ellipse, style=filled, fillcolor="#e8f5e9", label="START"];
    writing_director_node [shape=box, style="rounded,filled", fillcolor="#c5cae9", label="writing_director_node\n(大纲/任务拆分)"];
    parallel_writing_node [shape=box, style="rounded,filled", fillcolor="#c5cae9", label="parallel_writing_node\n(并行写作)"];
    END [shape=ellipse, style=filled, fillcolor="#fff3e0", label="END"];

    START -> writing_director_node;
    writing_director_node -> parallel_writing_node;
    parallel_writing_node -> END;
}
"""


# ---------- pygraphviz 绘制（需安装 pygraphviz + 系统 Graphviz 开发库）----------

def _draw_with_pygraphviz():
    import pygraphviz as pgv

    # 1. 主编排图
    g_main = pgv.AGraph(string=get_orchestrator_dot(), strict=False, directed=True)
    path_main_png = os.path.join(OUTPUT_DIR, "orchestrator_graph.png")
    path_main_svg = os.path.join(OUTPUT_DIR, "orchestrator_graph.svg")
    g_main.draw(path_main_png, prog="dot")
    g_main.draw(path_main_svg, prog="dot")
    print(f"主编排图已保存: {path_main_png}, {path_main_svg}")

    # 2. 写作子图
    g_writing = pgv.AGraph(string=get_writing_workflow_dot(), strict=False, directed=True)
    path_writing_png = os.path.join(OUTPUT_DIR, "writing_workflow_graph.png")
    path_writing_svg = os.path.join(OUTPUT_DIR, "writing_workflow_graph.svg")
    g_writing.draw(path_writing_png, prog="dot")
    g_writing.draw(path_writing_svg, prog="dot")
    print(f"写作子图已保存: {path_writing_png}, {path_writing_svg}")


# ---------- graphviz 包绘制（仅需系统 dot 可执行文件，无需编译）----------

def _draw_with_graphviz_package():
    import graphviz

    # 1. 主编排图
    path_main_png = os.path.join(OUTPUT_DIR, "orchestrator_graph.png")
    path_main_svg = os.path.join(OUTPUT_DIR, "orchestrator_graph.svg")
    graphviz.Source(get_orchestrator_dot()).render(
        filename="orchestrator_graph", format="png", directory=OUTPUT_DIR, cleanup=True
    )
    graphviz.Source(get_orchestrator_dot()).render(
        filename="orchestrator_graph", format="svg", directory=OUTPUT_DIR, cleanup=True
    )
    print(f"主编排图已保存: {path_main_png}, {path_main_svg}")

    # 2. 写作子图
    path_writing_png = os.path.join(OUTPUT_DIR, "writing_workflow_graph.png")
    path_writing_svg = os.path.join(OUTPUT_DIR, "writing_workflow_graph.svg")
    graphviz.Source(get_writing_workflow_dot()).render(
        filename="writing_workflow_graph", format="png", directory=OUTPUT_DIR, cleanup=True
    )
    graphviz.Source(get_writing_workflow_dot()).render(
        filename="writing_workflow_graph", format="svg", directory=OUTPUT_DIR, cleanup=True
    )
    print(f"写作子图已保存: {path_writing_png}, {path_writing_svg}")


def _write_dot_files():
    """始终写入 .dot 源文件，便于用在线查看器或本地 dot 命令渲染。"""
    path_main = os.path.join(OUTPUT_DIR, "orchestrator_graph.dot")
    path_writing = os.path.join(OUTPUT_DIR, "writing_workflow_graph.dot")
    with open(path_main, "w", encoding="utf-8") as f:
        f.write(get_orchestrator_dot().strip())
    with open(path_writing, "w", encoding="utf-8") as f:
        f.write(get_writing_workflow_dot().strip())
    print(f"DOT 源文件已保存: {path_main}, {path_writing}")
    print("  可将 .dot 内容复制到 https://dreampuf.github.io/GraphvizOnline/ 查看，或安装 Graphviz 后执行：")
    print("  dot -Tpng orchestrator_graph.dot -o orchestrator_graph.png")


def main():
    _write_dot_files()

    # 优先使用 pygraphviz（你指定的库），不可用时回退到 graphviz 包
    try:
        import pygraphviz as pgv  # noqa: F401
        _draw_with_pygraphviz()
    except ImportError:
        try:
            import graphviz  # noqa: F401
            _draw_with_graphviz_package()
            print("\n（未安装 pygraphviz，已使用 graphviz 包渲染）")
        except ImportError:
            print("\n未安装 pygraphviz。若需 PNG/SVG，请：pip install graphviz 并安装系统 Graphviz（dot 在 PATH）")
        except Exception as e:
            if "ExecutableNotFound" in type(e).__name__ or "dot" in str(e).lower():
                print("\n未检测到 Graphviz 的 dot 可执行文件。请安装 Graphviz 并加入 PATH：")
                print("  https://graphviz.org/download/  （Windows 可用 winget install Graphviz.Graphviz）")
            else:
                raise

    print(f"\n所有输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
