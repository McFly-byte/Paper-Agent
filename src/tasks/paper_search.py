import asyncio
import logging
import threading
import time
import arxiv
import requests
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

from src.core.config import config
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

# --- 进程级 arXiv 频率上限（根本措施）---
# arXiv ToU：同一 IP 约每 3 秒至多一次 API 请求。arxiv.Client 的 delay_seconds 只作用于**单个实例**，
# 多 run / 多标签页会创建多个 Client → 各实例互不感知 → 同一时刻多发请求 → 429。
# 做法：在 requests.Session.get 外包一层，全进程共享「下一次允许发起请求」的时间戳；
# 只在极短临界区里 sleep/改时间戳，**不**在持锁期间跑完整次检索或长退避（避免此前全局锁卡死）。
_arxiv_http_lock = threading.Lock()
_arxiv_next_request_monotonic = 0.0


def _arxiv_global_min_interval() -> float:
    return float(config.get("paper_search.arxiv_global_min_interval", 4.5) or 4.5)


def _arxiv_trust_env() -> bool:
    return config.get_bool("paper_search.arxiv_trust_env", False)


def _arxiv_client_inner_retries() -> int:
    return int(config.get_int("paper_search.arxiv_client_inner_retries", 0))


def _arxiv_retry_backoffs() -> Tuple[int, ...]:
    raw = config.get("paper_search.arxiv_retry_backoff_seconds")
    if isinstance(raw, list) and raw and all(isinstance(x, (int, float)) for x in raw):
        return tuple(int(x) for x in raw)
    return (20, 45, 90, 120)


def _arxiv_simplify_on_retry() -> bool:
    return config.get_bool("paper_search.arxiv_simplify_query_on_retry", True)


def _arxiv_reduce_max_from_attempt() -> int:
    return int(config.get_int("paper_search.arxiv_reduce_max_results_from_attempt", 2))


def _install_arxiv_global_rate_limit(session: Any) -> None:
    """对 arxiv 使用的 Session 打补丁：任意 Client、任意协程/线程，两次 GET 起始间隔可配置（默认 ≥4.5s）。"""
    if getattr(session, "_paper_agent_arxiv_gated", False):
        return
    raw_get = session.get

    def gated_get(url: str, **kwargs: Any):
        global _arxiv_next_request_monotonic
        interval = max(2.0, _arxiv_global_min_interval())
        with _arxiv_http_lock:
            now = time.monotonic()
            wait = _arxiv_next_request_monotonic - now
            if wait > 0:
                time.sleep(wait)
            _arxiv_next_request_monotonic = time.monotonic() + interval
        return raw_get(url, **kwargs)

    session.get = gated_get  # type: ignore[method-assign]
    session._paper_agent_arxiv_gated = True


_ARXIV_CLIENT_DELAY_SECONDS = 0.0
_ARXIV_REQUEST_TIMEOUT: Tuple[float, float] = (15.0, 180.0)
_ARXIV_USER_AGENT = (
    "Paper-Agent/1.0 (compatible; +https://arxiv.org/help/api/tou; academic research tool)"
)

class PaperSearcher:
    """论文搜索器，使用arxiv库搜索论文"""
    
    def __init__(self):
        """初始化论文搜索器"""
        pass

    def _build_arxiv_search_query(
        self,
        querys: List[str],
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
    ) -> str:
        """
        构造 arXiv search_query。旧逻辑用 all:%22 + 片段 + %22，与 LLM 已带引号的布尔式叠加后
        易产生畸形语法（如 all:\"\"Diffusion...）并加重服务端压力；改为 all:(...) 包裹每条子表达式。
        """
        parts: List[str] = []
        for raw in querys:
            t = (raw or "").strip()
            if not t:
                continue
            if not (t.startswith("(") and t.endswith(")")):
                t = f"({t})"
            parts.append(f"all:{t}")
        if not parts:
            return ""
        search_query = " OR ".join(parts)
        if start_date or end_date:
            start_date_str = self._format_date(start_date) if start_date else "190001010000"
            end_date_str = (
                self._format_date(end_date)
                if end_date
                else datetime.now().strftime("%Y%m%d2359")
            )
            date_filter = f"submittedDate:[{start_date_str} TO {end_date_str}]"
            search_query = f"({search_query}) AND {date_filter}"
        return search_query

    @staticmethod
    def _simplify_querys_for_retry(querys: List[str], attempt: int) -> List[str]:
        """重试时缩短布尔式，降低 arXiv 对超长 query 的 429 概率。"""
        if attempt <= 0 or not querys:
            return list(querys)
        cleaned = [(q or "").strip() for q in querys if (q or "").strip()]
        if not cleaned:
            return list(querys)
        if len(cleaned) == 1:
            q0 = cleaned[0]
            if attempt == 1 and len(q0) > 240:
                return [q0[:240].rstrip()]
            if attempt >= 2 and len(q0) > 120:
                return [q0[:120].rstrip()]
            return [q0]
        shortest = min(cleaned, key=len)
        if attempt == 1:
            return [shortest]
        s = shortest.strip()
        if len(s) > 160:
            s = s[:160].rstrip()
        return [s] if s else cleaned[:1]

    def _arxiv_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, arxiv.HTTPError):
            return getattr(exc, "status", None) in (429, 503)
        if isinstance(
            exc,
            (
                requests.exceptions.ProxyError,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ),
        ):
            return True
        return False

    def _fetch_with_retry_backoff(
        self,
        querys: List[str],
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
        max_results: int,
        sort_by: arxiv.SortCriterion,
        sort_order: arxiv.SortOrder,
    ) -> List[Dict]:
        """执行检索；遇 HTTP 429/503 或代理/连接类错误时退避后重试，并可逐步简化子查询。"""
        backoffs = _arxiv_retry_backoffs()
        n_backoffs = len(backoffs)
        simplify = _arxiv_simplify_on_retry()
        reduce_from = _arxiv_reduce_max_from_attempt()

        for attempt in range(n_backoffs + 1):
            q_use = self._simplify_querys_for_retry(querys, attempt) if simplify else list(querys)
            if not q_use:
                q_use = list(querys)
            mr = max_results
            if reduce_from > 0 and attempt >= reduce_from:
                mr = max(5, min(max_results, 25))

            search_query = self._build_arxiv_search_query(q_use, start_date, end_date)
            if not search_query.strip():
                logger.warning("论文搜索跳过：简化后 arXiv 查询为空")
                return []

            if attempt > 0:
                logger.info(
                    "arXiv 重试 attempt=%s：子式数=%s max_results=%s 查询长度=%s",
                    attempt,
                    len(q_use),
                    mr,
                    len(search_query),
                )

            page_size = max(1, min(50, mr))
            client = arxiv.Client(
                page_size=page_size,
                delay_seconds=_ARXIV_CLIENT_DELAY_SECONDS,
                num_retries=_arxiv_client_inner_retries(),
            )
            client._session.headers.update({"User-Agent": _ARXIV_USER_AGENT})
            client._session.timeout = _ARXIV_REQUEST_TIMEOUT
            client._session.trust_env = _arxiv_trust_env()
            _install_arxiv_global_rate_limit(client._session)

            try:
                search = arxiv.Search(
                    query=search_query,
                    max_results=mr,
                    sort_by=sort_by,
                    sort_order=sort_order,
                )
                return self.format_papers_list(client.results(search))
            except arxiv.HTTPError as e:
                status = getattr(e, "status", None)
                if status not in (429, 503) or attempt >= n_backoffs:
                    raise
                wait = backoffs[attempt]
                reason = "请求过频(429)" if status == 429 else "服务暂时不可用(503)"
                logger.warning(
                    "arXiv 返回 HTTP %s（%s），等待 %s 秒后重试 (%s/%s)",
                    status,
                    reason,
                    wait,
                    attempt + 1,
                    n_backoffs,
                )
                time.sleep(wait)
            except Exception as e:  # noqa: BLE001
                if not self._arxiv_retryable(e) or attempt >= n_backoffs:
                    raise
                wait = backoffs[attempt]
                logger.warning(
                    "arXiv 请求异常（%s: %s），等待 %s 秒后重试 (%s/%s)",
                    type(e).__name__,
                    str(e)[:500],
                    wait,
                    attempt + 1,
                    n_backoffs,
                )
                time.sleep(wait)

        raise RuntimeError("arXiv 检索：重试耗尽仍未返回结果")

    def _search_papers_sync(
        self,
        querys: List[str],
        max_results: int,
        sort_by: arxiv.SortCriterion,
        sort_order: arxiv.SortOrder,
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
    ) -> List[Dict]:
        """同步执行 arXiv 查询与解析（供 asyncio.to_thread 调用）。"""
        if not querys:
            logger.warning("论文搜索跳过：querys 为空")
            return []
        try:
            preview = self._build_arxiv_search_query(querys, start_date, end_date)
            if not preview.strip():
                logger.warning("论文搜索跳过：拼接后的 arXiv 查询为空")
                return []

            logger.info(
                "开始搜索论文: max_results=%s sort_by=%s 查询长度=%s",
                max_results,
                sort_by,
                len(preview),
            )
            logger.info("论文搜索查询条件: %s", preview[:4000] + ("…" if len(preview) > 4000 else ""))

            papers = self._fetch_with_retry_backoff(
                querys, start_date, end_date, max_results, sort_by, sort_order
            )

            logger.info(f"论文搜索完成，共找到 {len(papers)} 篇论文")
            return papers
        except Exception as e:
            logger.error(f"论文搜索失败: {str(e)}")
            raise

    async def search_papers(self,
                      querys: List[str],
                      max_results: int = 50,
                      sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
                      sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
                      start_date: Optional[Union[str, datetime]] = None,
                      end_date: Optional[Union[str, datetime]] = None) -> List[Dict]:
        """
        搜索 arXiv 论文（在线程池中执行同步 arxiv 客户端，避免阻塞事件循环）。

        参数:
            querys: 搜索关键词
            max_results: 最大返回结果数量
            sort_by: 排序方式 (Relevance, LastUpdatedDate, SubmittedDate)
            sort_order: 排序顺序 (Ascending, Descending)
            start_date: 开始日期，可以是字符串(YYYY-MM-DD)或 datetime
            end_date: 结束日期，可以是字符串(YYYY-MM-DD)或 datetime

        返回:
            论文列表，每项包含论文的详细信息
        """
        return await asyncio.to_thread(
            self._search_papers_sync,
            querys,
            max_results,
            sort_by,
            sort_order,
            start_date,
            end_date,
        )

    async def search_by_topic(self, 
                       topic: str, 
                       limit: int = 10, 
                       recent_days: Optional[int] = None) -> List[Dict]:
        """
        按主题搜索最近的论文
        
        参数:
            topic: 主题关键词
            limit: 返回结果数量限制
            recent_days: 搜索最近多少天的论文，None表示不限制
        
        返回:
            论文列表
        """
        logger.info(f"按主题搜索论文: topic='{topic}', limit={limit}, recent_days={recent_days}")
        
        # 计算开始日期
        start_date = None
        if recent_days:
            start_date = datetime.now() - timedelta(days=recent_days)
        
        # 调用搜索方法
        return self.search_papers(
            query=topic,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
            start_date=start_date
        )
    
    def format_papers_list(self, search_results) -> List[Dict]:
        """
        将搜索结果（迭代器或列表）格式化为论文信息字典列表
        
        参数:
            search_results: arxiv搜索结果对象（可能是迭代器）
        
        返回:
            格式化后的论文信息字典列表
        """
        # 将迭代器转换为列表以便后续处理
        results_list = list(search_results)
        
        # 格式化论文列表
        formatted_papers = [self._parse_paper_result(result) for result in results_list]
        
        logger.info(f"开始格式化论文列表，共 {len(results_list)} 篇论文")
        return formatted_papers

    def search_by_author(self, 
                        author_name: str, 
                        limit: int = 10) -> List[Dict]:
        """
        按作者搜索论文
        
        参数:
            author_name: 作者姓名
            limit: 返回结果数量限制
        
        返回:
            论文列表
        """
        logger.info(f"按作者搜索论文: author='{author_name}', limit={limit}")
        
        # 使用作者字段搜索
        query = f"au:{author_name}"
        return self.search_papers(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
    
    def _parse_paper_result(self, result: arxiv.Result) -> Dict:
        """
        解析arXiv搜索结果
        
        参数:
            result: arxiv.Result对象
        
        返回:
            包含论文信息的字典
        """
        # 从结果URL中提取论文ID
        paper_id = result.get_short_id()
        
        # 提取发布年份
        published_year = result.published.year if result.published else None
        
        return {
            "paper_id": paper_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": published_year,
            "published_date": result.published.isoformat() if result.published else None,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "doi": result.doi if hasattr(result, 'doi') else None
        }
    
    def _format_date(self, date: Union[str, datetime]) -> str:
        """
        
        格式化日期为arXiv API支持的格式YYYYMMDDTTTT
        
        参数:
            date: 日期字符串或datetime对象
        
        返回:
            格式化后的日期字符串YYYYMMDD0000
        """
        if isinstance(date, datetime):
            return date.strftime("%Y%m%d0000")
        elif isinstance(date, str):
            # 定义多种可能的日期格式
            date_formats = [
                "%Y-%m-%d",      # YYYY-MM-DD
                "%Y/%m/%d",      # YYYY/MM/DD
                "%Y.%m.%d",      # YYYY.MM.DD
                "%Y-%m",         # YYYY-MM
                "%Y/%m",         # YYYY/MM
                "%Y",            # YYYY
                "%Y年%m月%d日",  # 中文格式
                "%Y年%m月",      # 中文格式（年月）
                "%Y年",          # 中文格式（年）
            ]
            
            for fmt in date_formats:
                try:
                    if fmt == "%Y":  # 单独处理只有年份的情况
                        if len(date) == 4 and date.isdigit():
                            parsed_date = datetime(int(date), 1, 1)
                            return parsed_date.strftime("%Y%m%d0000")
                    elif fmt in ["%Y-%m", "%Y/%m", "%Y年%m月"]:  # 处理年月格式
                        try:
                            parsed_date = datetime.strptime(date, fmt)
                            return parsed_date.strftime("%Y%m%d0000")
                        except:
                            continue
                    else:
                        parsed_date = datetime.strptime(date, fmt)
                        return parsed_date.strftime("%Y%m%d0000")
                except ValueError:
                    continue
            
            # 如果所有格式都失败，尝试使用dateutil或返回默认值
            try:
                from dateutil import parser
                parsed_date = parser.parse(date)
                return parsed_date.strftime("%Y%m%d0000")
            except:
                # 最终fallback：当前日期
                return datetime.now().strftime("%Y%m%d0000")
        
        # 默认返回当前日期
        return datetime.now().strftime("%Y%m%d0000")

# 示例用法
if __name__ == "__main__":
    data = PaperSearcher()._format_date("2023")
    print(data)