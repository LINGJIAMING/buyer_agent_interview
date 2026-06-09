# retriever.py Hybrid Search版 - BM25 + 向量语义搜索
import os
import re
import json
import warnings
from typing import List, Dict, Tuple
from difflib import SequenceMatcher

# 处理可选依赖
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    warnings.warn("jieba not installed, using character-level tokenization")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    warnings.warn("rank-bm25 not installed, using fallback scoring")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    warnings.warn("sentence-transformers not installed, semantic search disabled")

# 映射 Router 返回的英文 key 到知识库中的中文场景名
SCENE_ZH_MAP = {
    "activity": "活动",
    "price_negotiation": "核价",
    "inventory": "库存",
    "product": "商品",
    "price_limit": "限流",
    "approval": "审版",
    "policy": "政策",
    "general": "通用"  # 这里的中文与.md 文件里的 **场景** 内容一致
}


class Retriever:
    # 可配置检索参数（可按数据效果微调）
    DEFAULT_BM25_WEIGHT = 0.4
    DEFAULT_SEMANTIC_WEIGHT = 0.6
    DEFAULT_SCENE_BONUS = 10.0
    DEFAULT_MIN_RELEVANCE_SCORE = 5.0
    DEFAULT_STRONG_HIT_SCORE = 50.0
    DEFAULT_LOW_CONFIDENCE_SCORE = 30.0
    DEFAULT_LOW_CONFIDENCE_GAP = 8.0

    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.docs = self._load_structured_kb()
        
        # 初始化BM25索引和向量模型
        self.bm25_index = None
        self.embedding_model = None
        self.doc_embeddings = None
        
        if self.docs:
            self._init_hybrid_search()
    
    def _init_hybrid_search(self):
        """初始化混合搜索的两个引擎"""
        # 初始化BM25索引
        if BM25_AVAILABLE:
            try:
                # 对文档进行分词
                tokenized_docs = []
                for doc in self.docs:
                    # 合并标题+内容+关键词进行分词
                    text = f"{doc.get('title', '')} {doc.get('content', '')} {' '.join(doc.get('keywords', []))}"
                    tokens = self._tokenize(text)
                    tokenized_docs.append(tokens)
                
                self.bm25_index = BM25Okapi(tokenized_docs)
                print(f"[SEARCH] BM25索引初始化成功，包含 {len(self.docs)} 个文档")
            except Exception as e:
                print(f"[WARN] BM25初始化失败: {e}")
        
        # 初始化向量embedding模型（如果可用）
        if EMBEDDING_AVAILABLE:
            try:
                # 支持环境变量指定模型，未指定时按候选列表依次尝试
                # 注意：此前默认模型名拼写有误，这里修正为官方模型名
                env_model = os.getenv("EMBEDDING_MODEL_NAME", "").strip()
                candidate_models = [env_model] if env_model else [
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "moka-ai/m3e-base",
                    "BAAI/bge-small-zh-v1.5",
                ]

                print("[SEARCH] 加载embedding模型（首次加载会下载，耗时取决于网络）...")
                self.embedding_model = None
                last_error = None
                for model_name in candidate_models:
                    if not model_name:
                        continue
                    try:
                        print(f"[SEARCH] 尝试模型: {model_name}")
                        self.embedding_model = SentenceTransformer(model_name)
                        print(f"[SEARCH] Embedding模型加载成功: {model_name}")
                        break
                    except Exception as e:
                        last_error = e
                        print(f"[WARN] 模型加载失败: {model_name} | {e}")

                if self.embedding_model is None:
                    raise RuntimeError(f"所有embedding模型加载失败: {last_error}")
                
                # 预先计算所有文档的embedding
                corpus_texts = [
                    f"{doc.get('title', '')} {doc.get('content', '')} {' '.join(doc.get('keywords', []))}"
                    for doc in self.docs
                ]
                self.doc_embeddings = self.embedding_model.encode(corpus_texts, convert_to_numpy=True)
                print(f"[SEARCH] 文档向量化完成，共 {len(self.docs)} 个文档")
            except Exception as e:
                print(f"[WARN] Embedding模型初始化失败: {e}")
                print("[INFO] 将使用BM25纯关键词搜索，不影响检索功能")
                self.embedding_model = None
                self.doc_embeddings = None
    
    def _tokenize(self, text: str) -> List[str]:
        """分词函数：优先使用jieba，否则降级为字符级"""
        if not text:
            return []
        
        text = text.lower().strip()
        
        if JIEBA_AVAILABLE:
            # 使用jieba分词（更精确）
            tokens = list(jieba.cut(text))
            # 过滤掉太短的词和纯空白
            tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 0]
            return tokens
        else:
            # 降级方案：字符级分词 + 2-4字滑动窗口
            tokens = []
            for i in range(len(text)):
                tokens.append(text[i])
                if i < len(text) - 1:
                    tokens.append(text[i:i+2])
                if i < len(text) - 2:
                    tokens.append(text[i:i+3])
            return list(dict.fromkeys(tokens))  # 去重

    def _load_structured_kb(self) -> List[Dict]:
        """加载结构化知识库，支持两种格式：
        1. JSONL: {"title": "...", "keywords": ["..."], "content": "..."}
        2. Markdown: ## 标题\n**关键词**: xxx\n**内容**: xxx\n**动作**: xxx\n**场景**: xxx
        """
        if not os.path.exists(self.kb_path):
            print(f"[WARN] KB not found: {self.kb_path}")
            return []

        with open(self.kb_path, "r", encoding="utf-8") as f:
            text = f.read()

        docs = []

        # 格式1: JSONL
        if self.kb_path.endswith('.jsonl'):
            for line in text.strip().split('\n'):
                if line.strip():
                    docs.append(json.loads(line))
            return docs

        # 格式2: Markdown 分隔
        if self.kb_path.endswith('.md'):
            raw_sections = re.split(r'\n## |\n\[政策标题\]', text)

            for section in raw_sections:
                if not section.strip():
                    continue

                doc = self._parse_section(section)
                if doc:
                    docs.append(doc)

            print(f"[KB] Loaded {len(docs)} policy documents")
            return docs

        return docs

    def _parse_section(self, text: str) -> Dict:
        """解析 Markdown 格式的政策段落"""
        lines = text.strip().split('\n')
        if not lines:
            return None

        doc = {
            "title": "",
            "keywords": [],
            "content": "",
            "actions": [],
            "scene": "",
            "links": []
        }

        # 第一行是标题（去除 ## 前缀）
        first_line = lines[0].strip()
        if first_line.startswith('## '):
            doc["title"] = first_line[3:].strip()
        elif first_line.startswith('##'):
            doc["title"] = first_line[2:].strip()
        else:
            doc["title"] = first_line

        current_field = None
        content_lines = []

        for line in lines[1:]:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # 检测字段标记
            if line_stripped.startswith('**关键词**') or line_stripped.startswith('【关键词】'):
                current_field = "keywords"
                content_part = re.sub(r'^\*\*关键词\*\*[:：]\s*|\【关键词】[:：]\s*', '', line_stripped)
                if content_part:
                    doc["keywords"] = [k.strip() for k in re.split(r'[,，、]', content_part) if k.strip()]

            elif line_stripped.startswith('**内容**') or line_stripped.startswith('【内容】'):
                current_field = "content"
                content_part = re.sub(r'^\*\*内容\*\*[:：]\s*|\【内容】[:：]\s*', '', line_stripped)
                if content_part:
                    content_lines = [content_part]
                else:
                    content_lines = []

            elif line_stripped.startswith('**动作**') or line_stripped.startswith('【动作】'):
                current_field = "actions"
                content_part = re.sub(r'^\*\*动作\*\*[:：]\s*|\【动作】[:：]\s*', '', line_stripped)
                if content_part:
                    doc["actions"] = [a.strip() for a in re.split(r'[,，、]', content_part) if a.strip()]
                else:
                    doc["actions"] = []

            elif line_stripped.startswith('**场景**') or line_stripped.startswith('【场景】'):
                current_field = "scene"
                content_part = re.sub(r'^\*\*场景\*\*[:：]\s*|\【场景】[:：]\s*', '', line_stripped)
                doc["scene"] = content_part.strip() if content_part else ""

            else:
                # 继续上一字段的内容
                if current_field == "content":
                    content_lines.append(line_stripped)
                elif current_field == "keywords" and '：' not in line_stripped and ':' not in line_stripped:
                    extra_keywords = [k.strip() for k in re.split(r'[,，、]', line_stripped) if k.strip()]
                    doc["keywords"].extend(extra_keywords)
                elif current_field == "actions" and '：' not in line_stripped and ':' not in line_stripped:
                    extra_actions = [a.strip() for a in re.split(r'[,，、]', line_stripped) if a.strip()]
                    doc["actions"].extend(extra_actions)

        # 合并内容
        if content_lines:
            doc["content"] = '\n'.join(content_lines).strip()

        # 自动提取正文中的链接
        if doc["content"]:
            extracted_links = re.findall(r'https?://[^\s，,；;]+', doc["content"])
            doc["links"] = list(dict.fromkeys(extracted_links))

            cleaned_content = doc["content"]
            for link in doc["links"]:
                cleaned_content = cleaned_content.replace(link, " ")

            cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
            cleaned_content = re.sub(r'\s+([，。；：])', r'\1', cleaned_content)
            doc["content"] = cleaned_content

        # 去重并保序
        doc["keywords"] = list(dict.fromkeys([k for k in doc["keywords"] if k]))
        doc["actions"] = list(dict.fromkeys([a for a in doc["actions"] if a]))
        doc["links"] = list(dict.fromkeys([u for u in doc["links"] if u]))

        return doc if (doc["title"] or doc["content"]) else None

    def _normalize_text(self, text: str) -> str:
        """做最轻量的归一化，方便标题精确匹配"""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'[【】\[\]（）()：:，,。！？!?\s]+', '', text)
        return text

    def _fuzzy_match(self, s1: str, s2: str) -> float:
        """模糊匹配分数"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def _get_bm25_scores(self, query: str) -> List[float]:
        """计算BM25相关度分数（0-100）"""
        if not self.bm25_index:
            return [0.0] * len(self.docs)
        
        tokens = self._tokenize(query)
        if not tokens:
            return [0.0] * len(self.docs)
        
        bm25_scores = self.bm25_index.get_scores(tokens)
        # 归一化到0-100
        max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
        normalized = [s / max_score * 100 for s in bm25_scores]
        return normalized
    
    def _get_semantic_scores(self, query: str) -> List[float]:
        """计算向量语义相似度（0-100）"""
        if not self.embedding_model or self.doc_embeddings is None:
            return [0.0] * len(self.docs)
        
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            
            # 计算余弦相似度（使用numpy而不是sklearn）
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return [0.0] * len(self.docs)
            
            similarities = []
            for doc_embedding in self.doc_embeddings:
                doc_norm = np.linalg.norm(doc_embedding)
                if doc_norm == 0:
                    similarities.append(0.0)
                else:
                    # 余弦相似度 = (A·B) / (||A|| * ||B||)
                    sim = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
                    similarities.append(float(sim))
            
            # 转换为0-100，处理负值
            scores = [max(0, float(sim)) * 100 for sim in similarities]
            return scores
        except Exception as e:
            print(f"[WARN] 向量计算失败: {e}")
            return [0.0] * len(self.docs)
    
    def _hybrid_score(
        self,
        bm25_scores: List[float],
        semantic_scores: List[float],
        bm25_weight: float = None,
        semantic_weight: float = None
    ) -> List[float]:
        """融合BM25和语义分数"""
        bm25_weight = self.DEFAULT_BM25_WEIGHT if bm25_weight is None else bm25_weight
        semantic_weight = self.DEFAULT_SEMANTIC_WEIGHT if semantic_weight is None else semantic_weight
        hybrid_scores = []
        for bm25, semantic in zip(bm25_scores, semantic_scores):
            # 加权融合：BM25(40%) + Semantic(60%)
            # 可根据实际效果调整权重
            score = bm25 * bm25_weight + semantic * semantic_weight
            hybrid_scores.append(score)
        return hybrid_scores
    
    def retrieve_context(
        self,
        user_input: str,
        scene: str = "",
        top_k: int = 1,
        use_hybrid: bool = True
    ) -> Dict:
        """改进版检索：支持Hybrid Search（BM25 + 向量语义搜索）
        
        Args:
            user_input: 用户输入
            scene: 场景标签（会进行映射和权重加成）
            top_k: 返回top-k个文档（当前仅支持top_k=1）
            use_hybrid: 是否使用混合搜索，True=hybrid, False=降级到原始方案
        
        Returns:
            Dict: {
                "context": str, "score": float, "title": str, "title_exact_match": bool,
                "strong_hit": bool, "retrieval_method": str, "candidates": List[Dict]
            }
        """
        try:
            top_k = max(1, int(top_k))
        except (TypeError, ValueError):
            top_k = 1

        if not self.docs:
            return {
                "context": "未检索到明确相关的政策资料。（知识库为空）",
                "score": 0.0,
                "title": "",
                "title_exact_match": False,
                "strong_hit": False,
                "retrieval_method": "empty",
                "candidates": [],
                "low_confidence": True,
                "follow_up_question": "我先帮你确认一下：你想处理的是哪个商品/SPU，以及你现在卡在哪一步？"
            }

        query = user_input.lower().strip()
        scene_zh = SCENE_ZH_MAP.get(scene, scene) if scene else ""
        
        print(f"[RETRIEVE] Query: '{query}' | Scene: '{scene}' -> '{scene_zh}'")
        print(f"[RETRIEVE] Using Hybrid Search: {use_hybrid and (BM25_AVAILABLE or EMBEDDING_AVAILABLE)}")
        
        # ===== 混合搜索版本 =====
        if use_hybrid and (BM25_AVAILABLE or EMBEDDING_AVAILABLE):
            # 获取BM25分数
            bm25_scores = self._get_bm25_scores(query) if BM25_AVAILABLE else [0.0] * len(self.docs)
            
            # 获取语义分数
            semantic_scores = self._get_semantic_scores(query) if EMBEDDING_AVAILABLE else [0.0] * len(self.docs)
            
            # 融合分数
            hybrid_scores = self._hybrid_score(bm25_scores, semantic_scores)
            
            # 加入场景匹配权重加成
            final_scores = []
            for i, score in enumerate(hybrid_scores):
                doc_scene = self.docs[i].get("scene", "").lower()
                # 如果匹配场景，加10分
                scene_bonus = self.DEFAULT_SCENE_BONUS if (scene_zh and scene_zh in doc_scene) else 0
                final_score = score + scene_bonus
                final_scores.append(final_score)

            # 获取top-k文档
            ranked_indices = sorted(
                range(len(final_scores)),
                key=lambda idx: final_scores[idx],
                reverse=True
            )[:top_k]
            max_score_idx = ranked_indices[0]
            top_score = final_scores[max_score_idx]
            top_doc = self.docs[max_score_idx]
            candidates = [
                {
                    "title": self.docs[idx].get("title", ""),
                    "score": float(final_scores[idx]),
                    "scene": self.docs[idx].get("scene", "")
                }
                for idx in ranked_indices
            ]
            
            print(f"[RETRIEVE] Scores - BM25: {bm25_scores[max_score_idx]:.1f}, "
                  f"Semantic: {semantic_scores[max_score_idx]:.1f}, "
                  f"Hybrid: {hybrid_scores[max_score_idx]:.1f}, "
                  f"Final: {top_score:.1f}")
            retrieval_method = "hybrid"
            
        # ===== 降级：原始字符级匹配 =====
        else:
            print("[RETRIEVE] 未启用Hybrid或库不可用，使用原始匹配方案")
            
            query_keywords = set()
            for length in [4, 2]:
                for i in range(len(query) - length + 1):
                    query_keywords.add(query[i:i + length])
            
            scored_docs = []
            for doc in self.docs:
                score = 0
                title = doc.get("title", "").lower()
                doc_scene = doc.get("scene", "").lower()
                doc_keywords = [k.lower() for k in doc.get("keywords", [])]
                content = doc.get("content", "").lower()
                
                if any(kw in title for kw in query_keywords):
                    score += 10
                if scene_zh and scene_zh in doc_scene:
                    score += 6
                for qk in query_keywords:
                    for dk in doc_keywords:
                        if qk in dk or dk in qk:
                            score += 2
                for qk in query_keywords:
                    if qk in content:
                        score += 1
                
                if score > 0:
                    scored_docs.append((score, doc))
            
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            if not scored_docs or scored_docs[0][0] < 1.0:
                return {
                    "context": "未检索到明确相关的政策资料。",
                    "score": 0.0,
                    "title": "",
                    "title_exact_match": False,
                    "strong_hit": False,
                    "retrieval_method": "fallback",
                    "candidates": [],
                    "low_confidence": True,
                    "follow_up_question": "我先确认一下你的具体场景：是政策咨询、审版、核价、库存还是活动问题？"
                }

            top_candidates = scored_docs[:top_k]
            top_score, top_doc = top_candidates[0]
            candidates = [
                {
                    "title": doc.get("title", ""),
                    "score": float(score),
                    "scene": doc.get("scene", "")
                }
                for score, doc in top_candidates
            ]
            retrieval_method = "fallback"
        
        # ===== 通用的结果后处理 =====
        if top_score < self.DEFAULT_MIN_RELEVANCE_SCORE:  # 分数太低，认为无关
            return {
                "context": "未检索到明确相关的政策资料。",
                "score": float(top_score),
                "title": top_doc.get("title", ""),
                "title_exact_match": False,
                "strong_hit": False,
                "retrieval_method": retrieval_method,
                "candidates": candidates,
                "low_confidence": True,
                "follow_up_question": "我先确认一下你的目标：你是想了解规则，还是希望我帮你推进处理？"
            }
        
        # 判断是否强命中
        normalized_query = self._normalize_text(user_input)
        normalized_title = self._normalize_text(top_doc.get("title", ""))
        
        title_exact_match = (
            normalized_query == normalized_title
            or normalized_title in normalized_query
            or normalized_query in normalized_title
        )
        
        strong_hit = title_exact_match or top_score >= self.DEFAULT_STRONG_HIT_SCORE
        second_score = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        score_gap = float(top_score) - second_score
        low_confidence = (
            (not strong_hit)
            and (
                float(top_score) < self.DEFAULT_LOW_CONFIDENCE_SCORE
                or score_gap < self.DEFAULT_LOW_CONFIDENCE_GAP
            )
        )

        follow_up_question = ""
        if low_confidence:
            follow_up_question = (
                "我先确认两个关键信息：你要处理的商品/SPU是哪个，"
                "以及你当前做到哪一步、报错或拦截提示是什么？"
            )
        
        # 根据强度返回不同长度的内容
        max_len = 800 if strong_hit else 400
        snippet = f"{top_doc['content'][:max_len]}{'...' if len(top_doc['content']) > max_len else ''}"
        
        if top_doc.get("actions"):
            snippet += "\n动作：" + "；".join(top_doc["actions"])
        
        if top_doc.get("links"):
            snippet += "\n链接：" + "；".join(top_doc["links"])
        
        context = (
            f"【{top_doc['title']}】\n"
            f"相关度: {top_score:.1f}\n"
            f"{snippet}"
        )
        
        return {
            "context": context,
            "score": float(top_score),
            "title": top_doc.get("title", ""),
            "title_exact_match": title_exact_match,
            "strong_hit": strong_hit,
            "retrieval_method": retrieval_method,
            "candidates": candidates,
            "low_confidence": low_confidence,
            "follow_up_question": follow_up_question
        }


# 兼容旧接口的简化函数（如果其他地方直接调用）
def load_policy_kb(path: str) -> str:
    """向后兼容：返回原始文本"""
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    retriever = Retriever("kb/policy_kb.md")
    print(f"[KB] Loaded {len(retriever.docs)} policy documents")

    test_cases = [
        ["极速起量是什么", 'policy'],
        ["水洗唛是什么", 'policy'],
        ["黑五活动什么时候开始", 'policy'],
        ["审版需要什么资料", 'approval'],
        ["核价流程是怎样的", 'price_negotiation'],
        ["如何加站", 'product'],
        ["skc是什么", 'product'],
        ["怎么下单", 'inventory'],
        ["怎么填写推广问卷", 'activity'],
    ]

    for query, scene in test_cases:
        print(f"\n{'=' * 50}")
        print(f"Query: '{query}' | Scene: '{scene}'")
        result = retriever.retrieve_context(query, scene=scene, top_k=1)
        print(result)