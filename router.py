# router.py — 关键词打分 + 优先级消歧
import re

# 场景优先级：分数相同时，排在前面的 scene 胜出
SCENE_PRIORITY = [
    "price_limit",
    "approval",
    "policy",
    "price_negotiation",
    "inventory",
    "activity",
    "product",
]

KEYWORD_MAP = {
    "price_limit": {
        "keywords": [
            "高价限流", "站外有同款", "站外同款", "比到的不是同款",
            "不是同款", "非同款", "低价同款", "限流", "比价", "比错",
            "比错了", "误判限流", "申诉解罚", "申诉",
        ],
        "patterns": [r"限.*流", r"比.*价", r"申.*诉", r"同.*款"],
    },
    "approval": {
        "keywords": [
            "审版不通过", "审版", "审办", "虚拟寄样", "复色",
            "底板移植", "审款", "审图", "寄样", "已审", "待审",
            "同款不同色", "免审", "套版",
        ],
        "patterns": [r"审.*版", r"审.*款", r"寄.*样"],
    },
    "policy": {
        "keywords": [
            "平台政策", "店铺政策", "平台规则", "全托管", "半托管",
            "开店", "资质", "罚款", "罚款规则", "考核", "违规",
            "极速起量", "水洗唛", "黑五", "大促政策", "政策",
            "账期", "结算", "规则",
        ],
        "patterns": [r"规.*则", r"政.*策"],
    },
    "price_negotiation": {
        "keywords": [
            "核价", "谈价", "涨价", "供货价", "供货价太低",
            "价格太低", "调价", "提价", "压价", "提一下",
            "往上谈", "赚不到钱", "合理区间",
        ],
        "patterns": [r"价.*[低高]", r"[高低].*价", r"涨.*价", r"压.*价"],
    },
    "inventory": {
        "keywords": [
            "备货单开白", "开白", "催仓库上架", "催仓库", "催上架",
            "质检不合格", "退供商品", "退供", "发起备货", "备货",
            "断货", "补货", "入库", "签收", "发货", "库存", "下个单",
        ],
        "patterns": [r"备.*货", r"发.*货", r"库.*存", r"下.*单"],
    },
    "activity": {
        "keywords": [
            "活动价格太低", "退出活动", "想退出", "报名途径",
            "报名入口", "报名链接", "活动报名", "找不到报名",
            "不想参加", "利润太低", "促销报名", "秒杀", "限时秒杀",
        ],
        "patterns": [r"报.*名", r"促.*销"],
    },
    "product": {
        "keywords": [
            "商品上架", "商品下架", "商品信息修改", "信息修改",
            "侵权", "商品图片更新", "图片更新", "商品链接",
            "上下架", "加站", "上新", "发布上线", "没在售",
        ],
        "patterns": [r"上.*架", r"下.*架", r"加.*站"],
    },
}

# 上下文加分：解决「黑五大促政策」vs activity 等典型冲突
CONTEXT_BOOST = [
    (lambda t: "政策" in t, "policy", 6),
    (lambda t: "限流" in t, "price_limit", 6),
    (lambda t: ("申诉" in t or "比价" in t) and "限流" in t, "price_limit", 4),
    (lambda t: "催仓库" in t, "inventory", 6),
    (lambda t: "供货价" in t, "price_negotiation", 5),
    (lambda t: "审款" in t or "审版" in t, "approval", 5),
    (lambda t: "复色" in t or "同款不同色" in t, "approval", 5),
    (lambda t: "站外" in t and "同款" in t, "price_limit", 5),
]


def _score_scene(text: str) -> dict[str, int]:
    scores = {scene: 0 for scene in KEYWORD_MAP}
    for scene, config in KEYWORD_MAP.items():
        for kw in sorted(config["keywords"], key=len, reverse=True):
            if kw in text:
                scores[scene] += len(kw)
        for pattern in config.get("patterns", []):
            if re.search(pattern, text):
                scores[scene] += 3
    for cond, scene, bonus in CONTEXT_BOOST:
        if cond(text):
            scores[scene] += bonus
    return scores


def route_query(user_input: str) -> str:
    """关键词打分 + 优先级消歧；无命中则 general。"""
    text = user_input.lower().strip()
    scores = _score_scene(text)
    max_score = max(scores.values())

    if max_score == 0:
        question_words = ["哪些", "介绍", "什么", "怎样", "如何"]
        business_terms = ["平台", "规则", "费用", "扣点", "账期", "结算"]
        if any(w in text for w in question_words) and any(
            t in text for t in business_terms
        ):
            return "policy"
        return "general"

    candidates = [s for s, sc in scores.items() if sc == max_score]
    if len(candidates) == 1:
        return candidates[0]

    for scene in SCENE_PRIORITY:
        if scene in candidates:
            return scene
    return candidates[0]


def detect_subtask(scene: str, user_input: str) -> str:
    text = user_input.lower()

    if scene == "inventory":
        if "开白" in text or "白名单" in text:
            return "备货单开白"
        if any(k in text for k in ["质检不合格", "拦截", "不合格"]):
            return "质检不合格拦截"
        if "退供" in text:
            return "退供商品"
        if any(k in text for k in ["催仓库", "催上架", "仓库"]):
            return "催仓库上架"
        if any(k in text for k in ["备货", "补货", "补单", "下单", "下个单", "断货"]):
            return "发起备货"
        if any(k in text for k in ["发货", "库存", "入库", "签收", "没货"]):
            return "库存发货咨询"

    if scene == "product":
        if "侵权" in text:
            return "商品侵权问题查询"
        if any(k in text for k in ["图片更新", "换图", "主图"]):
            return "商品图片更新"
        if any(k in text for k in ["信息修改", "改信息", "标题", "属性"]):
            return "商品信息修改"
        if "下架" in text:
            return "商品下架"
        if any(k in text for k in ["上架", "加站", "上新", "发布", "没在售"]):
            return "商品上架"

    if scene == "activity":
        if any(
            k in text
            for k in [
                "退出",
                "想退出",
                "不想参加",
                "利润太低",
                "活动价格太低",
                "报错了",
                "赚不到",
            ]
        ):
            return "活动价格太低需退出活动"
        if any(
            k in text
            for k in [
                "报名",
                "报名入口",
                "报名链接",
                "怎么报名",
                "在哪报名",
                "促销报名",
                "秒杀",
            ]
        ):
            return "找不到报名途径"

    if scene == "price_limit":
        if any(
            k in text
            for k in ["不是同款", "非同款", "误判", "比错", "比错了", "比错了"]
        ):
            return "比到的不是同款，需申诉解罚"
        return "站外有同款，价格太低，需申诉"

    if scene == "approval":
        if any(
            k in text
            for k in ["虚拟寄样", "复色", "同款不同色", "已审", "待审", "套版"]
        ):
            return "虚拟寄样（复色）流程"
        if "底板移植" in text:
            return "底板移植流程"
        if any(k in text for k in ["审版不通过", "审核不过", "被拒"]):
            return "审版不通过，查询原因"
        return "审版相关咨询"

    if scene == "price_negotiation":
        if any(
            k in text
            for k in ["涨价", "提价", "想涨", "能不能涨", "提一下", "往上谈"]
        ):
            return "商品供货价太低，需涨价"
        return "商品供货价太低，需谈价"

    if scene == "policy":
        if any(k in text for k in ["极速起量", "起量"]):
            return "咨询极速起量政策"
        if any(k in text for k in ["水洗唛", "洗标", "成分标"]):
            return "咨询水洗唛规范"
        if any(k in text for k in ["黑五", "黑色星期五", "大促政策", "大促"]):
            return "咨询大促政策"
        return "咨询现有平台政策"

    return "未细分"


if __name__ == "__main__":
    test_cases = [
        "极速起量是什么",
        "spu没在售，帮我加站一下",
        "水洗唛有什么要求",
        "平台罚款规则是怎样的",
        "黑五大促政策怎么准备",
        "供货价能不能提一下",
        "催仓库快点上架",
        "高价限流不是同款帮我申诉",
        "同款不同色怎么复色",
        "审款一直没结果",
    ]
    for q in test_cases:
        scene = route_query(q)
        sub = detect_subtask(scene, q)
        print(f"{q[:22]:<22} -> {scene:<18} | {sub}")
