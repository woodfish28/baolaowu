#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb_search.py — 离线知识快照检索（IMA 不可用或未配置时的兜底）
============================================================

用途
----
本技能的正路是检索 IMA 知识库。但接收方在下面两种情况下检索不到：
  1. 还没连接 IMA 连接器 / 还没加入共享知识库；
  2. IMA 接口临时不可用。

这时用本脚本在技能内置的离线快照里做关键词检索，让「老余观点」这一层
不至于完全空掉。没有它，模型就只能凭记忆编，那是这个技能最不能接受的事。

数据源（data/ 目录，均为 2026-08-16 的快照，会过期，回答时需留意时效）
------
  wx_articles.json           保来屋经纪公众号 567 篇（title + digest + url）
  reference_baozhitang.json  保知堂 10 篇（title + content 全文）
  reference_industry.json    保险业干货 535 篇（title + content 全文）

用法
----
  python kb_search.py "甲状腺结节 买什么医疗险"
  python kb_search.py "重疾险 误区" --kb baolaowu --top 5
  python kb_search.py "护工卡" --full          # 输出完整正文而非截断摘要
  python kb_search.py "重疾险保额" --json      # 机器可读输出

参数
----
  query            检索词，必填。支持空格分词，无需精确匹配。
  --kb             限定库：baolaowu / baozhitang / industry / all（默认 all）
  --top N          返回条数，默认 5，上限 20
  --full           输出完整正文（可能很长，慎用）
  --json           以 JSON 输出
  --snap-date      打印快照日期后退出（用于回答时标注时效）

依赖：仅 Python 标准库，无需 pip install。Python 3.7+。
"""

import argparse
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

SNAPSHOT_DATE = "2026-08-16"

# 库名 -> (文件名, 正文键, 权重, 展示名, 引用口径)
KBS = {
    "baolaowu": (
        "wx_articles.json",
        "digest",
        3,
        "保来屋经纪（老余观点）",
        "保来屋经纪观点",
    ),
    "baozhitang": (
        "reference_baozhitang.json",
        "content",
        2,
        "保知堂（内部参考）",
        "内部参考，不向用户展示原文链接",
    ),
    "industry": (
        "reference_industry.json",
        "content",
        1,
        "保险业干货（行业背景）",
        "行业背景，不作投保结论依据",
    ),
}

PUNCT = re.compile(r"[\s，。、；：！？（）【】《》\"'“”‘’\-—_/\\|·•,.;:!?()\[\]{}<>#*~`+=]+")

# 单字虚词：几乎出现在每篇里，无区分度
STOP_CHARS = set("的了是在和与及对为以将于把被而或之其这那我你他它们个也还就都又很")
# 疑问词与句式助词。必须过滤，否则「怎么买」这类表述会压过真正的主题词：
# bigram 切分会把「怎么买」切成「怎么 / 么买」，其中「么买」df 极低、idf 极高，
# 一次实测里它的权重(5.14)甚至超过了「宝宝」(4.94)，直接把车险文推到宝宝保险前面。
STOP_WORDS = {
    "怎么", "怎样", "咋么", "么买", "么办", "么选", "么配", "么投", "么算",
    "如何", "什么", "哪些", "哪个", "那种", "这种",
    "可以", "能不能", "要不要", "是不是", "有没有", "会不会",
    "多少", "请问", "一下", "需要", "应该", "是否", "能否",
}

# 命中分阈值。低于此值视为噪声——中文 bigram 很容易在长文里偶然撞上
# 一两个片段，没有阈值的话「随便打几个字」也会返回一堆不相干结果。
MIN_SCORE = 1.2
# 查询词覆盖率下限：命中了多少比例的 query token。只靠分数不够，
# 长文能靠堆偶然命中把分数刷上去，覆盖率能挡住这类误召回。
MIN_COVERAGE = 0.30

TAG = re.compile(r"<[^>]{1,200}>")
WS = re.compile(r"[ \t\u00a0]{2,}")
NL = re.compile(r"\n{3,}")
# 快照里存的是「字面转义序列」而非真实字节：\\x0a 是 4 个字符，不是换行
ESC = re.compile(r"\\x([0-9a-fA-F]{2})|\\u([0-9a-fA-F]{4})")


def _unesc(m):
    if m.group(1):
        return chr(int(m.group(1), 16))
    return chr(int(m.group(2), 16))


def clean(text):
    """快照里的正文混着双重 HTML 转义、字面 \\x0a 与公众号残留标签，统一洗成纯文本。"""
    if not text:
        return ""
    text = ESC.sub(_unesc, text)
    for _ in range(2):          # \x26lt; -> &lt; -> <
        new = html.unescape(text)
        if new == text:
            break
        text = new
    text = TAG.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WS.sub(" ", text)
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    text = NL.sub("\n\n", text)
    return text.strip()


def tokenize(text):
    """中文按 bigram + 单字，英文数字按词。够用且不引入依赖。"""
    text = (text or "").lower()
    tokens = []
    for seg in PUNCT.split(text):
        seg = seg.strip()
        if not seg:
            continue
        if re.fullmatch(r"[0-9a-z]+", seg):
            tokens.append(seg)
            continue
        for i, ch in enumerate(seg):
            if ch in STOP_CHARS:
                continue
            tokens.append(ch)
            if i + 1 < len(seg):
                bg = seg[i : i + 2]
                if bg not in STOP_WORDS and seg[i + 1] not in STOP_CHARS:
                    tokens.append(bg)
    return [t for t in tokens if t not in STOP_WORDS]


def load(kb_key):
    fname, body_key, weight, display, cite = KBS[kb_key]
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for it in raw:
        title = clean(it.get("title"))
        body = clean(it.get(body_key))
        if not title and not body:
            continue
        out.append(
            {
                "kb": kb_key,
                "kb_display": display,
                "cite": cite,
                "weight": weight,
                "title": title,
                "body": body,
                "url": (it.get("url") or "").strip(),
                "_hay": (title + "\n" + body).lower(),
            }
        )
    return out


def build_idf(pool, uniq_tokens):
    """词频倒排权重：'保险''怎么买' 这类词几乎篇篇都有，区分度接近 0；
    '宝宝''护工卡' 才是真正定位内容的词。不给它们加权，泛词查询就会跑偏。
    只对 query 涉及的词统计，避免全文分词带来的开销。
    """
    n = len(pool)
    idf = {}
    for t in uniq_tokens:
        df = sum(1 for it in pool if t in it["_hay"])
        idf[t] = __import__("math").log((n + 1) / (df + 1)) + 1.0
    return idf


def score(item, q_tokens, idf):
    """IDF 加权 + 标题命中加成；长度归一化避免长文刷分；覆盖率低判为噪声。

    返回 (最终分, 覆盖率)。
    """
    hay = item["_hay"]
    if not hay:
        return 0.0, 0.0
    title_l = item["title"].lower()
    uniq = set(q_tokens)
    hit = 0.0
    hit_w = 0.0
    total_w = 0.0
    for t in uniq:
        w = idf.get(t, 1.0)
        # 单字 token（「保」「险」「买」）几乎命中一切，噪声极大，重罚；
        # bigram（「宝宝」「护工」）才是真正定位内容的信号。
        if len(t) == 1:
            w *= 0.3
        total_w += w
        c = hay.count(t)
        if not c:
            continue
        hit_w += w
        hit += (1.0 + 0.35 * min(c, 6)) * w     # 重复出现有边际回报
        if t in title_l:
            hit += 2.0 * w                       # 标题命中加成
    if hit <= 0:
        return 0.0, 0.0
    # 覆盖率按 IDF 加权：命中「保险」「怎么买」不算数，没命中「宝宝」才是硬伤
    coverage = hit_w / total_w if total_w else 0.0
    # 长度归一：短而准的条目不该被长文压过
    norm = hit / (1.0 + 0.02 * (len(hay) ** 0.5))
    return norm * item["weight"] * coverage, coverage


def search(query, kb="all", top=5):
    q_tokens = [t for t in tokenize(query) if len(t) >= 1]
    if not q_tokens:
        return []
    keys = list(KBS.keys()) if kb == "all" else [kb]
    pool = []
    for k in keys:
        pool.extend(load(k))
    if not pool:
        return []
    idf = build_idf(pool, set(q_tokens))
    scored = []
    for it in pool:
        s, cov = score(it, q_tokens, idf)
        if s >= MIN_SCORE and cov >= MIN_COVERAGE:
            scored.append((s, it))
    scored.sort(key=lambda x: (-x[0], x[1]["title"]))
    return [it for _, it in scored[:top]]


def render(items, full=False):
    if not items:
        return "未命中任何离线条目。"
    lines = []
    for i, it in enumerate(items, 1):
        lines.append("### %d. %s" % (i, it["title"] or "(无题)"))
        lines.append("- 来源：%s" % it["kb_display"])
        lines.append("- 引用口径：%s" % it["cite"])
        if it["url"] and it["kb"] != "baozhitang":
            lines.append("- 链接：%s" % it["url"])
        body = it["body"]
        if not full and len(body) > 600:
            body = body[:600].rstrip() + " …（截断，加 --full 看全文）"
        if body:
            lines.append("")
            lines.append(body)
        lines.append("")
    lines.append("---")
    lines.append("> 离线快照日期：%s。内容可能已过时，涉及费率/条款/在售状态时必须以官方当前信息为准。" % SNAPSHOT_DATE)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="老余数字人 · 离线知识快照检索")
    ap.add_argument("query", nargs="?", help="检索词")
    ap.add_argument("--kb", default="all", choices=["all"] + list(KBS.keys()))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--full", action="store_true", help="输出完整正文")
    ap.add_argument("--json", action="store_true", help="JSON 输出")
    ap.add_argument("--snap-date", action="store_true", help="打印快照日期后退出")
    args = ap.parse_args()

    if args.snap_date:
        print(SNAPSHOT_DATE)
        return
    if not args.query:
        ap.error("缺少检索词")

    args.top = max(1, min(args.top, 20))
    items = search(args.query, args.kb, args.top)

    if args.json:
        print(json.dumps([{k: v for k, v in it.items() if k != "_hay"} for it in items],
                         ensure_ascii=False, indent=2))
    else:
        print("## 离线检索：%s（库=%s，top=%d）\n" % (args.query, args.kb, args.top))
        print(render(items, args.full))

    # 退出码：0 有结果，1 无结果（便于智能体判断是否要走联网兜底）
    sys.exit(0 if items else 1)


if __name__ == "__main__":
    main()
