"""업종별(시기·업종 집중) 분석 CSV → 대시보드용 JSON.

값은 팀원 CSV를 그대로 옮기고, 새로 계산하는 것은
(1) 업종 비중(06 파일 금액 합 ÷ 지역 연간 합, 명세서 V05 방식)
(2) 기존 구조 진단 5곳(dashboard.py 원본과 같은 규칙) 뿐이다.
"""
import json, re, sys
from pathlib import Path
import pandas as pd

APP = Path(sys.argv[1])
OUT = Path(sys.argv[2])
B = APP / "제주_관광소비_업종별분석" / "자료"
rd = lambda p: pd.read_csv(p, encoding="utf-8-sig")

F = rd(B / "01_상세분석/05_43지역_통합특성.csv")
M = rd(B / "01_상세분석/03_43지역_월별집단소비거래.csv")
C = rd(B / "01_상세분석/06_월업종소비거래.csv")
HL = rd(B / "01_상세분석/01_43지역_상하위월비교.csv")
ST = rd(B / "01_상세분석/07_상점거래참고.csv")
SEN = rd(B / "02_추가검증/01_민감도_요약.csv")
JC = rd(B / "02_추가검증/02_제주전체및업종구성보정.csv")
TH = rd(B / "판정경계값.csv")

PER = {1: "2023.08~2024.07", 2: "2024.08~2025.07", 3: "2025.08~2026.07"}
INDS = ["음식점업", "소매업", "숙박업", "예술스포츠여가업", "기타서비스업"]
r1 = lambda x, n=1: None if pd.isna(x) else round(float(x), n)

# ---- 지역·구간 특성 (V01~V03)
feat = {}
for _, r in F.iterrows():
    feat.setdefault(r["읍면동명"], {})[int(r["구간"])] = {
        "ty": r["종합유형"], "mty": r["월유형"], "ity": r["업종유형"],
        "top3": r1(r["상위3개월비중_pct"]), "hhi": r1(r["업종HHI"], 3),
        "maxInd": r["최대소비업종"], "maxIndPct": r1(r["최대업종비중_pct"]),
        "topM": str(r["상위3개월"]), "lowM": str(r["저조3개월"]),
        "tourEok": r1(r["관광객소비_억"]), "totEok": r1(r["전체소비_억"]),
        "tourShare": r1(r["관광객소비비중_pct"]), "resShare": r1(r["도민소비비중_pct"]),
        "season": r["최대소비계절"], "seasonPct": r1(r["최대계절비중_pct"]),
    }

# ---- 월별 관광객·도민 (V04, V08)
mon = {}
for (k, p), g in M.groupby(["읍면동명", "구간"]):
    g = g.sort_values("기준연월")
    months = sorted(g["기준연월"].astype(str).unique())
    t = g[g["집단"] == "관광객"].set_index(g[g["집단"] == "관광객"]["기준연월"].astype(str))
    d = g[g["집단"] == "도민"].set_index(g[g["집단"] == "도민"]["기준연월"].astype(str))
    mon.setdefault(k, {})[int(p)] = {
        "m": months,
        "t": [r1(t.loc[m, "소비금액(원)"] / 1e8, 2) if m in t.index else None for m in months],
        "r": [r1(d.loc[m, "소비금액(원)"] / 1e8, 2) if m in d.index else None for m in months],
        "tn": [int(t.loc[m, "이용건수"]) if m in t.index else None for m in months],
    }

# ---- 월×업종 관광객 소비 (V05, V06) : 관측행만, 빈 셀은 null
cat = {}
for (k, p), g in C.groupby(["읍면동명", "구간"]):
    months = mon[k][int(p)]["m"]
    tot = g["소비금액(원)"].sum()
    piv = g.pivot_table(index="업종명", columns=g["기준연월"].astype(str), values="소비금액(원)", aggfunc="sum")
    cells = {i: [float(piv.loc[i, m] / tot * 100) if (i in piv.index and m in piv.columns and pd.notna(piv.loc[i, m])) else None for m in months] for i in INDS}
    share = {i: r1(g.loc[g["업종명"] == i, "소비금액(원)"].sum() / tot * 100) for i in INDS}
    cat.setdefault(k, {})[int(p)] = {"share": share, "cell": cells}
# 업종 비중 43곳 중앙값 (V05 기준선)
cat_med = {}
for p in (1, 2, 3):
    cat_med[p] = {i: r1(pd.Series([cat[k][p]["share"][i] for k in cat if p in cat[k]]).median()) for i in INDS}

# ---- 상위월·저조월 (V07)
hl = {}
for _, r in HL.iterrows():
    hl.setdefault(r["읍면동명"], {})[int(r["구간"])] = {
        "top": str(r["상위3개월"]), "low": str(r["하위3개월"]),
        "spend": r1(100 + r["관광소비감소율_pct"]), "tx": r1(r["관광거래_하위상위비_pct"]),
        "per": r1(r["관광건당금액_하위상위비_pct"]), "res": r1(r["도민소비_하위상위비_pct"]),
        "sum": r1(100 + r["합산소비변화율_pct"]), "why": r["거래금액변동설명"],
    }

# ---- 상점·거래 구성 (V09)
st_ = {}
for (k, p), g in ST.groupby(["읍면동명", "구간"]):
    st_.setdefault(k, {})[int(p)] = [
        {"ind": r["업종"], "store": r1(r["상점비중_pct"]), "tx": r1(r["거래비중_pct"]), "n": int(r["상점수"])}
        for _, r in g.iterrows()
    ]

# ---- 민감도 (V10), 제주 비교 (V11)
sen = {}
for _, r in SEN.iterrows():
    sen.setdefault(r["읍면동명"], {})[int(r["구간"])] = int(r["선정조합수_9개중"])
jc = {}
for _, r in JC.iterrows():
    jc.setdefault(r["읍면동명"], {})[int(r["구간"])] = {
        "reg": r1(r["지역상위3월비중_pct"]), "rest": r1(r["나머지제주동일3월비중_pct"]),
        "adj": r1(r["업종구성보정_동일3월기대_pct"]), "main": r["주력업종"], "mainPct": r1(r["주력비중_pct"]),
        "restMain": r1(r["나머지제주주력업종비중_pct"]),
    }

th = {int(r["구간"]): {"m25": r1(r["월Q25"], 2), "m75": r1(r["월Q75"], 2), "h25": r1(r["HHI_Q25"], 3), "h75": r1(r["HHI_Q75"], 3)} for _, r in TH.iterrows()}

# ---- 기존 구조 진단 (dashboard.py 원본 legacy_tab 규칙: 도민 ≥75%, 소매 ≥45%, 총매출 ≤ 65백분위)
cons = rd(APP / "업종별분석" / "전체지역_소비유형_분류결과.csv")
conc = rd(APP / "업종별분석" / "지역별_편중도_전체결과.csv")
for c_ in ["지역총매출액", "매출순위", "도민매출비중", "관광객매출비중", "소매업매출비중"]:
    cons[c_] = pd.to_numeric(cons[c_], errors="coerce")
conc["매출기준_HHI"] = pd.to_numeric(conc["매출기준_HHI"], errors="coerce")
mg = cons.merge(conc[["읍면동명", "매출기준_HHI", "매출기준_최대업종"]], on="읍면동명", how="left")
cut = mg["지역총매출액"].quantile(0.65)
leg = mg[(mg["도민매출비중"] >= .75) & (mg["소매업매출비중"] >= .45) & (mg["지역총매출액"] <= cut)].sort_values("지역총매출액")
legacy = {
    "cut": r1(cut / 1e8), "rows": [
        {"k": r["읍면동명"], "sales": r1(r["지역총매출액"] / 1e8), "rank": int(r["매출순위"]),
         "res": r1(r["도민매출비중"] * 100), "retail": r1(r["소매업매출비중"] * 100),
         "hhi": r1(r["매출기준_HHI"], 3), "maxInd": r["매출기준_최대업종"]}
        for _, r in leg.iterrows()]
}

# ---- 종합진단 표 (05_종합진단과_원인가설.md 원문 그대로)
md = (APP / "제주_관광소비_업종별분석" / "05_종합진단과_원인가설.md").read_text(encoding="utf-8")
rows = []
for line in md.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 5 and cells[0] in ("구좌읍", "우도면", "화북동", "예래동"):
        rows.append(dict(zip(["k", "fact", "cause", "act", "kpi"], cells)))
overall = re.search(r"## 전체 진단\n(.+?)\n", md).group(1)
tail = md.strip().splitlines()[-1]

data = {"PER": PER, "INDS": INDS, "F": feat, "MON": mon, "CAT": cat, "CATMED": cat_med, "HL": hl,
        "ST": st_, "SEN": sen, "JC": jc, "TH": th, "LEG": legacy,
        "DX": {"rows": rows, "overall": overall, "tail": tail}}

# 검수 (명세서 5절)
cnt = {p: pd.Series([feat[k][p]["ty"] for k in feat]).value_counts().to_dict() for p in (1, 2, 3)}
assert cnt[1] == {"중간": 39, "집중형": 4}, cnt[1]
assert cnt[2] == {"중간": 38, "집중형": 5}, cnt[2]
assert cnt[3] == {"중간": 38, "집중형": 4, "분산형": 1}, cnt[3]
assert sen["예래동"][3] == 6
assert len(rows) == 4 and len(feat) == 43
OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("ok", cnt, "legacy", [r["k"] for r in legacy["rows"]], "bytes", OUT.stat().st_size)
