"""제주 관광소비 진단 대시보드 (v3: 첫 탭 결론 요약형).

실행:
    streamlit run dashboard_v3.py
"""

import json
import html as html_lib
import inspect
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).resolve().parent
CONSUMPTION_FILE = (
    BASE_DIR / "업종별분석" / "전체지역_소비유형_분류결과.csv"
)
CONCENTRATION_FILE = (
    BASE_DIR / "업종별분석" / "지역별_편중도_전체결과.csv"
)
VISITOR_FILE = BASE_DIR / "업종별분석" / "제주 지역별 차량 도착 수.csv"
GEOJSON_FILE = BASE_DIR / "hangjeongdong_제주특별자치도.geojson"
REGIONAL_DASHBOARD_FILE = BASE_DIR / "지역별분석" / "consumption_dashboard_v2.html"
INDUSTRY_DASHBOARD_FILE = BASE_DIR / "제주_관광소비_업종별분석" / "industry_dashboard.html"
EMBED_HEIGHT = 1900
INDUSTRY_ANALYSIS_DIR = BASE_DIR / "제주_관광소비_업종별분석"
INDUSTRY_FEATURES_FILE = (
    INDUSTRY_ANALYSIS_DIR / "자료" / "01_상세분석" / "05_43지역_통합특성.csv"
)
INDUSTRY_MONTHLY_FILE = (
    INDUSTRY_ANALYSIS_DIR / "자료" / "01_상세분석" / "03_43지역_월별집단소비거래.csv"
)
INDUSTRY_CATEGORY_MONTHLY_FILE = (
    INDUSTRY_ANALYSIS_DIR / "자료" / "01_상세분석" / "06_월업종소비거래.csv"
)
INDUSTRY_HIGH_LOW_FILE = (
    INDUSTRY_ANALYSIS_DIR / "자료" / "01_상세분석" / "01_43지역_상하위월비교.csv"
)
INDUSTRY_STORE_FILE = (
    INDUSTRY_ANALYSIS_DIR / "자료" / "01_상세분석" / "07_상점거래참고.csv"
)
INDUSTRY_SENSITIVITY_FILE = (
    INDUSTRY_ANALYSIS_DIR / "자료" / "02_추가검증" / "01_민감도_요약.csv"
)


st.set_page_config(
    page_title="제주 관광소비 진단",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """분석에 필요한 두 결과 파일을 읽고 공통 지표를 결합한다."""
    consumption = pd.read_csv(CONSUMPTION_FILE, encoding="utf-8-sig")
    concentration = pd.read_csv(CONCENTRATION_FILE, encoding="utf-8-sig")
    visitors = pd.read_csv(VISITOR_FILE, encoding="utf-8-sig")

    numeric_columns = [
        "지역총매출액",
        "매출순위",
        "도민매출비중",
        "관광객매출비중",
        "소매업매출비중",
        "소매업내_도민비중",
    ]
    for column in numeric_columns:
        if column in consumption:
            consumption[column] = pd.to_numeric(consumption[column], errors="coerce")

    concentration_columns = [
        "지역총매출액",
        "매출기준_HHI",
        "매출기준_업종엔트로피",
        "매출기준_최대업종비중",
    ]
    for column in concentration_columns:
        if column in concentration:
            concentration[column] = pd.to_numeric(
                concentration[column], errors="coerce"
            )

    merged = consumption.merge(
        concentration[
            [
                "읍면동명",
                "매출기준_HHI",
                "매출기준_업종엔트로피",
                "매출기준_최대업종",
                "매출기준_최대업종비중",
                "편중판정",
            ]
        ],
        on="읍면동명",
        how="left",
        suffixes=("", "_편중도"),
    )
    visitor_summary = (
        visitors.groupby("읍면동명", as_index=False)["도착건수"]
        .sum()
        .rename(columns={"도착건수": "차량 도착건수"})
    )
    visitor_summary["차량 도착건수"] = pd.to_numeric(
        visitor_summary["차량 도착건수"], errors="coerce"
    )
    merged = merged.merge(visitor_summary, on="읍면동명", how="left")
    return merged, concentration, visitor_summary


def format_won(value: float) -> str:
    """큰 금액을 억원 단위로 표시한다."""
    if pd.isna(value):
        return "-"
    return f"{value / 100_000_000:,.1f}억원"


@st.cache_data
def load_boundary_data() -> dict:
    """행정구역 GeoJSON을 읽고 지도 표시용 지역명을 복원한다."""
    with GEOJSON_FILE.open(encoding="utf-8") as file:
        geojson = json.load(file)

    for feature in geojson["features"]:
        raw_name = feature["properties"].get("adm_nm", "")
        if "\\u" in raw_name:
            try:
                decoded_name = raw_name.encode("ascii").decode("unicode_escape")
            except UnicodeError:
                decoded_name = raw_name
        else:
            decoded_name = raw_name
        feature["properties"]["행정구역명"] = decoded_name.split()[-1]
    return geojson


def build_admin_map(
    boundary_data: dict,
    data: pd.DataFrame,
    selected_region: str,
    target_data: pd.DataFrame,
    map_metric: str,
) -> pdk.Deck:
    """행정구역별 분석 지표를 색으로 표시하는 pydeck 지도를 만든다."""
    metric_columns = {
        "총매출": ("지역총매출액", "억원"),
        "관광객 매출 비중": ("관광객매출비중", "%"),
        "업종 편중도": ("매출기준_HHI", "지수"),
        "차량 도착건수": ("차량 도착건수", "건"),
    }
    source_column, unit = metric_columns[map_metric]
    map_data = data[["읍면동명", source_column]].copy()
    map_data["표시값"] = map_data[source_column].fillna(0)
    if map_metric == "총매출":
        map_data["표시값"] = map_data["표시값"] / 100_000_000
    elif map_metric == "관광객 매출 비중":
        map_data["표시값"] = map_data["표시값"] * 100
    map_data = map_data.set_index("읍면동명")
    values = map_data["표시값"]
    minimum, maximum = float(values.min()), float(values.max())
    span = maximum - minimum or 1
    target_regions = set(target_data["읍면동명"])
    is_full_scope = len(target_regions) == len(data)

    for feature in boundary_data["features"]:
        properties = feature["properties"]
        region = properties["행정구역명"]
        value = float(map_data.loc[region, "표시값"]) if region in map_data.index else 0
        ratio = (value - minimum) / span
        is_target = region in target_regions
        if is_full_scope:
            fill = [
                int(226 + (15 - 226) * ratio),
                int(232 + (95 - 232) * ratio),
                int(230 + (112 - 230) * ratio),
                255,
            ]
        else:
            fill = [42, 120, 214, 255] if is_target else [226, 232, 230, 255]
        properties.update(
            {
                "표시값": value,
                "단위": unit,
                "분석 대상": (
                    "전체 지역"
                    if is_full_scope
                    else ("문제 지역" if is_target else "일반 지역")
                ),
                "fill_color": fill,
                "line_color": [255, 255, 255, 230],
            }
        )
        if region == selected_region:
            properties["line_color"] = [21, 32, 30, 255]

    layer = pdk.Layer(
        "GeoJsonLayer",
        boundary_data,
        pickable=True,
        auto_highlight=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color="properties.line_color",
        line_width_min_pixels=1,
    )
    view_state = pdk.ViewState(latitude=33.38, longitude=126.55, zoom=9.4)
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={
            "html": (
                "<b>{행정구역명}</b><br/>"
                f"{map_metric}: {{표시값}} {unit}<br/>"
                "상태: {분석 대상}"
            ),
            "style": {
                "backgroundColor": "#ffffff",
                "color": "#15201e",
                "fontSize": "12.5px",
                "border": "1px solid #dde4e2",
                "borderRadius": "8px",
            },
        },
    )


def render_segmented_filter() -> str:
    """분석 기준을 세그먼트 컨트롤로 렌더링한다."""
    options = [
        "전체 지역",
        "관광소비 하락",
        "시기·업종 집중",
        "상점·거래 불일치",
    ]
    current = st.session_state.get("dashboard_quick_filter", "전체 지역")
    if current not in options:
        current = "전체 지역"
    st.session_state.dashboard_quick_filter = current
    segmented_control = getattr(st, "segmented_control", None)
    if segmented_control is not None:
        return segmented_control(
            "빠른 분석 기준",
            options,
            key="dashboard_quick_filter",
        )
    return st.radio(
        "빠른 분석 기준",
        options,
        index=options.index(current),
        horizontal=True,
        key="dashboard_quick_filter",
    )


def safe_filter_filename(value: str | None) -> str:
    """파일명에 사용할 수 있는 필터 이름을 정리한다."""
    label = str(value or "전체 지역")
    return label.replace("·", "_").replace(" ", "_")


def get_map_selected_region(event: object) -> str | None:
    """PyDeck 선택 이벤트에서 행정구역명을 안전하게 추출한다."""
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if not isinstance(selection, dict):
        return None

    objects = selection.get("objects", {})
    if isinstance(objects, dict):
        candidates = objects.values()
    else:
        candidates = [objects]
    for candidate_group in candidates:
        if isinstance(candidate_group, dict):
            candidate_group = [candidate_group]
        if not isinstance(candidate_group, list):
            continue
        for feature in candidate_group:
            if not isinstance(feature, dict):
                continue
            properties = feature.get("properties", feature)
            if isinstance(properties, dict) and properties.get("행정구역명"):
                return str(properties["행정구역명"])
    return None


def render_admin_map(
    deck: pdk.Deck,
    selected_region: str,
) -> str | None:
    """지원 버전에 따라 지도 클릭 선택을 활성화하고 선택 지역을 반환한다."""
    supports_selection = "on_select" in inspect.signature(st.pydeck_chart).parameters
    if not supports_selection:
        st.pydeck_chart(deck, use_container_width=True)
        st.info(
            "지도 클릭 선택을 사용하려면 Streamlit 1.35 이상이 필요합니다. "
            "현재 버전에서는 hover와 지역 바로가기 버튼을 사용할 수 있습니다."
        )
        return None

    event = st.pydeck_chart(
        deck,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-object",
        key="dashboard_admin_map",
    )
    clicked_region = get_map_selected_region(event)
    if clicked_region and clicked_region != selected_region:
        return clicked_region
    return None


def build_threshold_chart(
    chart_data: pd.DataFrame,
    x_field: str,
    y_field: str,
    x_title: str,
    y_title: str,
    x_threshold: float,
    y_threshold: float,
    x_threshold_label: str,
    y_threshold_label: str,
    x_format: str = ".1f",
    y_format: str = ".1f",
) -> alt.Chart:
    """판정 기준선과 문제 지역 라벨을 포함한 사분면 비교 차트를 만든다."""
    base = alt.Chart(chart_data).encode(
        x=alt.X(
            f"{x_field}:Q",
            title=x_title,
            axis=alt.Axis(format=x_format),
        ),
        y=alt.Y(
            f"{y_field}:Q",
            title=y_title,
            axis=alt.Axis(format=y_format),
        ),
        tooltip=[
            alt.Tooltip("읍면동명:N", title="지역"),
            alt.Tooltip(f"{x_field}:Q", title=x_title, format=x_format),
            alt.Tooltip(f"{y_field}:Q", title=y_title, format=y_format),
        ],
    )
    points = base.mark_circle(size=110).encode(
        color=alt.Color(
            "문제 지역:N",
            title="판정",
            scale=alt.Scale(
                domain=["문제 지역", "일반 지역"],
                range=["#d62728", "#9aa7b8"],
            ),
        )
    )
    labels = (
        base        .transform_filter("datum['문제 지역'] == '문제 지역'")
        .mark_text(align="left", dx=7, dy=-7, fontSize=12, fontWeight="bold")
        .encode(text="읍면동명:N", color=alt.value("#b22222"))
    )
    x_rule = (
        alt.Chart(pd.DataFrame({x_field: [x_threshold]}))
        .mark_rule(color="#d62728", strokeDash=[5, 4])
        .encode(x=alt.X(f"{x_field}:Q"))
    )
    y_rule = (
        alt.Chart(pd.DataFrame({y_field: [y_threshold]}))
        .mark_rule(color="#d62728", strokeDash=[5, 4])
        .encode(y=alt.Y(f"{y_field}:Q"))
    )
    x_text = (
        alt.Chart(pd.DataFrame({x_field: [x_threshold], "label": [x_threshold_label]}))
        .mark_text(align="left", dx=5, dy=12, color="#b22222", fontSize=11)
        .encode(x=alt.X(f"{x_field}:Q"), text="label:N")
    )
    y_text = (
        alt.Chart(pd.DataFrame({y_field: [y_threshold], "label": [y_threshold_label]}))
        .mark_text(align="left", dx=5, dy=-7, color="#b22222", fontSize=11)
        .encode(y=alt.Y(f"{y_field}:Q"), text="label:N")
    )
    return (points + labels + x_rule + y_rule + x_text + y_text).properties(
        height=430
    ).interactive()


def build_structure_heatmap(data: pd.DataFrame) -> alt.Chart:
    """지역별 소비 구조와 문제 지역 여부를 비교하는 히트맵을 만든다."""
    heatmap_data = data[
        [
            "읍면동명",
            "도민매출비중",
            "소매업매출비중",
            "매출기준_HHI",
            "문제 지역",
        ]
    ].rename(
        columns={
            "도민매출비중": "도민 매출 비중",
            "소매업매출비중": "소매업 매출 비중",
            "매출기준_HHI": "업종 편중도(HHI)",
        }
    )
    long_data = heatmap_data.melt(
        id_vars=["읍면동명", "문제 지역"],
        var_name="구조 지표",
        value_name="값",
    )
    long_data["표시값"] = long_data.apply(
        lambda row: (
            f"{row['값']:.3f}"
            if row["구조 지표"] == "업종 편중도(HHI)"
            else f"{row['값']:.1%}"
        ),
        axis=1,
    )
    order = (
        heatmap_data.assign(
            문제순서=heatmap_data["문제 지역"].eq("문제 지역").astype(int)
        )
        .sort_values(["문제순서", "읍면동명"], ascending=[False, True])["읍면동명"]
        .tolist()
    )
    return (
        alt.Chart(long_data)
        .mark_rect(stroke="#ffffff", strokeWidth=1)
        .encode(
            x=alt.X("구조 지표:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "읍면동명:N",
                title="지역",
                sort=order,
                axis=alt.Axis(labelLimit=100),
            ),
            color=alt.Color(
                "값:Q",
                title="비중·편중도",
                scale=alt.Scale(scheme="blues", domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("읍면동명:N", title="지역"),
                alt.Tooltip("구조 지표:N", title="지표"),
                alt.Tooltip("표시값:N", title="값"),
                alt.Tooltip("문제 지역:N", title="판정"),
            ],
        )
        .properties(height=max(420, len(order) * 22))
    )


def show_criterion_marker(
    criterion: str, target: int, current: int, description: str
) -> None:
    """슬라이더의 고정 판정 기준점을 시각적으로 표시한다."""
    position = max(0, min(100, target))
    current_color = "#1677ff" if current == target else "#666666"
    st.markdown(
        f"""
        <div style="font-size:0.78rem; color:#666; margin-top:-0.55rem;">
            <span style="color:#d62728; font-weight:700;">◆ 판정 기준 {target}%</span>
            <span style="float:right; color:{current_color};">
                현재 {current}% · {description}
            </span>
        </div>
        <div style="position:relative; height:0.7rem; margin:0.05rem 0 0.8rem;">
            <div style="position:absolute; left:{position}%; top:0; transform:translateX(-50%);
                        color:#d62728; font-size:0.8rem; line-height:0.7rem;">│</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def snap_slider_to_criterion(
    key: str, target: int, snap_distance: int = 2
) -> None:
    """판정 기준점에 가까운 슬라이더 값을 기준값으로 맞춘다."""
    value = st.session_state[key]
    if abs(value - target) <= snap_distance:
        st.session_state[key] = target


def render_topic1_filters() -> tuple[int, int, int]:
    st.header("업종 구조 진단 기준")
    st.caption("도민 소비와 소매업 편중이 동시에 나타나는 지역을 찾는 기준입니다.")
    resident = st.slider(
        "도민 매출 비중 하한",
        0,
        100,
        75,
        1,
        "%d%%",
        key="topic1_resident_threshold",
        on_change=snap_slider_to_criterion,
        args=("topic1_resident_threshold", 75),
    )
    show_criterion_marker("도민 매출 비중", 75, resident, "도민 비중 ≥ 기준")
    retail = st.slider(
        "소매업 매출 비중 하한",
        0,
        100,
        45,
        1,
        "%d%%",
        key="topic1_retail_threshold",
        on_change=snap_slider_to_criterion,
        args=("topic1_retail_threshold", 45),
    )
    show_criterion_marker("소매업 매출 비중", 45, retail, "소매업 비중 ≥ 기준")
    sales = st.slider(
        "낮은 총매출 기준 (하위 백분위)",
        10,
        80,
        65,
        1,
        "하위 %d%%",
        key="topic1_sales_percentile",
        on_change=snap_slider_to_criterion,
        args=("topic1_sales_percentile", 65),
    )
    st.markdown(
        "**문제 지역 판정**\n\n"
        "도민 매출 비중 ≥ 75% · 소매업 매출 비중 ≥ 45% · "
        "총매출 ≤ 하위 65%"
    )
    return resident, retail, sales


def diagnosis(row: pd.Series) -> tuple[str, str]:
    """선별된 지역의 관측된 문제와 소비 구조를 요약한다."""
    resident_share = row["도민매출비중"]
    retail_share = row["소매업매출비중"]
    top_share = row["매출기준_최대업종비중"]

    causes = [
        f"도민 소비 비중 {resident_share:.1%}",
        f"소매업 매출 비중 {retail_share:.1%}",
        f"최대 업종 집중도 {top_share:.1%}",
    ]
    cause = " · ".join(causes)
    problem = (
        f"{row['읍면동명']} — 총매출 {format_won(row['지역총매출액'])}, "
        f"제주 전체 매출 순위 {int(row['매출순위'])}위"
    )
    return problem, cause


def integrated_diagnosis(
    filtered: pd.DataFrame, resident_threshold: float, retail_threshold: float
) -> str:
    """선별 지역 전체에 적용되는 하나의 통합 진단을 만든다."""
    median_sales = filtered["지역총매출액"].median()
    return (
        f"선별된 {len(filtered)}개 지역은 도민 매출 비중이 {resident_threshold:.0%} "
        f"이상이고 소매업 매출 비중이 {retail_threshold:.0%} 이상인 동시에, "
        f"총매출이 설정한 저매출 기준에 해당합니다. 이는 소비가 소매업에 집중되어 "
        f"있고 관광·체험·식음 등 다른 업종으로 분산되는 정도가 낮은 공통 패턴을 "
        f"보여주는 것이지, 매출 부진의 원인을 확정하는 결과는 아닙니다. "
        f"선별 지역의 총매출 중앙값은 {format_won(median_sales)}입니다."
    )


@st.cache_data
def load_industry_analysis() -> dict[str, pd.DataFrame]:
    """시기·업종 집중 분석의 사전 계산 결과를 읽는다."""
    files = {
        "features": INDUSTRY_FEATURES_FILE,
        "monthly": INDUSTRY_MONTHLY_FILE,
        "category_monthly": INDUSTRY_CATEGORY_MONTHLY_FILE,
        "high_low": INDUSTRY_HIGH_LOW_FILE,
        "store": INDUSTRY_STORE_FILE,
        "sensitivity": INDUSTRY_SENSITIVITY_FILE,
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return {
        name: pd.read_csv(path, encoding="utf-8-sig", dtype={"기준연월": str})
        for name, path in files.items()
    }


def format_analysis_period(period: object) -> str:
    period_labels = {
        1: "2023.08~2024.07",
        2: "2024.08~2025.07",
        3: "2025.08~2026.07",
    }
    period_number = int(period)
    date_range = period_labels.get(period_number)
    if date_range is None:
        return f"{period_number}차 구간"
    return f"{period_number}차 구간 ({date_range})"


def render_industry_analysis() -> None:
    """인계본의 시기·업종 집중 분석과 기존 업종 분석을 통합 표시한다."""
    try:
        analysis = load_industry_analysis()
    except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"업종별 분석 자료를 읽을 수 없습니다: {error}")
        return

    features = analysis["features"].copy()
    period_options = sorted(features["구간"].dropna().unique())
    selected_period = st.selectbox(
        "분석 구간",
        period_options,
        index=len(period_options) - 1,
        format_func=format_analysis_period,
        key="industry_analysis_period",
    )
    period_features = features[features["구간"] == selected_period].copy()
    default_region = (
        "전체 지역"
        if "전체 지역" in st.session_state.get("industry_analysis_region", "")
        else period_features.iloc[0]["읍면동명"]
    )
    region_options = ["전체 지역"] + sorted(period_features["읍면동명"].tolist())
    selected_region = st.selectbox(
        "상세 지역",
        region_options,
        index=region_options.index(default_region)
        if default_region in region_options
        else 0,
        key="industry_analysis_region",
    )
    region_features = (
        period_features
        if selected_region == "전체 지역"
        else period_features[period_features["읍면동명"] == selected_region]
    )

    st.info(
        "핵심 질문: 관광객 소비가 특정 시기와 업종에 반복적으로 집중되는 지역은 "
        "어디인가? 집중형은 위험이나 매출 감소의 확정 판정이 아니라 43개 지역 "
        "분포에서 상대적으로 높은 지역을 뜻합니다."
    )
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    concentrated = period_features[period_features["종합유형"] == "집중형"]
    metric_1.metric("분석 지역", f"{len(period_features)}개")
    metric_2.metric("집중형 지역", f"{len(concentrated)}개")
    metric_3.metric(
        "월 집중도 기준(Q75)",
        f"{period_features['상위3개월비중_pct'].quantile(.75):.1f}%",
    )
    metric_4.metric(
        "업종 HHI 기준(Q75)",
        f"{period_features['업종HHI'].quantile(.75):.3f}",
    )

    overview_tab, detail_tab, evidence_tab, legacy_tab, action_tab = st.tabs(
        ["전체 결과", "월·업종 상세", "추가 검증", "기존 구조 진단", "종합 진단·대응"]
    )
    with overview_tab:
        st.subheader(f"{format_analysis_period(selected_period)} 지역별 집중 결과")
        chart = (
            alt.Chart(period_features)
            .mark_circle(size=110)
            .encode(
                x=alt.X("상위3개월비중_pct:Q", title="상위 3개월 소비 비중(%)"),
                y=alt.Y("업종HHI:Q", title="업종 HHI"),
                color=alt.Color(
                    "종합유형:N",
                    scale=alt.Scale(
                        domain=["집중형", "중간", "분산형"],
                        range=["#d95f02", "#7f8c8d", "#1b9e77"],
                    ),
                ),
                size=alt.Size("관광객소비_억:Q", title="관광객 소비(억원)"),
                tooltip=[
                    "읍면동명",
                    "종합유형",
                    alt.Tooltip("상위3개월비중_pct:Q", format=".1f"),
                    alt.Tooltip("업종HHI:Q", format=".3f"),
                    "최대소비업종",
                    alt.Tooltip("관광객소비_억:Q", format=".1f"),
                ],
            )
            .properties(height=430)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "월 집중도와 업종 HHI가 모두 해당 구간의 43개 지역 Q75 이상이면 "
            "집중형으로 분류합니다. 상위 3개월은 비연속일 수 있습니다."
        )
        display = period_features[
            [
                "읍면동명",
                "종합유형",
                "월유형",
                "업종유형",
                "상위3개월비중_pct",
                "업종HHI",
                "최대소비업종",
                "최대업종비중_pct",
            ]
        ].sort_values(["종합유형", "상위3개월비중_pct"], ascending=[True, False])
        st.dataframe(
            display.rename(
                columns={
                    "읍면동명": "지역",
                    "종합유형": "종합 유형",
                    "월유형": "시기 유형",
                    "업종유형": "업종 유형",
                    "상위3개월비중_pct": "상위 3개월 비중(%)",
                    "업종HHI": "업종 HHI",
                    "최대소비업종": "최대 소비 업종",
                    "최대업종비중_pct": "최대 업종 비중(%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with detail_tab:
        if selected_region == "전체 지역":
            st.subheader("집중형 지역 요약")
            st.dataframe(
                concentrated[
                    [
                        "읍면동명",
                        "상위3개월",
                        "저조3개월",
                        "최대소비업종",
                        "상위3개월주력업종_연간비중_pct",
                        "주력업종중상위3개월비중_pct",
                    ]
                ].rename(
                    columns={
                        "읍면동명": "지역",
                        "상위3개월": "상위 3개월",
                        "저조3개월": "저조 3개월",
                        "최대소비업종": "최대 소비 업종",
                        "상위3개월주력업종_연간비중_pct": "주력 업종 연간 비중(%)",
                        "주력업종중상위3개월비중_pct": "주력 업종의 상위월 비중(%)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("지역을 선택하면 월별 소비와 업종 구성의 상세 차트를 확인할 수 있습니다.")
        else:
            row = region_features.iloc[0]
            st.subheader(f"{selected_region} · 관측된 소비 구조")
            detail_metric = st.columns(4)
            detail_metric[0].metric("종합 유형", row["종합유형"])
            detail_metric[1].metric("상위 3개월 비중", f"{row['상위3개월비중_pct']:.1f}%")
            detail_metric[2].metric("업종 HHI", f"{row['업종HHI']:.3f}")
            detail_metric[3].metric("최대 업종", row["최대소비업종"])

            monthly = analysis["monthly"]
            monthly = monthly[
                (monthly["읍면동명"] == selected_region)
                & (monthly["구간"] == selected_period)
            ].copy()
            monthly["월"] = monthly["기준연월"].astype(str)
            monthly_chart = (
                alt.Chart(monthly)
                .mark_line(point=True)
                .encode(
                    x=alt.X("월:O", title="기준 연월", sort="ascending"),
                    y=alt.Y("소비금액(원):Q", title="소비 금액(억원)", scale=alt.Scale(zero=False)),
                    color=alt.Color("집단:N", title="집단"),
                    tooltip=["월", "집단", alt.Tooltip("소비금액(원):Q", format=",.0f")],
                )
                .transform_calculate(소비금액_억원="datum['소비금액(원)']/100000000")
                .properties(height=330)
            )
            st.altair_chart(monthly_chart, use_container_width=True)
            category = analysis["category_monthly"]
            category = category[
                (category["읍면동명"] == selected_region)
                & (category["구간"] == selected_period)
            ].copy()
            category["월"] = category["기준연월"].astype(str)
            category_chart = (
                alt.Chart(category)
                .mark_bar()
                .encode(
                    x=alt.X("월:O", title="기준 연월", sort="ascending"),
                    y=alt.Y("소비금액(원):Q", stack="normalize", title="업종별 소비 구성"),
                    color=alt.Color("업종명:N", title="업종"),
                    tooltip=["월", "업종명", alt.Tooltip("소비금액(원):Q", format=",.0f")],
                )
                .properties(height=330)
            )
            st.altair_chart(category_chart, use_container_width=True)

            high_low = analysis["high_low"]
            high_low = high_low[
                (high_low["읍면동명"] == selected_region)
                & (high_low["구간"] == selected_period)
            ]
            if not high_low.empty:
                hl = high_low.iloc[0]
                st.markdown("#### 상위월·저조월 비교")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "지표": ["관광 소비", "관광 거래", "관광 건당 금액", "도민 소비", "합산 소비"],
                            "하위월/상위월(%)": [
                                hl["관광소비_하위상위비_pct"],
                                hl["관광거래_하위상위비_pct"],
                                hl["관광건당금액_하위상위비_pct"],
                                hl["도민소비_하위상위비_pct"],
                                100 + hl["합산소비변화율_pct"],
                            ],
                        }
                    ).round(1),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    f"상위월: {hl['상위3개월']} · 저조월: {hl['하위3개월']} · "
                    f"{hl['거래금액변동설명']}. 이는 전년 대비 감소나 인과 원인 분석이 아닙니다."
                )

    with evidence_tab:
        st.subheader("민감도·상점·거래 추가 검증")
        sensitivity = analysis["sensitivity"]
        sensitivity_period = sensitivity[sensitivity["구간"] == selected_period]
        st.markdown("#### 9개 기준 조합에서의 선정 횟수")
        st.dataframe(
            sensitivity_period.sort_values("선정조합수_9개중", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("선정 횟수는 확률이 아닙니다. 기본 유형을 대체하지 않는 민감도 참고값입니다.")

        store = analysis["store"]
        store = store[
            (store["구간"] == selected_period)
            & (
                store["읍면동명"].eq(selected_region)
                if selected_region != "전체 지역"
                else store["읍면동명"].isin(concentrated["읍면동명"])
            )
        ]
        if not store.empty:
            st.markdown("#### 상점 구성과 관광객 거래 구성")
            store_chart = store.melt(
                id_vars=["읍면동명", "업종"],
                value_vars=["상점비중_pct", "거래비중_pct"],
                var_name="구성",
                value_name="비중(%)",
            )
            st.altair_chart(
                alt.Chart(store_chart)
                .mark_bar()
                .encode(
                    x=alt.X("업종:N", title="업종"),
                    y=alt.Y("비중(%):Q", title="비중(%)"),
                    color=alt.Color("구성:N", title="기준"),
                    column=alt.Column("읍면동명:N", title="지역"),
                    tooltip=["읍면동명", "업종", "구성", "비중(%)"],
                )
                .properties(height=300),
                use_container_width=True,
            )
            st.caption("상점 목록은 한 시점 자료이며, 상점 비중과 거래 비중의 차이는 수요 부족을 뜻하지 않습니다.")

    with action_tab:
        st.subheader("종합 진단과 원인 가설")
        st.warning(
            "집중형은 관측된 시기·업종 편중입니다. 상권 위험, 소비 감소, 원인 또는 "
            "정책 효과를 확정하지 않습니다."
        )
        if selected_region == "전체 지역":
            st.markdown(
                "**3개년 지속 집중형:** 구좌읍·우도면·화북동·예래동\n\n"
                "네 지역은 관광객 소비가 연중 고르게 분산되기보다 특정 3개월과 "
                "주력 업종에 함께 몰리는 패턴이 세 구간에서 반복됩니다. 따라서 "
                "문제는 단순히 연간 매출 규모가 아니라, 관광객 수요가 약해지는 시기에 "
                "대체 소비를 만들어낼 업종과 동선이 충분하지 않다는 구조적 취약성으로 "
                "해석하는 것이 적절합니다. 다만 아래 진단은 카드 소비에서 관측된 "
                "패턴에 근거한 가설이며 방문객 감소나 영업 중단의 원인을 확정하지 않습니다."
            )
            st.markdown(
                "**지역별 진단 방향**\n\n"
                "- **구좌읍:** 음식점업 중심의 소비가 상위월에 집중되고 저조월에는 "
                "거래 수가 크게 줄어드는 유형입니다. 관광객이 방문해도 식사 외에 "
                "머무르며 지출할 체험·지역상품 선택지가 부족하면 비수기 매출을 "
                "받쳐주기 어렵습니다. 저조월에 음식점·숙박·체험을 묶은 예약형 "
                "동선을 우선 시범 운영하는 방향이 적합합니다.\n"
                "- **우도면:** 음식점업 집중과 함께 상·저조월의 소비 및 거래 차이가 "
                "가장 크게 나타나는 유형입니다. 섬 접근성이나 운항 가능일에 따라 "
                "방문 기회 자체가 달라질 수 있어, 방문객을 늘리는 홍보보다 운영 가능한 "
                "날짜에 식사·지역상품·체험을 연결해 한 번의 방문당 소비를 늘리는 "
                "전략이 우선입니다.\n"
                "- **화북동:** 소매업 비중과 업종 HHI가 높아 소비가 특정 소매 업종에 "
                "강하게 의존합니다. 관광객 소비가 약한 시기에도 도민 소비 기반은 "
                "상대적으로 유지될 수 있으므로, 기존 생활권 상권을 관광객 대상 "
                "음식·여가·지역상품 소비로 확장하는 방식이 적합합니다.\n"
                "- **예래동:** 숙박업이 관광 소비를 주도하지만 상위월과 저조월의 "
                "거래 수·건당 금액이 함께 낮아지는 유형입니다. 숙박 결제가 지역 "
                "내 음식·체험·소매로 이어지는 연결고리가 약할 가능성이 있으므로, "
                "숙박 예약 단계에서 주변 업종 쿠폰과 체험을 결합하는 전략을 "
                "우선 검토할 수 있습니다."
            )
            st.dataframe(
                period_features[
                    period_features["종합유형"] == "집중형"
                ][
                    ["읍면동명", "최대소비업종", "상위3개월", "저조3개월", "거래금액변동설명"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            hypotheses = {
                "구좌읍": "음식점업이 주력 업종이고 상위월의 소비가 저조월보다 큰 폭으로 나타나며, 차이는 건당 금액보다 거래 수에서 더 크게 설명됩니다. 관광객이 적게 쓰는 것보다 방문 기회가 있는 달에 식사 외 소비로 이어지는 경로가 약한지 우선 살펴볼 필요가 있습니다. 저조월에 음식·숙박·체험을 묶은 예약형 상품과 지역상품 구매 동선을 시범 운영하는 것이 적절합니다.",
                "우도면": "음식점업 중심 구조와 함께 상·저조월의 관광 소비 및 거래 수 차이가 크게 나타납니다. 섬이라는 입지 특성상 운항일·기상·방문 가능일의 영향을 먼저 고려해야 하며, 단순 방문객 확대보다 운영 가능한 날짜에 식사·지역상품·체험을 묶어 방문 1건당 소비를 높이는 접근이 현실적입니다.",
                "화북동": "소매업 집중도가 높고 관광객 소비가 약한 시기에도 도민 소비가 상대적으로 버티는 생활권형 구조로 볼 수 있습니다. 기존 상점의 고객 기반은 있으므로 관광객 전용 업종을 새로 만드는 것보다 소매점과 음식·여가·지역상품을 연결하는 짧은 동선을 만들어 체류 소비를 늘리는 방향이 우선입니다.",
                "예래동": "숙박업이 주력 업종이지만 저조월에는 거래 수와 건당 금액이 함께 낮아져 숙박 수요가 주변 소비로 충분히 전환되지 않을 가능성이 있습니다. 숙박 예약 단계에서 음식·체험·소매 쿠폰을 결합하고, 숙박업소와 주변 상점이 공동으로 저조기 상품을 운영하는 방안을 우선 검토할 수 있습니다.",
            }
            st.markdown(
                f"**{selected_region}에서 확인된 사실**\n\n"
                f"- 종합 유형: {row['종합유형']}\n"
                f"- 최대 소비 업종: {row['최대소비업종']} ({row['최대업종비중_pct']:.1f}%)\n"
                f"- 상위 소비월: {row['상위3개월']}\n"
                f"- 저조 소비월: {row['저조3개월']}\n\n"
                f"**원인 가설과 확인·대응 방향**\n\n"
                f"{hypotheses.get(selected_region, '방문객 수와 체류시간, 영업일·휴업일, 상품 구성과 가격대, 예약·운항·기상 자료를 월별로 대조해 소비 변동이 수요·운영·상품 중 어느 요인과 함께 나타나는지 확인합니다.')}"
            )
            st.markdown(
                "**진단을 구체화하기 위한 다음 단계:** 이 지역의 소비 저하는 하나의 "
                "원인으로 단정하기보다, `방문 기회 감소 → 영업 가능일 감소 → "
                "주력 업종 외 소비 부족` 중 어떤 경로가 큰지 나눠 보는 것이 좋습니다. "
                "상위월·저조월의 방문객 수와 체류시간을 소비 거래 수와 맞춰 보면 "
                "유입 자체가 줄었는지, 방문은 있었지만 결제가 줄었는지 구분할 수 "
                "있습니다. 여기에 영업일·휴업일, 상품 구성·가격대, 예약·결제 시점, "
                "기상·운항·행사 일정을 겹쳐 보면 지역별 대응 우선순위가 선명해집니다. "
                "그 결과에 따라 구좌읍·우도면은 저조기 체험·음식 연계, 화북동은 "
                "생활권 소매의 관광 동선화, 예래동은 숙박 예약과 주변 소비의 결합을 "
                "작게 시행하고, 관광객 거래 수·연계 업종 매출·건당 금액·연간 소비 "
                "유지 여부를 비교지역과 함께 확인하는 방식이 적절합니다."
            )
        st.caption(
            "대응 효과는 저조기 소비·거래 증가, 연간 소비 유지, 연계 업종 실적을 "
            "동시에 확인해야 합니다. 분석 결과는 기술통계이며 인과분석이 아닙니다."
        )

    with legacy_tab:
        st.subheader("기존 업종별 분석 기준")
        st.caption(
            "기존 화면의 도민 소비·소매업 편중·저매출 조건을 별도 참고 분석으로 "
            "유지합니다. 핵심 집중형 판정과 섞지 않고 결과를 나란히 비교합니다."
        )
        resident_threshold = (
            st.session_state.get("topic1_resident_threshold", 75) / 100
        )
        retail_threshold = st.session_state.get("topic1_retail_threshold", 45) / 100
        sales_percentile = st.session_state.get("topic1_sales_percentile", 65)
        sales_cutoff = data["지역총매출액"].quantile(sales_percentile / 100)
        legacy = data[
            (data["도민매출비중"] >= resident_threshold)
            & (data["소매업매출비중"] >= retail_threshold)
            & (data["지역총매출액"] <= sales_cutoff)
        ].copy()
        legacy = legacy.sort_values("지역총매출액")
        st.markdown(
            "#### 관찰된 구조와 개선 방향\n"
            "기존 기준에 해당하는 지역에서는 소매업 비중이 크고 도민 소비 비중이 "
            "상대적으로 높은 가운데, 관광소비가 낮은 패턴이 함께 관찰됩니다. "
            "이는 관광객 소비가 소매업과 일부 업종에 머물러 지역 내 소비 선택지가 "
            "충분히 확장되지 않았을 가능성을 보여주는 신호이며, 소비 자료만으로 "
            "원인을 확정하는 결과는 아닙니다.\n\n"
            "**제안:** 기존 소매 상권을 기반으로 지역 음식·체험·여가·숙박과 연결한 "
            "짧은 동선형 상품을 만들고, 소매점에서 사용할 수 있는 연계 쿠폰이나 "
            "QR 안내를 도입합니다. 저조기에는 상점 공동 프로모션과 주말 체험을 "
            "시범 운영하고, 관광객 거래 수·연계 업종 매출·재방문 지표를 전후 "
            "비교해 효과를 확인합니다."
        )
        lm = st.columns(4)
        lm[0].metric("기존 기준 지역", f"{len(legacy)}개")
        lm[1].metric("도민 비중 기준", f"{resident_threshold:.0%}")
        lm[2].metric("소매업 비중 기준", f"{retail_threshold:.0%}")
        lm[3].metric("저매출 기준", format_won(sales_cutoff))
        if legacy.empty:
            st.info("현재 기존 기준을 만족하는 지역이 없습니다. 사이드바 기준을 조정해 보세요.")
        else:
            st.dataframe(
                legacy[
                    [
                        "읍면동명",
                        "지역총매출액",
                        "매출순위",
                        "도민매출비중",
                        "소매업매출비중",
                        "매출기준_HHI",
                        "매출기준_최대업종",
                    ]
                ].rename(
                    columns={
                        "읍면동명": "지역",
                        "지역총매출액": "총매출",
                        "매출순위": "매출 순위",
                        "도민매출비중": "도민 매출 비중",
                        "소매업매출비중": "소매업 매출 비중",
                        "매출기준_HHI": "업종 HHI",
                        "매출기준_최대업종": "최대 매출 업종",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "기존 기준 결과 CSV 다운로드",
                data=legacy.to_csv(index=False).encode("utf-8-sig"),
                file_name="제주_기존업종구조_참고결과.csv",
                mime="text/csv",
                key="industry_legacy_download",
            )


def build_region_grid(data: pd.DataFrame, selected_region: str) -> alt.Chart:
    """지역을 비교 가능한 카드형 그리드로 표현한다."""
    grid = data[["읍면동명", "지역총매출액", "관광객매출비중"]].copy()
    grid = grid.sort_values("지역총매출액", ascending=False).reset_index(drop=True)
    grid["열"] = grid.index % 8
    grid["행"] = -(grid.index // 8)
    grid["선택 지역"] = grid["읍면동명"].eq(selected_region).map(
        {True: "선택 지역", False: "그 외 지역"}
    )
    grid["총매출 (억원)"] = grid["지역총매출액"] / 100_000_000
    return (
        alt.Chart(grid)
        .mark_rect(cornerRadius=7, stroke="#ffffff", strokeWidth=2)
        .encode(
            x=alt.X("열:O", title=None, axis=None),
            y=alt.Y("행:O", title=None, axis=None, sort="descending"),
            color=alt.Color(
                "선택 지역:N",
                scale=alt.Scale(
                    domain=["선택 지역", "그 외 지역"],
                    range=["#0868d7", "#dce3ec"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("읍면동명:N", title="지역"),
                alt.Tooltip("총매출 (억원):Q", title="총매출", format=",.1f"),
                alt.Tooltip("관광객매출비중:Q", title="관광객 매출 비중", format=".1%"),
            ],
        )
        .properties(height=355)
    )


def reset_dashboard_state() -> None:
    """홈 화면의 탐색 조건을 기본값으로 되돌린다."""
    st.session_state.dashboard_quick_filter = "전체 지역"
    st.session_state.dashboard_region = None
    st.session_state.dashboard_detail_region = "선택 안 함"


def describe_dashboard_criterion(
    quick_filter: str, data: pd.DataFrame, target_data: pd.DataFrame
) -> tuple[str, str]:
    """현재 주제의 선별 근거와 실제 임계값을 짧게 요약한다."""
    share = len(target_data) / len(data) if len(data) else 0
    if quick_filter == "관광소비 하락":
        threshold = data["지역총매출액"].quantile(0.25)
        return (
            "총매출이 낮은 지역",
            f"지역총매출액 하위 25% (약 {format_won(threshold)} 이하)를 "
            f"관광소비 하락 후보로 선별했습니다. {len(target_data)}개 지역 "
            f"({share:.0%})이 해당합니다.",
        )
    if quick_filter == "시기·업종 집중":
        threshold = data["매출기준_HHI"].quantile(0.75)
        return (
            "업종 소비가 특정 업종에 집중된 지역",
            f"매출기준 HHI 상위 25% (약 {threshold:.3f} 이상)를 기준으로 "
            f"선별했습니다. HHI가 높을수록 소수 업종 의존도가 큽니다. "
            f"{len(target_data)}개 지역 ({share:.0%})이 해당합니다.",
        )
    if quick_filter == "상점·거래 불일치":
        threshold = data["소매업매출비중"].quantile(0.75)
        return (
            "소매업 매출 비중이 높은 지역",
            f"소매업매출비중 상위 25% (약 {threshold:.1%} 이상)를 "
            f"상점·거래 불일치 후보로 선별했습니다. {len(target_data)}개 지역 "
            f"({share:.0%})이 해당합니다.",
        )
    return (
        "전체 지역 비교",
        f"별도 제외 기준 없이 {len(data)}개 읍면동 전체를 비교합니다. "
        "지도 색상 기준은 상단의 지도 색상 기준에서 변경할 수 있습니다.",
    )


def set_dashboard_detail_region() -> None:
    """선별 지역 목록에서 고른 지역을 상세 패널의 대상에 반영한다."""
    region = st.session_state.dashboard_detail_region
    st.session_state.dashboard_region = None if region == "선택 안 함" else region


def select_dashboard_region(region: str) -> None:
    """전체 지역 바로가기 버튼에서 선택한 지역을 위젯 상태와 함께 갱신한다."""
    st.session_state.dashboard_region = region
    st.session_state.dashboard_detail_region = region


def render_dashboard_home(data: pd.DataFrame, boundary_data: dict) -> None:
    """이미지의 정보 구조를 따르는 대시보드 홈을 렌더링한다."""
    region_names = sorted(data["읍면동명"].dropna().unique().tolist())
    if "dashboard_region" not in st.session_state:
        st.session_state.dashboard_region = None

    with st.container(border=True):
        st.markdown("##### 1. 분석 조건 설정<span class=\'panel-mark\'></span>", unsafe_allow_html=True)
        st.caption("분석 주제를 선택하면 지도와 선별 지역 목록이 함께 갱신됩니다.")
        filter_left, map_metric_col, reset_col = st.columns(
            [1.7, 1, 0.55]
        )
        with filter_left:
            quick_filter = render_segmented_filter() or st.session_state.get(
                "dashboard_quick_filter", "전체 지역"
            )
        with map_metric_col:
            map_metric = st.selectbox(
                "지도 색상 기준",
                ["총매출", "관광객 매출 비중", "업종 편중도", "차량 도착건수"],
                key="dashboard_map_metric",
            )
        with reset_col:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            st.button(
                "초기화",
                key="dashboard_reset",
                on_click=reset_dashboard_state,
            )

    if quick_filter == "관광소비 하락":
        target_data = data.nsmallest(max(1, len(data) // 4), "지역총매출액")
    elif quick_filter == "시기·업종 집중":
        target_data = data.nlargest(max(1, len(data) // 4), "매출기준_HHI")
    elif quick_filter == "상점·거래 불일치":
        target_data = data.nlargest(max(1, len(data) // 4), "소매업매출비중")
    else:
        target_data = data
    customer_column = "관광객매출비중"
    customer_label = "관광객 매출 비중"
    selected_region = st.session_state.dashboard_region
    selected_rows = data[data["읍면동명"].eq(selected_region)]
    selected = selected_rows.iloc[0] if not selected_rows.empty else None
    criterion_title, criterion_description = describe_dashboard_criterion(
        quick_filter, data, target_data
    )

    cards = st.columns(4)
    cards[0].metric("분석 지역", f"{len(data)}개")
    cards[1].metric(
        "표시 지역" if quick_filter == "전체 지역" else "문제 지역",
        f"{len(target_data)}개",
        delta="전체 기준" if quick_filter == "전체 지역" else "분석 기준 해당",
        delta_color="off",
    )
    cards[2].metric("지도 지표", map_metric)
    cards[3].metric("상세 선택", selected_region or "지도에서 선택")

    map_col, detail_col = st.columns([1.65, 1])
    with map_col:
        with st.container(border=True):
            st.markdown("##### 2. 행정구역별 분포<span class=\'panel-mark\'></span>", unsafe_allow_html=True)
            st.caption(f"색상 기준: {map_metric} · 강조 기준: {quick_filter}")
            clicked_region = render_admin_map(
                build_admin_map(
                    boundary_data,
                    data,
                    selected_region,
                    target_data,
                    map_metric,
                ),
                selected_region,
            )
            if clicked_region in region_names:
                st.session_state.dashboard_region = clicked_region
                st.session_state.dashboard_detail_region = clicked_region
                st.rerun()
            st.caption(
                f"행정구역을 클릭하면 상세 정보가 열립니다. 청록색이 진할수록 {map_metric} 값이 큽니다."
                if quick_filter == "전체 지역"
                else "행정구역을 클릭하면 상세 정보가 열립니다. 파란 영역은 현재 기준의 선별 지역, "
                "연회색 영역은 그 외 지역입니다."
            )
    with detail_col:
        with st.container(border=True):
            st.markdown("##### 3. 분석 결과<span class=\'panel-mark\'></span>", unsafe_allow_html=True)
            if selected is None:
                st.markdown("#### 분석 기준 요약")
                st.markdown(
                    f"<div class='detail-badge'>{quick_filter} · 전체 소비 기준</div>",
                    unsafe_allow_html=True,
                )
                headline = (
                    f"{len(data)}개 읍면동 전체 비교"
                    if quick_filter == "전체 지역"
                    else f"{len(target_data)}개 지역이 선별되었습니다"
                )
                st.markdown(
                    f"<h3>{headline}</h3>"
                    "<p>지도에서 행정구역을 클릭하면 해당 지역의 상세 정보가 표시됩니다.</p>",
                    unsafe_allow_html=True,
                )
                st.divider()
                st.markdown(f"**선별 근거 | {criterion_title}**")
                st.caption(criterion_description)
                st.markdown("**43곳 평균 지표**" if quick_filter == "전체 지역" else "**선별 지역의 공통 지표**")
                scope_label = "43곳" if quick_filter == "전체 지역" else "선별 지역"
                st.metric(f"{scope_label} 평균 매출", format_won(target_data["지역총매출액"].mean()))
                st.metric(
                    f"{scope_label} 평균 {customer_label}",
                    f"{target_data[customer_column].mean():.1%}",
                )
                st.download_button(
                    "현재 기준 선별 지역 다운로드",
                    data=target_data.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"제주_{safe_filter_filename(quick_filter)}_선별지역.csv",
                    mime="text/csv",
                    key="dashboard_target_download",
                )
            else:
                st.markdown(f"#### {selected['읍면동명']}")
                st.markdown(
                    f"<div class='detail-badge'>{quick_filter} · 전체 소비 기준</div>"
                    f"<h3>{format_won(selected['지역총매출액'])}</h3>"
                    f"<p>제주 전체 매출 <b>{int(selected['매출순위'])}위</b> · "
                    f"{customer_label} <b>{selected[customer_column]:.1%}</b></p>",
                    unsafe_allow_html=True,
                )
                st.divider()
                st.markdown("**주요 소비 업종**")
                st.markdown(
                    f"<div class='detail-value'>{selected['매출기준_최대업종']}</div>"
                    f"<div class='detail-sub'>업종 매출 비중 "
                    f"{selected['매출기준_최대업종비중']:.1%}</div>",
                    unsafe_allow_html=True,
                )
                st.progress(float(min(selected["매출기준_최대업종비중"], 1)))
                st.download_button(
                    "지역 상세 데이터 다운로드",
                    data=selected.to_frame().T.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"제주_{selected['읍면동명']}_상세.csv",
                    mime="text/csv",
                    key="dashboard_region_download",
                )

    with st.container(border=True):
        st.markdown("##### 4. 지역 상세 조회<span class=\'panel-mark\'></span>", unsafe_allow_html=True)
        st.caption(
            "현재 분석 기준에 해당하는 지역을 목록에서 바로 선택하거나, "
            "전체 지역에서 특정 읍면동을 찾을 수 있습니다."
        )
        detail_options = ["선택 안 함"] + sorted(target_data["읍면동명"].tolist())
        current_detail = st.session_state.get("dashboard_detail_region", "선택 안 함")
        if current_detail not in detail_options:
            current_detail = (
                selected_region if selected_region in detail_options else "선택 안 함"
            )
        if st.session_state.get("dashboard_detail_region") != current_detail:
            st.session_state.dashboard_detail_region = current_detail
        st.selectbox(
            "선별 지역 빠른 조회",
            detail_options,
            index=detail_options.index(current_detail),
            key="dashboard_detail_region",
            on_change=set_dashboard_detail_region,
        )
        with st.expander("전체 지역에서 찾기"):
            region_columns = st.columns(5)
            for index, region in enumerate(region_names):
                region_columns[index % 5].button(
                    region,
                    key=f"dashboard_region_{region}",
                    type="primary" if region == st.session_state.dashboard_region else "secondary",
                    use_container_width=True,
                    on_click=select_dashboard_region,
                    args=(region,),
                )

    with st.expander("분석 기준과 데이터 주의사항"):
        st.write(
            "카드 매출과 차량 도착건수를 읍면동 단위로 비교한 탐색 화면입니다. "
            "차량 도착건수는 실제 체류·소비·동선을 직접 측정하지 않으며, "
            "기간과 목적이 완전히 일치하지 않을 수 있습니다."
        )


try:
    data, _, visitor_summary = load_data()
    boundary_data = load_boundary_data()
except FileNotFoundError as error:
    st.error(f"분석 결과 파일을 찾을 수 없습니다: {error.filename}")
    st.stop()
except (pd.errors.ParserError, UnicodeDecodeError, json.JSONDecodeError) as error:
    st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {error}")
    st.stop()


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');
    :root {
        --bg:#eaf6fd; --surface:#ffffff; --ink:#0d2f42; --basalt:#2f3a3d; --yuchae:#f2c230; --ink-2:#4a5855; --ink-3:#7f8c89;
        --rule:#cfe6f3; --rule-2:#e3f2fa; --accent:#0b84c6; --cD:#2a78d6;
        --sans:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * {
        font-family: var(--sans);
    }
    [data-testid="stIconMaterial"], .material-symbols-rounded { font-family: "Material Symbols Rounded" !important; }
    [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-left: 2rem !important; padding-right: 2rem !important; }
    h1 { font-size: 32px !important; font-weight: 700 !important; letter-spacing: -.015em; color: #06344b !important; }
    h3, h4, h5 { color: var(--ink) !important; font-weight: 700 !important; }
    h5 { font-size: 19px !important; }
    [data-testid="stCaptionContainer"], .stCaption { color: var(--ink-3) !important; }

    /* 사이드바 = 지역별 대시보드의 왼쪽 메뉴 (Streamlit 1.40·최신 버전 모두 대응) */
    [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--rule); }
    [data-testid="stSidebar"] h2 { font-size: 17px !important; font-weight: 800 !important; color: var(--ink) !important; }
    [data-testid="stSidebar"] div[data-testid="stRadio"] > label { display: none; }
    [data-testid="stSidebar"] div[role="radiogroup"] { display: grid !important; gap: 4px; }
    [data-testid="stSidebar"] label[data-baseweb="radio"],
    [data-testid="stSidebar"] label[data-testid="stRadioOption"] {
        background: transparent; border: 0; border-radius: 8px; color: var(--ink-2);
        padding: 10px 12px; margin: 0; cursor: pointer; width: 100%;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:hover,
    [data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover { background: var(--rule-2); }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked),
    [data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {
        background: rgba(11,132,198,.12); color: var(--accent);
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"] > div:not(:has([data-testid="stMarkdownContainer"])),
    [data-testid="stSidebar"] label[data-testid="stRadioOption"] > div:not(:has([data-testid="stMarkdownContainer"])),
    [data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display: none; }
    [data-testid="stSidebar"] label[data-baseweb="radio"] p,
    [data-testid="stSidebar"] label[data-testid="stRadioOption"] p { font-size: 14.5px; color: inherit !important; }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p,
    [data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p { font-weight: 700; color: var(--accent) !important; }

    /* 테두리 상자 = .panel (제목 옆 .panel-mark 표시로 찾음) */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .panel-mark),
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .panel-mark) {
        background: var(--surface);
        border-color: var(--rule) !important;
        border-radius: 10px !important;
    }

    /* 지표 카드 = .ovk .k */
    div[data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
        padding: 14px 18px; box-shadow: none;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p { font-size: 12.5px; color: var(--ink-3); }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 24px; font-weight: 800; color: var(--ink); }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 12.5px; color: var(--ink-2); }

    /* 빠른 분석 기준·심층분석 주제 = .pills */
    [data-testid="stButtonGroup"] { gap: 8px; flex-wrap: wrap; }
    [data-testid="stButtonGroup"] > div { gap: 8px; flex-wrap: wrap; }
    [data-testid="stButtonGroup"] button {
        border: 1px solid var(--rule) !important; background: var(--surface) !important; color: var(--ink) !important;
        border-radius: 8px !important; padding: 8px 16px !important; min-height: 0 !important; box-shadow: none !important;
    }
    [data-testid="stButtonGroup"] button p { font-size: 14px; font-weight: 600; color: inherit; }
    [data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
    [data-testid="stButtonGroup"] button[aria-checked="true"] {
        background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important;
    }

    /* 버튼·선택 상자 */
    .stButton > button, .stDownloadButton > button {
        border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); color: var(--ink);
        font-weight: 600; box-shadow: none;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent); }
    .stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; }
    div[data-baseweb="select"] > div { border-radius: 8px; border-color: var(--rule); background: var(--surface); }
    [data-testid="stExpander"] details { border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); }

    /* 상세 패널 글자 = .ovd */
    .detail-badge {
        display: inline-flex; align-items: center; gap: 6px; background: var(--surface); color: var(--accent);
        border: 1px solid var(--accent); border-radius: 999px; padding: 1px 10px; font-weight: 600; font-size: 12.5px;
    }
    .detail-value { color: var(--ink); font-size: 15px; font-weight: 700; }
    .detail-sub { color: var(--ink-3); font-size: 12.5px; margin-top: 0.2rem; }
    [data-testid="stProgress"] > div > div > div > div { background: var(--accent); }
    hr { border-color: var(--rule) !important; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================================
# 대시보드 탭 = 심층분석 결론 요약 (v3)
# ======================================================================
SUMMARY_FILE = BASE_DIR / "대시보드요약" / "summary.json"
KIND_LABEL = {"both": "두 기준 모두", "spend": "방문 대비 소비", "conc": "시기·업종 집중"}
KIND_HEX = {"spend": "#2a78d6", "conc": "#e08a1e", "both": "#7a4ac4"}
KIND_RGB = {
    "spend": [42, 120, 214, 255],
    "conc": [224, 138, 30, 255],
    "both": [122, 74, 196, 255],
}
OTHER_RGB = [240, 237, 228, 255]
SUMMARY_VIEWS = ["종합", "방문 대비 소비", "시기·업종 집중"]

st.markdown(
    """
    <style>
    .jeju-banner { position: relative; min-height: 150px; border-radius: 16px; overflow: hidden; margin: 2px 0 14px; word-break: keep-all;
        background: radial-gradient(ellipse at 70% 40%, #7fd6fb 0%, #3cbcf7 55%, #22a8ee 100%); border-bottom: 4px solid #ff8a1f; }
    .jeju-banner svg { position: absolute; right: 0; bottom: 0; width: 100%; height: 150px; }
    .jeju-banner .txt { position: relative; z-index: 2; padding: 34px 28px; color: #06344b; max-width: 470px; }
    .jeju-banner .txt b { display: block; font-size: 23px; font-weight: 800; line-height: 1.3; }
    .jeju-banner .txt span { display: block; margin-top: 8px; font-size: 14px; color: #0b4a66; }
    @media (max-width: 700px) {
        .jeju-banner { min-height: 210px; }
        .jeju-banner .txt { padding: 20px 18px 110px; max-width: none; }
        .jeju-banner .txt b { font-size: 19px; }
        .jeju-banner svg { height: 100px; transform: scale(1.7); transform-origin: 100% 100%; }
        h1 { font-size: 25px !important; word-break: keep-all; }
    }
    /* 밝은 톤 */
    [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #d9f1fd 0, #eaf7fd 260px, #f4fbff 100%) !important; }
    [data-testid="stSidebar"] { background: #f4fbff !important; border-right: 1px solid #cfe6f3 !important; border-top: 4px solid #ff8a1f; }
    div[data-testid="stMetric"] { border-top: 4px solid #3cbcf7 !important; border-radius: 16px !important; box-shadow: 0 2px 10px rgba(30,150,220,.10); }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: #cfe6f3 !important; border-radius: 16px !important; }
    .stButton > button { border-radius: 999px !important; border-color: #9fd3ee !important; }
    .sum-head { background: var(--surface); border: 1px solid var(--rule); border-left: 6px solid #ff8a1f;
        border-radius: 16px; box-shadow: 0 2px 10px rgba(30,150,220,.10); padding: 18px 22px; margin: 4px 0 14px; }
    .sum-kick { font-size: 12.5px; font-weight: 700; color: #e8740c; letter-spacing: .02em; }
    .sum-head h2 { font-size: 24px; font-weight: 800; margin: 4px 0 8px; padding: 0; color: var(--ink); line-height: 1.35; }
    .sum-head p { margin: 0 0 4px; color: var(--ink-2); font-size: 15px; line-height: 1.65; }
    .sum-head p.sub { color: var(--ink-3); font-size: 13px; }
    .kchip { display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 12px; font-weight: 700;
        color: #fff; margin: 0 4px 2px 0; white-space: nowrap; }
    .kchip.spend { background: #2a78d6; } .kchip.conc { background: #e08a1e; } .kchip.both { background: #7a4ac4; }
    .sum-legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; color: var(--ink-2); margin: 2px 0 4px; }
    .sum-legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
    [data-testid="stDeckGlJsonChart"] { border-radius: 10px; overflow: hidden; }
    .sum-pane { font-size: 14px; line-height: 1.6; color: var(--ink); }
    .sum-head, .sum-head *, .sum-pane, .sum-pane *, .sum-tbl, .sum-tbl * { word-break: keep-all !important; overflow-wrap: break-word; }
    .sum-pane h4 { font-size: 18px !important; margin: 0 0 6px; }
    .sum-pane .sec { border-top: 1px solid var(--rule-2); padding-top: 8px; margin-top: 8px; }
    .sum-pane .sec b.t { display: block; font-size: 13px; color: var(--ink-3); font-weight: 600; margin-bottom: 2px; }
    .sum-pane .muted, .sum-tbl .muted { color: var(--ink-3); font-size: 12.5px; }
    .sum-pane ul { margin: 2px 0 6px; padding-left: 18px; }
    .sum-wrap { overflow-x: auto; }
    .sum-tbl { width: 100%; border-collapse: collapse; font-size: 13.5px; min-width: 760px; }
    .sum-tbl th { text-align: left; color: var(--ink-3); font-weight: 600; font-size: 12.5px; padding: 8px 10px;
        border-bottom: 1px solid var(--rule); white-space: nowrap; }
    .sum-tbl td { padding: 10px; border-bottom: 1px solid var(--rule-2); vertical-align: top; line-height: 1.55; }
    .sum-tbl td.k { font-weight: 700; white-space: nowrap; }
    .sum-tbl tr.both td { background: rgba(122,74,196,.05); }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_summary() -> dict:
    with SUMMARY_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def region_kind(summary: dict, region: str) -> str | None:
    in_spend, in_conc = region in summary["spend"], region in summary["conc"]
    if in_spend and in_conc:
        return "both"
    if in_spend:
        return "spend"
    if in_conc:
        return "conc"
    return None


def esc(value: object) -> str:
    return html_lib.escape(str(value), quote=True)


def kchips(summary: dict, region: str) -> str:
    kind = region_kind(summary, region)
    if kind is None:
        return ""
    chips = []
    if kind in ("spend", "both"):
        chips.append("<span class='kchip spend'>방문 대비 소비</span>")
    if kind in ("conc", "both"):
        chips.append("<span class='kchip conc'>시기·업종 집중</span>")
    return "".join(chips)


def eok(won: float) -> str:
    return f"{won / 1e8:,.0f}억 원"


MAP_UNIFY_CSS = """
<style id="jeju-map-unify">
.jmap, .ovmap svg { background:#a0d8f4 !important; border-radius:10px; }
.jmap path.rg, .ovmap path.rg { stroke:#ffffff; stroke-width:1; }
.ovmap path.rg.dim { opacity:1 !important; }
.jmap path.rg[fill="var(--land)"], .ovmap path.rg[fill="var(--land)"] { fill:#f0ede4; }
.jmap text.lbl, .ovmap text.lbl { font-size:14px; font-weight:800; fill:#ffffff; paint-order:stroke; stroke:#08263a; stroke-width:5px; stroke-linejoin:round; }
.jmap text.lbl2, .ovmap text.lbl2, .jmap .sea { fill:#2f5f78; }
.jmap .inset { stroke:#2f5f78; }
</style>
<script id="jeju-map-coast">
(function(){
  var NS="http://www.w3.org/2000/svg", queued=false;
  function coast(svg){
    var ps=svg.querySelectorAll("path.rg");
    if(!ps.length||svg.querySelector("g.coast"))return;
    var g=document.createElementNS(NS,"g"); g.setAttribute("class","coast"); g.setAttribute("pointer-events","none");
    ps.forEach(function(p){var c=document.createElementNS(NS,"path");c.setAttribute("d",p.getAttribute("d"));
      c.setAttribute("fill","none");c.setAttribute("stroke","#0b6ea0");c.setAttribute("stroke-width","6");c.setAttribute("stroke-linejoin","round");g.appendChild(c);});
    svg.insertBefore(g,ps[0]);
  }
  function ring(svg){
    var sel=svg.querySelector("path.rg.sel"),old=svg.querySelector("g.selring");
    var key=sel?sel.getAttribute("d").length+":"+sel.getAttribute("d").slice(0,40):"";
    if(old&&old.getAttribute("data-k")===key)return;
    if(old)old.remove();
    if(!sel)return;
    var ps=svg.querySelectorAll("path.rg"),g=document.createElementNS(NS,"g");
    g.setAttribute("class","selring");g.setAttribute("data-k",key);g.setAttribute("pointer-events","none");
    [["#08263a","8",""],["#ffffff","3.4","7 5"]].forEach(function(c){var p=document.createElementNS(NS,"path");p.setAttribute("d",sel.getAttribute("d"));
      p.setAttribute("fill","none");p.setAttribute("stroke",c[0]);p.setAttribute("stroke-width",c[1]);p.setAttribute("stroke-linejoin","round");if(c[2])p.setAttribute("stroke-dasharray",c[2]);g.appendChild(p);});
    ps[ps.length-1].after(g);
  }
  function run(){queued=false;document.querySelectorAll(".jmap, .ovmap svg").forEach(function(n){var sv=n.tagName.toLowerCase()==="svg"?n:n.querySelector("svg");coast(sv);ring(sv);});}
  function q(){if(!queued){queued=true;requestAnimationFrame(run);}}
  new MutationObserver(q).observe(document.documentElement,{childList:true,subtree:true,attributes:true,attributeFilter:["class"]});
  q();
})();
</script>
"""


def with_focus(page: str, region: str | None) -> str:
    """지도 형식을 통일하고, 요약 탭에서 넘어온 지역이 있으면 '지역별 상세' 화면을 그 지역으로 연다."""
    page = page.replace("</body>", MAP_UNIFY_CSS + "</body>", 1)
    if not region:
        return page
    script = (
        "<script>try{go('why'," + json.dumps(region, ensure_ascii=False) + ");}catch(e){}</script>"
    )
    return page.replace("</body>", script + "</body>", 1)


def open_deep(tab: str, region: str) -> None:
    """심층분석의 해당 탭·지역으로 이동한다 (버튼 콜백)."""
    st.session_state["main_menu"] = "심층분석"
    st.session_state["deep_analysis_tab"] = tab
    st.session_state["deep_analysis_last"] = tab
    st.session_state["deep_focus"] = region


def _S(text: str):
    """pydeck 버전에 따라 문자열 리터럴이 식(@@=)으로 바뀌는 것을 막는다."""
    types = getattr(pdk, "types", None)
    return types.String(text) if hasattr(types, "String") else text


def _ring_centroid(ring):
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        c = x0 * y1 - x1 * y0
        a += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    if abs(a) < 1e-12:
        return ring[0][0], ring[0][1], 0.0
    return cx / (3 * a), cy / (3 * a), abs(a) / 2


LABEL_OFFSETS = {  # 제주시 중심부처럼 좁은 곳은 이름표를 바다 쪽으로 비켜 놓는다 (화면 픽셀)
    "도두동": [-52, -26], "삼도1동": [-14, -46], "건입동": [30, -54], "일도2동": [54, -24],
    "오라동": [-28, 4], "아라동": [28, 10], "중문동": [-8, -16], "대천동": [12, 10],
}


def label_points(boundary_data: dict) -> list:
    """각 행정구역의 가장 큰 면 중심에 지역명 라벨 위치를 만든다."""
    out = []
    for f in boundary_data["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        best = max(
            (_ring_centroid([tuple(pt[:2]) for pt in poly[0]]) for poly in polys),
            key=lambda t: t[2],
        )
        size = 13 if best[2] > 4e-4 else (12 if best[2] > 6e-5 else 11)
        name = f["properties"]["행정구역명"]
        out.append({"name": name, "pos": [best[0], best[1]], "size": size, "off": LABEL_OFFSETS.get(name, [0, 0])})
    return out


BANNER_HTML = (
    "<div class='jeju-banner'>"
    '<svg viewBox="0 0 1000 150" preserveAspectRatio="xMaxYMax meet"><defs><radialGradient id="jj_glow"><stop offset="0" stop-color="#fff" stop-opacity=".9"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient><linearGradient id="jj_gm" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#4fae55"/><stop offset="1" stop-color="#2a7d3c"/></linearGradient><radialGradient id="jj_og" cx=".35" cy=".3" r=".8"><stop offset="0" stop-color="#ffb35c"/><stop offset="1" stop-color="#f27a12"/></radialGradient><radialGradient id="jj_rg" cx=".4" cy=".35" r=".8"><stop offset="0" stop-color="#e5484d"/><stop offset="1" stop-color="#b5232b"/></radialGradient><linearGradient id="jj_sg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#7b8798"/><stop offset="1" stop-color="#5b6677"/></linearGradient><filter id="jj_sh" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="3" stdDeviation="2.5" flood-color="#06344b" flood-opacity=".28"/></filter></defs><!-- orange path --><path d="M430 150 L430 120 Q430 104 446 104 L930 104 Q960 104 960 74 Q960 56 940 56" fill="none" stroke="#ff8a1f" stroke-width="8" stroke-linecap="round" opacity="0"/><path d="M440 108 Q470 108 500 108 T600 108 T700 108 T800 108 T900 108 T1000 108" fill="none" stroke="#ff8a1f" stroke-width="8" stroke-linecap="round" opacity="0"/><path d="M0 120 L1000 120" stroke="#ff8a1f" stroke-width="9" stroke-linecap="round" opacity="0"/><path d="M450 126 Q480 126 505 126 L840 126 Q872 126 872 126 L1000 126" fill="none" stroke="#ff8a1f" stroke-width="9" stroke-linecap="round"/><g fill="#ff8a1f"><circle cx="545" cy="126" r="9"/><circle cx="665" cy="126" r="9"/><circle cx="785" cy="126" r="9"/><circle cx="905" cy="126" r="9"/></g><g fill="url(#jj_glow)"><circle cx="545" cy="126" r="20"/><circle cx="665" cy="126" r="20"/><circle cx="785" cy="126" r="20"/><circle cx="905" cy="126" r="20"/></g><g fill="#ff8a1f"><circle cx="545" cy="126" r="9"/><circle cx="665" cy="126" r="9"/><circle cx="785" cy="126" r="9"/><circle cx="905" cy="126" r="9"/></g><!-- 한라산 --><g transform="translate(497,38)" filter="url(#jj_sh)"><path d="M2 80 Q14 62 28 46 L34 36 Q48 22 62 36 L70 46 Q84 62 96 80 Z" fill="url(#jj_gm)"/><path d="M28 46 L34 36 Q48 22 62 36 L70 46 Q64 44 60 50 Q55 42 48 50 Q42 42 36 50 Q32 44 28 46Z" fill="#fff"/><ellipse cx="48" cy="33" rx="10" ry="3.6" fill="#3d9ee8"/></g><!-- 동백 --><g transform="translate(617,38)" filter="url(#jj_sh)"><path d="M12 72 Q22 62 34 66 Q26 78 12 72Z" fill="#3c9a4a"/><path d="M84 72 Q74 62 62 66 Q70 78 84 72Z" fill="#3c9a4a"/><g fill="url(#jj_rg)"><circle cx="48" cy="26" r="19"/><circle cx="30" cy="40" r="19"/><circle cx="66" cy="40" r="19"/><circle cx="36" cy="58" r="18"/><circle cx="60" cy="58" r="18"/></g><circle cx="48" cy="44" r="12" fill="#d9363e"/><g fill="#ffd74a"><circle cx="42" cy="42" r="2.2"/><circle cx="48" cy="38" r="2.2"/><circle cx="54" cy="42" r="2.2"/><circle cx="45" cy="48" r="2.2"/><circle cx="52" cy="48" r="2.2"/></g></g><!-- 귤 --><g transform="translate(737,36)" filter="url(#jj_sh)"><path d="M48 12 Q50 2 62 2 Q58 12 48 14Z" fill="#4fae55"/><rect x="46" y="8" width="4" height="9" rx="2" fill="#3a7d34"/><path d="M48 16 Q70 12 80 34 Q86 58 70 74 Q48 86 26 74 Q10 58 16 34 Q26 12 48 16Z" fill="url(#jj_og)"/><ellipse cx="34" cy="34" rx="8" ry="5" fill="#fff" opacity=".35" transform="rotate(-30 34 34)"/><g fill="#d86a0d" opacity=".55"><circle cx="58" cy="44" r="1.4"/><circle cx="50" cy="56" r="1.4"/><circle cx="64" cy="58" r="1.4"/><circle cx="40" cy="62" r="1.4"/><circle cx="56" cy="30" r="1.4"/></g></g><!-- 돌하르방 --><g transform="translate(857,32)" filter="url(#jj_sh)"><rect x="22" y="10" width="52" height="66" rx="12" fill="url(#jj_sg)"/><path d="M20 22 Q20 4 48 4 Q76 4 76 22 L76 26 L20 26 Z" fill="#4f5a6b"/><path d="M30 6 Q48 -1 66 6" stroke="#fff" stroke-opacity=".25" stroke-width="2.5" fill="none"/><path d="M30 35 Q37 31 43 34" stroke="#2b3340" stroke-width="2.4" stroke-linecap="round" fill="none"/><path d="M66 35 Q59 31 53 34" stroke="#2b3340" stroke-width="2.4" stroke-linecap="round" fill="none"/><ellipse cx="37" cy="44" rx="4.6" ry="5.4" fill="#222a36"/><ellipse cx="59" cy="44" rx="4.6" ry="5.4" fill="#222a36"/><circle cx="38.6" cy="42.2" r="1.7" fill="#fff"/><circle cx="60.6" cy="42.2" r="1.7" fill="#fff"/><ellipse cx="29" cy="54" rx="5" ry="3" fill="#ff9aa2" opacity=".75"/><ellipse cx="67" cy="54" rx="5" ry="3" fill="#ff9aa2" opacity=".75"/><ellipse cx="48" cy="51" rx="4.6" ry="3" fill="#4b5667"/><path d="M39 58 Q48 68 57 58 Q48 62 39 58Z" fill="#2b3340"/><path d="M42 60.5 Q48 65 54 60.5 Q48 62.4 42 60.5Z" fill="#ff8f8f"/><ellipse cx="22" cy="68" rx="9" ry="8" fill="#6b7788"/><ellipse cx="74" cy="68" rx="9" ry="8" fill="#6b7788"/></g></svg>'
    "<div class='txt'><b>제주 관광, 방문을 넘어 소비를 진단하다</b>"
    "<span>43개 읍면동의 방문 대비 소비 수준과 시기·업종별 집중 현상을 분석합니다.</span></div></div>"
)


@st.cache_data
def edge_paths() -> tuple[list, list]:
    """행정구역 경계선을 짧은 선분으로 쪼개 바깥(해안선) / 안쪽(구역 사이)으로 나눈다.
    (경계를 지역 면 레이어에 함께 그리면 지도 클릭 선택이 되지 않아 별도 선 레이어로 그린다.)"""
    import collections

    with GEOJSON_FILE.open(encoding="utf-8") as file:
        gj = json.load(file)
    count: collections.Counter = collections.Counter()
    seg: dict = {}
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for ring in poly:
                for a, b in zip(ring, ring[1:]):
                    pa, pb = (a[0], a[1]), (b[0], b[1])
                    key = tuple(sorted([(round(pa[0], 6), round(pa[1], 6)), (round(pb[0], 6), round(pb[1], 6))]))
                    count[key] += 1
                    seg.setdefault(key, [list(pa), list(pb)])
    outer = [{"path": seg[k]} for k, n in count.items() if n == 1]
    inner = [{"path": seg[k]} for k, n in count.items() if n > 1]
    return outer, inner


def coast_layer(boundary_data: dict) -> pdk.Layer:
    """해안선(바깥 경계)을 짙은 청록 선으로 깔아 섬 윤곽을 또렷하게 한다."""
    return pdk.Layer(
        "PathLayer",
        edge_paths()[0],
        id="jeju-coast",
        pickable=False,
        get_path="path",
        get_color=[11, 110, 160, 255],
        get_width=4,
        width_units=_S("pixels"),
        cap_rounded=True,
        joint_rounded=True,
    )


def sea_layer() -> pdk.Layer:
    """바다를 단색으로 깔아 배경 지도 상태와 상관없이 바다/육지가 분명히 구분되게 한다."""
    box = [[124.6, 32.2], [128.6, 32.2], [128.6, 34.6], [124.6, 34.6]]
    return pdk.Layer(
        "PolygonLayer",
        [{"polygon": box}],
        id="jeju-sea",
        get_polygon="polygon",
        get_fill_color=[160, 216, 244, 255],
        stroked=False,
        pickable=False,
    )


def _dash_features(feats: list) -> dict:
    """선택 지역의 윤곽선을 짧은 선분(점선)으로 잘라 LineString 모음으로 만든다."""
    import math
    lines = []
    for f in feats:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = [tuple(pt[:2]) for pt in poly[0]]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ring, ring[1:])]
            total = sum(seg)
            if total <= 0:
                continue
            n = max(10, round(total / 0.012))
            period = total / n
            on, off = period * 0.6, period * 0.4

            def at(d):
                d %= total
                for (a, b), l in zip(zip(ring, ring[1:]), seg):
                    if d <= l:
                        t = d / l if l else 0
                        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                    d -= l
                return ring[-1]

            for i in range(n):
                start = i * period
                pts = [at(start + on * k / 4) for k in range(5)]
                lines.append({"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [list(p) for p in pts]}})
    return {"type": "FeatureCollection", "features": lines}


def select_rings(boundary_data: dict, selected: str) -> list:
    """선택한 지역 윤곽을 맨 위에 그린다: 남색 실선 바탕 위에 흰 점선. 이웃 지역에 가려지지 않고 어떤 배경에서도 보인다."""
    if not selected:
        return []
    feats = [f for f in boundary_data["features"] if f["properties"]["행정구역명"] == selected]
    if not feats:
        return []
    data = {"type": "FeatureCollection", "features": feats}
    unit = {"line_width_units": _S("pixels")}
    return [
        pdk.Layer("GeoJsonLayer", data, id="jeju-ring-base", pickable=False, stroked=True, filled=False,
                  get_line_color=[8, 38, 58, 255], get_line_width=7, **unit),
        pdk.Layer("GeoJsonLayer", _dash_features(feats), id="jeju-ring-dash", pickable=False, stroked=True, filled=False,
                  get_line_color=[255, 255, 255, 255], get_line_width=3, **unit),
    ]


def build_summary_map(boundary_data: dict, summary: dict, view: str, selected: str) -> pdk.Deck:
    colored = set()
    for feature in boundary_data["features"]:
        properties = feature["properties"]
        region = properties["행정구역명"]
        kind = region_kind(summary, region)
        if view == "종합":
            fill = KIND_RGB[kind] if kind else OTHER_RGB
        elif view == "방문 대비 소비":
            fill = KIND_RGB["spend"] if kind in ("spend", "both") else OTHER_RGB
        else:
            fill = KIND_RGB["conc"] if kind in ("conc", "both") else OTHER_RGB
        if fill != OTHER_RGB:
            colored.add(region)
        properties.update(
            {
                "fill_color": fill,
                "line_color": [255, 255, 255, 230],
                "line_w": 1,
                "상태": KIND_LABEL[kind] if kind else "선별되지 않음",
            }
        )
    layer = pdk.Layer(
        "GeoJsonLayer",
        boundary_data,
        id="jeju-regions",
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 90],
        stroked=False,  # 테두리를 같이 그리면 클릭 선택이 되지 않아 별도 레이어로 그린다
        filled=True,
        get_fill_color="properties.fill_color",
    )
    borders = pdk.Layer(
        "PathLayer",
        edge_paths()[1],
        id="jeju-borders",
        pickable=False,
        get_path="path",
        get_color=[255, 255, 255, 230],
        get_width=1,
        width_units=_S("pixels"),
    )
    labels = pdk.Layer(
        "TextLayer",
        [pt for pt in label_points(boundary_data) if pt["name"] in colored or pt["name"] == selected],
        id="jeju-labels",
        pickable=False,
        get_position="pos",
        get_text="name",
        get_size="size",
        get_pixel_offset="off",
        size_units=_S("pixels"),
        get_color=[255, 255, 255, 255],
        get_text_anchor=_S("middle"),
        get_alignment_baseline=_S("center"),
        font_family=_S('"Pretendard","Noto Sans KR","Malgun Gothic","Apple SD Gothic Neo",sans-serif'),
        font_weight=800,
        character_set=_S("auto"),
        font_settings={"sdf": True, "fontSize": 96, "buffer": 8, "radius": 8},
        background=True,
        get_background_color=[8, 38, 58, 215],
        background_padding=[4, 2],
    )
    return pdk.Deck(
        layers=[sea_layer(), coast_layer(boundary_data), layer, borders, *select_rings(boundary_data, selected), labels],
        initial_view_state=pdk.ViewState(latitude=33.37, longitude=126.56, zoom=9.1, min_zoom=8.8, max_zoom=12.5),
        views=[pdk.View(type="MapView", controller={"dragRotate": False, "touchRotate": False})],
        map_style="light",
        tooltip={
            "html": "<b>{행정구역명}</b><br/>{상태}",
            "style": {
                "backgroundColor": "#ffffff",
                "color": "#15201e",
                "fontSize": "12.5px",
                "border": "1px solid #dde4e2",
                "borderRadius": "8px",
            },
        },
    )


def summary_legend(view: str) -> str:
    items = {
        "종합": [("both", "두 기준 모두"), ("spend", "방문 대비 소비"), ("conc", "시기·업종 집중")],
        "방문 대비 소비": [("spend", "방문 대비 소비 (지역별 분석)")],
        "시기·업종 집중": [("conc", "시기·업종 집중 (업종별 분석)")],
    }[view]
    parts = [f"<span><i style='background:{KIND_HEX[k]}'></i>{label}</span>" for k, label in items]
    parts.append("<span><i style='background:#f0ede4'></i>그 외 지역</span>")
    return "<div class='sum-legend'>" + "".join(parts) + "</div>"


def region_detail_html(summary: dict, region: str) -> str:
    kind = region_kind(summary, region)
    if kind is None:
        return (
            f"<div class='sum-pane'><h4>{esc(region)}</h4>"
            "<p class='muted'>방문 대비 소비·시기·업종 집중 두 기준 모두에서 우선 확인 지역으로 "
            "선별되지 않았습니다.</p></div>"
        )
    out = [f"<div class='sum-pane'><h4>{esc(region)}</h4>{kchips(summary, region)}"]
    if region in summary["spend"]:
        d = summary["spend"][region]
        who = " · ".join(f"{p['who']} {p['grade']}" for p in d["persp"])
        out.append(
            "<div class='sec'><b class='t'>방문 대비 소비 (지역별 분석)</b>"
            f"<div><b>{esc(who)}</b> · {esc(d['character'])} · {esc(d['type'])}"
            + (" <span class='muted'>(잠정)</span>" if d["prov"] else "")
            + "</div><ul>"
            + "".join(f"<li>{esc(p['who'])}: {esc(p['why'])}</li>" for p in d["persp"])
            + "</ul>"
            f"<div><b>대응 방향</b> {esc(d['rx'])}</div>"
            f"<div class='muted'>잠재 소비 약 {eok(d['won'])}/년 · 43곳 중간 수준에 맞춘 가정치이며 손실액이 아닙니다</div></div>"
        )
    if region in summary["conc"]:
        c = summary["conc"][region]
        out.append(
            "<div class='sec'><b class='t'>시기·업종 집중 (업종별 분석)</b>"
            f"<div>{esc(c['fact'])}</div>"
            f"<div><b>확인할 점</b> {esc(c['cause'])}</div>"
            f"<div><b>대응 방향</b> {esc(c['act'])}</div>"
            f"<div class='muted'>3개 구간 연속 집중형 · 최신 구간 9개 기준 조합 중 {c['sen']}개에서 집중형</div></div>"
        )
    out.append("</div>")
    return "".join(out)


def summary_table_html(summary: dict) -> str:
    rows = []
    for region in summary["order"]:
        kind = region_kind(summary, region)
        seen, act, note = [], [], []
        if region in summary["spend"]:
            d = summary["spend"][region]
            who = " · ".join(f"{p['who']} {p['grade']}" for p in d["persp"])
            seen.append(
                f"<div><b>{esc(d['character'])} · {esc(d['type'])}</b>"
                + (" <span class='muted'>(잠정)</span>" if d["prov"] else "")
                + f"<br/>{esc(d['persp'][0]['why'])}</div><div class='muted'>{esc(who)}</div>"
            )
            act.append(f"<div>{esc(d['rx'])}</div>")
            note.append(f"<div>잠재 소비 약 {eok(d['won'])}/년 <span class='muted'>(가정치)</span></div>")
        if region in summary["conc"]:
            c = summary["conc"][region]
            seen.append(f"<div>{esc(c['fact'])}</div>")
            act.append(f"<div>{esc(c['act'])}</div>")
            note.append(f"<div>{esc(c['cause'])}</div>")
        rows.append(
            f"<tr class='{kind}'><td class='k'>{esc(region)}</td><td>{kchips(summary, region)}</td>"
            f"<td>{'<hr/>'.join(seen)}</td><td>{'<hr/>'.join(act)}</td><td>{'<hr/>'.join(note)}</td></tr>"
        )
    return (
        "<div class='sum-wrap'><table class='sum-tbl'><thead><tr><th>지역</th><th>해당 기준</th>"
        "<th>데이터에서 보이는 것</th><th>대응 방향</th><th>규모 · 확인할 점</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_summary_home(boundary_data: dict) -> None:
    summary = load_summary()
    order = summary["order"]
    n_spend, n_conc, both = len(summary["spend"]), len(summary["conc"]), summary["both"]
    both_text = "·".join(both)

    st.markdown(
        "<div class='sum-head'><div class='sum-kick'>핵심 결론</div>"
        f"<h2>{summary['n_regions']}개 읍면동 중 {len(order)}곳을 우선 확인할 필요가 있습니다</h2>"
        f"<p><b>{n_spend}곳</b>은 방문에 비해 소비가 낮고(방문 대비 소비), "
        f"<b>{n_conc}곳</b>은 관광객 소비가 특정 시기·업종에 쏠립니다(시기·업종 집중). "
        f"두 기준에 모두 해당하는 곳은 <b>{both_text}</b> {len(both)}곳입니다.</p>"
        "<p class='sub'>두 분석은 서로 다른 질문을 봅니다. 방문 대비 소비는 방문객 수에 견준 소비 수준이고, "
        "시기·업종 집중은 한 해 안에서 소비가 몰리는 달과 업종을 분석합니다. "
        f"분석 기간 {summary['period']} · 근거는 심층분석 탭에서 확인합니다.</p></div>",
        unsafe_allow_html=True,
    )

    cards = st.columns(4)
    cards[0].metric(
        "방문 대비 소비가 낮은 지역",
        f"{n_spend}곳",
        delta=f"내국인 {summary['n_by']['내국인']} · 외국인 {summary['n_by']['외국인']} (둘 다 {summary['n_by_both']})",
        delta_color="off",
    )
    cards[1].metric("시기·업종 쏠림 지역", f"{n_conc}곳", delta="3개 구간 연속 집중형", delta_color="off")
    cards[2].metric("두 기준 모두 해당", f"{len(both)}곳", delta=both_text, delta_color="off")
    cards[3].metric(
        "잠재 소비 합계(가정치)",
        f"약 {summary['won_total'] / 1e8:,.0f}억 원",
        delta="연간 · 손실액 아님",
        delta_color="off",
    )

    with st.container(border=True):
        st.markdown("##### 1. 우선 확인 지역 지도<span class='panel-mark'></span>", unsafe_allow_html=True)
        st.caption("지도에서 지역을 클릭하거나 오른쪽에서 선택하면 두 분석의 결과가 함께 열립니다. 색칠된 지역만 이름이 표시되며, 나머지는 마우스를 올리면 나타납니다.")
        picked = st.segmented_control(
            "지도 보기",
            SUMMARY_VIEWS,
            default="종합",
            key="summary_view",
            label_visibility="collapsed",
        )
        view = picked or st.session_state.get("summary_view_last", "종합")
        st.session_state["summary_view_last"] = view

        map_col, side_col = st.columns([2.2, 1])
        with map_col:
            selected = st.session_state.get("summary_sel", "")
            deck = build_summary_map(boundary_data, summary, view, selected)
            if "on_select" in inspect.signature(st.pydeck_chart).parameters:
                event = st.pydeck_chart(
                    deck,
                    use_container_width=True,
                    height=640,
                    on_select="rerun",
                    selection_mode="single-object",
                    key="summary_map",
                )
                clicked = get_map_selected_region(event)
                if clicked and clicked != st.session_state.get("summary_last_click"):
                    st.session_state["summary_last_click"] = clicked
                    if clicked != selected:
                        st.session_state["summary_sel"] = clicked
                        st.rerun()
            else:
                st.pydeck_chart(deck, use_container_width=True, height=640)
            st.markdown(summary_legend(view), unsafe_allow_html=True)
        with side_col:
            others = sorted(
                f["properties"]["행정구역명"]
                for f in boundary_data["features"]
                if f["properties"]["행정구역명"] not in order
            )
            st.session_state["summary_pick"] = st.session_state.get("summary_sel", "")
            region = st.selectbox(
                "지역 선택",
                [""] + order + others,
                key="summary_pick",
                on_change=lambda: st.session_state.__setitem__("summary_sel", st.session_state["summary_pick"]),
                format_func=lambda r: (
                    "지역을 선택하세요"
                    if not r
                    else (f"{r} · {KIND_LABEL[region_kind(summary, r)]}" if region_kind(summary, r) else r)
                ),
            )
            if region:
                st.markdown(region_detail_html(summary, region), unsafe_allow_html=True)
                if region in summary["spend"]:
                    st.button(
                        "지역별 분석에서 자세히 보기 ›",
                        key=f"go_reg_{region}",
                        on_click=open_deep,
                        args=("지역별 분석", region),
                    )
                if region in summary["conc"]:
                    st.button(
                        "업종별 분석에서 자세히 보기 ›",
                        key=f"go_ind_{region}",
                        on_click=open_deep,
                        args=("업종별 분석", region),
                    )
            else:
                st.markdown(
                    "<div class='sum-pane'><h4>읽는 법</h4><ul>"
                    "<li><b>보라</b>: 두 기준 모두에서 뽑힌 지역 — 가장 먼저 확인합니다.</li>"
                    "<li><b>파랑</b>: 방문은 많은데 방문당 소비가 낮은 지역입니다.</li>"
                    "<li><b>주황</b>: 소비가 특정 달·업종에 몰리는 지역입니다.</li></ul>"
                    "<p class='muted'>지도 위 버튼으로 한 기준만 골라 볼 수 있습니다.</p></div>",
                    unsafe_allow_html=True,
                )

    with st.container(border=True):
        st.markdown("##### 2. 우선 확인 지역 목록<span class='panel-mark'></span>", unsafe_allow_html=True)
        st.caption("두 기준 모두 해당하는 지역을 먼저, 이어서 방문 대비 소비(잠재 소비 순)와 시기·업종 집중 순입니다.")
        st.markdown(summary_table_html(summary), unsafe_allow_html=True)
        st.caption(
            "방문객은 통신 데이터 추정이고 소비는 카드 결제만 집계합니다. 잠재 소비는 43곳 중간 수준에 맞춘 가정치이며 "
            "손실액이 아닙니다. 시기·업종 집중은 관측된 편중으로 소비 감소나 원인을 확정하지 않으며, "
            "원인·대응은 검증이 필요한 가설입니다."
        )


with st.sidebar:
    st.markdown("## 📊 제주 관광소비 분석")
    st.caption("전체 현황에서 심층 분석까지 단계적으로 확인하세요")
    main_menu = st.radio(
        "메뉴",
        ["대시보드", "심층분석"],
        key="main_menu",
        index=0,
        horizontal=True,
    )
    st.divider()
    if main_menu == "대시보드":
        st.markdown("**결론 요약**")
        st.caption("두 분석의 결론을 먼저 보고, 근거는 심층분석에서 확인합니다.")
    else:
        st.markdown("**심층분석**")
        st.caption("지역별·업종별 주제를 심층적으로 비교합니다.")

if main_menu == "대시보드":
    st.title("제주 관광소비 진단 대시보드")
    st.markdown(BANNER_HTML, unsafe_allow_html=True)
    st.caption("심층분석의 결론만 모아 먼저 보여줍니다. 지역을 고르면 근거 화면으로 이어집니다.")
    render_summary_home(boundary_data)
else:
    tab_names = ["지역별 분석", "업종별 분석"]
    if "deep_analysis_tab" not in st.session_state:
        st.session_state["deep_analysis_tab"] = "지역별 분석"
    deep_tab = st.segmented_control(
        "심층분석 주제",
        tab_names,
        key="deep_analysis_tab",
        label_visibility="collapsed",
    ) or st.session_state.get("deep_analysis_last", "지역별 분석")
    focus_region = st.session_state.pop("deep_focus", None)
    st.session_state["deep_analysis_last"] = deep_tab
    st.markdown(
        "<div style='height: 0.1rem;'></div>",
        unsafe_allow_html=True,
    )

    if deep_tab == "지역별 분석":
        if not REGIONAL_DASHBOARD_FILE.is_file():
            st.error(
                "지역별 분석 HTML 파일을 찾을 수 없습니다: "
                f"{REGIONAL_DASHBOARD_FILE}"
            )
            st.stop()

        components.html(
            with_focus(REGIONAL_DASHBOARD_FILE.read_text(encoding="utf-8"), focus_region),
            height=EMBED_HEIGHT,
            scrolling=True,
        )
    else:
        if not INDUSTRY_DASHBOARD_FILE.is_file():
            st.error(
                "업종별 분석 HTML 파일을 찾을 수 없습니다: "
                f"{INDUSTRY_DASHBOARD_FILE}"
            )
            st.stop()
        components.html(
            with_focus(INDUSTRY_DASHBOARD_FILE.read_text(encoding="utf-8"), focus_region),
            height=EMBED_HEIGHT,
            scrolling=True,
        )
        st.stop()

        st.title("업종별 분석 · 소비 구조와 업종 편중")
        st.caption(
            "도민·관광객 소비 구조와 업종별 매출 집중도를 비교해 "
            "지역의 소비 취약 구조를 진단합니다."
        )
        resident_threshold_value = st.session_state.get("topic1_resident_threshold", 75)
        retail_threshold_value = st.session_state.get("topic1_retail_threshold", 45)
        sales_percentile = st.session_state.get("topic1_sales_percentile", 65)
        resident_threshold = resident_threshold_value / 100
        retail_threshold = retail_threshold_value / 100
        sales_cutoff = data["지역총매출액"].quantile(sales_percentile / 100)
        filtered = data[
            (data["도민매출비중"] >= resident_threshold)
            & (data["소매업매출비중"] >= retail_threshold)
            & (data["지역총매출액"] <= sales_cutoff)
        ].copy()
        filtered["진단"] = filtered.apply(
            lambda row: "도민 소비 + 소매업 집중 + 저매출", axis=1
        )
        filtered = filtered.sort_values("지역총매출액")

        st.subheader("도민 소비와 소매업에 집중된 저매출 지역")
        st.info(
            "이 화면은 대시보드의 빠른 선별 결과를 다시 보여주는 화면이 아니라, "
            "도민 소비·소매업 편중·매출 수준을 함께 사용해 지역별 구조를 세부 진단합니다."
        )
        st.markdown(
            "업종 구성이 **소매업에 편중**된 지역 중, "
            "**도민 소비 비중은 높지만 매출이 낮은 곳**을 우선 문제 지역으로 선정"
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("분석 지역", f"{len(data)}개")
        metric_2.metric("선별 문제 지역", f"{len(filtered)}개")
        metric_3.metric("저매출 기준", format_won(sales_cutoff))
        metric_4.metric("전체 매출 중앙값", format_won(data["지역총매출액"].median()))

        left, right = st.columns([1.45, 1])
        with left:
            st.markdown("#### 지역별 소비 구조 히트맵")
            heatmap_data = data.copy()
            heatmap_data["문제 지역"] = heatmap_data["읍면동명"].isin(
                filtered["읍면동명"]
            ).map({True: "문제 지역", False: "일반 지역"})
            st.altair_chart(build_structure_heatmap(heatmap_data), use_container_width=True)
            st.caption(
                "짙은 색일수록 해당 구조 지표가 높습니다. 문제 지역은 목록과 "
                "진단 카드에서 함께 확인할 수 있습니다."
            )
        with right:
            st.markdown("#### 선별 지역의 총매출")
            if filtered.empty:
                st.info("현재 기준을 만족하는 지역이 없습니다. 사이드바 기준을 완화해 보세요.")
            else:
                chart_data = filtered.set_index("읍면동명")[["지역총매출액"]].copy()
                chart_data["총매출 (억원)"] = chart_data.pop("지역총매출액") / 100_000_000
                st.bar_chart(chart_data, y="총매출 (억원)", height=440)

        st.markdown("#### 문제 지역 · 지역별 원인")
        if filtered.empty:
            st.info("선별된 문제 지역이 없어 진단 카드를 표시할 수 없습니다.")
        else:
            for _, row in filtered.iterrows():
                problem, cause = diagnosis(row)
                with st.container(border=True):
                    st.markdown(f"**문제 지역 | {problem}**")
                    st.markdown(f"**소비 데이터에서 확인된 구조**  \n{cause}")

            st.markdown("#### 선별 지역 통합 진단")
            st.info(integrated_diagnosis(filtered, resident_threshold, retail_threshold))
            st.caption(
                "주의: 이 진단은 소비 데이터에서 함께 관찰된 패턴을 요약한 것입니다. "
                "교통량, 유동인구, 관광자원, 임대료 등 외부 요인을 포함하지 않으므로 "
                "지역별 원인이나 정책 효과를 확정하지 않습니다."
            )

            table = filtered[
                [
                    "읍면동명",
                    "지역총매출액",
                    "매출순위",
                    "도민매출비중",
                    "소매업매출비중",
                    "매출기준_HHI",
                    "매출기준_최대업종",
                ]
            ].rename(
                columns={
                    "읍면동명": "지역",
                    "지역총매출액": "총매출",
                    "매출순위": "매출 순위",
                    "도민매출비중": "도민 매출 비중",
                    "소매업매출비중": "소매업 매출 비중",
                    "매출기준_HHI": "업종 편중도(HHI)",
                    "매출기준_최대업종": "최대 매출 업종",
                }
            )
            table["총매출"] = table["총매출"].map(format_won)
            for column in ["도민 매출 비중", "소매업 매출 비중"]:
                table[column] = table[column].map(lambda value: f"{value:.1%}")
            table["업종 편중도(HHI)"] = table["업종 편중도(HHI)"].map(
                lambda value: f"{value:.3f}"
            )
            st.dataframe(table, use_container_width=True, hide_index=True)

        st.download_button(
            "선별 결과 CSV 다운로드",
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="제주_업종편중_저매출_문제지역.csv",
            mime="text/csv",
            disabled=filtered.empty,
        )
