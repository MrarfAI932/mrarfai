#!/usr/bin/env python3
"""
MRARFAI 品牌配置系统 v2.0
============================
白标方案 — 换Logo、公司名、主题色即可变成任何公司的产品
Colors aligned with ui_theme.COLORS design tokens.
"""

# ============================================================
# 品牌配置（修改这里即可白标）
# ============================================================

BRAND_CONFIG = {
    # 基础信息
    "company_name": "MRARFAI",
    "product_name": "Sales Intelligence Agent",
    "product_subtitle": "AI-Powered Sales Analytics Platform",
    "stock_code": "Sprocomm 禾苗通讯 · 01401.HK",
    "version": "v4.3",
    "copyright": "© 2025 MrarfAI. All rights reserved.",

    # Logo（支持URL或本地路径）
    "logo_url": "",  # 留空则用默认emoji logo
    "logo_emoji": "🧠",
    "favicon": "🧠",

    # Theme — Dark + neutral grays
    "colors": {
        "primary": "#FAFAFA",
        "primary_light": "#A1A1AA",
        "secondary": "#71717A",
        "accent": "#FAFAFA",
        "success": "#FAFAFA",
        "warning": "#A1A1AA",
        "danger": "#555555",
        "bg_dark": "#000000",
        "bg_base": "#0A0A0A",
        "bg_card": "#121212",
        "bg_overlay": "#1C1C1C",
        "text_primary": "#FAFAFA",
        "text_secondary": "#A1A1AA",
        "text_tertiary": "#71717A",
        "text_muted": "#52525B",
    },

    # 功能开关（SaaS化：不同客户开放不同功能）
    "features": {
        "multi_agent": True,
        "health_score": True,
        "anomaly_detection": True,
        "wechat_notify": True,
        "pdf_report": True,
        "forecast": True,
        "benchmark": True,
        "ai_narrator": True,
    },

    # 数据配置（SaaS化：不同行业不同术语）
    "industry": {
        "name": "手机ODM/OEM",
        "revenue_unit": "万",
        "quantity_unit": "台",
        "typical_customers": "品牌商",
        "competitors": ["华勤", "闻泰", "龙旗", "天珑"],
    },

    # 侧边栏
    "sidebar": {
        "show_stock_code": True,
        "show_ai_engine": True,
        "show_upload_hint": True,
        "upload_file_types": ["xlsx"],
        "max_file_size_mb": 50,
    },

    # 首页
    "landing": {
        "hero_title": "Sales Intelligence",
        "hero_highlight": "Agent",
        "hero_subtitle": "上传{company}销售数据，用自然语言对话\n获取深度洞察与战略建议",
        "feature_cards": [
            {"icon": "🗣️", "title": "对话式分析", "desc": "用中文提问，Agent 自动选择分析工具，理解数据含义，给出专业建议"},
            {"icon": "📊", "title": "12维深度分析", "desc": "客户分级 · 流失预警 · 价量分解 · 行业对标 · 预测引擎 · CEO备忘录"},
            {"icon": "🔮", "title": "智能预测", "desc": "Q1 2026 营收预测 · 情景分析 · 客户级别预测 · AI 战略叙事"},
        ],
    },
}


# ============================================================
# 预设品牌模板（SaaS客户快速配置）
# ============================================================

BRAND_TEMPLATES = {
    "mrarfai_default": BRAND_CONFIG,

    "blue_corporate": {
        **BRAND_CONFIG,
        "company_name": "YourCompany",
        "product_name": "Sales Analytics Pro",
        "colors": {
            **BRAND_CONFIG["colors"],
            "primary": "#FFFFFF",
            "primary_light": "#A0A0A0",
            "secondary": "#707070",
        },
    },

    "green_tech": {
        **BRAND_CONFIG,
        "company_name": "YourCompany",
        "product_name": "Revenue Intelligence",
        "colors": {
            **BRAND_CONFIG["colors"],
            "primary": "#FFFFFF",
            "primary_light": "#A0A0A0",
            "secondary": "#707070",
        },
    },

    "red_enterprise": {
        **BRAND_CONFIG,
        "company_name": "YourCompany",
        "product_name": "Sales Command Center",
        "colors": {
            **BRAND_CONFIG["colors"],
            "primary": "#FFFFFF",
            "primary_light": "#A0A0A0",
            "secondary": "#707070",
        },
    },
}


# ============================================================
# 辅助函数
# ============================================================

def get_brand():
    """获取当前品牌配置"""
    return BRAND_CONFIG


def get_color(key: str) -> str:
    """获取颜色"""
    return BRAND_CONFIG["colors"].get(key, "#FFFFFF")


def is_feature_enabled(feature: str) -> bool:
    """检查功能是否启用"""
    return BRAND_CONFIG["features"].get(feature, True)


def get_css_variables() -> str:
    """生成CSS变量，注入到页面"""
    colors = BRAND_CONFIG["colors"]
    return f"""
    <style>
    :root {{
        --brand-primary: {colors['primary']};
        --brand-primary-light: {colors['primary_light']};
        --brand-secondary: {colors['secondary']};
        --brand-accent: {colors['accent']};
        --brand-success: {colors['success']};
        --brand-warning: {colors['warning']};
        --brand-danger: {colors['danger']};
        --brand-bg-dark: {colors['bg_dark']};
        --brand-bg-base: {colors['bg_base']};
        --brand-bg-card: {colors['bg_card']};
        --brand-bg-overlay: {colors['bg_overlay']};
        --brand-text-primary: {colors['text_primary']};
        --brand-text-secondary: {colors['text_secondary']};
        --brand-text-tertiary: {colors['text_tertiary']};
        --brand-text-muted: {colors['text_muted']};
    }}
    </style>
    """


def render_brand_settings():
    """在Streamlit中渲染品牌设置面板"""
    import streamlit as st

    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
        <div style="font-size:24px;">🎨</div>
        <div>
            <div style="font-size:18px; font-weight:700; color:#FFFFFF; font-family:'Fira Sans',sans-serif;">品牌与白标配置</div>
            <div style="font-size:14px; color:#707070;">
                修改品牌信息，一键变成你的产品
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 基础信息")
        new_company = st.text_input("公司名称", value=BRAND_CONFIG["company_name"])
        new_product = st.text_input("产品名称", value=BRAND_CONFIG["product_name"])
        new_subtitle = st.text_input("产品副标题", value=BRAND_CONFIG["product_subtitle"])
        new_stock = st.text_input("底部标识", value=BRAND_CONFIG["stock_code"])

        st.markdown("#### 主题色")
        new_primary = st.color_picker("主色", value=BRAND_CONFIG["colors"]["primary"])
        new_secondary = st.color_picker("辅助色", value=BRAND_CONFIG["colors"]["secondary"])
        new_accent = st.color_picker("强调色", value=BRAND_CONFIG["colors"]["accent"])

    with col2:
        st.markdown("#### 行业配置")
        new_industry = st.text_input("行业", value=BRAND_CONFIG["industry"]["name"])
        new_unit = st.text_input("金额单位", value=BRAND_CONFIG["industry"]["revenue_unit"])
        new_qty_unit = st.text_input("数量单位", value=BRAND_CONFIG["industry"]["quantity_unit"])
        competitors = st.text_input(
            "竞争对手（逗号分隔）",
            value=", ".join(BRAND_CONFIG["industry"]["competitors"])
        )

        st.markdown("#### 功能开关")
        for feat, enabled in BRAND_CONFIG["features"].items():
            BRAND_CONFIG["features"][feat] = st.toggle(
                feat, value=enabled, key=f"brand_feat_{feat}"
            )

    st.markdown("---")

    # 预设模板
    st.markdown("#### 预设主题模板")
    template_cols = st.columns(3)
    templates = [
        ("blue_corporate", "企业白", "#FFFFFF"),
        ("green_tech", "科技灰", "#A0A0A0"),
        ("red_enterprise", "商务深", "#707070"),
    ]
    for i, (key, name, color) in enumerate(templates):
        with template_cols[i]:
            st.markdown(f"""
            <div style="padding:12px; background:{color}15; border:1px solid {color}30;
                 border-radius:0; text-align:center; cursor:pointer;">
                <div style="font-size:19px; margin-bottom:4px; font-family:'Fira Sans',sans-serif;">{name}</div>
                <div style="width:100%; height:4px; background:{color};"></div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"应用 {name}", key=f"apply_{key}", use_container_width=True):
                template = BRAND_TEMPLATES[key]
                BRAND_CONFIG.update(template)
                st.toast(f"已应用 {name} 主题")
                st.rerun()

    # 预览
    st.markdown("---")
    st.markdown("#### 预览")
    st.markdown(f"""
    <div style="padding:20px; background:{BRAND_CONFIG['colors']['bg_card']};
         border:1px solid rgba(255,255,255,0.12);
         border-radius:0;">
        <div style="font-size:21px; font-weight:800; color:{BRAND_CONFIG['colors']['text_primary']};
             font-family:'Fira Sans',sans-serif;">
            {BRAND_CONFIG['logo_emoji']} {BRAND_CONFIG['company_name']}
        </div>
        <div style="font-size:14px; color:{BRAND_CONFIG['colors']['text_secondary']};
             font-family:'Fira Sans',sans-serif;">
            {BRAND_CONFIG['product_name']}
        </div>
        <div style="margin-top:12px; display:flex; gap:8px;">
            <div style="padding:4px 12px; background:{BRAND_CONFIG['colors']['primary']};
                 border-radius:0; font-size:13px; color:#060606; font-weight:600;
                 font-family:'Fira Sans',sans-serif; letter-spacing:0.5px;">
                {BRAND_CONFIG['version']}
            </div>
            <div style="padding:4px 12px; background:rgba(255,255,255,0.06);
                 border:1px solid rgba(255,255,255,0.12);
                 border-radius:0; font-size:13px; color:{BRAND_CONFIG['colors']['text_secondary']};
                 font-family:'Fira Sans',sans-serif;">
                {BRAND_CONFIG['industry']['name']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
