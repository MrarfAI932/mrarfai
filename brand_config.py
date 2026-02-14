#!/usr/bin/env python3
"""
MRARFAI 品牌配置系统 v1.0
============================
白标方案 — 换Logo、公司名、主题色即可变成任何公司的产品

使用方式：
  1. 修改 BRAND_CONFIG 中的字段
  2. 替换 logo 文件
  3. 重启即可
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
    
    # 主题色
    "colors": {
        "primary": "#7c3aed",       # 紫色主色
        "primary_light": "#a78bfa",
        "secondary": "#06b6d4",     # 青色辅助色
        "accent": "#f59e0b",        # 强调色
        "success": "#10b981",
        "warning": "#f97316",
        "danger": "#ef4444",
        "bg_dark": "#0f172a",       # 深色背景
        "bg_card": "#1e293b",       # 卡片背景
        "text_primary": "#f8fafc",
        "text_secondary": "#94a3b8",
        "text_muted": "#64748b",
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
            "primary": "#2563eb",
            "primary_light": "#60a5fa",
            "secondary": "#0891b2",
            "bg_dark": "#0c1222",
            "bg_card": "#1a2332",
        },
    },
    
    "green_tech": {
        **BRAND_CONFIG,
        "company_name": "YourCompany",
        "product_name": "Revenue Intelligence",
        "colors": {
            **BRAND_CONFIG["colors"],
            "primary": "#059669",
            "primary_light": "#34d399",
            "secondary": "#0891b2",
            "bg_dark": "#0a1a14",
            "bg_card": "#132a20",
        },
    },
    
    "red_enterprise": {
        **BRAND_CONFIG,
        "company_name": "YourCompany",
        "product_name": "Sales Command Center",
        "colors": {
            **BRAND_CONFIG["colors"],
            "primary": "#dc2626",
            "primary_light": "#f87171",
            "secondary": "#ea580c",
            "bg_dark": "#1a0a0a",
            "bg_card": "#2a1515",
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
    return BRAND_CONFIG["colors"].get(key, "#7c3aed")


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
        --brand-bg-card: {colors['bg_card']};
        --brand-text-primary: {colors['text_primary']};
        --brand-text-secondary: {colors['text_secondary']};
        --brand-text-muted: {colors['text_muted']};
    }}
    </style>
    """


def render_brand_settings():
    """在Streamlit中渲染品牌设置面板"""
    import streamlit as st
    
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
        <div style="font-size:1.5rem;">🎨</div>
        <div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">品牌与白标配置</div>
            <div style="font-size:0.8rem; color:#64748b;">
                修改品牌信息，一键变成你的产品
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏢 基础信息")
        new_company = st.text_input("公司名称", value=BRAND_CONFIG["company_name"])
        new_product = st.text_input("产品名称", value=BRAND_CONFIG["product_name"])
        new_subtitle = st.text_input("产品副标题", value=BRAND_CONFIG["product_subtitle"])
        new_stock = st.text_input("底部标识", value=BRAND_CONFIG["stock_code"])
        
        st.markdown("#### 🎨 主题色")
        new_primary = st.color_picker("主色", value=BRAND_CONFIG["colors"]["primary"])
        new_secondary = st.color_picker("辅助色", value=BRAND_CONFIG["colors"]["secondary"])
        new_accent = st.color_picker("强调色", value=BRAND_CONFIG["colors"]["accent"])
    
    with col2:
        st.markdown("#### 🏭 行业配置")
        new_industry = st.text_input("行业", value=BRAND_CONFIG["industry"]["name"])
        new_unit = st.text_input("金额单位", value=BRAND_CONFIG["industry"]["revenue_unit"])
        new_qty_unit = st.text_input("数量单位", value=BRAND_CONFIG["industry"]["quantity_unit"])
        competitors = st.text_input(
            "竞争对手（逗号分隔）",
            value=", ".join(BRAND_CONFIG["industry"]["competitors"])
        )
        
        st.markdown("#### ⚙️ 功能开关")
        for feat, enabled in BRAND_CONFIG["features"].items():
            BRAND_CONFIG["features"][feat] = st.toggle(
                feat, value=enabled, key=f"brand_feat_{feat}"
            )
    
    st.markdown("---")
    
    # 预设模板
    st.markdown("#### 📦 预设主题模板")
    template_cols = st.columns(3)
    templates = [
        ("blue_corporate", "🔵 企业蓝", "#2563eb"),
        ("green_tech", "🟢 科技绿", "#059669"),
        ("red_enterprise", "🔴 商务红", "#dc2626"),
    ]
    for i, (key, name, color) in enumerate(templates):
        with template_cols[i]:
            st.markdown(f"""
            <div style="padding:12px; background:{color}15; border:1px solid {color}30;
                 border-radius:10px; text-align:center; cursor:pointer;">
                <div style="font-size:1.2rem; margin-bottom:4px;">{name}</div>
                <div style="width:100%; height:8px; background:{color}; border-radius:4px;"></div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"应用 {name}", key=f"apply_{key}", use_container_width=True):
                template = BRAND_TEMPLATES[key]
                BRAND_CONFIG.update(template)
                st.toast(f"已应用 {name} 主题")
                st.rerun()
    
    # 预览
    st.markdown("---")
    st.markdown("#### 👁️ 预览")
    st.markdown(f"""
    <div style="padding:20px; background:{BRAND_CONFIG['colors']['bg_card']};
         border:1px solid {BRAND_CONFIG['colors']['primary']}30;
         border-radius:14px;">
        <div style="font-size:1.3rem; font-weight:800; color:{BRAND_CONFIG['colors']['text_primary']};">
            {BRAND_CONFIG['logo_emoji']} {BRAND_CONFIG['company_name']}
        </div>
        <div style="font-size:0.8rem; color:{BRAND_CONFIG['colors']['text_secondary']};">
            {BRAND_CONFIG['product_name']}
        </div>
        <div style="margin-top:12px; display:flex; gap:8px;">
            <div style="padding:4px 12px; background:{BRAND_CONFIG['colors']['primary']};
                 border-radius:6px; font-size:0.75rem; color:white;">
                ✨ {BRAND_CONFIG['version']}
            </div>
            <div style="padding:4px 12px; background:{BRAND_CONFIG['colors']['secondary']}20;
                 border:1px solid {BRAND_CONFIG['colors']['secondary']}40;
                 border-radius:6px; font-size:0.75rem; color:{BRAND_CONFIG['colors']['secondary']};">
                {BRAND_CONFIG['industry']['name']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
