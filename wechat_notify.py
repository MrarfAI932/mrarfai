#!/usr/bin/env python3
"""
MRARFAI 预警通知系统 v1.0
============================
支持：
  - 企业微信群机器人 (Webhook)
  - 企业微信应用消息 (API)
  - 邮件通知 (复用 pdf_report 的邮件功能)

触发条件：
  - 高风险流失客户
  - 月度异常波动
  - 健康评分 D/F 级
"""

import json
import hashlib
import base64
import time
from datetime import datetime
from typing import Optional

# 企业微信用 requests（如果不可用则标记）
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ============================================================
# 企业微信群机器人 (最简单，只需Webhook URL)
# ============================================================

def send_wecom_bot(
    webhook_url: str,
    title: str,
    content: str,
    mentioned_list: list = None,
) -> tuple:
    """
    发送企业微信群机器人消息
    
    参数:
        webhook_url: 机器人Webhook地址
            格式: https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxx
        title: 标题
        content: Markdown内容
        mentioned_list: @的人 (userid列表, ["@all"] 表示所有人)
    
    返回: (success: bool, message: str)
    """
    if not HAS_REQUESTS:
        return False, "需要安装 requests: pip install requests"
    
    if not webhook_url:
        return False, "请配置企业微信群机器人 Webhook URL"
    
    # Markdown格式消息
    md_content = f"## {title}\n\n{content}"
    
    if mentioned_list:
        md_content += "\n\n" + " ".join(f"<@{u}>" for u in mentioned_list)
    
    payload = {
        "msgtype": "markdown",
        "markdown": {
            "content": md_content
        }
    }
    
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        result = resp.json()
        if result.get('errcode') == 0:
            return True, "企业微信通知发送成功"
        else:
            return False, f"发送失败: {result.get('errmsg', '未知错误')}"
    except Exception as e:
        return False, f"请求失败: {str(e)}"


def send_wecom_bot_card(
    webhook_url: str,
    title: str,
    description: str,
    url: str = "",
) -> tuple:
    """发送卡片消息（更醒目）"""
    if not HAS_REQUESTS:
        return False, "需要安装 requests"
    
    payload = {
        "msgtype": "template_card",
        "template_card": {
            "card_type": "text_notice",
            "main_title": {"title": title},
            "sub_title_text": description[:120],
            "card_action": {
                "type": 1,
                "url": url or "https://mrarfai-oezd879u8nrtxnyunxyyuq.streamlit.app"
            }
        }
    }
    
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        result = resp.json()
        if result.get('errcode') == 0:
            return True, "卡片通知发送成功"
        return False, f"发送失败: {result.get('errmsg')}"
    except Exception as e:
        return False, f"请求失败: {str(e)}"


# ============================================================
# 预警消息构建
# ============================================================

def build_risk_alert_message(results: dict, health_scores: list = None) -> dict:
    """
    从分析结果构建预警消息
    
    返回: {
        'title': str,
        'content': str,        # Markdown格式
        'level': str,          # critical / warning / info
        'summary': str,        # 一句话摘要
    }
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    alerts = results.get('流失预警', [])
    high_risk = [a for a in alerts if '高' in a.get('风险', '')]
    
    # 判断预警等级
    if len(high_risk) >= 3:
        level = 'critical'
        emoji = '🔴'
    elif high_risk:
        level = 'warning'
        emoji = '🟡'
    else:
        level = 'info'
        emoji = '🟢'
    
    lines = [f"**{emoji} MRARFAI 预警报告** | {now}\n"]
    
    # 高风险客户
    if high_risk:
        lines.append(f"### ⚠️ 高风险客户 ({len(high_risk)}家)")
        total_risk_amount = 0
        for a in high_risk[:5]:
            amt = a.get('年度金额', 0)
            total_risk_amount += amt
            lines.append(
                f"- **{a['客户']}** | ¥{amt:,.0f}万 | "
                f"{a.get('原因', '趋势下滑')}"
            )
        lines.append(f"\n> 高风险总金额: **¥{total_risk_amount:,.0f}万**\n")
    
    # 中风险
    med_risk = [a for a in alerts if '中' in a.get('风险', '') and a not in high_risk]
    if med_risk:
        lines.append(f"### 🟡 中风险客户 ({len(med_risk)}家)")
        for a in med_risk[:3]:
            lines.append(f"- {a['客户']} | ¥{a.get('年度金额', 0):,.0f}万")
    
    # 健康评分D/F
    if health_scores:
        df_scores = [s for s in health_scores if s['等级'] in ('D', 'F')]
        if df_scores:
            lines.append(f"\n### 🔴 健康评分低 ({len(df_scores)}家)")
            for s in df_scores[:5]:
                tags = ' '.join(s.get('风险标签', []))
                lines.append(f"- **{s['客户']}** | {s['总分']}分({s['等级']}级) | {tags}")
    
    # 异常波动
    anomalies = results.get('MoM异常', [])
    if anomalies:
        big_drops = [a for a in anomalies if a.get('变动率', 0) < -0.3]
        if big_drops:
            lines.append(f"\n### 📉 月度暴跌 ({len(big_drops)}次)")
            for a in big_drops[:3]:
                lines.append(
                    f"- {a.get('客户', '?')} | "
                    f"{a.get('月份', '?')} | "
                    f"{a.get('变动率', 0)*100:+.0f}%"
                )
    
    # 行动建议
    lines.append("\n### 📋 建议行动")
    if high_risk:
        lines.append(f"1. 立即安排拜访: {', '.join(a['客户'] for a in high_risk[:3])}")
    if med_risk:
        lines.append(f"2. 本周关注: {', '.join(a['客户'] for a in med_risk[:3])}")
    lines.append(f"3. [查看完整报告]"
                 f"(https://mrarfai-oezd879u8nrtxnyunxyyuq.streamlit.app)")
    
    content = "\n".join(lines)
    summary = f"{emoji} {len(high_risk)}个高风险 | {len(med_risk)}个中风险 | {len(alerts)}个预警"
    
    return {
        'title': f'MRARFAI 预警 | {summary}',
        'content': content,
        'level': level,
        'summary': summary,
    }


def build_weekly_digest(data: dict, results: dict, health_scores: list = None) -> dict:
    """构建每周摘要消息"""
    now = datetime.now().strftime('%Y-%m-%d')
    total = data.get('总营收', 0)
    yoy = data.get('总YoY', {})
    growth = yoy.get('增长率', 0) * 100
    
    lines = [f"**📊 MRARFAI 周报** | {now}\n"]
    lines.append(f"### 核心指标")
    lines.append(f"- 全年营收: **¥{total:,.0f}万** ({growth:+.1f}% YoY)")
    
    active = sum(1 for c in data.get('客户金额', []) if c.get('年度金额', 0) > 0)
    lines.append(f"- 活跃客户: **{active}家**")
    
    if health_scores:
        avg = sum(s['总分'] for s in health_scores) / len(health_scores)
        lines.append(f"- 平均健康分: **{avg:.1f}/100**")
    
    # 核心发现
    findings = results.get('核心发现', [])
    if findings:
        lines.append(f"\n### 本周发现")
        for f in findings[:3]:
            lines.append(f"- {f}")
    
    content = "\n".join(lines)
    return {
        'title': f'MRARFAI 周报 | {now}',
        'content': content,
        'level': 'info',
        'summary': f'营收{total:,.0f}万 | {active}家活跃',
    }


# ============================================================
# Streamlit 渲染
# ============================================================

def render_notification_settings(results: dict, health_scores: list = None):
    """在Streamlit中渲染通知配置和预览"""
    import streamlit as st
    
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
        <div style="font-size:1.5rem;">🔔</div>
        <div>
            <div style="font-size:1.1rem; font-weight:700; color:#FFFFFF; font-family:'Space Grotesk',sans-serif;">预警通知</div>
            <div style="font-size:0.8rem; color:#6a6a6a; font-family:'JetBrains Mono',monospace;">
                企业微信群机器人 · 自动推送风险预警
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚙️ 企业微信配置")
        webhook = st.text_input(
            "Webhook URL",
            value=st.session_state.get('wecom_webhook', ''),
            placeholder="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxx",
            help="企业微信群 → 群设置 → 群机器人 → 添加 → 复制Webhook地址"
        )
        if webhook:
            st.session_state['wecom_webhook'] = webhook
        
        mention = st.text_input(
            "@提醒 (可选)",
            placeholder="userid1,userid2 或 @all",
            help="输入企业微信userid，多个用逗号分隔"
        )
        
        st.markdown("")
        st.markdown("**推送设置**")
        auto_alert = st.toggle("🔴 高风险自动推送", value=True, 
                               help="发现高风险客户时自动推送")
        weekly_digest = st.toggle("📊 每周摘要", value=False,
                                  help="每周自动发送分析周报")
    
    with col2:
        st.markdown("#### 📱 预览 & 手动发送")
        
        msg_type = st.radio(
            "消息类型",
            ["⚠️ 风险预警", "📊 每周摘要"],
            horizontal=True,
            key="notify_msg_type"
        )
        
        if msg_type == "⚠️ 风险预警":
            msg = build_risk_alert_message(results, health_scores)
        else:
            # 需要data，但这里只有results，构建简化版
            msg = {
                'title': 'MRARFAI 周报预览',
                'content': '（需要完整数据才能生成周报）',
                'level': 'info',
                'summary': '周报预览',
            }
        
        # 预览
        st.markdown(f"""
        <div style="padding:12px 16px; background:#080808;
             border:1px solid rgba(255,255,255,0.06);
             max-height:300px; overflow-y:auto; font-size:0.82rem; color:#8a8a8a;
             font-family:'JetBrains Mono',monospace;">
            <div style="font-weight:700; margin-bottom:8px; color:#FFFFFF;">
                {msg['title']}
            </div>
            <div style="white-space:pre-wrap; line-height:1.6;">
{msg['content'][:800]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 发送按钮
        if webhook:
            if st.button("📨 立即发送", type="primary", use_container_width=True):
                mention_list = None
                if mention:
                    mention_list = [m.strip() for m in mention.split(',')]
                
                ok, result_msg = send_wecom_bot(
                    webhook_url=webhook,
                    title=msg['title'],
                    content=msg['content'],
                    mentioned_list=mention_list,
                )
                if ok:
                    st.success(f"✅ {result_msg}")
                else:
                    st.error(f"❌ {result_msg}")
        else:
            st.info("👈 请先配置 Webhook URL")
        
        # 配置说明
        with st.expander("📖 如何获取 Webhook URL", expanded=False):
            st.markdown("""
            1. 打开**企业微信** → 进入目标群
            2. 点击群名称 → **群机器人** → **添加**
            3. 填写机器人名称（如 "MRARFAI预警"）
            4. 复制 **Webhook地址**，粘贴到上方
            
            格式：`https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxx`
            
            ⚠️ 请妥善保管Webhook地址，泄露后他人可向群发消息
            """)
