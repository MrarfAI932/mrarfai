#!/usr/bin/env python3
"""
MRARFAI PDF Report Generator + Email Sender
=============================================
自动生成专业PDF分析报告 + 邮件推送
"""

import os
import io
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

logger = logging.getLogger("mrarfai.pdf_report")

# ============================================================
# 颜色系统
# ============================================================
INDIGO = HexColor("#6366f1")
INDIGO_LIGHT = HexColor("#a5b4fc")
CYAN = HexColor("#22d3ee")
GREEN = HexColor("#10b981")
RED = HexColor("#ef4444")
ORANGE = HexColor("#f59e0b")
DARK_BG = HexColor("#0f172a")
DARK2 = HexColor("#1e293b")
TEXT_PRIMARY = HexColor("#1e293b")
TEXT_SECONDARY = HexColor("#475569")
TEXT_MUTED = HexColor("#94a3b8")
WHITE = HexColor("#ffffff")
LIGHT_BG = HexColor("#f8fafc")
BORDER = HexColor("#e2e8f0")

# ============================================================
# 中文字体处理
# ============================================================
def _register_chinese_font():
    """尝试注册中文字体，返回可用字体名"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # 尝试常见中文字体路径
    font_paths = [
        # macOS
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/STSong.ttf",
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont("ChineseFont", fp))
                return "ChineseFont"
            except Exception as e:
                logger.debug(f"Font path {fp} registration failed: {type(e).__name__}")
                continue
    
    # 如果没找到，返回默认字体（中文会显示为方块，但不会crash）
    return "Helvetica"

CN_FONT = _register_chinese_font()
CN_FONT_BOLD = CN_FONT  # TTF通常不分粗体

# ============================================================
# 样式定义
# ============================================================
def _get_styles():
    """返回报告用的所有样式"""
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='CoverTitle',
        fontName=CN_FONT, fontSize=28, leading=34,
        textColor=DARK_BG, alignment=TA_CENTER,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name='CoverSubtitle',
        fontName=CN_FONT, fontSize=14, leading=20,
        textColor=TEXT_SECONDARY, alignment=TA_CENTER,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name='SectionTitle',
        fontName=CN_FONT, fontSize=16, leading=22,
        textColor=INDIGO, spaceAfter=10, spaceBefore=16,
    ))
    styles.add(ParagraphStyle(
        name='SubSection',
        fontName=CN_FONT, fontSize=12, leading=16,
        textColor=TEXT_PRIMARY, spaceAfter=6, spaceBefore=10,
    ))
    styles.add(ParagraphStyle(
        name='BodyText2',
        fontName=CN_FONT, fontSize=10, leading=15,
        textColor=TEXT_SECONDARY, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name='SmallNote',
        fontName=CN_FONT, fontSize=8, leading=11,
        textColor=TEXT_MUTED, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name='MetricValue',
        fontName=CN_FONT, fontSize=20, leading=24,
        textColor=INDIGO, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name='MetricLabel',
        fontName=CN_FONT, fontSize=9, leading=12,
        textColor=TEXT_MUTED, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name='TableHeader',
        fontName=CN_FONT, fontSize=9, leading=12,
        textColor=WHITE, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name='TableCell',
        fontName=CN_FONT, fontSize=9, leading=12,
        textColor=TEXT_PRIMARY, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name='Footer',
        fontName=CN_FONT, fontSize=7, leading=10,
        textColor=TEXT_MUTED, alignment=TA_CENTER,
    ))
    
    return styles


# ============================================================
# 辅助函数
# ============================================================
def _make_table(headers, rows, col_widths=None):
    """创建统一风格的表格"""
    s = _get_styles()
    
    # 表头
    header_cells = [Paragraph(h, s['TableHeader']) for h in headers]
    # 数据行
    data_rows = []
    for row in rows:
        data_rows.append([Paragraph(str(c), s['TableCell']) for c in row])
    
    table_data = [header_cells] + data_rows
    
    if col_widths:
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
    else:
        t = Table(table_data, repeatRows=1)
    
    # 样式
    style_commands = [
        # 表头背景
        ('BACKGROUND', (0, 0), (-1, 0), INDIGO),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, -1), CN_FONT),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        # 对齐
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # 边框
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, INDIGO),
        # 内边距
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]
    
    # 斑马纹
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_commands.append(('BACKGROUND', (0, i), (-1, i), LIGHT_BG))
    
    t.setStyle(TableStyle(style_commands))
    return t


def _metric_card(label, value):
    """创建指标卡片"""
    s = _get_styles()
    return [
        Paragraph(str(value), s['MetricValue']),
        Paragraph(label, s['MetricLabel']),
    ]


def _divider():
    """分隔线"""
    return HRFlowable(
        width="100%", thickness=1, 
        color=BORDER, spaceBefore=8, spaceAfter=8,
    )


def _fmt(v):
    """格式化数字"""
    if v is None: return "-"
    try:
        v = float(v)
        if abs(v) >= 100: return f"{v:,.0f}"
        elif abs(v) >= 1: return f"{v:,.1f}"
        else: return f"{v:.2f}"
    except: return str(v)


# ============================================================
# PDF报告生成
# ============================================================
def generate_pdf_report(data, results, benchmark, forecast, output_path=None):
    """
    生成完整的PDF分析报告
    
    参数:
        data: 原始数据字典
        results: 分析结果
        benchmark: 行业对标
        forecast: 预测结果
        output_path: 输出路径（默认自动命名）
    
    返回:
        bytes: PDF二进制数据（可直接下载）
    """
    s = _get_styles()
    now = datetime.now()
    
    if output_path:
        buffer = output_path
    else:
        buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title="MRARFAI Sales Analysis Report",
        author="MRARFAI Agent v4.0",
    )
    
    story = []
    page_width = A4[0] - 4*cm  # 可用宽度
    
    # ========== 封面 ==========
    story.append(Spacer(1, 60))
    
    # Logo区
    story.append(Paragraph("🧠 MRARFAI", s['CoverTitle']))
    story.append(Paragraph("Sales Intelligence Agent", s['CoverSubtitle']))
    story.append(Spacer(1, 30))
    
    story.append(HRFlowable(width="40%", thickness=2, color=INDIGO, spaceBefore=0, spaceAfter=20))
    
    story.append(Paragraph(
        f"<b>禾苗通讯 销售深度分析报告</b>",
        ParagraphStyle('BigTitle', fontName=CN_FONT, fontSize=22, leading=28,
                       textColor=TEXT_PRIMARY, alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Sprocomm Technology · 01401.HK",
        s['CoverSubtitle']
    ))
    story.append(Spacer(1, 40))
    
    # 核心数据一览
    yoy = data['总YoY']
    active = sum(1 for c in data['客户金额'] if c['年度金额'] > 0)
    high_risk = [a for a in results['流失预警'] if '高' in a['风险']]
    
    cover_metrics = [
        ['全年营收', '活跃客户', '增长率', '高风险客户'],
        [f"{data['总营收']:,.0f}万", f"{active}家",
         f"{yoy['增长率']*100:+.1f}%", f"{len(high_risk)}家"],
    ]
    
    cover_table = Table(cover_metrics, colWidths=[page_width/4]*4)
    cover_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), CN_FONT),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, 1), 16),
        ('TEXTCOLOR', (0, 0), (-1, 0), TEXT_MUTED),
        ('TEXTCOLOR', (0, 1), (-1, 1), INDIGO),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 1, BORDER),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, BORDER),
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
    ]))
    story.append(cover_table)
    
    story.append(Spacer(1, 40))
    story.append(Paragraph(
        f"报告生成时间：{now.strftime('%Y年%m月%d日 %H:%M')}",
        s['SmallNote']
    ))
    story.append(Paragraph("Powered by MRARFAI Agent v4.0", s['SmallNote']))
    
    story.append(PageBreak())
    
    # ========== 目录 ==========
    story.append(Paragraph("目录", s['SectionTitle']))
    story.append(_divider())
    
    toc_items = [
        "1. 核心发现与执行摘要",
        "2. 客户分级分析 (ABC)",
        "3. 流失预警",
        "4. 增长机会",
        "5. 价量分解",
        "6. 区域分析",
        "7. 行业对标",
        "8. 营收预测 (Q1 2026)",
        "9. CEO行动建议",
    ]
    for item in toc_items:
        story.append(Paragraph(item, s['BodyText2']))
    
    story.append(PageBreak())
    
    # ========== 1. 核心发现 ==========
    story.append(Paragraph("1. 核心发现与执行摘要", s['SectionTitle']))
    story.append(_divider())
    
    for i, finding in enumerate(results['核心发现']):
        story.append(Paragraph(f"<b>发现 {i+1}：</b>{finding}", s['BodyText2']))
    
    story.append(Spacer(1, 12))
    
    # 月度营收趋势表
    story.append(Paragraph("月度营收趋势 (万元)", s['SubSection']))
    months = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
    m_data = data['月度总营收']
    month_rows = [[months[i], f"{m_data[i]:,.0f}"] for i in range(12)]
    
    # 分两行显示
    row1 = [months[:6], [f"{m_data[i]:,.0f}" for i in range(6)]]
    row2 = [months[6:], [f"{m_data[i]:,.0f}" for i in range(6, 12)]]
    
    m_table = Table(
        [months[:6], [f"{v:,.0f}" for v in m_data[:6]],
         months[6:], [f"{v:,.0f}" for v in m_data[6:]]],
        colWidths=[page_width/6]*6
    )
    m_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), CN_FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 0), (-1, 0), INDIGO),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('BACKGROUND', (0, 2), (-1, 2), INDIGO),
        ('TEXTCOLOR', (0, 2), (-1, 2), WHITE),
        ('TEXTCOLOR', (0, 1), (-1, 1), TEXT_PRIMARY),
        ('TEXTCOLOR', (0, 3), (-1, 3), TEXT_PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(m_table)
    
    # 季度汇总
    story.append(Spacer(1, 10))
    q = [sum(m_data[i:i+3]) for i in range(0, 12, 3)]
    q_table = Table(
        [['Q1', 'Q2', 'Q3', 'Q4'],
         [f"{q[0]:,.0f}", f"{q[1]:,.0f}", f"{q[2]:,.0f}", f"{q[3]:,.0f}"]],
        colWidths=[page_width/4]*4
    )
    q_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), CN_FONT),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, 1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TEXTCOLOR', (0, 0), (-1, 0), TEXT_MUTED),
        ('TEXTCOLOR', (0, 1), (-1, 1), INDIGO),
        ('BOX', (0, 0), (-1, -1), 1, BORDER),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(q_table)
    
    story.append(PageBreak())
    
    # ========== 2. 客户分级 ==========
    story.append(Paragraph("2. 客户分级分析 (ABC)", s['SectionTitle']))
    story.append(_divider())
    
    tiers = results['客户分级']
    tier_counts = {t: sum(1 for x in tiers if x['等级']==t) for t in ['A','B','C']}
    tier_rev = {t: sum(x['年度金额'] for x in tiers if x['等级']==t) for t in ['A','B','C']}
    
    story.append(Paragraph(
        f"A级客户 {tier_counts['A']}家 (营收{tier_rev['A']:,.0f}万, "
        f"占{tier_rev['A']/data['总营收']*100:.1f}%) | "
        f"B级 {tier_counts['B']}家 | C级 {tier_counts['C']}家",
        s['BodyText2']
    ))
    story.append(Spacer(1, 8))
    
    # Top15客户表
    top15 = tiers[:15]
    headers = ['排名', '客户', '等级', '年度金额', '占比', '累计占比']
    rows = []
    for i, t in enumerate(top15):
        rows.append([
            str(i+1), t['客户'], t['等级'],
            f"{t['年度金额']:,.0f}", f"{t['占比']}%", f"{t['累计占比']}%"
        ])
    
    story.append(_make_table(headers, rows, 
        col_widths=[30, page_width*0.25, 30, 80, 50, 55]))
    
    story.append(PageBreak())
    
    # ========== 3. 流失预警 ==========
    story.append(Paragraph("3. 流失预警", s['SectionTitle']))
    story.append(_divider())
    
    alerts = results['流失预警']
    if alerts:
        total_risk = sum(a['年度金额'] for a in alerts)
        story.append(Paragraph(
            f"预警客户 {len(alerts)}家 | 高风险 {len(high_risk)}家 | "
            f"风险金额 {total_risk:,.0f}万 (占总营收 {total_risk/data['总营收']*100:.1f}%)",
            s['BodyText2']
        ))
        story.append(Spacer(1, 8))
        
        headers = ['客户', '风险', '得分', '年度金额', '原因']
        rows = []
        for a in alerts[:15]:
            rows.append([
                a['客户'], a['风险'], str(a.get('得分', '-')),
                f"{a['年度金额']:,.0f}", a.get('原因', '-')
            ])
        story.append(_make_table(headers, rows,
            col_widths=[page_width*0.2, 40, 35, 65, page_width*0.35]))
    else:
        story.append(Paragraph("无高风险预警客户", s['BodyText2']))
    
    story.append(PageBreak())
    
    # ========== 4. 增长机会 ==========
    story.append(Paragraph("4. 增长机会", s['SectionTitle']))
    story.append(_divider())
    
    growth = results['增长机会']
    if growth:
        types = sorted(set(g['类型'] for g in growth))
        type_str = " | ".join([f"{t}: {sum(1 for g in growth if g['类型']==t)}个" for t in types])
        story.append(Paragraph(type_str, s['BodyText2']))
        story.append(Spacer(1, 8))
        
        headers = ['客户', '类型', '金额', '说明']
        rows = []
        for g in growth[:15]:
            rows.append([
                g.get('客户', '-'), g['类型'],
                f"{g.get('金额', 0):,.0f}" if g.get('金额') else '-',
                g.get('说明', '-')
            ])
        story.append(_make_table(headers, rows,
            col_widths=[page_width*0.2, 60, 65, page_width*0.35]))
    else:
        story.append(Paragraph("暂无显著增长信号", s['BodyText2']))
    
    # ========== 5. 价量分解 ==========
    story.append(Spacer(1, 12))
    story.append(Paragraph("5. 价量分解", s['SectionTitle']))
    story.append(_divider())
    
    pv = results['价量分解']
    if pv:
        quality_map = {}
        for p in pv:
            q = p['质量评估']
            if '优质' in q: k = '优质增长'
            elif '以价补量' in q: k = '以价补量'
            elif '量换价' in q: k = '以量换价'
            elif '齐跌' in q: k = '量价齐跌'
            else: k = '价格稳定'
            quality_map[k] = quality_map.get(k, 0) + 1
        
        q_str = " | ".join([f"{k}: {v}家" for k, v in quality_map.items()])
        story.append(Paragraph(q_str, s['BodyText2']))
        story.append(Spacer(1, 8))
        
        headers = ['客户', '年度金额', '均价(元)', '价格变动', '质量评估']
        rows = []
        for p in pv[:12]:
            rows.append([
                p['客户'], f"{p['年度金额']:,.0f}",
                f"{p.get('均价(元)', 0):,.1f}" if p.get('均价(元)') else '-',
                p.get('价格变动', '-'), p['质量评估']
            ])
        story.append(_make_table(headers, rows,
            col_widths=[page_width*0.22, 65, 60, 60, page_width*0.22]))
    
    story.append(PageBreak())
    
    # ========== 6. 区域分析 ==========
    story.append(Paragraph("6. 区域分析", s['SectionTitle']))
    story.append(_divider())
    
    reg = results['区域洞察']
    story.append(Paragraph(
        f"覆盖 {len(reg['详细'])}个区域 | Top3集中度 {reg['Top3集中度']}% | "
        f"HHI指数 {reg['赫芬达尔指数']}"
        f"{'（高度集中）' if reg['赫芬达尔指数'] > 2500 else ''}",
        s['BodyText2']
    ))
    story.append(Spacer(1, 8))
    
    headers = ['区域', '金额', '占比', '客户数']
    rows = []
    for r in reg['详细']:
        rows.append([
            r['区域'], f"{r['金额']:,.0f}",
            f"{r.get('占比', '-')}%", str(r.get('客户数', '-'))
        ])
    story.append(_make_table(headers, rows,
        col_widths=[page_width*0.3, 80, 60, 60]))
    
    # ========== 7. 行业对标 ==========
    story.append(Spacer(1, 12))
    story.append(Paragraph("7. 行业对标", s['SectionTitle']))
    story.append(_divider())
    
    if benchmark:
        # 市场定位
        mp = benchmark.get('市场定位', {})
        for k, v in mp.items():
            story.append(Paragraph(f"<b>{k}：</b>{v}", s['BodyText2']))
        
        story.append(Spacer(1, 8))
        
        # 竞争对标
        cb = benchmark.get('竞争对标', {})
        if cb:
            headers = ['公司', '营收(亿)', '增速', '毛利率']
            rows = []
            for name in ['华勤', '闻泰', '龙旗', '禾苗']:
                rows.append([
                    f"{'→ ' if name=='禾苗' else ''}{name}",
                    str(cb.get('营收', {}).get(name, '-')),
                    str(cb.get('增速', {}).get(name, '-')),
                    str(cb.get('毛利率', {}).get(name, '-')),
                ])
            story.append(_make_table(headers, rows,
                col_widths=[page_width*0.25, 80, 80, 80]))
    
    story.append(PageBreak())
    
    # ========== 8. 营收预测 ==========
    story.append(Paragraph("8. 营收预测 (Q1 2026)", s['SectionTitle']))
    story.append(_divider())
    
    if forecast:
        t = forecast['总营收预测']
        ci = t['置信区间']
        
        pred_table = Table(
            [['乐观 (+15%)', '基准预测', '悲观 (-15%)'],
             [f"{ci['乐观(+15%)']:,.0f}万", f"{ci['基准']:,.0f}万", f"{ci['悲观(-15%)']:,.0f}万"]],
            colWidths=[page_width/3]*3
        )
        pred_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), CN_FONT),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TEXTCOLOR', (0, 0), (-1, 0), TEXT_MUTED),
            ('TEXTCOLOR', (0, 1), (0, 1), GREEN),
            ('TEXTCOLOR', (1, 1), (1, 1), INDIGO),
            ('TEXTCOLOR', (2, 1), (2, 1), RED),
            ('BOX', (0, 0), (-1, -1), 1, BORDER),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(pred_table)
        
        story.append(Spacer(1, 12))
        
        # 情景分析
        story.append(Paragraph("年度情景分析", s['SubSection']))
        scenarios = forecast.get('风险场景', {})
        if scenarios:
            headers = ['情景', '全年预测', '假设']
            rows = []
            for name, sc in scenarios.items():
                rows.append([
                    name.split('(')[0],
                    f"{sc['全年预测']/10000:.1f}亿",
                    sc['假设']
                ])
            story.append(_make_table(headers, rows,
                col_widths=[60, 60, page_width*0.55]))
        
        story.append(Spacer(1, 12))
        
        # Top10客户预测
        story.append(Paragraph("Top10 客户 Q1 预测", s['SubSection']))
        cpred = forecast.get('客户预测', [])
        if cpred:
            headers = ['客户', 'Q4实际', 'Q1预测', '趋势']
            rows = []
            for cp in cpred[:10]:
                rows.append([
                    cp['客户'],
                    f"{cp.get('Q4实际', 0):,.0f}" if cp.get('Q4实际') else '-',
                    f"{cp.get('Q1预测', 0):,.0f}" if cp.get('Q1预测') else '-',
                    cp.get('趋势', '-')
                ])
            story.append(_make_table(headers, rows,
                col_widths=[page_width*0.3, 70, 70, 80]))
    
    story.append(PageBreak())
    
    # ========== 9. CEO行动建议 ==========
    story.append(Paragraph("9. CEO行动建议", s['SectionTitle']))
    story.append(_divider())
    
    # 从各分析中汇总建议
    suggestions = []
    
    # 来自流失预警
    if high_risk:
        hr_names = ", ".join([a['客户'] for a in high_risk[:3]])
        suggestions.append(f"立即关注高流失风险客户：{hr_names}，安排专项拜访")
    
    # 来自增长机会
    if growth:
        top_growth = growth[:2]
        for g in top_growth:
            suggestions.append(f"增长机会 - {g.get('客户', '未知')}：{g.get('说明', g['类型'])}")
    
    # 来自行业对标
    if benchmark:
        for risk in benchmark.get('结构性风险', [])[:2]:
            suggestions.append(f"结构性风险 - {risk['风险']}：{risk['建议']}")
        for opp in benchmark.get('战略机会', [])[:2]:
            suggestions.append(f"战略机会 - {opp['机会']}：{opp['行动']}")
    
    # 来自预测
    if forecast:
        ci = forecast['总营收预测']['置信区间']
        suggestions.append(f"Q1 2026基准预测 {ci['基准']:,.0f}万，确保核心客户订单落实")
    
    for i, s_text in enumerate(suggestions):
        story.append(Paragraph(f"<b>{i+1}.</b> {s_text}", s['BodyText2']))
    
    story.append(Spacer(1, 20))
    story.append(_divider())
    story.append(Paragraph(
        f"— 报告结束 · {now.strftime('%Y年%m月%d日')} · MRARFAI Agent v4.0 —",
        s['SmallNote']
    ))
    
    # ========== 生成 ==========
    doc.build(story)
    
    if isinstance(buffer, io.BytesIO):
        buffer.seek(0)
        return buffer.getvalue()
    return None


# ============================================================
# 邮件发送
# ============================================================
def send_report_email(
    pdf_bytes: bytes,
    to_emails: list,
    subject: str = None,
    body: str = None,
    smtp_server: str = "smtp.qq.com",
    smtp_port: int = 465,
    sender_email: str = "",
    sender_password: str = "",
):
    """
    发送PDF报告邮件
    
    参数:
        pdf_bytes: PDF二进制数据
        to_emails: 收件人列表
        subject: 邮件主题
        body: 邮件正文
        smtp_server: SMTP服务器
        smtp_port: 端口
        sender_email: 发件人邮箱
        sender_password: 授权码（非登录密码）
    
    返回:
        (bool, str): (成功?, 信息)
    """
    now = datetime.now()
    
    if not subject:
        subject = f"禾苗通讯 销售分析报告 - {now.strftime('%Y年%m月%d日')}"
    
    if not body:
        body = f"""您好，

附件为 MRARFAI 自动生成的销售分析报告。

报告包含：
- 核心发现与执行摘要
- 客户分级分析
- 流失预警
- 增长机会
- 营收预测

报告生成时间：{now.strftime('%Y-%m-%d %H:%M')}

—
MRARFAI Sales Intelligence Agent v4.0
"""
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ", ".join(to_emails)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 附件
        attachment = MIMEBase('application', 'pdf')
        attachment.set_payload(pdf_bytes)
        encoders.encode_base64(attachment)
        filename = f"禾苗销售分析_{now.strftime('%Y%m%d')}.pdf"
        attachment.add_header('Content-Disposition', 'attachment', 
                            filename=('utf-8', '', filename))
        msg.attach(attachment)
        
        # 发送
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_emails, msg.as_string())
        server.quit()
        
        return True, f"邮件发送成功，收件人：{', '.join(to_emails)}"
    
    except Exception as e:
        return False, f"发送失败：{str(e)}"


# ============================================================
# Streamlit 集成函数
# ============================================================
def render_report_section(data, results, benchmark, forecast):
    """在Streamlit中渲染报告生成+邮件发送界面"""
    import streamlit as st
    
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
        <div style="font-size:1.5rem;">📄</div>
        <div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">自动PDF报告</div>
            <div style="font-size:0.8rem; color:#64748b;">一键生成专业分析报告 · 支持邮件推送</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 生成PDF报告")
        if st.button("🧠 立即生成报告", type="primary", use_container_width=True):
            with st.spinner("⚡ 正在生成PDF报告..."):
                pdf_bytes = generate_pdf_report(data, results, benchmark, forecast)
            
            if pdf_bytes:
                now = datetime.now().strftime('%Y%m%d')
                st.success("✅ PDF报告生成成功！")
                st.download_button(
                    "📥 下载PDF报告",
                    pdf_bytes,
                    f"禾苗销售分析_{now}.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
                st.session_state['last_pdf'] = pdf_bytes
            else:
                st.error("生成失败，请检查数据")
    
    with col2:
        st.markdown("#### 📧 邮件推送")
        with st.expander("配置邮箱", expanded=False):
            smtp_server = st.selectbox("SMTP服务器", 
                ["smtp.qq.com", "smtp.163.com", "smtp.gmail.com", "smtp.exmail.qq.com"],
                help="企业邮箱用 smtp.exmail.qq.com")
            sender = st.text_input("发件人邮箱", placeholder="your@qq.com")
            password = st.text_input("授权码", type="password", 
                help="QQ邮箱：设置→账户→POP3/SMTP→生成授权码")
            recipients = st.text_area("收件人（每行一个）", 
                placeholder="ceo@sprocomm.com\nmanager@sprocomm.com")
        
        pdf_ready = st.session_state.get('last_pdf')
        if pdf_ready and sender and password and recipients:
            if st.button("📨 发送报告邮件", use_container_width=True):
                to_list = [e.strip() for e in recipients.strip().split('\n') if e.strip()]
                with st.spinner("📤 发送中..."):
                    ok, msg = send_report_email(
                        pdf_bytes=pdf_ready,
                        to_emails=to_list,
                        smtp_server=smtp_server,
                        sender_email=sender,
                        sender_password=password,
                    )
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(msg)
        elif not pdf_ready:
            st.info("👈 先生成PDF报告")
        else:
            st.info("请配置邮箱信息")
