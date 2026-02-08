#!/usr/bin/env python3
"""
MRARFAI 销售数据分析 Agent v2.0
===================================
升级内容：
  1. 价量分解（金额=数量×单价，追踪单价趋势）
  2. 同比环比 + 异常检测
  3. 客户-产品-区域交叉分析
  4. 产品结构（FP/SP/PAD）× 订单模式（CKD/SKD/CBU）
  5. 目标达成追踪（季度级别）
  6. AI API 集成（Claude/DeepSeek）
  7. 销售团队深度绩效

Usage:
    python analyze_clients_v2.py --revenue <金额> --quantity <数量> [--ai] [--output <dir>]
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import hashlib
import argparse
from datetime import datetime
from pathlib import Path

MONTHS = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']


# ============================================================
# 模块一：数据加载（增强版 - 全维度）
# ============================================================

class SprocommDataLoaderV2:
    """全维度数据加载器"""

    def __init__(self, revenue_file: str, quantity_file: str):
        self.rf = revenue_file
        self.qf = quantity_file
        self.data = {}

    def load_all(self) -> dict:
        print("=" * 50)
        print("  数据加载中...")
        print("=" * 50)
        self._load_client_revenue()
        self._load_client_quantity()
        self._load_category_yoy()
        self._load_regional()
        self._load_product_breakdown()
        self._load_order_type_distribution()
        self._load_quarterly_targets()
        self._load_yoy_quarterly()
        self._load_sales_team()
        self._compute_price_volume()
        print(f"✅ 加载完成：{len(self.data)} 个数据模块\n")
        return self.data

    def _load_client_revenue(self):
        """客户月度金额明细"""
        df = pd.read_excel(self.rf, sheet_name='2025数据', header=None)
        clients = []
        for i in range(4, 42):
            row = df.iloc[i]
            name = row[1]
            if pd.isna(name) or name == '汇总':
                continue
            monthly = [float(row[j]) if pd.notna(row[j]) else 0.0 for j in range(3, 15)]
            total = float(row[2]) if pd.notna(row[2]) else sum(monthly)
            clients.append({
                '客户': str(name).strip(),
                '销售负责人': str(row[0]).strip() if pd.notna(row[0]) else '',
                '年度金额': total,
                '月度金额': monthly,
                '年度占比': float(row[18]) if pd.notna(row[18]) else 0,
                'local客户占比': float(row[19]) if pd.notna(row[19]) else None,
            })
        total_row = df.iloc[3]
        total_monthly = [float(total_row[j]) if pd.notna(total_row[j]) else 0 for j in range(3, 15)]
        self.data['客户金额'] = clients
        self.data['总营收'] = float(total_row[2]) if pd.notna(total_row[2]) else sum(total_monthly)
        self.data['月度总营收'] = total_monthly

    def _load_client_quantity(self):
        """客户月度数量（计划 vs 实际）"""
        df = pd.read_excel(self.qf, sheet_name='数量汇总', header=None)
        clients = []
        for i in range(3, 29):
            row = df.iloc[i]
            name = row[1]
            if pd.isna(name) or name == '汇总':
                continue
            planned = [float(row[j]) if pd.notna(row[j]) else 0 for j in range(2, 26, 2)]
            actual = [float(row[j]) if pd.notna(row[j]) else 0 for j in range(3, 27, 2)]
            tp = float(row[26]) if pd.notna(row[26]) else sum(planned)
            ta = float(row[27]) if pd.notna(row[27]) else sum(actual)
            clients.append({
                '客户': str(name).strip(),
                '月度计划': planned,
                '月度实际': actual,
                '全年计划': tp,
                '全年实际': ta,
                '完成率': ta / tp if tp > 0 else 0,
            })
        total_row = df.iloc[29]
        self.data['客户数量'] = clients
        self.data['数量汇总'] = {
            '全年计划': float(total_row[26]) if pd.notna(total_row[26]) else 0,
            '全年实际': float(total_row[27]) if pd.notna(total_row[27]) else 0,
            '月度计划': [float(total_row[j]) if pd.notna(total_row[j]) else 0 for j in range(2, 26, 2)],
            '月度实际': [float(total_row[j]) if pd.notna(total_row[j]) else 0 for j in range(3, 27, 2)],
        }

    def _load_category_yoy(self):
        """业务类别同比（2024 vs 2025）"""
        df = pd.read_excel(self.rf, sheet_name='Sheet2', header=None)
        cats = []
        for i in range(1, 7):
            row = df.iloc[i]
            cats.append({
                '类别': str(row[0]),
                '2025金额': float(row[1]) if pd.notna(row[1]) else 0,
                '2025占比': float(row[2]) if pd.notna(row[2]) else 0,
                '2024金额': float(row[3]) if pd.notna(row[3]) else 0,
                '2024占比': float(row[4]) if pd.notna(row[4]) else 0,
                '增长额': float(row[5]) if pd.notna(row[5]) else 0,
                '增长率': float(row[6]) if pd.notna(row[6]) else 0,
                '贡献比变化': float(row[7]) if pd.notna(row[7]) else 0,
            })
        # 汇总
        row7 = df.iloc[7]
        self.data['类别YoY'] = cats
        self.data['总YoY'] = {
            '2025总额': float(row7[1]) if pd.notna(row7[1]) else 0,
            '2024总额': float(row7[3]) if pd.notna(row7[3]) else 0,
            '增长率': float(row7[6]) if pd.notna(row7[6]) else 0,
        }

    def _load_regional(self):
        """区域分布"""
        df = pd.read_excel(self.rf, sheet_name='Sheet3', header=None)
        regions = []
        for i in range(1, len(df)):
            if pd.isna(df.iloc[i, 0]):
                break
            regions.append({
                '区域': str(df.iloc[i, 0]),
                '金额': float(df.iloc[i, 1]) if pd.notna(df.iloc[i, 1]) else 0,
            })
        self.data['区域'] = regions

    def _load_product_breakdown(self):
        """产品类型分布 FP/SP/PAD"""
        df = pd.read_excel(self.qf, sheet_name='数量汇总', header=None)
        products = {}
        for i in range(33, 36):
            row = df.iloc[i]
            ptype = str(row[1]).strip() if pd.notna(row[1]) else ''
            if not ptype or ptype == 'Total':
                continue
            monthly_plan = [float(row[j]) if pd.notna(row[j]) else 0 for j in range(2, 26, 2)]
            monthly_actual = [float(row[j]) if pd.notna(row[j]) else 0 for j in range(3, 27, 2)]
            products[ptype] = {
                '月度计划': monthly_plan,
                '月度实际': monthly_actual,
                '全年计划': float(row[26]) if pd.notna(row[26]) else sum(monthly_plan),
                '全年实际': float(row[27]) if pd.notna(row[27]) else sum(monthly_actual),
            }
        self.data['产品类型'] = products

    def _load_order_type_distribution(self):
        """订单模式分布 CKD/SKD/CBU"""
        df = pd.read_excel(self.qf, sheet_name='数量汇总', header=None)
        orders = {}
        for i in range(33, 39):
            row = df.iloc[i]
            otype = str(row[31]).strip() if pd.notna(row[31]) else ''
            if not otype or otype == 'Total':
                continue
            monthly = [float(row[j]) if pd.notna(row[j]) else 0 for j in range(32, 44)]
            total = float(row[44]) if pd.notna(row[44]) else sum(monthly)
            orders[otype] = {'月度': monthly, '全年': total}
        self.data['订单模式'] = orders

    def _load_quarterly_targets(self):
        """季度目标对比"""
        df = pd.read_excel(self.rf, sheet_name='与年度目标对比', header=None)
        targets = []
        for i in range(3, 9):
            row = df.iloc[i]
            name = str(row[0]).strip() if pd.notna(row[0]) else ''
            if not name or name == '合计':
                continue
            targets.append({
                '客户': name,
                '全年保底': float(row[1]) if pd.notna(row[1]) else 0,
                '占比': float(row[2]) if pd.notna(row[2]) else 0,
                'Q1目标': float(row[3]) if pd.notna(row[3]) else 0,
                'Q2目标': float(row[6]) if pd.notna(row[6]) else 0,
                'Q3目标': float(row[9]) if pd.notna(row[9]) else 0,
                'Q4目标': float(row[12]) if pd.notna(row[12]) else 0,
            })
        # 合计行
        row9 = df.iloc[9]
        self.data['季度目标'] = targets
        self.data['总目标'] = float(row9[1]) if pd.notna(row9[1]) else 0

    def _load_yoy_quarterly(self):
        """2024 vs 2025 季度对比"""
        df = pd.read_excel(self.rf, sheet_name='Sheet1', header=None)
        self.data['季度YoY'] = {
            'Q1': {
                '2025': float(df.iloc[1, 1]) if pd.notna(df.iloc[1, 1]) else 0,
                '2024': float(df.iloc[2, 1]) if pd.notna(df.iloc[2, 1]) else 0,
            },
            'Q2': {
                '2024': float(df.iloc[7, 1]) if pd.notna(df.iloc[7, 1]) else 0,
            },
            'Q3': {
                '2024': float(df.iloc[12, 1]) if pd.notna(df.iloc[12, 1]) else 0,
            }
        }

    def _load_sales_team(self):
        """销售团队维度"""
        df = pd.read_excel(self.rf, sheet_name='2025数据', header=None)
        team = {}
        for i in range(47, 55):
            row = df.iloc[i]
            name = str(row[1]).strip() if pd.notna(row[1]) else ''
            if not name or name == '合计':
                continue
            monthly = [float(row[j]) if pd.notna(row[j]) else 0 for j in range(3, 15)]
            total = float(row[2]) if pd.notna(row[2]) else sum(monthly)
            team[name] = {'月度金额': monthly, '年度总额': total}
        self.data['销售团队'] = team

    def _compute_price_volume(self):
        """价量分解：单价 = 金额 / 数量"""
        pv = []
        for cr in self.data['客户金额']:
            name = cr['客户']
            cq = next((q for q in self.data['客户数量'] if q['客户'] == name), None)
            if not cq or cq['全年实际'] == 0:
                continue
            monthly_price = []
            for m in range(12):
                amt = cr['月度金额'][m]
                qty = cq['月度实际'][m]
                if qty > 0 and amt > 0:
                    # 金额单位万元，数量单位台 => 单价=万元/台*10000=元/台
                    price = amt / qty * 10000
                    monthly_price.append(price)
                else:
                    monthly_price.append(None)

            valid_prices = [p for p in monthly_price if p is not None]
            if len(valid_prices) < 2:
                continue

            avg_price = np.mean(valid_prices)
            # 价格趋势：前半 vs 后半
            h1_prices = [p for p in monthly_price[:6] if p is not None]
            h2_prices = [p for p in monthly_price[6:] if p is not None]
            h1_avg = np.mean(h1_prices) if h1_prices else None
            h2_avg = np.mean(h2_prices) if h2_prices else None

            price_trend = None
            if h1_avg and h2_avg:
                price_trend = (h2_avg - h1_avg) / h1_avg

            pv.append({
                '客户': name,
                '年度金额': cr['年度金额'],
                '年度数量': cq['全年实际'],
                '年均单价': round(avg_price, 2),
                '月度单价': monthly_price,
                'H1均价': round(h1_avg, 2) if h1_avg else None,
                'H2均价': round(h2_avg, 2) if h2_avg else None,
                '价格趋势': price_trend,
                '数量完成率': cq['完成率'],
            })
        self.data['价量分解'] = pv


# ============================================================
# 模块二：深度分析引擎 V2
# ============================================================

class DeepAnalyzer:
    """深度分析引擎 - 7维分析 + 交叉洞察"""

    def __init__(self, data: dict):
        self.d = data
        self.results = {}

    def run_all(self) -> dict:
        print("=" * 50)
        print("  深度分析中...")
        print("=" * 50)
        self.results['客户分级'] = self._tier_clients()
        self.results['流失预警'] = self._churn_detection()
        self.results['增长机会'] = self._growth_opportunities()
        self.results['价量分解'] = self._price_volume_insights()
        self.results['MoM异常'] = self._anomaly_detection()
        self.results['类别趋势'] = self._category_deep_analysis()
        self.results['产品结构'] = self._product_structure()
        self.results['订单模式'] = self._order_type_analysis()
        self.results['区域洞察'] = self._regional_deep()
        self.results['目标达成'] = self._target_tracking()
        self.results['销售绩效'] = self._team_performance()
        self.results['核心发现'] = self._synthesize_findings()
        print(f"✅ 完成：{len(self.results)} 个分析维度\n")
        return self.results

    # ---------- 1. 客户ABC分级 ----------
    def _tier_clients(self):
        clients = sorted(self.d['客户金额'], key=lambda x: x['年度金额'], reverse=True)
        total = self.d['总营收']
        cum = 0
        results = []
        for c in clients:
            if c['年度金额'] <= 0:
                continue
            cum += c['年度金额']
            pct = cum / total
            tier = 'A' if pct <= 0.70 else ('B' if pct <= 0.90 else 'C')
            m = c['月度金额']
            h1, h2 = sum(m[:6]), sum(m[6:])
            q = [sum(m[i:i+3]) for i in range(0, 12, 3)]

            # 活跃月数
            active = sum(1 for v in m if v > 0)
            # 最大单月
            peak_month = MONTHS[m.index(max(m))] if max(m) > 0 else '-'

            results.append({
                '客户': c['客户'], '等级': tier,
                '年度金额': round(c['年度金额'], 2),
                '占比': round(c['年度金额'] / total * 100, 2),
                '累计占比': round(pct * 100, 2),
                'H1': round(h1, 2), 'H2': round(h2, 2),
                'H2vsH1': round((h2/h1-1)*100, 1) if h1 > 0 else None,
                'Q1': round(q[0], 2), 'Q2': round(q[1], 2),
                'Q3': round(q[2], 2), 'Q4': round(q[3], 2),
                '活跃月数': active, '峰值月': peak_month,
                '负责人': c['销售负责人'],
            })
        return results

    # ---------- 2. 流失预警（增强版） ----------
    def _churn_detection(self):
        alerts = []
        for c in self.d['客户金额']:
            if c['年度金额'] <= 0:
                continue
            m = c['月度金额']
            score = 0
            reasons = []

            # R1: 近3月零出货
            zeros = sum(1 for v in m[-3:] if v == 0)
            if zeros >= 1:
                score += 20 * zeros
                reasons.append(f'近3月{zeros}个月零出货')

            # R2: 连续下滑
            declining = 0
            for i in range(len(m)-1, 0, -1):
                if m[i] < m[i-1] and m[i-1] > 0:
                    declining += 1
                else:
                    break
            if declining >= 3:
                score += 25
                reasons.append(f'连续{declining}月下滑')

            # R3: H2 vs H1 大幅下降
            h1, h2 = sum(m[:6]), sum(m[6:])
            if h1 > 0 and h2 < h1 * 0.6:
                score += 25
                pct = (1 - h2/h1) * 100
                reasons.append(f'H2较H1下降{pct:.0f}%')

            # R4: 数量完成率低
            cq = next((q for q in self.d['客户数量'] if q['客户'] == c['客户']), None)
            if cq and cq['全年计划'] > 0 and cq['完成率'] < 0.5:
                score += 20
                reasons.append(f'数量完成率{cq["完成率"]*100:.0f}%')

            # R5: 单价下滑（来自价量分解）
            pv = next((p for p in self.d['价量分解'] if p['客户'] == c['客户']), None)
            if pv and pv['价格趋势'] is not None and pv['价格趋势'] < -0.2:
                score += 15
                reasons.append(f'单价下滑{abs(pv["价格趋势"])*100:.0f}%')

            # R6: 波动过大
            nonzero = [v for v in m if v > 0]
            if len(nonzero) >= 3:
                cv = np.std(nonzero) / np.mean(nonzero)
                if cv > 1.0:
                    score += 10
                    reasons.append(f'出货极不稳定(CV={cv:.1f})')

            if score >= 30:
                level = '🔴高' if score >= 60 else '🟡中'
                alerts.append({
                    '客户': c['客户'], '风险': level, '得分': score,
                    '年度金额': round(c['年度金额'], 2),
                    '原因': '；'.join(reasons),
                    '月度趋势': [round(v, 1) for v in m],
                })
        return sorted(alerts, key=lambda x: x['得分'], reverse=True)

    # ---------- 3. 增长机会 ----------
    def _growth_opportunities(self):
        opps = []
        for c in self.d['客户金额']:
            if c['年度金额'] <= 0:
                continue
            m = c['月度金额']
            h1, h2 = sum(m[:6]), sum(m[6:])

            # 高增长
            if h1 > 0 and h2 > h1 * 1.3:
                rate = (h2/h1-1)*100
                opps.append({'客户': c['客户'], '类型': '🚀高增长',
                            '指标': f'H2较H1增{rate:.0f}%', '金额': round(c['年度金额'], 2)})

            # 超额完成
            cq = next((q for q in self.d['客户数量'] if q['客户'] == c['客户']), None)
            if cq and cq['完成率'] > 1.5 and cq['全年计划'] > 50000:
                opps.append({'客户': c['客户'], '类型': '📈超额完成',
                            '指标': f'完成率{cq["完成率"]*100:.0f}%', '金额': round(c['年度金额'], 2)})

            # 新兴放量
            first = next((i for i, v in enumerate(m) if v > 0), -1)
            if first >= 3 and sum(m[first:]) > 1000:
                opps.append({'客户': c['客户'], '类型': '🌱新兴放量',
                            '指标': f'{MONTHS[first]}起开始出货', '金额': round(c['年度金额'], 2)})

            # 单价上升（量价齐升是最好的信号）
            pv = next((p for p in self.d['价量分解'] if p['客户'] == c['客户']), None)
            if pv and pv['价格趋势'] and pv['价格趋势'] > 0.1 and h2 > h1:
                opps.append({'客户': c['客户'], '类型': '💰量价齐升',
                            '指标': f'单价升{pv["价格趋势"]*100:.0f}%+量增', '金额': round(c['年度金额'], 2)})

        return opps

    # ---------- 4. 价量分解洞察 ----------
    def _price_volume_insights(self):
        pvs = self.d['价量分解']
        insights = []
        for pv in sorted(pvs, key=lambda x: x['年度金额'], reverse=True):
            # 利润质量判断
            if pv['价格趋势'] is not None:
                if pv['价格趋势'] > 0.05:
                    quality = '✅优质增长（量价齐升）' if pv['数量完成率'] > 0.8 else '⚠️以价补量'
                elif pv['价格趋势'] < -0.1:
                    quality = '⚠️以量换价' if pv['数量完成率'] > 1.0 else '❌量价齐跌'
                else:
                    quality = '→价格稳定'
            else:
                quality = '-数据不足'

            insights.append({
                '客户': pv['客户'],
                '年度金额': pv['年度金额'],
                '年度数量': pv['年度数量'],
                '均价(元)': pv['年均单价'],
                'H1均价': pv['H1均价'],
                'H2均价': pv['H2均价'],
                '价格变动': f"{pv['价格趋势']*100:+.1f}%" if pv['价格趋势'] else '-',
                '质量评估': quality,
            })
        return insights

    # ---------- 5. 月度异常检测 ----------
    def _anomaly_detection(self):
        anomalies = []
        # 公司级别
        total_m = self.d['月度总营收']
        for i in range(1, 12):
            if total_m[i-1] > 0:
                mom = (total_m[i] - total_m[i-1]) / total_m[i-1]
                if abs(mom) > 0.3:
                    anomalies.append({
                        '层级': '公司总计', '月份': MONTHS[i],
                        '当月': round(total_m[i], 0), '上月': round(total_m[i-1], 0),
                        '环比': f'{mom*100:+.1f}%',
                        '类型': '📈暴增' if mom > 0 else '📉暴跌',
                    })

        # 客户级别
        for c in self.d['客户金额']:
            if c['年度金额'] < 1000:
                continue
            m = c['月度金额']
            avg = np.mean([v for v in m if v > 0]) if any(v > 0 for v in m) else 0
            if avg == 0:
                continue
            for i in range(12):
                if m[i] > avg * 2.5 and m[i] > 1000:
                    anomalies.append({
                        '层级': c['客户'], '月份': MONTHS[i],
                        '当月': round(m[i], 0), '月均': round(avg, 0),
                        '环比': f'{m[i]/avg:.1f}x均值',
                        '类型': '⚡异常高峰',
                    })
                elif i > 0 and m[i-1] > 1000 and m[i] < m[i-1] * 0.3:
                    anomalies.append({
                        '层级': c['客户'], '月份': MONTHS[i],
                        '当月': round(m[i], 0), '上月': round(m[i-1], 0),
                        '环比': f'{(m[i]/m[i-1]-1)*100:+.0f}%',
                        '类型': '🔻断崖下跌',
                    })
        return anomalies

    # ---------- 6. 类别深度分析 ----------
    def _category_deep_analysis(self):
        cats = self.d['类别YoY']
        total_2025 = self.d['总YoY']['2025总额']
        total_2024 = self.d['总YoY']['2024总额']
        results = []
        for c in cats:
            # 增长贡献度 = 该类别增长额 / 总增长额
            total_growth = total_2025 - total_2024
            contribution = c['增长额'] / total_growth * 100 if total_growth != 0 else 0

            results.append({
                '类别': c['类别'],
                '2025金额': round(c['2025金额'], 2),
                '2024金额': round(c['2024金额'], 2),
                '增长率': f"{c['增长率']*100:.1f}%",
                '增长额': round(c['增长额'], 2),
                '增长贡献度': f"{contribution:.1f}%",
                '占比变化': f"{c['2024占比']*100:.1f}%→{c['2025占比']*100:.1f}%",
                '趋势判断': self._judge_category_trend(c),
            })
        return results

    def _judge_category_trend(self, c):
        rate = c['增长率']
        share_change = c['贡献比变化']
        if rate > 1.0:
            return '🔥爆发式增长，关注产能瓶颈'
        elif rate > 0.3:
            if share_change > 0.03:
                return '📈强劲增长，占比提升，加大投入'
            else:
                return '📈增长良好，但占比未显著提升'
        elif rate > 0:
            return '→温和增长，维持现状'
        elif rate > -0.2:
            return '⚠️小幅下滑，需关注原因'
        else:
            return '🚨大幅下滑，需立即制定挽回计划'

    # ---------- 7. 产品结构分析 ----------
    def _product_structure(self):
        pt = self.d['产品类型']
        total = sum(v['全年实际'] for v in pt.values())
        results = []
        for name, data in pt.items():
            completion = data['全年实际'] / data['全年计划'] if data['全年计划'] > 0 else 0
            # 月度趋势
            actual = data['月度实际']
            h1 = sum(actual[:6])
            h2 = sum(actual[6:])
            results.append({
                '类型': name,
                '全年计划': data['全年计划'],
                '全年实际': data['全年实际'],
                '完成率': f"{completion*100:.1f}%",
                '占比': f"{data['全年实际']/total*100:.1f}%" if total > 0 else '0%',
                'H2vsH1': f"{(h2/h1-1)*100:+.1f}%" if h1 > 0 else '-',
            })
        return results

    # ---------- 8. 订单模式分析 ----------
    def _order_type_analysis(self):
        ot = self.d['订单模式']
        total = sum(v['全年'] for v in ot.values())
        results = []
        for name, data in ot.items():
            m = data['月度']
            h1 = sum(m[:6])
            h2 = sum(m[6:])
            results.append({
                '模式': name,
                '全年数量': data['全年'],
                '占比': f"{data['全年']/total*100:.1f}%" if total > 0 else '0%',
                'H1': h1, 'H2': h2,
                '趋势': f"{(h2/h1-1)*100:+.1f}%" if h1 > 0 else 'N/A',
            })
        return results

    # ---------- 9. 区域深度 ----------
    def _regional_deep(self):
        regions = sorted(self.d['区域'], key=lambda x: x['金额'], reverse=True)
        total = sum(r['金额'] for r in regions)
        cum = 0
        for r in regions:
            r['占比'] = round(r['金额'] / total * 100, 1) if total > 0 else 0
            cum += r['金额']
            r['累计占比'] = round(cum / total * 100, 1)
        top3_pct = regions[2]['累计占比'] if len(regions) >= 3 else 100
        return {
            '详细': regions,
            '总额': total,
            'Top3集中度': top3_pct,
            '赫芬达尔指数': round(sum((r['金额']/total*100)**2 for r in regions), 1) if total > 0 else 0,
        }

    # ---------- 10. 目标达成追踪 ----------
    def _target_tracking(self):
        targets = self.d['季度目标']
        cats = self.d['类别YoY']
        results = []
        for t in targets:
            cat = next((c for c in cats if c['类别'] in t['客户'] or t['客户'] in c['类别']), None)
            actual = cat['2025金额'] if cat else 0
            target = t['全年保底']
            gap = actual - target
            results.append({
                '客户': t['客户'],
                '全年保底目标': round(target, 2),
                '实际完成': round(actual, 2),
                '差异': round(gap, 2),
                '完成率': f"{actual/target*100:.1f}%" if target > 0 else '-',
                'Q分布': f"Q1:{t['Q1目标']:,.0f} Q2:{t['Q2目标']:,.0f} Q3:{t['Q3目标']:,.0f} Q4:{t['Q4目标']:,.0f}",
                '状态': '✅超额' if gap > 0 else ('⚠️差距<20%' if gap > -target*0.2 else '❌严重落后'),
            })
        return results

    # ---------- 11. 销售绩效 ----------
    def _team_performance(self):
        team = self.d['销售团队']
        total = self.d['总营收']
        results = []
        for name, data in sorted(team.items(), key=lambda x: x[1]['年度总额'], reverse=True):
            m = data['月度金额']
            h1, h2 = sum(m[:6]), sum(m[6:])
            peak = MONTHS[m.index(max(m))] if max(m) > 0 else '-'
            # 稳定性
            nonzero = [v for v in m if v > 0]
            cv = np.std(nonzero) / np.mean(nonzero) if len(nonzero) > 1 else 0
            results.append({
                '销售': name,
                '年度总额': round(data['年度总额'], 2),
                '占比': f"{data['年度总额']/total*100:.1f}%",
                'H1': round(h1, 2), 'H2': round(h2, 2),
                'H2增长': f"{(h2/h1-1)*100:+.1f}%" if h1 > 0 else '-',
                '峰值月': peak,
                '稳定性': '稳定' if cv < 0.3 else ('一般' if cv < 0.6 else '波动大'),
            })
        return results

    # ---------- 12. 核心发现综合 ----------
    def _synthesize_findings(self):
        findings = []
        # F1: 总体增长
        yoy = self.d['总YoY']
        findings.append(f"年度总出货金额{self.d['总营收']:,.0f}万元，同比增长{yoy['增长率']*100:.1f}%")

        # F2: HMD下滑
        hmd = next((c for c in self.d['类别YoY'] if c['类别'] == 'HMD'), None)
        if hmd and hmd['增长率'] < 0:
            findings.append(f"⚠️ HMD同比下滑{abs(hmd['增长率'])*100:.1f}%，减少{abs(hmd['增长额']):,.0f}万元，是最大风险点")

        # F3: 最大增长引擎
        cats = sorted(self.d['类别YoY'], key=lambda x: x['增长额'], reverse=True)
        if cats:
            top = cats[0]
            findings.append(f"🔥 {top['类别']}是最大增长引擎，增长{top['增长率']*100:.1f}%，贡献增量{top['增长额']:,.0f}万元")

        # F4: 区域集中度
        reg = self.results.get('区域洞察', {})
        if reg:
            findings.append(f"区域高度集中：Top3占{reg.get('Top3集中度', 0):.1f}%（HHI={reg.get('赫芬达尔指数', 0)}）")

        # F5: 数量缺口
        qs = self.d['数量汇总']
        gap = qs['全年计划'] - qs['全年实际']
        findings.append(f"全年数量缺口{gap:,.0f}台（完成率{qs['全年实际']/qs['全年计划']*100:.1f}%），需排查原因")

        # F6: 价格洞察
        price_up = sum(1 for p in self.d['价量分解'] if p['价格趋势'] and p['价格趋势'] > 0.05)
        price_down = sum(1 for p in self.d['价量分解'] if p['价格趋势'] and p['价格趋势'] < -0.05)
        findings.append(f"单价趋势：{price_up}家客户价格上升，{price_down}家下降，关注以量换价风险")

        # F7: 流失风险
        high_risk = [a for a in self.results.get('流失预警', []) if '高' in a['风险']]
        if high_risk:
            total_at_risk = sum(a['年度金额'] for a in high_risk)
            findings.append(f"🚨 {len(high_risk)}家客户高风险预警，涉及金额{total_at_risk:,.0f}万元")

        return findings


# ============================================================
# 模块三：报告生成器 V2（深度版）
# ============================================================

class ReportGeneratorV2:
    """深度报告生成器"""

    def __init__(self, data: dict, results: dict):
        self.d = data
        self.r = results

    def generate(self) -> str:
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        rpt = f"""# 禾苗通讯 2025年度销售深度分析报告
> **MRARFAI 智能分析系统 v2.0** | {now}
> 数据源：2025年1-12月出货数据（金额+数量双维度）| 单位：万元人民币

---

## 〇、核心发现

"""
        for i, f in enumerate(self.r['核心发现'], 1):
            rpt += f"**{i}.** {f}\n\n"

        # === 一、年度总览 ===
        rpt += self._section_overview()
        # === 二、类别YoY ===
        rpt += self._section_category()
        # === 三、客户分级 ===
        rpt += self._section_tiers()
        # === 四、价量分解 ===
        rpt += self._section_price_volume()
        # === 五、流失预警 ===
        rpt += self._section_churn()
        # === 六、增长机会 ===
        rpt += self._section_growth()
        # === 七、月度异常 ===
        rpt += self._section_anomaly()
        # === 八、产品+订单结构 ===
        rpt += self._section_structure()
        # === 九、区域分布 ===
        rpt += self._section_regional()
        # === 十、目标达成 ===
        rpt += self._section_target()
        # === 十一、销售绩效 ===
        rpt += self._section_team()
        # === 十二、行动清单 ===
        rpt += self._section_actions()

        rpt += """
---
> MRARFAI 销售分析 Agent v2.0 | 分析维度：客户分级 / 价量分解 / 流失预警 / 异常检测 / 产品结构 / 区域集中度 / 目标追踪
> 数据口径：未税/离厂出货，汇率6.4
"""
        return rpt

    def _section_overview(self):
        t = self.d['总营收']
        m = self.d['月度总营收']
        qs = self.d['数量汇总']
        yoy = self.d['总YoY']
        s = f"""
---
## 一、年度业绩总览

| 指标 | 数值 | 同比 |
|------|------|------|
| 全年出货金额 | **{t:,.0f} 万元** | +{yoy['增长率']*100:.1f}%（2024年：{yoy['2024总额']:,.0f}万元） |
| 月均出货 | {t/12:,.0f} 万元 | - |
| 全年出货数量 | {qs['全年实际']:,.0f} 台 | 完成率 {qs['全年实际']/qs['全年计划']*100:.1f}% |
| 计划缺口 | {qs['全年计划']-qs['全年实际']:,.0f} 台 | - |
| 活跃客户 | {sum(1 for c in self.d['客户金额'] if c['年度金额'] > 0)} 家 | - |

### 月度出货金额趋势（万元）
```
"""
        max_v = max(m)
        for i, v in enumerate(m):
            bar = '█' * int(v / max_v * 35)
            s += f"  {MONTHS[i]:4s} | {bar} {v:,.0f}\n"
        s += "```\n\n"

        # 季度对比
        q = [sum(m[i:i+3]) for i in range(0, 12, 3)]
        s += f"**季度分布**：Q1 {q[0]:,.0f} → Q2 {q[1]:,.0f} → Q3 {q[2]:,.0f} → Q4 {q[3]:,.0f}\n\n"
        s += f"**趋势**：Q3为全年高峰（{q[2]:,.0f}万元），Q4回落至{q[3]:,.0f}万元，环比{(q[3]/q[2]-1)*100:+.1f}%\n\n"
        return s

    def _section_category(self):
        s = "---\n## 二、业务类别分析（2024 vs 2025）\n\n"
        s += "| 类别 | 2025年 | 2024年 | 增长率 | 增长额 | 增长贡献度 | 占比变化 | 趋势判断 |\n"
        s += "|------|--------|--------|--------|--------|-----------|---------|----------|\n"
        for c in self.r['类别趋势']:
            s += f"| {c['类别']} | {c['2025金额']:,.0f} | {c['2024金额']:,.0f} | {c['增长率']} | {c['增长额']:,.0f} | {c['增长贡献度']} | {c['占比变化']} | {c['趋势判断']} |\n"
        s += "\n"

        # 重点分析
        hmd = next((c for c in self.r['类别趋势'] if c['类别'] == 'HMD'), None)
        if hmd:
            s += f"**🚨 重点关注 HMD**：同比下滑{hmd['增长率']}，减少{abs(hmd['增长额']):,.0f}万元。"
            s += f"占比从{hmd['占比变化'].split('→')[0]}降至{hmd['占比变化'].split('→')[1]}，"
            s += "需深入分析：是丢了哪些型号订单？竞争对手是谁在接？价格因素还是产品因素？\n\n"

        local = next((c for c in self.r['类别趋势'] if c['类别'] == 'local'), None)
        if local:
            s += f"**💡 亮点 local客户**：增长{local['增长率']}，增量{local['增长额']:,.0f}万元，成功弥补了HMD的下滑。\n\n"
        return s

    def _section_tiers(self):
        s = "---\n## 三、客户分级（ABC分析）\n\n"
        s += "| # | 客户 | 等级 | 年度金额 | 占比 | 累计 | H2vsH1 | 活跃月 | 峰值月 | 负责人 |\n"
        s += "|---|------|------|---------|------|------|--------|--------|--------|--------|\n"
        for i, c in enumerate(self.r['客户分级'][:20], 1):
            h2h1 = f"{c['H2vsH1']:+.0f}%" if c['H2vsH1'] is not None else '-'
            s += f"| {i} | {c['客户']} | {c['等级']} | {c['年度金额']:,.0f} | {c['占比']}% | {c['累计占比']}% | {h2h1} | {c['活跃月数']} | {c['峰值月']} | {c['负责人']} |\n"

        a = sum(1 for c in self.r['客户分级'] if c['等级'] == 'A')
        b = sum(1 for c in self.r['客户分级'] if c['等级'] == 'B')
        cc = sum(1 for c in self.r['客户分级'] if c['等级'] == 'C')
        a_rev = sum(c['年度金额'] for c in self.r['客户分级'] if c['等级'] == 'A')
        s += f"\n**结构**：A级{a}家（{a_rev:,.0f}万元，{a_rev/self.d['总营收']*100:.1f}%）| B级{b}家 | C级{cc}家\n\n"
        s += f"**集中度风险**：前4家客户占比约{self.r['客户分级'][3]['累计占比']}%，客户集中度偏高。\n\n"
        return s

    def _section_price_volume(self):
        s = "---\n## 四、价量分解（核心洞察）\n\n"
        s += '> 单价 = 出货金额 ÷ 出货数量，追踪金额增长是靠走量还是靠提价\n\n'
        s += "| 客户 | 年度金额 | 年度数量 | 均价(元/台) | H1均价 | H2均价 | 价格变动 | 质量评估 |\n"
        s += "|------|---------|---------|------------|--------|--------|---------|----------|\n"
        for p in self.r['价量分解'][:15]:
            s += f"| {p['客户']} | {p['年度金额']:,.0f} | {p['年度数量']:,.0f} | {p['均价(元)']:,.0f} | {p['H1均价'] or '-'} | {p['H2均价'] or '-'} | {p['价格变动']} | {p['质量评估']} |\n"
        s += "\n**关键洞察**：\n"
        up = [p for p in self.r['价量分解'] if '优质' in p['质量评估'] or '量价齐升' in p['质量评估']]
        down = [p for p in self.r['价量分解'] if '量换价' in p['质量评估'] or '齐跌' in p['质量评估']]
        if up:
            s += f"- ✅ 优质增长客户：{'、'.join(p['客户'] for p in up)}\n"
        if down:
            s += f"- ⚠️ 以量换价/齐跌客户：{'、'.join(p['客户'] for p in down)}，需关注利润率\n"
        s += "\n"
        return s

    def _section_churn(self):
        s = "---\n## 五、客户流失预警\n\n"
        if not self.r['流失预警']:
            return s + "无高风险预警。\n\n"
        s += "| 客户 | 风险 | 得分 | 年度金额 | 预警原因 |\n"
        s += "|------|------|------|---------|----------|\n"
        for a in self.r['流失预警'][:12]:
            s += f"| {a['客户']} | {a['风险']} | {a['得分']} | {a['年度金额']:,.0f} | {a['原因']} |\n"
        total_risk = sum(a['年度金额'] for a in self.r['流失预警'])
        high_risk = sum(a['年度金额'] for a in self.r['流失预警'] if '高' in a['风险'])
        s += f"\n**风险敞口**：预警客户合计 {total_risk:,.0f} 万元（其中高风险 {high_risk:,.0f} 万元，占总营收 {high_risk/self.d['总营收']*100:.1f}%）\n\n"
        return s

    def _section_growth(self):
        s = "---\n## 六、增长机会\n\n"
        if not self.r['增长机会']:
            return s + "暂无显著增长信号。\n\n"
        s += "| 客户 | 类型 | 关键指标 | 年度金额 |\n"
        s += "|------|------|---------|----------|\n"
        for g in self.r['增长机会']:
            s += f"| {g['客户']} | {g['类型']} | {g['指标']} | {g['金额']:,.0f} |\n"
        s += "\n"
        return s

    def _section_anomaly(self):
        s = "---\n## 七、月度异常检测\n\n"
        s += "> 自动标记环比波动>30%或偏离均值>2.5倍的数据点\n\n"
        if not self.r['MoM异常']:
            return s + "无显著异常。\n\n"
        s += "| 客户/层级 | 月份 | 类型 | 详情 |\n"
        s += "|----------|------|------|------|\n"
        for a in self.r['MoM异常'][:20]:
            detail = f"当月{a.get('当月', '-'):,.0f}"
            if '上月' in a:
                detail += f" vs 上月{a['上月']:,.0f}"
            if '月均' in a:
                detail += f"（月均{a['月均']:,.0f}）"
            detail += f"，{a['环比']}"
            s += f"| {a['层级']} | {a['月份']} | {a['类型']} | {detail} |\n"
        s += "\n"
        return s

    def _section_structure(self):
        s = "---\n## 八、产品结构 & 订单模式\n\n"
        s += "### 产品类型（FP功能机 / SP智能机 / PAD平板）\n\n"
        s += "| 类型 | 全年计划 | 全年实际 | 完成率 | 占比 | H2vsH1 |\n"
        s += "|------|---------|---------|--------|------|--------|\n"
        for p in self.r['产品结构']:
            s += f"| {p['类型']} | {p['全年计划']:,.0f} | {p['全年实际']:,.0f} | {p['完成率']} | {p['占比']} | {p['H2vsH1']} |\n"

        s += "\n### 订单模式（CKD散件/SKD半散件/CBU整机）\n\n"
        s += "| 模式 | 全年数量 | 占比 | H1 | H2 | 趋势 |\n"
        s += "|------|---------|------|-----|-----|------|\n"
        for o in self.r['订单模式']:
            s += f"| {o['模式']} | {o['全年数量']:,.0f} | {o['占比']} | {o['H1']:,.0f} | {o['H2']:,.0f} | {o['趋势']} |\n"
        s += "\n"
        return s

    def _section_regional(self):
        s = "---\n## 九、区域出货分布\n\n"
        reg = self.r['区域洞察']
        s += f"**覆盖{len(reg['详细'])}个区域** | Top3集中度：{reg['Top3集中度']}% | HHI指数：{reg['赫芬达尔指数']}\n\n"
        s += "| 区域 | 金额(万元) | 占比 | 累计占比 |\n"
        s += "|------|----------|------|--------|\n"
        for r in reg['详细']:
            s += f"| {r['区域']} | {r['金额']:,.0f} | {r['占比']}% | {r['累计占比']}% |\n"
        if reg['赫芬达尔指数'] > 3000:
            s += f"\n**⚠️ 区域高度集中**：HHI={reg['赫芬达尔指数']}（>2500为高度集中），过度依赖单一市场，建议分散风险。\n\n"
        return s

    def _section_target(self):
        s = "---\n## 十、年度目标达成追踪\n\n"
        s += "| 客户/类别 | 全年保底 | 实际完成 | 差异 | 完成率 | 状态 |\n"
        s += "|----------|---------|---------|------|--------|------|\n"
        for t in self.r['目标达成']:
            s += f"| {t['客户']} | {t['全年保底目标']:,.0f} | {t['实际完成']:,.0f} | {t['差异']:+,.0f} | {t['完成率']} | {t['状态']} |\n"
        s += "\n"
        return s

    def _section_team(self):
        s = "---\n## 十一、销售团队绩效\n\n"
        s += "| 销售 | 年度总额 | 占比 | H1 | H2 | H2增长 | 峰值月 | 稳定性 |\n"
        s += "|------|---------|------|-----|-----|--------|--------|--------|\n"
        for t in self.r['销售绩效']:
            s += f"| {t['销售']} | {t['年度总额']:,.0f} | {t['占比']} | {t['H1']:,.0f} | {t['H2']:,.0f} | {t['H2增长']} | {t['峰值月']} | {t['稳定性']} |\n"
        s += "\n"
        return s

    def _section_actions(self):
        s = "---\n## 十二、本周重点行动清单\n\n"
        n = 1

        # 流失客户
        for a in self.r['流失预警'][:3]:
            s += f"**{n}. 【紧急-流失预警】** 回访 **{a['客户']}**（{a['风险']}，{a['年度金额']:,.0f}万元）\n"
            s += f"   原因：{a['原因']}\n\n"
            n += 1

        # HMD专项
        hmd = next((c for c in self.r['类别趋势'] if c['类别'] == 'HMD'), None)
        if hmd and '-' in hmd['增长率']:
            s += f"**{n}. 【战略】** 召开HMD专项分析会，下滑{hmd['增长率']}，定位原因并制定Q1挽回计划\n\n"
            n += 1

        # 增长机会
        for g in self.r['增长机会'][:2]:
            s += f"**{n}. 【机会】** 跟进 **{g['客户']}**（{g['类型']}，{g['指标']}），评估追加合作空间\n\n"
            n += 1

        # 价格
        down = [p for p in self.r['价量分解'] if '量换价' in p['质量评估']]
        if down:
            s += f"**{n}. 【利润】** 排查以量换价客户（{'、'.join(p['客户'] for p in down[:3])}），评估毛利率影响\n\n"
            n += 1

        s += f"**{n}. 【常规】** 更新下月销售预测，对照目标差异调整资源分配\n\n"
        return s

    def generate_ai_prompt(self) -> str:
        """生成 AI 深度分析 Prompt"""
        return f"""你是禾苗通讯（Sprocomm, 01401.HK）的资深销售分析总监，年营收约25亿元的智能终端ODM上市公司。

请基于以下2025年度全维度分析数据，写一份管理层级别的战略分析备忘录。

===== 核心发现 =====
{json.dumps(self.r['核心发现'], ensure_ascii=False, indent=2)}

===== 类别同比分析 =====
{json.dumps(self.r['类别趋势'], ensure_ascii=False, indent=2)}

===== 客户分级 Top15 =====
{json.dumps(self.r['客户分级'][:15], ensure_ascii=False, indent=2)}

===== 价量分解 Top10 =====
{json.dumps(self.r['价量分解'][:10], ensure_ascii=False, indent=2)}

===== 流失预警 =====
{json.dumps(self.r['流失预警'][:8], ensure_ascii=False, indent=2)}

===== 增长机会 =====
{json.dumps(self.r['增长机会'], ensure_ascii=False, indent=2)}

===== 月度异常 =====
{json.dumps(self.r['MoM异常'][:15], ensure_ascii=False, indent=2)}

===== 区域分布 =====
{json.dumps(self.r['区域洞察'], ensure_ascii=False, indent=2)}

===== 目标达成 =====
{json.dumps(self.r['目标达成'], ensure_ascii=False, indent=2)}

请输出：
1. **三段式摘要**（50字→200字→500字，适合不同层级阅读）
2. **HMD下滑深度分析**：可能原因、影响范围、挽回建议
3. **价量分解解读**：哪些客户的增长是健康的，哪些是隐患
4. **最值得投入的3个增长方向**（附具体行动）
5. **区域战略建议**：是否过度依赖印度市场
6. **本月CEO应该关注的3件事**

语气要求：简洁、数据驱动、直接给结论和建议，不要废话。
"""


# ============================================================
# 模块四：AI API 集成
# ============================================================

class AIAnalyst:
    """AI 深度分析接口"""

    @staticmethod
    def call_claude(prompt: str, api_key: str = None) -> str:
        """调用 Claude API 进行深度分析"""
        key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not key:
            return "[未配置 ANTHROPIC_API_KEY，跳过 AI 分析。设置环境变量后重试。]"

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except ImportError:
            return "[需安装 anthropic 库：pip install anthropic]"
        except Exception as e:
            return f"[AI 分析出错：{str(e)}]"

    @staticmethod
    def call_deepseek(prompt: str, api_key: str = None) -> str:
        """调用 DeepSeek API（低成本 fallback）"""
        key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        if not key:
            return "[未配置 DEEPSEEK_API_KEY]"
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[DeepSeek 出错：{str(e)}]"


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='MRARFAI 销售分析 Agent v2.0')
    parser.add_argument('--revenue', required=True, help='金额报表')
    parser.add_argument('--quantity', required=True, help='数量报表')
    parser.add_argument('--output', default='./output_v2', help='输出目录')
    parser.add_argument('--ai', action='store_true', help='启用 AI 深度分析')
    parser.add_argument('--ai-provider', choices=['claude', 'deepseek'], default='claude')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("\n" + "=" * 60)
    print("  MRARFAI 销售数据分析 Agent v2.0 (Deep Analytics)")
    print("  禾苗通讯 Sprocomm | 01401.HK")
    print("=" * 60 + "\n")

    # 1. 加载
    loader = SprocommDataLoaderV2(args.revenue, args.quantity)
    data = loader.load_all()

    # 2. 分析
    analyzer = DeepAnalyzer(data)
    results = analyzer.run_all()

    # 3. 报告
    gen = ReportGeneratorV2(data, results)
    report = gen.generate()

    rpath = os.path.join(args.output, '深度销售分析报告_v2.md')
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"📄 深度报告：{rpath}")

    # 4. AI Prompt
    ai_prompt = gen.generate_ai_prompt()
    ppath = os.path.join(args.output, 'ai_deep_prompt.txt')
    with open(ppath, 'w', encoding='utf-8') as f:
        f.write(ai_prompt)
    print(f"🤖 AI Prompt：{ppath}")

    # 5. 可选：AI 分析
    if args.ai:
        print("\n🧠 正在调用 AI 深度分析...")
        if args.ai_provider == 'claude':
            ai_result = AIAnalyst.call_claude(ai_prompt)
        else:
            ai_result = AIAnalyst.call_deepseek(ai_prompt)

        ai_path = os.path.join(args.output, 'ai_analysis_result.md')
        with open(ai_path, 'w', encoding='utf-8') as f:
            f.write(f"# AI 深度分析报告\n\n{ai_result}")
        print(f"💡 AI 分析结果：{ai_path}")

    # 6. JSON 数据
    jpath = os.path.join(args.output, 'analysis_v2.json')
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"📊 分析数据：{jpath}")

    print("\n" + "=" * 60)
    print("  ✅ 分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
