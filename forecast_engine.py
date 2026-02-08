"""MRARFAI Forecast Engine"""
import numpy as np

class ForecastEngine:
    def __init__(self, data, results):
        self.d = data
        self.r = results

    def run(self):
        return {
            '总营收预测': self._total(),
            '客户预测': self._clients(),
            '品类预测': self._categories(),
            '数量预测': self._quantity(),
            '风险场景': self._scenarios(),
            '关键假设': self._assumptions(),
        }

    def _total(self):
        m = self.d['月度总营收']
        q = [sum(m[i:i+3]) for i in range(0, 12, 3)]
        yoy = self.d.get('季度YoY', {})
        q1_2024 = yoy.get('Q1', {}).get('2024', 62547)
        q1_yoy = (q[0] - q1_2024) / q1_2024 if q1_2024 > 0 else 0
        total_2024 = self.d['总YoY']['2024总额']
        q1_share = q1_2024 / total_2024 if total_2024 > 0 else 0.23
        trend_adj = 1 + (self.d['总YoY']['增长率'] * 0.3)
        seasonal = q[0] / q[3] if q[3] > 0 else 0.8

        e1 = q[0] * (1 + q1_yoy * 0.5)
        e2 = q[3] * seasonal
        e3 = self.d['总营收'] * q1_share * trend_adj
        fc = e1 * 0.35 + e2 * 0.35 + e3 * 0.30

        return {
            'Q1_2026预测': round(fc, 0),
            '方法说明': {
                '同比外推(35%%)': 'Q1_2025(%s) x (1+%.1f%%) = %s' % (
                    format(q[0], ',.0f'), q1_yoy * 50, format(e1, ',.0f')),
                '季节性(35%%)': 'Q4_2025(%s) x 季节因子(%.2f) = %s' % (
                    format(q[3], ',.0f'), seasonal, format(e2, ',.0f')),
                '占比法(30%%)': '全年 x Q1占比(%.2f) x 趋势修正 = %s' % (
                    q1_share, format(e3, ',.0f')),
            },
            '置信区间': {
                '乐观(+15%)': round(fc * 1.15, 0),
                '基准': round(fc, 0),
                '悲观(-15%)': round(fc * 0.85, 0),
            },
            '参考': {
                'Q1_2025实际': round(q[0], 0),
                'Q1_2024实际': round(q1_2024, 0),
                'Q4_2025实际': round(q[3], 0),
            },
        }

    def _clients(self):
        preds = []
        for c in self.r['客户分级'][:10]:
            name = c['客户']
            m = next((x['月度金额'] for x in self.d['客户金额'] if x['客户'] == name), None)
            if not m:
                continue
            q = [sum(m[i:i+3]) for i in range(0, 12, 3)]
            recent = q[3]
            first_active = next((i for i, v in enumerate(m) if v > 0), 12)
            is_new = first_active >= 3
            if is_new:
                ratio = 0.85
            else:
                ratio = q[0] / q[3] if q[3] > 0 else 0.8
                ratio = min(ratio, 2.0)

            h2h1 = c['H2vsH1']
            if h2h1 is not None and h2h1 > 30:
                trend, adj = '上升', 1.10
            elif h2h1 is not None and h2h1 < -30:
                trend, adj = '下降', 0.85
            else:
                trend, adj = '平稳', 1.0

            pred = recent * ratio * adj
            if pred < 0:
                pred = 0

            churn = next((a for a in self.r['流失预警'] if a['客户'] == name), None)
            if churn and '高' in churn['风险']:
                pred *= 0.5
                trend += '(流失折扣)'

            label = '动量' if is_new else '季节'
            preds.append({
                '客户': name,
                'Q4实际': round(q[3], 0),
                'Q1预测': round(pred, 0),
                '趋势': trend,
                '预测逻辑': 'Q4(%s) x %s(%.2f) x 趋势(%.2f)' % (
                    format(q[3], ',.0f'), label, ratio, adj),
            })
        return preds

    def _categories(self):
        preds = []
        for c in self.d.get('类别YoY', []):
            rate = c['增长率']
            if rate > 1.0:
                nr, note = rate * 0.4, '爆发增长不可持续,预计大幅回落'
            elif rate > 0.3:
                nr, note = rate * 0.7, '高增长惯性,速度放缓'
            elif rate > 0:
                nr, note = rate * 0.9, '温和增长延续'
            elif rate > -0.2:
                nr, note = rate * 1.1, '下滑可能加深'
            else:
                nr, note = rate * 0.8, '大幅下滑后可能企稳'
            preds.append({
                '类别': c['类别'],
                '2025增长率': '%+.1f%%' % (rate * 100),
                '2026E增长率': '%+.1f%%' % (nr * 100),
                '预期': note,
            })
        return preds

    def _quantity(self):
        qs = self.d['数量汇总']
        actual = qs['全年实际']
        planned = qs['全年计划']
        monthly = qs['月度实际']
        q1s = sum(monthly[:3]) / actual if actual > 0 else 0.15
        pred = actual * 1.10
        return {
            '2025全年实际': actual,
            '2025完成率': '%.1f%%' % (actual / planned * 100),
            '2026E全年': round(pred, 0),
            '2026E_Q1': round(pred * q1s, 0),
        }

    def _scenarios(self):
        total = self.d['总营收']
        hmd = next((c for c in self.d['类别YoY'] if c['类别'] == 'HMD'), None)
        hmd_rev = hmd['2025金额'] if hmd else 0
        return {
            '乐观(+25%)': {
                '全年预测': round(total * 1.25, 0),
                '假设': 'ZTE/平板持续高增长+HMD企稳+AIOT放量',
                '关键驱动': 'ZYB平板订单翻倍/ZTE海外2-3个新市场突破',
            },
            '基准(+15%)': {
                '全年预测': round(total * 1.15, 0),
                '假设': '平板增长放缓+HMD小幅下滑+local稳定',
                '关键驱动': '现有客户自然增长,无重大变化',
            },
            '悲观(-5%)': {
                '全年预测': round(total * 0.95, 0),
                '假设': 'HMD继续大幅萎缩+ZYB订单缩减+印度政策变化',
                '关键驱动': 'HMD再跌50%%(损失约%s万)+印度关税上调' % format(hmd_rev * 0.5, ',.0f'),
            },
            '极端(-20%)': {
                '全年预测': round(total * 0.80, 0),
                '假设': 'ZYB/ZTE同时缩减+HMD退出+地缘风险爆发',
                '关键驱动': '多个大客户同时下滑,需要紧急拓展新客户',
            },
        }

    def _assumptions(self):
        return [
            '预测基于2025年数据外推,未考虑未公开的新客户pipeline',
            '季节性因子基于2025年单年数据,样本量有限',
            '行业趋势参考IDC/Counterpoint公开预测',
            '情景分析用于风险评估,实际取决于客户订单确认',
            'HMD预测基于品牌全球趋势+禾苗份额流失双重因素',
        ]


def generate_forecast_section(forecast):
    f = forecast
    s = "---\n## 附二、2026年前瞻预测\n\n"

    t = f['总营收预测']
    s += "### 1. Q1 2026 营收预测\n\n"
    s += "| 场景 | Q1预测(万元) |\n|------|------------|\n"
    for k, v in t['置信区间'].items():
        marker = ' <-' if '基准' in k else ''
        s += "| %s | %s%s |\n" % (k, format(v, ',.0f'), marker)
    s += "\n**预测方法(三因子加权)**:\n\n"
    for k, v in t['方法说明'].items():
        s += "- %s: %s\n" % (k, v)
    s += "\n**参考**: Q1_2025实际 %s万 | Q4_2025 %s万\n\n" % (
        format(t['参考']['Q1_2025实际'], ',.0f'),
        format(t['参考']['Q4_2025实际'], ',.0f'))

    s += "### 2. Top10 客户 Q1 预测\n\n"
    s += "| 客户 | Q4实际 | Q1预测 | 趋势 | 逻辑 |\n|------|--------|--------|------|------|\n"
    for p in f['客户预测']:
        s += "| %s | %s | %s | %s | %s |\n" % (
            p['客户'], format(p['Q4实际'], ',.0f'),
            format(p['Q1预测'], ',.0f'), p['趋势'], p['预测逻辑'])

    s += "\n### 3. 业务类别增长预测\n\n"
    s += "| 类别 | 2025增长 | 2026E增长 | 预期 |\n|------|---------|----------|------|\n"
    for p in f['品类预测']:
        s += "| %s | %s | %s | %s |\n" % (
            p['类别'], p['2025增长率'], p['2026E增长率'], p['预期'])

    s += "\n### 4. 2026年度情景分析\n\n"
    for sc, data in f['风险场景'].items():
        s += "**%s** -> %s万元\n- 假设: %s\n- 驱动: %s\n\n" % (
            sc, format(data['全年预测'], ',.0f'), data['假设'], data['关键驱动'])

    s += "### 5. 关键假设\n\n"
    for a in f['关键假设']:
        s += "- %s\n" % a
    s += "\n"
    return s
