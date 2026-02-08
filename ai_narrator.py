"""MRARFAI AI Narrative Engine"""
import json
import os

class AINarrator:
    def __init__(self, data, results, benchmark, forecast):
        self.d = data
        self.r = results
        self.b = benchmark
        self.f = forecast

    def generate(self, api_key=None, provider='claude'):
        key = api_key or os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
        if key:
            prompt = self._build_prompt()
            result = self._call_api(prompt, key, provider)
            if not result.startswith('['):
                return result
        return self._template_narrative()

    def _call_api(self, prompt, key, provider):
        try:
            if provider == 'claude':
                import anthropic
                client = anthropic.Anthropic(api_key=key)
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}])
                return resp.content[0].text
            else:
                from openai import OpenAI
                client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
                resp = client.chat.completions.create(
                    model="deepseek-chat", max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}])
                return resp.choices[0].message.content
        except Exception as e:
            return "[API Error: %s]" % str(e)

    def _build_prompt(self):
        return """你是禾苗通讯(Sprocomm, 01401.HK)的资深销售分析总监。
基于以下数据,写一份给CEO的战略分析备忘录。
核心发现: %s
类别YoY: %s
客户Top10: %s
价量分解: %s
流失预警: %s
行业对标: %s
客户情报: %s
预测: %s
要求:
1. 三段式摘要(50字/200字/500字)
2. HMD下滑根因分析(结合行业数据)
3. 增长质量评估(谁的增长健康/不健康)
4. CEO本月必须做的3件事
语气: 简洁、直接、数据驱动。""" % (
            json.dumps(self.r['核心发现'], ensure_ascii=False),
            json.dumps(self.r['类别趋势'][:6], ensure_ascii=False),
            json.dumps(self.r['客户分级'][:10], ensure_ascii=False),
            json.dumps(self.r['价量分解'][:8], ensure_ascii=False),
            json.dumps(self.r['流失预警'][:5], ensure_ascii=False),
            json.dumps(self.b['市场定位'], ensure_ascii=False),
            json.dumps(self.b['客户外部视角'], ensure_ascii=False),
            json.dumps(self.f['风险场景'], ensure_ascii=False, default=str))

    def _template_narrative(self):
        total = self.d['总营收']
        yoy = self.d['总YoY']['增长率']
        fc_base = self.f['风险场景']['基准(+15%)']['全年预测']

        hmd = next((c for c in self.d['类别YoY'] if c['类别'] == 'HMD'), None)
        pad = next((c for c in self.d['类别YoY'] if c['类别'] == '平板'), None)
        local = next((c for c in self.d['类别YoY'] if c['类别'] == 'local'), None)

        zyb_pv = next((p for p in self.d.get('价量分解', []) if p['客户'] == 'ZYB'), None)
        ai_pv = next((p for p in self.d.get('价量分解', []) if p['客户'] == 'AI'), None)

        hr_total = sum(a['年度金额'] for a in self.r['流失预警'] if '高' in a['风险'])

        rev_b = total / 10000

        s = """## 附三、管理层战略备忘录

> 致: CEO / 销售VP
> 来源: MRARFAI 智能分析系统
> 数据: 2025全年出货(金额+数量) | 行业对标: IDC/Counterpoint

---

### 一句话摘要

2025年营收%.1f亿元(+%.0f%%),平板和local客户爆发式增长成功对冲了HMD的大幅下滑,但增长结构存在隐患——过度依赖ZYB单一客户和印度单一市场。

### 200字版

禾苗2025年完成出货%s万元,同比增长%.1f%%,增速远超行业头部(华勤+28%%/龙旗+22%%),但营收规模(%.1f亿)仍不足华勤的5%%。增长由两个引擎驱动: ZYB平板业务从配角跃升为第二大客户(占比24.2%%,+110%%),以及local客户群爆发(+146%%)。但HMD从占比30%%暴跌至11%%,下滑42%%——且禾苗降幅远超HMD全球水平(-21%%),说明我们正在被华勤/富士康抢走HMD份额。%s万元客户处于高流失风险,Top3集中度达61%%远超行业50%%安全线。2026年基准预测约%.1f亿元。

### 500字版

**增长全貌: 数字亮眼,结构需要审视**

%.1f亿的成绩单证明禾苗在功能机萎缩的大环境下找到了新的增长路径。""" % (
            rev_b, yoy * 100,
            format(total, ',.0f'), yoy * 100, rev_b,
            format(hr_total, ',.0f'), fc_base / 10000,
            rev_b)

        if pad:
            zyb_rev = next((c['年度金额'] for c in self.r['客户分级'] if c['客户'] == 'ZYB'), 0)
            s += "平板业务全年增长%.0f%%,增量%s万元,是最大的增长贡献者。ZYB一家就贡献了%s万元,教育平板赛道的爆发让禾苗找到了第二曲线。" % (
                pad['增长率'] * 100, format(pad['增长额'], ',.0f'), format(zyb_rev, ',.0f'))

        if local:
            s += "local客户群增长%.0f%%,贡献增量%s万元,这批中小客户的崛起说明禾苗在非头部客户市场有竞争力。" % (
                local['增长率'] * 100, format(local['增长额'], ',.0f'))

        s += "\n\n但增长质量需要仔细审视:\n"

        if zyb_pv:
            s += "\n**ZYB: 唯一的优质增长标杆。** 均价从%s元升至%s元," % (
                zyb_pv.get('H1均价', 'N/A'), zyb_pv.get('H2均价', 'N/A'))
            pt = zyb_pv.get('价格趋势')
            if pt:
                s += "涨幅%+.1f%%,真正实现了量价齐升。这是最健康的增长模式——客户愿意为产品付更高价格,说明禾苗在平板品类有议价能力。\n" % (pt * 100)

        if ai_pv:
            s += "\n**AI客户: 危险信号。** 虽然出货量暴增,但均价从%s元跌至%s元," % (
                ai_pv.get('H1均价', 'N/A'), ai_pv.get('H2均价', 'N/A'))
            pt = ai_pv.get('价格趋势')
            if pt:
                s += "下跌%.0f%%。这是典型的以量换价——用利润换规模,短期好看但不可持续。\n" % (abs(pt) * 100)

        hmd_section = """
**HMD: 不只是客户自身的问题。** HMD全球出货下滑21%%,但禾苗的HMD业务跌了42%%——差额说明我们在HMD供应链中的份额被竞争对手(大概率是华勤)蚕食。原因可能包括: 华勤的规模成本优势、HMD功能机产品线缩减(直接影响CKD/FP核心订单)、以及HMD向自有品牌转型减少ODM依赖。这不是可以通过降价挽回的趋势,而是结构性的。

**两个最大的风险:**

第一,客户集中度。Top3占61%%,ZYB一家24%%——如果ZYB订单波动30%%,直接影响全年营收7个点。行业健康线是Top3<50%%。

第二,市场集中度。印度占出货61.7%%,HHI指数4120(>2500为高度集中)。印度关税政策或地缘变化可以瞬间冲击超过一半的业务。

---

### CEO本月必须关注的3件事

**1. [紧急] HMD份额保卫战**
召集销售+研发开专项会。确认: 还有哪些型号是我们有优势的?哪些已经确定丢给华勤了?止损比挽回更重要。同时更新2026年HMD营收预期——建议按再跌30-50%%做预算。

**2. [战略] ZYB风险对冲**
ZYB已经是第二大客户,但集中度太高。本月开始评估: 能否拓展2-3个同级别平板客户?教育平板/AI学习机赛道还有哪些品牌在找ODM?目标是12个月内将平板业务从ZYB一家独大变成3-4家分散。

**3. [利润] 价量分解复盘**
安排财务+销售做一次全客户毛利率复盘。重点看AI客户和其他单价下滑的客户,确认以量换价策略是否真的带来了利润。如果跑量不赚钱,需要重新谈价或减少低利润订单。

---

### 2026年前瞻

"""

        s += hmd_section

        for scenario, sc_data in self.f['风险场景'].items():
            s += "- **%s**: %s万元 — %s\n" % (
                scenario, format(sc_data['全年预测'], ',.0f'), sc_data['假设'])

        s += """
基准预期%.1f亿元(+15%%),但这高度依赖ZYB平板和ZTE海外两个引擎不熄火,同时HMD不再断崖式下跌。如果平板增速从+110%%回落到+30%%(更合理的预期),且HMD再跌30%%,实际增长可能只有5-8%%。

建议将2026年内部目标定在%.1f-%.1f亿区间(+10%%到+20%%),既有挑战性又不脱离现实。
""" % (fc_base / 10000, total * 1.10 / 10000, total * 1.20 / 10000)

        return s


def generate_narrative_section(narrator, api_key=None, provider='claude'):
    return narrator.generate(api_key, provider)
