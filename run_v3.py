#!/usr/bin/env python3
"""MRARFAI v3.0 CLI"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_clients_v2 import SprocommDataLoaderV2, DeepAnalyzer, ReportGeneratorV2
from industry_benchmark import IndustryBenchmark, generate_benchmark_section
from forecast_engine import ForecastEngine, generate_forecast_section
from ai_narrator import AINarrator

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--revenue', default=None)
    p.add_argument('--quantity', default=None)
    p.add_argument('--output', default='./output_v3')
    p.add_argument('--ai-key', default=None)
    p.add_argument('--provider', default='claude', choices=['claude','deepseek'])
    a = p.parse_args()

    rev = a.revenue or '/mnt/user-data/uploads/25年出货数据by金额报表_最终.xlsx'
    qty = a.quantity or '/mnt/user-data/uploads/手机_平板出货by数量2025_-_最终汇总.xlsx'
    os.makedirs(a.output, exist_ok=True)

    print("\n" + "="*60)
    print("  MRARFAI v3.0 [分析+行业对标+预测+AI叙事]")
    print("="*60)

    print("\n[1/5] 数据加载...")
    loader = SprocommDataLoaderV2(rev, qty)
    data = loader.load_all()

    print("[2/5] 深度分析...")
    results = DeepAnalyzer(data).run_all()

    print("[3/5] 行业对标...")
    benchmark = IndustryBenchmark(data, results).run()

    print("[4/5] 预测...")
    forecast = ForecastEngine(data, results).run()
    q1 = forecast['总营收预测']['Q1_2026预测']
    print("  Q1 2026: %s万元" % format(q1, ',.0f'))

    print("[5/5] AI叙事...")
    narrator = AINarrator(data, results, benchmark, forecast)
    narrative = narrator.generate(a.ai_key, a.provider)

    # Assemble report
    gen = ReportGeneratorV2(data, results)
    base = gen.generate()
    bench_s = generate_benchmark_section(benchmark)
    fore_s = generate_forecast_section(forecast)

    footer = "\n---\n> MRARFAI"
    if footer in base:
        parts = base.split(footer, 1)
        full = parts[0] + bench_s + fore_s + narrative + footer + parts[1]
    else:
        full = base + bench_s + fore_s + narrative
    full = full.replace("v2.0", "v3.0")

    rp = os.path.join(a.output, '禾苗2025销售分析_v3.md')
    with open(rp, 'w', encoding='utf-8') as f:
        f.write(full)

    jp = os.path.join(a.output, 'analysis_v3.json')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump({'分析': results, '行业': benchmark, '预测': forecast},
                  f, ensure_ascii=False, indent=2, default=str)

    print("\n  Report: %s (%s chars)" % (rp, format(len(full), ',')))
    print("  Data: %s" % jp)
    print("  Done!\n")

if __name__ == '__main__':
    main()
