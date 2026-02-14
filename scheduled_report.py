#!/usr/bin/env python3
"""
MRARFAI — 定时报告引擎
========================
自动生成各 Agent 分析报告，支持:
  - 邮件发送 (SMTP)
  - 微信推送 (已有 wechat_notify 模块)
  - 文件导出 (Markdown / JSON)

配置方法 (.env):
  REPORT_SMTP_HOST=smtp.qq.com
  REPORT_SMTP_PORT=465
  REPORT_SMTP_USER=user@qq.com
  REPORT_SMTP_PASS=授权码
  REPORT_RECIPIENTS=boss@company.com,team@company.com
  REPORT_SCHEDULE=daily  # daily / weekly / monthly
"""

import json
import os
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional

logger = logging.getLogger("mrarfai.report")


# ============================================================
# 报告配置
# ============================================================
class ReportConfig:
    """报告调度配置"""

    def __init__(self):
        self.smtp_host = os.environ.get("REPORT_SMTP_HOST", "")
        self.smtp_port = int(os.environ.get("REPORT_SMTP_PORT", "465") or "465")
        self.smtp_user = os.environ.get("REPORT_SMTP_USER", "")
        self.smtp_pass = os.environ.get("REPORT_SMTP_PASS", "")
        self.recipients = [
            r.strip() for r in os.environ.get("REPORT_RECIPIENTS", "").split(",")
            if r.strip()
        ]
        self.schedule = os.environ.get("REPORT_SCHEDULE", "daily")
        self.enabled = bool(self.smtp_host and self.smtp_user and self.recipients)


# ============================================================
# 报告生成器
# ============================================================
class ReportGenerator:
    """从 Agent 引擎自动生成分析报告"""

    def __init__(self, gateway=None):
        self.gateway = gateway

    def generate_daily_report(self) -> str:
        """生成日报 (Markdown)"""
        now = datetime.now()
        report = f"# MRARFAI 日报 — {now.strftime('%Y年%m月%d日')}\n\n"
        report += f"> 自动生成于 {now.strftime('%H:%M:%S')} · MRARFAI V10.0\n\n"

        if not self.gateway:
            report += "⚠ Gateway 未初始化\n"
            return report

        # 各 Agent 快速摘要
        agents_queries = {
            "procurement": "今日采购概况摘要",
            "quality": "良率概况总结",
            "finance": "应收账款和毛利情况",
            "market": "竞品市场概况",
        }

        for agent, query in agents_queries.items():
            report += f"---\n\n## {agent.upper()} Agent\n\n"
            try:
                resp = self.gateway.ask(query, user="system_report")
                answer = resp.get("answer", "")
                try:
                    data = json.loads(answer)
                    report += f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)[:800]}\n```\n\n"
                except (json.JSONDecodeError, TypeError):
                    report += f"{str(answer)[:800]}\n\n"
            except Exception as e:
                report += f"⚠ 生成失败: {e}\n\n"

        # 平台统计
        try:
            stats = self.gateway.get_platform_stats()
            report += "---\n\n## 平台统计\n\n"
            report += f"- 总请求数: {stats.get('total_requests', 0)}\n"
            report += f"- Agent 数: {stats.get('agent_count', 0)}\n"
            report += f"- 平均响应: {stats.get('avg_duration_ms', 0):.0f}ms\n"
        except Exception:
            pass

        report += f"\n---\n\n*MRARFAI V10.0 · Automated Report · {now.isoformat()}*\n"
        return report

    def generate_weekly_report(self) -> str:
        """生成周报"""
        now = datetime.now()
        week_start = now - timedelta(days=now.weekday())
        report = f"# MRARFAI 周报 — {week_start.strftime('%m月%d日')} ~ {now.strftime('%m月%d日')}\n\n"
        report += f"> 自动生成于 {now.strftime('%Y-%m-%d %H:%M')} · MRARFAI V10.0\n\n"

        # 周报包含更详细的分析
        report += self._generate_section("procurement", "本周采购分析：供应商表现、延迟情况、成本趋势")
        report += self._generate_section("quality", "本周品质分析：良率趋势、退货分析、投诉汇总")
        report += self._generate_section("finance", "本周财务分析：应收账款、毛利率、现金流")
        report += self._generate_section("market", "本周市场分析：竞品动态、行业趋势")

        report += f"\n---\n\n*MRARFAI V10.0 · Weekly Report · {now.isoformat()}*\n"
        return report

    def _generate_section(self, agent: str, query: str) -> str:
        """生成单个 Agent 的报告段落"""
        section = f"---\n\n## {agent.upper()} — {query.split('：')[0] if '：' in query else agent}\n\n"
        if not self.gateway:
            section += "⚠ Gateway 未初始化\n\n"
            return section
        try:
            resp = self.gateway.ask(query, user="system_report")
            answer = resp.get("answer", "")
            try:
                data = json.loads(answer)
                section += f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)[:1200]}\n```\n\n"
            except (json.JSONDecodeError, TypeError):
                section += f"{str(answer)[:1200]}\n\n"
        except Exception as e:
            section += f"⚠ 生成失败: {e}\n\n"
        return section


# ============================================================
# 邮件发送器
# ============================================================
class EmailSender:
    """SMTP 邮件发送"""

    def __init__(self, config: ReportConfig):
        self.config = config

    def send(self, subject: str, body_markdown: str, recipients: List[str] = None) -> Dict:
        """发送邮件 (Markdown 转 HTML)"""
        if not self.config.enabled:
            return {"status": "disabled", "message": "邮件未配置 (REPORT_SMTP_* 未设置)"}

        to_list = recipients or self.config.recipients
        if not to_list:
            return {"status": "error", "message": "无收件人"}

        try:
            # Markdown → simple HTML
            html_body = self._markdown_to_html(body_markdown)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.smtp_user
            msg["To"] = ", ".join(to_list)

            # 纯文本 + HTML
            msg.attach(MIMEText(body_markdown, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            # 发送
            if self.config.smtp_port == 465:
                server = smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port, timeout=15)
            else:
                server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=15)
                server.starttls()

            server.login(self.config.smtp_user, self.config.smtp_pass)
            server.sendmail(self.config.smtp_user, to_list, msg.as_string())
            server.quit()

            logger.info(f"邮件已发送: {subject} → {to_list}")
            return {"status": "ok", "recipients": to_list}

        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    def _markdown_to_html(md: str) -> str:
        """简单的 Markdown → HTML 转换"""
        html = "<html><head><style>"
        html += "body{font-family:'Helvetica Neue',sans-serif;background:#0c0c0c;color:#e0e0e0;padding:20px;max-width:800px;margin:0 auto;}"
        html += "h1{color:#00FF88;border-bottom:1px solid #333;padding-bottom:8px;}"
        html += "h2{color:#00A0C8;margin-top:24px;}"
        html += "pre{background:#111;padding:12px;border:1px solid #333;overflow-x:auto;font-size:13px;}"
        html += "code{color:#00FF88;font-family:'JetBrains Mono',monospace;}"
        html += "blockquote{border-left:3px solid #00FF88;padding-left:12px;color:#888;}"
        html += "hr{border:none;border-top:1px solid #333;margin:20px 0;}"
        html += "</style></head><body>"

        for line in md.split("\n"):
            if line.startswith("# "):
                html += f"<h1>{line[2:]}</h1>"
            elif line.startswith("## "):
                html += f"<h2>{line[3:]}</h2>"
            elif line.startswith("### "):
                html += f"<h3>{line[4:]}</h3>"
            elif line.startswith("> "):
                html += f"<blockquote>{line[2:]}</blockquote>"
            elif line.startswith("- "):
                html += f"<li>{line[2:]}</li>"
            elif line.startswith("---"):
                html += "<hr>"
            elif line.startswith("```"):
                if "json" in line:
                    html += "<pre><code>"
                else:
                    html += "</code></pre>"
            elif line.strip():
                html += f"<p>{line}</p>"

        html += "</body></html>"
        return html


# ============================================================
# 报告调度器
# ============================================================
class ReportScheduler:
    """定时报告调度管理"""

    LAST_RUN_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".report_last_run.json")

    def __init__(self, gateway=None):
        self.config = ReportConfig()
        self.generator = ReportGenerator(gateway)
        self.sender = EmailSender(self.config)
        self._last_runs = self._load_last_runs()

    def _load_last_runs(self) -> Dict:
        if os.path.exists(self.LAST_RUN_FILE):
            try:
                with open(self.LAST_RUN_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_last_runs(self):
        with open(self.LAST_RUN_FILE, "w") as f:
            json.dump(self._last_runs, f, ensure_ascii=False, indent=2)

    def check_and_run(self) -> Optional[Dict]:
        """检查是否该运行定时报告，如果是则自动生成+发送"""
        if not self.config.enabled:
            return None

        now = datetime.now()
        schedule = self.config.schedule

        # 检查上次运行时间
        last_run_str = self._last_runs.get(schedule, "")
        if last_run_str:
            last_run = datetime.fromisoformat(last_run_str)
            if schedule == "daily" and (now - last_run).total_seconds() < 82800:  # 23h
                return None
            elif schedule == "weekly" and (now - last_run).total_seconds() < 604800:  # 7d
                return None
            elif schedule == "monthly" and now.month == last_run.month:
                return None

        # 生成并发送
        if schedule == "daily":
            report = self.generator.generate_daily_report()
            subject = f"MRARFAI 日报 — {now.strftime('%Y年%m月%d日')}"
        elif schedule == "weekly":
            report = self.generator.generate_weekly_report()
            subject = f"MRARFAI 周报 — {now.strftime('%Y年第%W周')}"
        else:
            report = self.generator.generate_daily_report()
            subject = f"MRARFAI 月报 — {now.strftime('%Y年%m月')}"

        result = self.sender.send(subject, report)

        # 保存文件
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".reports")
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"{schedule}_{now.strftime('%Y%m%d_%H%M')}.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        # 记录运行时间
        self._last_runs[schedule] = now.isoformat()
        self._save_last_runs()

        return {
            "type": schedule,
            "subject": subject,
            "email_result": result,
            "saved_to": report_file,
            "timestamp": now.isoformat(),
        }

    def generate_now(self, report_type: str = "daily") -> Dict:
        """立即生成报告 (手动触发)"""
        now = datetime.now()
        if report_type == "weekly":
            report = self.generator.generate_weekly_report()
            subject = f"MRARFAI 周报 — {now.strftime('%Y年第%W周')}"
        else:
            report = self.generator.generate_daily_report()
            subject = f"MRARFAI 日报 — {now.strftime('%Y年%m月%d日')}"

        # 保存到文件
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".reports")
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"manual_{report_type}_{now.strftime('%Y%m%d_%H%M')}.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        return {
            "report": report,
            "saved_to": report_file,
            "subject": subject,
        }

    def get_status(self) -> Dict:
        """获取定时报告状态"""
        return {
            "enabled": self.config.enabled,
            "schedule": self.config.schedule,
            "recipients": self.config.recipients,
            "smtp_host": self.config.smtp_host or "(未配置)",
            "last_runs": self._last_runs,
        }
