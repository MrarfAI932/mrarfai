# MRARFAI 部署指南 — Week 5-6

## 方案一：Streamlit Cloud（推荐 ⭐）

**优势**：免费、最简单、自动HTTPS、自动部署

### 步骤

1. **推送代码到GitHub**
```bash
git init
git add .
git commit -m "MRARFAI v3.2 - deployment ready"
git remote add origin https://github.com/YOUR_USERNAME/mrarfai.git
git push -u origin main
```

2. **访问 Streamlit Cloud**
- 打开 https://share.streamlit.io
- 用GitHub账号登录
- 点击 "New app"

3. **配置部署**
- Repository: `YOUR_USERNAME/mrarfai`
- Branch: `main`
- Main file path: `app.py`
- 点击 "Deploy!"

4. **分享链接**
- 部署成功后会生成类似 `https://mrarfai.streamlit.app` 的链接
- 发送给禾苗内部用户试用

### 注意事项
- Streamlit Cloud 免费版有资源限制（1GB RAM）
- 上传文件不会持久保存（安全✅）
- 反馈数据在每次重启后丢失（云端需接数据库）

---

## 方案二：Railway

**优势**：更灵活、支持后台任务、有免费额度

### 步骤

1. **注册 Railway**：https://railway.app
2. **连接GitHub仓库**
3. **Railway会自动检测Procfile并部署**
4. **设置环境变量**（可选）：
   - `ANTHROPIC_API_KEY`：Claude API密钥
   - `FEEDBACK_DIR`：反馈存储路径

### Procfile已配置
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

---

## 方案三：阿里云/腾讯云（合规推荐）

**优势**：数据在国内、符合数据安全法

### Docker部署
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

```bash
# 构建并运行
docker build -t mrarfai .
docker run -p 8501:8501 mrarfai
```

---

## Week 5-6 检查清单

- [ ] 代码推送到GitHub
- [ ] 选择部署平台并完成部署
- [ ] 生成访问链接
- [ ] 发送给禾苗销售/运营负责人试用
- [ ] 收集至少3条使用反馈（目标满意度>80%）
- [ ] 根据反馈整理Week 7-8迭代计划

## 反馈收集机制

Dashboard内置了「💬 反馈」Tab：
- 满意度评分（1-10）
- 最有用模块投票
- 痛点和建议文本框
- 所有反馈自动保存为JSON

管理员可在反馈Tab底部查看汇总统计。

## 安全提醒

⚠️ 部署前确认：
1. 不要在代码中硬编码API密钥
2. 反馈数据不包含客户敏感信息
3. 如果使用Streamlit Cloud/Railway，数据在境外 → 只上传脱敏数据
4. 正式用于禾苗生产数据时，应使用国内云服务（方案三）
