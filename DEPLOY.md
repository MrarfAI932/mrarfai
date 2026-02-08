# MRARFAI 云部署指南
## Streamlit Community Cloud（免费）

### 第一步：确保 GitHub 仓库完整
```bash
cd ~/Desktop/mrarfai
git add -A
git commit -m "v4.3: 异常检测 + 品牌白标 + 部署就绪"
git push
```

### 第二步：部署到 Streamlit Cloud
1. 打开 https://share.streamlit.io
2. 用 GitHub 账号登录
3. 点 **New app**
4. 选择：
   - Repository: `MrarfAI932/mrarfai`
   - Branch: `main`
   - Main file: `app.py`
5. 点 **Deploy**
6. 等 2-3 分钟，自动安装依赖并启动

### 第三步：配置 Secrets（API Key）
部署后在 Settings → Secrets 中添加：
```toml
[api_keys]
deepseek = "sk-xxx"
claude = "sk-ant-xxx"
```

### 访问地址
部署成功后会得到类似：
`https://mrarfai-xxx.streamlit.app`

分享给同事即可直接访问，不需要本地安装。

---

## 进阶：自定义域名
1. 升级 Streamlit Teams（$250/月）
2. 或用 Cloudflare Tunnel 免费映射

## 进阶：Docker 部署到自有服务器
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
docker build -t mrarfai .
docker run -p 8501:8501 mrarfai
```
