# 解决 Hugging Face 模型下载问题

## 问题描述
在运行 `python app.py` 时，出现以下错误：
```
'[WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。'
```

这是因为无法从 Hugging Face 官方服务器下载 BGE-M3 模型文件。

## 解决方案

### 方案一：使用镜像站点（已配置）
已在 `.env` 文件中添加了 Hugging Face 镜像配置：
```
HF_ENDPOINT=https://hf-mirror.com
```

### 方案二：手动下载模型
如果镜像站点也无法访问，可以手动下载模型：

1. 访问 https://hf-mirror.com/BAAI/bge-m3
2. 下载所有模型文件
3. 将文件放置在本地目录，例如 `./models/bge-m3/`
4. 修改 `.env` 文件中的模型路径：
   ```
   EMBEDDING_MODEL=./models/bge-m3
   ```

### 方案三：使用其他模型
可以考虑使用其他可用的嵌入模型，如：
- `sentence-transformers/all-MiniLM-L6-v2`（较小，速度快）
- `BAAI/bge-base-zh`（中文优化）

## 验证配置
重新启动应用：
```bash
cd backend
python app.py
```

如果配置正确，应该能够成功加载模型而不会出现网络错误。