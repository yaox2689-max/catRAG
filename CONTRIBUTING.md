# 贡献指南

感谢您对 catRAG 项目的关注！我们欢迎任何形式的贡献。

## 如何贡献

### 报告 Bug

1. 使用 GitHub Issues 报告 bug
2. 使用 **Bug Report** 模板
3. 提供详细的复现步骤和环境信息

### 建议新功能

1. 使用 GitHub Issues 提出建议
2. 使用 **Feature Request** 模板
3. 说明使用场景和预期行为

### 提交代码

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature develop`
3. 提交更改：`git commit -m 'feat: add your feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request 到 `develop` 分支

## 开发规范

### 分支命名

- `feature/*` - 新功能
- `fix/*` - Bug 修复
- `docs/*` - 文档更新
- `refactor/*` - 重构
- `test/*` - 测试相关

### 提交信息

使用 **Conventional Commits** 格式：

```
<type>(<scope>): <description>

[optional body]
[optional footer]
```

**类型：**
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

**示例：**
```
feat(rag): add semantic chunking with auto-merging

- Implement three-level chunking strategy
- Add auto-merging for parent chunks
- Update Milvus vector storage

Closes #123
```

### 代码规范

- Python: 遵循 PEP 8
- JavaScript/Vue: 遵循 ESLint 配置
- 使用类型注解
- 编写清晰的文档字符串

### 测试

- 新功能必须包含测试
- 确保所有测试通过：`uv run pytest`
- 测试覆盖率不低于 80%

### 文档

- 更新 README.md（如有必要）
- 添加代码注释
- 更新 API 文档

## Pull Request 流程

1. **创建 PR**
   - 标题清晰描述变更内容
   - 使用 PR 模板填写详细信息
   - 关联相关 Issue

2. **代码审查**
   - 至少一位审查者批准
   - 解决所有审查意见
   - 确保 CI 通过

3. **合并**
   - 使用 "Squash and merge" 或 "Rebase and merge"
   - 删除已合并的功能分支

## 开发环境设置

### 1. 克隆仓库

```bash
git clone https://github.com/yaox2689-max/catRAG.git
cd catRAG
```

### 2. 创建功能分支

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature
```

### 3. 安装依赖

```bash
uv sync
cd frontend && npm install
```

### 4. 启动开发环境

```bash
docker compose up -d
cd backend && uv run python app.py
cd frontend && npm run dev
```

### 5. 提交更改

```bash
git add .
git commit -m 'feat: add your feature'
git push origin feature/your-feature
```

### 6. 创建 Pull Request

在 GitHub 上创建 PR 到 `develop` 分支。

## 代码审查清单

- [ ] 代码符合项目规范
- [ ] 包含必要的测试
- [ ] 测试全部通过
- [ ] 文档已更新
- [ ] 提交信息符合规范
- [ ] 没有引入安全漏洞
- [ ] 性能没有明显下降

## 社区准则

- 尊重他人
- 建设性反馈
- 欢迎新手
- 保持专业

## 联系方式

如有问题，请通过以下方式联系：

- GitHub Issues
- Pull Requests

感谢您的贡献！
