# 项目网页使用说明

项目页入口为 `docs/index.html`，使用原生 HTML、CSS 和 JavaScript，无需前端依赖或构建步骤。页面采用简洁的学术项目页布局：居中标题与作者、论文和代码链接、主视频、摘要与方法结构图。

## 本地预览

在仓库根目录执行：

```bash
python -m http.server 8000 --directory docs
```

在运行该服务的机器上打开 `http://localhost:8000/`。如果编辑器支持端口转发，可转发 8000 端口后预览。不要直接使用 `file://` 打开页面：HTTP 预览更接近实际 Pages 部署，也便于验证视频与相对资源路径。

页面使用黑白文字排版；结构图、视频和封面均保留原始彩色，不应用灰度或去色处理。正文采用单栏布局，标题居中，内容宽度受限；手机端自动调整字号和视频排列。不使用固定导航、装饰卡片或数据图表。

## 添加仿真视频

页面接入 `videos/` 中的仿真视频，保留装配编号和 `_01` / `_02` 等片段编号。按文件名排序的第一个片段作为主视频，其余片段放入 Simulation 网格，主视频不会在网格中重复出现。桌面端为三列，手机端自动调整为两列或一列。

新增视频后，在仓库根目录执行：

```bash
python scripts/sync_site_videos.py
```

脚本将 MP4/WebM 文件复制到网页目录并更新列表，保留仓库地址等其他页面配置。安装了 OpenCV 的环境会同时从原始彩色帧提取封面；没有 OpenCV 时仍可同步视频，已有封面会被保留。当前目录中没有视频时，脚本会报告错误，不会清空配置。脚本不删除源视频，也不会自动提交或发布。

网页副本及封面保存在 `docs/assets/videos/`，不修改仓库根目录 `videos/` 中的原始文件。`docs/assets/site-config.js` 中每个条目定义一个视频：

```javascript
{
  id: "overview",
  title: "Assembly 00062 · Clip 01",
  src: "assets/videos/00062_01.mp4",
  poster: "assets/videos/00062_01-poster.jpg",
  captions: "",
  type: "video/mp4"
}
```

`overview` 对应标题下方的主视频。其他条目使用唯一 ID，例如 `00186_02`；在配置数组中增加条目后，页面会自动生成视频位，不必手写 HTML。新增文件需同时放入 `docs/assets/videos/`，以便 `/docs` 可独立发布。

路径相对于 `docs/index.html`。`poster` 为可选封面，暂时没有封面时保持空字符串。`captions` 可填写 WebVTT 字幕路径，默认语言为英文；可额外设置 `captionsLanguage` 和 `captionsLabel`。也可在 `src` 中使用直接提供视频文件的 HTTPS URL，不能填写普通视频平台的观看页面。

`src` 为空的条目不会生成额外视频位；没有补充视频时，Simulation 栏目保持隐藏。填写路径后自动使用原生播放器，包含播放进度、音量和全屏控件；默认不自动播放。主视频预载元数据，网格视频在播放时才加载，避免打开页面就下载全部视频。视频加载失败时显示不可用提示，不会保留失效的播放器。

## 论文、结构图与代码链接

- 网页论文链接：`docs/assets/faca-paper.pdf`。该文件与仓库根目录的研究 PDF 相同，是为 Pages 独立发布保留的副本。
- 模型结构图：`docs/assets/faca_architecture.png`，点击打开 `docs/1.pdf`。
- 仓库地址：在 `docs/assets/site-config.js` 设置 `repositoryUrl`，例如 `https://github.com/OWNER/REPOSITORY`。
- 文档分支：配置中的 `sourceBranch` 默认为 `main`，其他分支可改为实际名称。

标准 `OWNER.github.io/REPOSITORY/` 地址可自动推断对应仓库；使用自定义域名时应显式填写 `repositoryUrl`。配置仓库地址后，“[Model]”会变为“[Code]”，文档链接也会指向对应分支。没有配置仓库地址的本地页面仍可使用页内模型入口与本地文档。

修改页面布局和正文使用 `docs/index.html`，样式使用 `docs/assets/site.css`，视频和链接使用 `docs/assets/site-config.js`。页面展示项目的研究结果；模型示例使用随机权重，只用于演示接口调用。

## 发布到 GitHub Pages

等页面检查通过并由维护者提交、推送到 GitHub 后，在仓库设置中选择：

```text
Settings → Pages
Source: Deploy from a branch
Branch: main
Folder: /docs
```

保存后由 GitHub Pages 发布。项目已包含 `docs/.nojekyll`，按静态文件提供页面，无需安装主题或执行前端构建。具体配置见 [GitHub Pages 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。

发布操作由维护者在审查并提交页面后执行。
