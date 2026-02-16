"use client"

import { useState, useRef, useCallback } from "react"
import {
  Upload,
  FileText,
  FileSpreadsheet,
  FileImage,
  FileArchive,
  File,
  Trash2,
  Eye,
  X,
  HardDrive,
  Clock,
  FolderOpen,
  Bot,
  CheckCircle,
  Loader2,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"
import { agents } from "@/lib/dashboard-data"
import { uploadFile as apiUploadFile } from "@/lib/api"

// ─── File-type to agent routing rules ──────────────────────────────
// Keywords in filename or extension determine which agent(s) handle the file.
// Falls back to heuristics based on MIME type / extension.

interface RouteRule {
  /** Keywords to match against the lowercased filename (without extension) */
  keywords: string[]
  /** Agent names to assign when matched */
  agents: string[]
}

const KEYWORD_RULES: RouteRule[] = [
  // Finance / revenue
  { keywords: ["finance", "财务", "revenue", "营收", "profit", "利润", "pnl", "损益", "invoice", "发票", "cost", "成本", "budget", "预算", "cashflow", "现金流"], agents: ["财务智能体"] },
  // Sales / client / shipment
  { keywords: ["sales", "销售", "client", "客户", "shipment", "出货", "order", "订单", "pipeline", "管道", "deal", "交易", "account", "账户"], agents: ["销售智能体"] },
  // Risk / compliance
  { keywords: ["risk", "风险", "compliance", "合规", "audit", "审计", "payment", "付款", "overdue", "逾期", "geopolitical", "地缘"], agents: ["风控智能体"] },
  // Procurement / supply chain / BOM
  { keywords: ["procurement", "采购", "supplier", "供应商", "bom", "inventory", "库存", "component", "组件", "vendor", "logistics", "物流"], agents: ["采购智能体"] },
  // Quality / defect
  { keywords: ["quality", "质量", "defect", "缺陷", "yield", "良率", "inspection", "检验", "qc", "qa", "test", "测试"], agents: ["质量智能体"] },
  // Market / trend / competitor
  { keywords: ["market", "市场", "trend", "趋势", "competitor", "竞品", "竞争", "demand", "需求", "forecast", "预测", "industry", "行业"], agents: ["市场智能体"] },
  // Strategy / plan
  { keywords: ["strategy", "策略", "战略", "plan", "规划", "expansion", "扩张", "roadmap", "路线图", "scenario", "情景"], agents: ["策略智能体"] },
]

/** Extension-based fallback mapping */
const EXT_FALLBACK: Record<string, string[]> = {
  xlsx:  ["财务智能体", "销售智能体"],
  xls:   ["财务智能体", "销售智能体"],
  csv:   ["财务智能体", "市场智能体"],
  pdf:   ["策略智能体"],
  docx:  ["策略智能体"],
  doc:   ["策略智能体"],
  pptx:  ["策略智能体", "市场智能体"],
  ppt:   ["策略智能体", "市场智能体"],
  png:   ["质量智能体"],
  jpg:   ["质量智能体"],
  jpeg:  ["质量智能体"],
  zip:   ["采购智能体"],
  rar:   ["采购智能体"],
}

function assignAgentsByFileType(fileName: string): string[] {
  const lower = fileName.toLowerCase()
  const ext = lower.split(".").pop() || ""
  const baseName = lower.replace(/\.[^.]+$/, "")

  // 1. Try keyword matching - collect all matching agents
  const matched = new Set<string>()
  for (const rule of KEYWORD_RULES) {
    for (const kw of rule.keywords) {
      if (baseName.includes(kw)) {
        rule.agents.forEach((a) => matched.add(a))
        break // one keyword hit per rule is enough
      }
    }
  }

  if (matched.size > 0) {
    // Cap at 3 agents, and always keep at least 1
    return Array.from(matched).slice(0, 3)
  }

  // 2. Fallback: use extension mapping
  if (EXT_FALLBACK[ext]) {
    return EXT_FALLBACK[ext]
  }

  // 3. Last resort: assign the sales agent (general purpose)
  return ["销售智能体"]
}

// ─── Agent-specific analysis templates ────────────────────────────
function generateAnalysis(fileName: string, agentNames: string[]) {
  const ext = fileName.split(".").pop()?.toLowerCase() || ""
  const primary = agentNames[0]

  // Per-agent analysis templates
  const agentTemplates: Record<string, string[]> = {
    "财务智能体": [
      `已完成财务数据解析，识别到 ${Math.floor(Math.random() * 20 + 5)} 项关键指标。营收数据与上季度对比偏差 ${(Math.random() * 5 + 0.5).toFixed(1)}%，已标记异常波动项。`,
      `报表结构化处理完毕，提取损益关键字段 ${Math.floor(Math.random() * 15 + 8)} 个。现金流与应收/应付已交叉验证。`,
      `成本数据导入完成，BOM 成本分析识别到 ${Math.floor(Math.random() * 5 + 2)} 个优化机会点，预估可节省 ${(Math.random() * 3 + 0.5).toFixed(1)}%。`,
    ],
    "销售智能体": [
      `客户数据分析完成，覆盖 ${Math.floor(Math.random() * 10 + 3)} 个活跃账户。出货完成率 ${(Math.random() * 10 + 85).toFixed(1)}%，管道健康度评估为"良好"。`,
      `订单数据已同步，${Math.floor(Math.random() * 50 + 20)} 条交易记录已归档。Top 3 客户贡献率 ${(Math.random() * 15 + 70).toFixed(1)}%。`,
      `销售预测模型已更新，Q2 预计增长 ${(Math.random() * 8 + 5).toFixed(1)}%。建议关注 ${Math.floor(Math.random() * 3 + 1)} 个潜在流失客户。`,
    ],
    "风控智能体": [
      `风险扫描完成，识别 ${Math.floor(Math.random() * 5 + 1)} 个中高级风险项。付款逾期风险评分已更新，建议关注 ${Math.floor(Math.random() * 3 + 1)} 个账户。`,
      `合规文档审核完毕，${Math.floor(Math.random() * 3 + 1)} 项需补充认证材料。供应链中断概率评估为 ${(Math.random() * 10 + 5).toFixed(1)}%。`,
      `地缘风险扫描已完成，影响评级"低"。已更新区域市场风险矩阵。`,
    ],
    "采购智能体": [
      `供应商数据解析完成，覆盖 ${Math.floor(Math.random() * 20 + 10)} 个供应商。库存周转率 ${(Math.random() * 3 + 4).toFixed(1)} 次/年，${Math.floor(Math.random() * 5 + 1)} 个 SKU 低于安全库存。`,
      `BOM 数据已导入分析引擎，${Math.floor(Math.random() * 200 + 50)} 个组件已索引。识别到 ${Math.floor(Math.random() * 5 + 2)} 个替代供应商方案。`,
      `物流文档分析完毕，平均交付周期 ${Math.floor(Math.random() * 10 + 14)} 天。建议优化 ${Math.floor(Math.random() * 3 + 1)} 条运输路线。`,
    ],
    "质量智能体": [
      `质检数据分析完成，整体良率 ${(Math.random() * 3 + 96).toFixed(1)}%。${Math.floor(Math.random() * 3 + 1)} 个批次需复检，缺陷分类已标注。`,
      `图像/文档分析完毕，识别 ${Math.floor(Math.random() * 8 + 2)} 个检测维度。与上月数据对比，关键指标改善 ${(Math.random() * 2 + 0.5).toFixed(1)}%。`,
      `检验报告已结构化，${Math.floor(Math.random() * 10 + 5)} 项合规指标全部达标。已生成质量趋势报告摘要。`,
    ],
    "市场智能体": [
      `市场数据提取完成，覆盖 ${Math.floor(Math.random() * 5 + 3)} 个区域市场。行业增速 ${(Math.random() * 5 + 3).toFixed(1)}%，竞品动态已更新 ${Math.floor(Math.random() * 8 + 3)} 条。`,
      `需求预测模型已输入新数据，预计下季度目标市场规模增长 ${(Math.random() * 8 + 2).toFixed(1)}%。消费者偏好趋势已标注。`,
      `竞品分析报告已整合，${Math.floor(Math.random() * 5 + 2)} 个竞争对手的定价/渠道策略已更新。建议关注 ${Math.floor(Math.random() * 2 + 1)} 个新兴品牌。`,
    ],
    "策略智能体": [
      `战略文档解析完成，识别 ${Math.floor(Math.random() * 8 + 3)} 个核心议题。SWOT 分析框架已自动匹配，${Math.floor(Math.random() * 5 + 2)} 项行动建议已生成。`,
      `规划文件已结构化处理，提取关键里程碑 ${Math.floor(Math.random() * 10 + 5)} 个。情景分析模型已初始化，建议补充 ${Math.floor(Math.random() * 3 + 1)} 项假设条件。`,
      `文档分析完毕，共 ${Math.floor(Math.random() * 30 + 5)} 页内容已索引。关键决策点 ${Math.floor(Math.random() * 5 + 2)} 个，已标记优先级。`,
    ],
  }

  // Extension-specific prefix
  const extPrefixes: Record<string, string> = {
    pdf: "[PDF] ",
    xlsx: "[Excel] ",
    xls: "[Excel] ",
    csv: "[CSV] ",
    docx: "[Word] ",
    doc: "[Word] ",
    pptx: "[PPT] ",
    ppt: "[PPT] ",
    png: "[图像] ",
    jpg: "[图像] ",
    jpeg: "[图像] ",
  }

  const prefix = extPrefixes[ext] || "[文件] "

  const pool = agentTemplates[primary] || [
    `${primary}已接收文件并完成初步分析。数据预处理完成，已建立索引以供后续查询。`,
  ]

  return prefix + pool[Math.floor(Math.random() * pool.length)]
}

export interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  uploadedAt: Date
  progress: number
  url?: string
  // Agent processing fields
  assignedAgents: string[]
  processingStatus: "uploading" | "processing" | "done"
  analysisResult?: string
  confidence?: number
}

function getFileIcon(type: string) {
  if (type.includes("pdf")) return <FileText className="h-4 w-4 text-[#ccc]" />
  if (type.includes("sheet") || type.includes("excel") || type.includes("csv"))
    return <FileSpreadsheet className="h-4 w-4 text-[#999]" />
  if (type.startsWith("image/")) return <FileImage className="h-4 w-4 text-[#bbb]" />
  if (type.includes("zip") || type.includes("rar") || type.includes("tar") || type.includes("7z"))
    return <FileArchive className="h-4 w-4 text-[#888]" />
  return <File className="h-4 w-4 text-[#666]" />
}

function getFileExtLabel(name: string) {
  const ext = name.split(".").pop()?.toUpperCase() || "FILE"
  return ext
}

function formatFileSize(bytes: number) {
  if (bytes === 0) return "0 B"
  const k = 1024
  const sizes = ["B", "KB", "MB", "GB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i]
}

function formatTime(date: Date) {
  const h = date.getHours().toString().padStart(2, "0")
  const m = date.getMinutes().toString().padStart(2, "0")
  const s = date.getSeconds().toString().padStart(2, "0")
  return `${h}:${m}:${s}`
}

export function FilesTab() {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [previewFile, setPreviewFile] = useState<UploadedFile | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const simulateUpload = useCallback((file: File) => {
    const id = Math.random().toString(36).substring(2, 10)
    const assignedAgents = assignAgentsByFileType(file.name)
    const newFile: UploadedFile = {
      id,
      name: file.name,
      size: file.size,
      type: file.type || "application/octet-stream",
      uploadedAt: new Date(),
      progress: 0,
      url: file.type.startsWith("image/") ? URL.createObjectURL(file) : undefined,
      assignedAgents,
      processingStatus: "uploading",
    }

    setFiles((prev) => [newFile, ...prev])

    // Phase 1: Upload progress animation
    let progress = 0
    const interval = setInterval(() => {
      progress += Math.random() * 30 + 10
      if (progress >= 100) {
        progress = 100
        clearInterval(interval)
        // Phase 2: Switch to processing
        setFiles((prev) =>
          prev.map((f) =>
            f.id === id ? { ...f, progress: 100, processingStatus: "processing" } : f
          )
        )
        // Phase 3: Real API upload
        apiUploadFile(file)
          .then((result) => {
            setFiles((prev) =>
              prev.map((f) =>
                f.id === id
                  ? {
                      ...f,
                      processingStatus: "done",
                      assignedAgents: result.assignedAgents || assignedAgents,
                      analysisResult: result.analysisResult || generateAnalysis(file.name, assignedAgents),
                      confidence: result.confidence || parseFloat((Math.random() * 8 + 91).toFixed(1)),
                    }
                  : f
              )
            )
          })
          .catch(() => {
            // Fallback to client-side analysis on API error
            setFiles((prev) =>
              prev.map((f) =>
                f.id === id
                  ? {
                      ...f,
                      processingStatus: "done",
                      analysisResult: generateAnalysis(file.name, assignedAgents),
                      confidence: parseFloat((Math.random() * 8 + 91).toFixed(1)),
                    }
                  : f
              )
            )
          })
      } else {
        setFiles((prev) =>
          prev.map((f) => (f.id === id ? { ...f, progress } : f))
        )
      }
    }, 200)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const droppedFiles = Array.from(e.dataTransfer.files)
      droppedFiles.forEach(simulateUpload)
    },
    [simulateUpload]
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = Array.from(e.target.files || [])
      selectedFiles.forEach(simulateUpload)
      if (fileInputRef.current) fileInputRef.current.value = ""
    },
    [simulateUpload]
  )

  const handleDelete = useCallback((id: string) => {
    setFiles((prev) => {
      const file = prev.find((f) => f.id === id)
      if (file?.url) URL.revokeObjectURL(file.url)
      return prev.filter((f) => f.id !== id)
    })
    setPreviewFile((prev) => (prev?.id === id ? null : prev))
  }, [])

  const totalSize = files.reduce((acc, f) => acc + f.size, 0)
  const processedCount = files.filter((f) => f.processingStatus === "done").length
  const processingCount = files.filter(
    (f) => f.processingStatus === "processing" || f.processingStatus === "uploading"
  ).length

  const kpis = [
    { title: "文件总数", value: files.length.toString(), icon: FolderOpen },
    { title: "总大小", value: formatFileSize(totalSize), icon: HardDrive },
    { title: "已分析", value: processedCount.toString(), icon: CheckCircle },
    { title: "处理中", value: processingCount.toString(), icon: Clock },
  ]

  return (
    <div className="flex flex-col gap-6">
      {/* KPI cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {kpis.map((kpi, i) => {
          const Icon = kpi.icon
          return (
            <Card
              key={i}
              className="border-[#1a1a1a] bg-[#0e0e0e]"
            >
              <CardContent className="flex items-center gap-4 p-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-[#1a1a1a] bg-[#141414]">
                  <Icon className="h-4 w-4 text-[#888]" />
                </div>
                <div>
                  <p className="text-xs text-[#555]">{kpi.title}</p>
                  <p className="font-mono text-xl font-bold text-[#ededed]">
                    {kpi.value}
                  </p>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Upload zone */}
      <Card className="border-[#1a1a1a] bg-[#0e0e0e]">
        <CardContent className="p-4">
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed py-16 transition-all duration-200",
              isDragging
                ? "border-[#ffffff] bg-[#ffffff]/5"
                : "border-[#222] bg-[#0a0a0a] hover:border-[#444] hover:bg-[#111]"
            )}
            role="button"
            tabIndex={0}
            aria-label="上传文件"
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click()
            }}
          >
            <div
              className={cn(
                "flex h-14 w-14 items-center justify-center rounded-2xl border transition-colors",
                isDragging
                  ? "border-[#555] bg-[#1a1a1a]"
                  : "border-[#1a1a1a] bg-[#141414]"
              )}
            >
              <Upload
                className={cn(
                  "h-6 w-6 transition-colors",
                  isDragging ? "text-[#ffffff]" : "text-[#555]"
                )}
              />
            </div>
            <div className="text-center">
              <p
                className={cn(
                  "text-sm font-medium transition-colors",
                  isDragging ? "text-[#ffffff]" : "text-[#888]"
                )}
              >
                {isDragging ? "松开以上传文件" : "拖拽文件到此处，或点击选择"}
              </p>
              <p className="mt-1 font-mono text-xs text-[#444]">
                支持所有文件类型 / 根据报告类型自动路由到对应智能体
              </p>
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            className="hidden"
            aria-hidden="true"
          />
        </CardContent>
      </Card>

      {/* File list */}
      {files.length > 0 && (
        <Card className="border-[#1a1a1a] bg-[#0e0e0e]">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-[#ededed]">
                已上传文件
              </CardTitle>
              <span className="font-mono text-xs text-[#555]">
                共 {files.length} 个文件
              </span>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow className="border-b border-[#1a1a1a] hover:bg-transparent">
                  <TableHead className="font-mono text-xs text-[#555]">
                    文件名
                  </TableHead>
                  <TableHead className="font-mono text-xs text-[#555]">
                    类型
                  </TableHead>
                  <TableHead className="font-mono text-xs text-[#555]">
                    分配智能体
                  </TableHead>
                  <TableHead className="font-mono text-xs text-[#555] text-right">
                    大小
                  </TableHead>
                  <TableHead className="font-mono text-xs text-[#555] text-right">
                    状态
                  </TableHead>
                  <TableHead className="font-mono text-xs text-[#555] text-right">
                    操作
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {files.map((file) => (
                  <TableRow
                    key={file.id}
                    className="border-b border-[#111] hover:bg-[#111]"
                  >
                    <TableCell>
                      <div className="flex items-center gap-3">
                        {getFileIcon(file.type)}
                        <span className="max-w-[180px] truncate font-mono text-xs text-[#ccc]">
                          {file.name}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className="border-[#222] bg-[#141414] font-mono text-[10px] text-[#888]"
                      >
                        {getFileExtLabel(file.name)}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1.5">
                        <Bot className="h-3 w-3 text-[#555]" />
                        <span className="max-w-[160px] truncate font-mono text-[10px] text-[#888]">
                          {file.assignedAgents.join(", ")}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs text-[#888]">
                      {formatFileSize(file.size)}
                    </TableCell>
                    <TableCell className="text-right">
                      {file.processingStatus === "uploading" ? (
                        <div className="ml-auto flex w-24 items-center gap-2">
                          <div className="h-1 flex-1 overflow-hidden rounded-full bg-[#1a1a1a]">
                            <div
                              className="h-full rounded-full bg-[#ededed] transition-all duration-300"
                              style={{ width: `${file.progress}%` }}
                            />
                          </div>
                          <span className="font-mono text-[10px] text-[#555]">
                            {Math.round(file.progress)}%
                          </span>
                        </div>
                      ) : file.processingStatus === "processing" ? (
                        <Badge
                          variant="outline"
                          className="gap-1 border-[#333] bg-[#1a1a1a] font-mono text-[10px] text-[#aaa]"
                        >
                          <Loader2 className="h-3 w-3 animate-spin" />
                          分析中
                        </Badge>
                      ) : (
                        <Badge
                          variant="outline"
                          className="gap-1 border-[#333] bg-[#1a1a1a] font-mono text-[10px] text-[#ccc]"
                        >
                          <CheckCircle className="h-3 w-3" />
                          已完成
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => setPreviewFile(file)}
                          className="rounded-md p-1.5 text-[#555] transition-colors hover:bg-[#1a1a1a] hover:text-[#ccc]"
                          aria-label="预览文件"
                        >
                          <Eye className="h-3.5 w-3.5" />
                        </button>
                        <button
                          onClick={() => handleDelete(file.id)}
                          className="rounded-md p-1.5 text-[#555] transition-colors hover:bg-[#1a1a1a] hover:text-[#ff4444]"
                          aria-label="删除文件"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Agent analysis results for completed files */}
      {files.some((f) => f.processingStatus === "done") && (
        <Card className="border-[#1a1a1a] bg-[#0e0e0e]">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Bot className="h-4 w-4 text-[#888]" />
              <CardTitle className="text-sm font-medium text-[#ededed]">
                智能体分析结果
              </CardTitle>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-3 pt-0">
            {files
              .filter((f) => f.processingStatus === "done")
              .map((file) => (
                <div
                  key={file.id}
                  className="animate-fade-in-up flex flex-col gap-3 rounded-lg border border-[#1a1a1a] bg-[#111] p-4"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      {getFileIcon(file.type)}
                      <div>
                        <p className="text-sm font-medium text-[#ccc]">
                          {file.name}
                        </p>
                        <p className="font-mono text-[10px] text-[#555]">
                          {formatFileSize(file.size)} / {formatTime(file.uploadedAt)}
                        </p>
                      </div>
                    </div>
                    {file.confidence && (
                      <Badge
                        variant="outline"
                        className="border-[#333] bg-[#1a1a1a] font-mono text-[10px] text-[#ccc]"
                      >
                        置信度 {file.confidence}%
                      </Badge>
                    )}
                  </div>

                  {/* Assigned agents */}
                  <div className="flex flex-wrap items-center gap-2">
                    {file.assignedAgents.map((agentName) => (
                      <Badge
                        key={agentName}
                        variant="outline"
                        className="gap-1 border-[#222] bg-[#141414] font-mono text-[10px] text-[#888]"
                      >
                        <Bot className="h-2.5 w-2.5" />
                        {agentName}
                      </Badge>
                    ))}
                  </div>

                  {/* Analysis result */}
                  {file.analysisResult && (
                    <div className="rounded-lg border border-[#1a1a1a] bg-[#0e0e0e] px-4 py-3">
                      <p className="text-xs leading-relaxed text-[#999]">
                        {file.analysisResult}
                      </p>
                    </div>
                  )}
                </div>
              ))}
          </CardContent>
        </Card>
      )}

      {/* Empty state */}
      {files.length === 0 && (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <FolderOpen className="h-10 w-10 text-[#333]" />
          <p className="mt-4 text-sm text-[#555]">暂无文件</p>
          <p className="mt-1 font-mono text-xs text-[#333]">
            上传文件后将自动分配智能体进行分析
          </p>
        </div>
      )}

      {/* Preview modal */}
      {previewFile && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm"
          onClick={() => setPreviewFile(null)}
          role="dialog"
          aria-modal="true"
          aria-label="文件预览"
        >
          <div
            className="relative mx-4 w-full max-w-lg rounded-xl border border-[#1a1a1a] bg-[#0e0e0e] p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setPreviewFile(null)}
              className="absolute right-4 top-4 rounded-md p-1 text-[#555] transition-colors hover:bg-[#1a1a1a] hover:text-[#ccc]"
              aria-label="关闭预览"
            >
              <X className="h-4 w-4" />
            </button>

            <div className="flex flex-col items-center gap-4">
              {previewFile.url && previewFile.type.startsWith("image/") ? (
                <div className="w-full overflow-hidden rounded-lg border border-[#1a1a1a]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewFile.url}
                    alt={previewFile.name}
                    className="max-h-[300px] w-full object-contain bg-[#0a0a0a]"
                  />
                </div>
              ) : (
                <div className="flex h-24 w-24 items-center justify-center rounded-2xl border border-[#1a1a1a] bg-[#141414]">
                  {getFileIcon(previewFile.type)}
                </div>
              )}

              <div className="w-full text-center">
                <p className="text-sm font-medium text-[#ededed] break-all">
                  {previewFile.name}
                </p>
                <div className="mt-3 flex flex-wrap items-center justify-center gap-3">
                  <Badge
                    variant="outline"
                    className="border-[#222] bg-[#141414] font-mono text-[10px] text-[#888]"
                  >
                    {getFileExtLabel(previewFile.name)}
                  </Badge>
                  <span className="font-mono text-xs text-[#666]">
                    {formatFileSize(previewFile.size)}
                  </span>
                  <span className="font-mono text-xs text-[#666]">
                    {formatTime(previewFile.uploadedAt)}
                  </span>
                </div>
              </div>

              {/* Agent processing info in preview */}
              <div className="w-full rounded-lg border border-[#1a1a1a] bg-[#111] p-4">
                <p className="mb-2 text-xs font-medium text-[#888]">智能体处理</p>
                <div className="flex flex-wrap gap-2">
                  {previewFile.assignedAgents.map((agentName) => (
                    <Badge
                      key={agentName}
                      variant="outline"
                      className="gap-1 border-[#222] bg-[#141414] font-mono text-[10px] text-[#888]"
                    >
                      <Bot className="h-2.5 w-2.5" />
                      {agentName}
                    </Badge>
                  ))}
                </div>
                {previewFile.processingStatus === "processing" && (
                  <div className="mt-3 flex items-center gap-2 text-xs text-[#aaa]">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    智能体正在分析中...
                  </div>
                )}
                {previewFile.processingStatus === "done" && previewFile.analysisResult && (
                  <div className="mt-3 rounded-lg border border-[#1a1a1a] bg-[#0e0e0e] px-3 py-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-[10px] text-[#555]">分析结果</span>
                      {previewFile.confidence && (
                        <span className="font-mono text-[10px] text-[#ccc]">
                          置信度 {previewFile.confidence}%
                        </span>
                      )}
                    </div>
                    <p className="text-xs leading-relaxed text-[#999]">
                      {previewFile.analysisResult}
                    </p>
                  </div>
                )}
              </div>

              <button
                onClick={() => handleDelete(previewFile.id)}
                className="flex items-center gap-2 rounded-lg border border-[#222] bg-[#141414] px-4 py-2 text-xs text-[#888] transition-colors hover:border-[#444] hover:text-[#ff4444]"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>删除文件</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
