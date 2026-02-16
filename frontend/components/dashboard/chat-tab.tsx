"use client"

import { useState, useRef, useEffect } from "react"
import { Send, Bot, User, Clock, Gauge, Users, Loader2 } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  sampleMessages,
  quickPrompts,
  type ChatMessage,
} from "@/lib/dashboard-data"
import { askAgent } from "@/lib/api"

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user"

  return (
    <div
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      {/* Avatar */}
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
          isUser
            ? "bg-[#ffffff] text-[#000000]"
            : "bg-[#1a1a1a] text-[#888]"
        }`}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      {/* Content */}
      <div className={`flex max-w-[75%] flex-col ${isUser ? "items-end" : "items-start"}`}>
        {/* Agent header */}
        {!isUser && message.agent && (
          <div className="mb-1.5 flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-[#ccc]">
              {message.agent}
            </span>
            {message.confidence && (
              <Badge className="gap-1 border-[#333] bg-[#1a1a1a] px-2 py-0 text-[10px] text-[#999] hover:bg-[#1a1a1a]">
                <Gauge className="h-2.5 w-2.5" />
                {message.confidence}%
              </Badge>
            )}
            {message.latency && (
              <Badge className="gap-1 border-[#333] bg-[#1a1a1a] px-2 py-0 text-[10px] text-[#666] hover:bg-[#1a1a1a]">
                <Clock className="h-2.5 w-2.5" />
                {message.latency}ms
              </Badge>
            )}
            {message.collaborators && message.collaborators.length > 0 && (
              <Badge className="gap-1 border-[#333] bg-[#1a1a1a] px-2 py-0 text-[10px] text-[#888] hover:bg-[#1a1a1a]">
                <Users className="h-2.5 w-2.5" />
                +{message.collaborators.length} 个智能体
              </Badge>
            )}
          </div>
        )}

        {/* Bubble */}
        <div
          className={`rounded-xl px-4 py-3 ${
            isUser
              ? "bg-[#ffffff] text-[#000000]"
              : "border border-[#1a1a1a] bg-[#111] text-[#ccc]"
          }`}
        >
          <div className="whitespace-pre-wrap text-sm leading-relaxed">
            {message.content}
          </div>
        </div>

        {/* Collaborators detail */}
        {!isUser && message.collaborators && message.collaborators.length > 0 && (
          <div className="mt-1.5 flex items-center gap-1.5">
            <span className="font-mono text-[10px] text-[#444]">
              协同: {message.collaborators.join(", ")}
            </span>
          </div>
        )}

        {/* Timestamp */}
        <span className="mt-1 font-mono text-[10px] text-[#444]">
          {message.timestamp}
        </span>
      </div>
    </div>
  )
}

export function ChatTab() {
  const [messages, setMessages] = useState<ChatMessage[]>(sampleMessages)
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return
    const userQuery = input.trim()
    const newMsg: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: userQuery,
      timestamp: new Date().toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      }),
    }
    setMessages((prev) => [...prev, newMsg])
    setInput("")
    setIsLoading(true)

    try {
      const response = await askAgent(userQuery)
      setMessages((prev) => [...prev, response])
    } catch {
      // Fallback on API error
      const fallback: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: "assistant",
        content: "抱歉，处理请求时出现错误。请稍后再试。",
        agent: "系统",
        timestamp: new Date().toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          hour12: false,
        }),
      }
      setMessages((prev) => [...prev, fallback])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex h-[calc(100vh-10rem)] flex-col">
      <Card className="flex flex-1 flex-col overflow-hidden border-[#1a1a1a] bg-[#111]">
        {/* Messages */}
        <ScrollArea className="flex-1 p-6" ref={scrollRef}>
          <div className="flex flex-col gap-6">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
          </div>
        </ScrollArea>

        {/* Quick prompts */}
        <div className="border-t border-[#1a1a1a] px-6 pt-4">
          <div className="flex flex-wrap gap-2">
            {quickPrompts.map((prompt) => (
              <button
                key={prompt}
                onClick={() => setInput(prompt)}
                className="rounded-lg border border-[#1a1a1a] bg-[#0e0e0e] px-3 py-1.5 font-mono text-xs text-[#666] transition-colors hover:border-[#333] hover:text-[#ccc]"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>

        {/* Input */}
        <CardContent className="p-4">
          <div className="flex items-center gap-3 rounded-xl border border-[#1a1a1a] bg-[#0e0e0e] px-4 py-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend() } }}
              placeholder="向智能情报平台提问..."
              className="flex-1 bg-transparent font-mono text-sm text-[#ededed] placeholder:text-[#333] focus:outline-none"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#ffffff] text-[#000000] transition-opacity disabled:opacity-30"
              aria-label="发送消息"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
