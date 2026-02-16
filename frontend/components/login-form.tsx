"use client"

import { useState } from "react"
import Image from "next/image"
import { Eye, EyeOff, ArrowRight, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { login } from "@/lib/api"

interface LoginFormProps {
  onLogin: () => void
}

export function LoginForm({ onLogin }: LoginFormProps) {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (!email || !password) {
      setError("请填写所有字段")
      return
    }

    setIsLoading(true)

    try {
      await login(email, password)
      setIsLoading(false)
      onLogin()
    } catch (err) {
      setIsLoading(false)
      setError(err instanceof Error ? err.message : "登录失败，请检查凭据")
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#0a0a0a] px-4">
      {/* Subtle grid background */}
      <div
        className="pointer-events-none fixed inset-0"
        style={{
          backgroundImage:
            "linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      <div className="relative z-10 w-full max-w-[400px]">
        {/* Logo and branding */}
        <div className="mb-10 flex flex-col items-center">
          <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl border border-[#222] bg-[#111]">
            <Image
              src="/images/logo_horse.png"
              alt="MRARFAI Logo"
              width={40}
              height={40}
              className="invert"
            />
          </div>
          <h1 className="mb-1 text-2xl font-semibold tracking-tight text-[#ededed]">
            MRARFAI
          </h1>
          <p className="text-sm text-[#666]">
            多智能体销售情报平台
          </p>
        </div>

        {/* Login card */}
        <div className="rounded-xl border border-[#1a1a1a] bg-[#0e0e0e] p-8">
          <div className="mb-6">
            <h2 className="text-lg font-medium text-[#ededed]">登录</h2>
            <p className="mt-1 text-sm text-[#555]">
              请输入您的凭据以访问仪表盘
            </p>
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col gap-5">
            <div className="flex flex-col gap-2">
              <Label htmlFor="email" className="text-[#999] text-xs tracking-wider">
                邮箱
              </Label>
              <Input
                id="email"
                type="email"
                placeholder="admin@sprocomm.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading}
                className="h-11 border-[#1a1a1a] bg-[#111] text-[#ededed] placeholder:text-[#444] focus-visible:ring-[#333] focus-visible:ring-offset-0 focus-visible:ring-offset-[#0e0e0e]"
              />
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="password" className="text-[#999] text-xs tracking-wider">
                  密码
                </Label>
                <button
                  type="button"
                  className="text-xs text-[#555] transition-colors hover:text-[#999]"
                >
                  忘记密码?
                </button>
              </div>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="请输入密码"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isLoading}
                  className="h-11 border-[#1a1a1a] bg-[#111] pr-10 text-[#ededed] placeholder:text-[#444] focus-visible:ring-[#333] focus-visible:ring-offset-0 focus-visible:ring-offset-[#0e0e0e]"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[#555] transition-colors hover:text-[#999]"
                  aria-label={showPassword ? "隐藏密码" : "显示密码"}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {error && (
              <p className="text-sm text-[#ff4444]">{error}</p>
            )}

            <Button
              type="submit"
              disabled={isLoading}
              className="mt-1 h-11 bg-[#ededed] text-[#0a0a0a] hover:bg-[#fff] font-medium"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>登录中...</span>
                </>
              ) : (
                <>
                  <span>登录</span>
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </Button>
          </form>

          {/* Divider */}
          <div className="my-6 flex items-center gap-3">
            <div className="h-px flex-1 bg-[#1a1a1a]" />
            <span className="text-xs text-[#444]">或</span>
            <div className="h-px flex-1 bg-[#1a1a1a]" />
          </div>

          {/* SSO button */}
          <Button
            variant="outline"
            className="h-11 w-full border-[#1a1a1a] bg-transparent text-[#999] hover:bg-[#151515] hover:text-[#ededed]"
            disabled={isLoading}
          >
            企业 SSO 登录
          </Button>
        </div>

        {/* Footer */}
        <div className="mt-6 flex items-center justify-center gap-1.5 text-xs text-[#444]">
          <span>SPROCOMM (01401.HK)</span>
          <span>{"/"}</span>
          <span>安全访问</span>
        </div>
      </div>
    </div>
  )
}
