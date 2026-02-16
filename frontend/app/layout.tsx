import type { Metadata, Viewport } from 'next'
import { DM_Sans, JetBrains_Mono } from 'next/font/google'

import './globals.css'

const dmSans = DM_Sans({
  subsets: ['latin'],
  variable: '--font-dm-sans',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
})

export const metadata: Metadata = {
  title: 'MRARFAI - 多智能体销售情报平台',
  description: 'SPROCOMM ODM/OEM 移动设备智能分析仪表盘',
}

export const viewport: Viewport = {
  themeColor: '#0a0a0a',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN" className={`${dmSans.variable} ${jetbrainsMono.variable}`}>
      <body className="font-sans antialiased" style={{ backgroundColor: '#0a0a0a' }}>{children}</body>
    </html>
  )
}
