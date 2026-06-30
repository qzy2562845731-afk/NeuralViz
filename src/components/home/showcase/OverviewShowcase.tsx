import { motion } from 'framer-motion';

/* ============================================
   OverviewShowcase — 神经网络训练可视化总览
   左侧：标题+描述+能力点+CTA
   右侧：训练流程可视化（Loss曲线/指标卡片）
   动画：缩放淡入
   ============================================ */

const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08, delayChildren: 0.15 } },
};

const item = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
};

export function OverviewShowcase() {
  return (
    <div className="mx-auto grid max-w-6xl gap-12 px-6 py-20 lg:grid-cols-2 lg:items-center">
      {/* ===== 左侧：文案 + CTA ===== */}
      <motion.div variants={stagger} initial="hidden" animate="visible" className="space-y-6">
        <motion.div variants={item}>
          <span className="inline-block rounded-full border border-primary/15 bg-primary/[0.05] px-3 py-1 text-[11px] font-bold tracking-wider text-primary uppercase">
            Platform Overview
          </span>
        </motion.div>

        <motion.h2 variants={item} className="text-3xl font-bold leading-tight text-foreground sm:text-4xl">
          全流程{' '}
          <span className="bg-gradient-to-r from-primary to-[oklch(0.72_0.12_230)] bg-clip-text text-transparent">
            AI 驱动
          </span>{' '}
          训练洞察
        </motion.h2>

        <motion.p variants={item} className="text-base leading-relaxed text-muted-foreground">
          从数据输入到模型部署，每一个训练环节都清晰可见。
          实时监控 Loss 曲线、梯度分布、权重变化，AI 自动诊断异常并给出优化建议。
        </motion.p>

        {/* 能力点列表 */}
        <motion.ul variants={stagger} className="space-y-3" initial="hidden" animate="visible">
          {[
            { icon: '📊', label: '实时训练可视化', desc: 'Loss / Accuracy / 梯度全维度监控' },
            { icon: '✨', label: 'AI 智能诊断', desc: '自动检测过拟合、梯度爆炸等异常' },
            { icon: '🧊', label: '3D 模型结构', desc: '交互式神经网络架构探索' },
            { icon: '🔀', label: '多模型支持', desc: 'CNN / RNN / Transformer / BERT 一键切换' },
          ].map((f) => (
            <motion.li key={f.label} variants={item} className="flex items-start gap-3 group">
              <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-[#0f141f]/80 text-sm transition-colors group-hover:bg-primary/10">
                {f.icon}
              </span>
              <div>
                <div className="text-sm font-semibold text-foreground">{f.label}</div>
                <div className="mt-0.5 text-xs text-muted-foreground">{f.desc}</div>
              </div>
            </motion.li>
          ))}
        </motion.ul>

        {/* CTA */}
        <motion.div variants={item}>
          <a
            href="#features"
            className="group inline-flex items-center gap-2 rounded-xl bg-primary px-7 py-3.5 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:shadow-xl hover:shadow-primary/30 active:scale-[0.98]"
          >
            探索全部功能
            <svg className="size-4 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
          </a>
        </motion.div>
      </motion.div>

      {/* ===== 右侧：训练流程可视化面板 ===== */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.55, delay: 0.25, ease: [0.22, 1, 0.36, 1] }}
        className="relative"
      >
        {/* 主面板 */}
        <div className="overflow-hidden rounded-2xl border border-white/[0.08] bg-[#080c14]/60 shadow-lg shadow-black/20 backdrop-blur-md">
          {/* 面板头部 */}
          <div className="flex items-center justify-between border-b border-white/[0.04] px-6 py-4">
            <div className="flex items-center gap-2">
              <span className="size-2.5 rounded-full bg-green-500 animate-pulse" style={{ boxShadow: '0 0 6px rgba(34,197,94,0.5)' }} />
              <span className="text-xs font-medium text-foreground">Training Session Active</span>
            </div>
            <span className="font-mono text-[10px] text-muted-foreground tabular-nums">Epoch 47 / 100</span>
          </div>

          {/* 模拟 Loss 曲线区域 */}
          <div className="relative h-48 px-6 pt-6">
            {/* 网格线 */}
            <svg className="absolute inset-0 h-full w-full" preserveAspectRatio="none">
              {[25, 50, 75].map((y) => (
                <line key={y} x1="0" y1={`${y}%`} x2="100%" y2={`${y}%`} stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
              ))}
              {[20, 40, 60, 80].map((x) => (
                <line key={x} x1={`${x}%`} y1="0" x2={`${x}%`} y2="100%" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
              ))}
            </svg>
            {/* Loss 曲线 (SVG path) */}
            <svg className="relative z-10 h-full w-full" viewBox="0 0 400 160" preserveAspectRatio="none">
              <defs>
                <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="oklch(0.68 0.16 155 / 0.35)" />
                  <stop offset="100%" stopColor="oklch(0.68 0.16 155 / 0)" />
                </linearGradient>
              </defs>
              <path
                d="M0,140 C30,135 60,120 90,105 S150,70 180,58 S240,38 270,32 S330,22 360,18 L400,14"
                fill="none"
                stroke="oklch(0.68 0.16 155)"
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
              <path
                d="M0,140 C30,135 60,120 90,105 S150,70 180,58 S240,38 270,32 S330,22 360,18 L400,14 L400,160 L0,160 Z"
                fill="url(#lossGrad)"
              />
            </svg>
            {/* Y轴标签 */}
            <div className="absolute left-1 top-2 font-mono text-[9px] text-muted-foreground">Loss</div>
            <div className="absolute right-1 bottom-2 font-mono text-[9px] text-muted-foreground tabular-nums">0.023</div>
          </div>

          {/* 底部指标条 */}
          <div className="grid grid-cols-4 border-t border-white/[0.04]">
            {[
              { label: 'Loss', value: '0.023', color: 'text-red-400' },
              { label: 'Acc', value: '96.8%', color: 'text-green-400' },
              { label: 'LR', value: '1e-4', color: 'text-blue-400' },
              { label: 'Status', value: '●', color: 'text-green-400' },
            ].map((m) => (
              <div key={m.label} className="border-r border-white/[0.04] px-4 py-3 last:border-r-0 text-center">
                <div className={`font-mono text-sm font-bold tabular-nums ${m.color}`}>{m.value}</div>
                <div className="mt-0.5 text-[9px] text-muted-foreground">{m.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* 装饰光效 */}
        <div
          className="absolute -right-6 -top-6 h-32 w-32 rounded-full blur-3xl opacity-50"
          style={{ background: 'radial-gradient(circle, oklch(0.68 0.16 155 / 0.2), transparent 65%)' }}
        />
      </motion.div>
    </div>
  );
}
