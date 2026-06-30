import { motion } from 'framer-motion';

/* ============================================
   FlipText — 逐字跳跃翻转进入动画
   - 每个字符独立 3D 翻转 + 弹性落位
   - 支持渐变文字（渐变类作用于单字，避免 inline-block 导致消失）
   ============================================ */
function FlipText({
  text,
  className = '',
  charClassName = '',
  baseDelay = 0.3,
  stagger = 0.05,
}: {
  text: string;
  className?: string;
  charClassName?: string;
  baseDelay?: number;
  stagger?: number;
}) {
  return (
    <span
      className={`inline-block ${className}`}
      style={{ perspective: '1000px' }}
      aria-label={text}
    >
      {text.split('').map((char, i) => (
        <motion.span
          key={`${char}-${i}`}
          className={`inline-block will-change-transform ${charClassName}`}
          style={{ transformStyle: 'preserve-3d' }}
          initial={{
            opacity: 0,
            rotateX: -95,
            rotateY: 25,
            y: -35,
            scale: 0.7,
          }}
          animate={{
            opacity: 1,
            rotateX: 0,
            rotateY: 0,
            y: 0,
            scale: 1,
          }}
          transition={{
            type: 'spring',
            stiffness: 360,
            damping: 14,
            mass: 0.8,
            delay: baseDelay + i * stagger,
          }}
        >
          {char === ' ' ? '\u00A0' : char}
        </motion.span>
      ))}
    </span>
  );
}

/* ============================================
   HeroSection — 产品舞台级主视觉（增强版）
   - 强聚焦大标题 + 克制科技背景
   - 双 CTA 均可点击跳转
   - 向下滚动自然衔接，向上返回重新聚焦
   ============================================ */

export function HeroSection() {
  const scrollToShowcase = () => {
    const el = document.getElementById('showcase');
    if (el) {
      window.scrollTo({ top: el.offsetTop - 64, behavior: 'smooth' });
    }
  };

  return (
    <section
      id="hero"
      className="relative flex min-h-screen items-center justify-center overflow-hidden"
    >
      {/* ===== 背景层：与全局动态背景兼容的轻柔叠加 ===== */}
      <div className="absolute inset-0">
        {/* 底部暗角，让内容更聚焦 */}
        <div
          className="absolute inset-0"
          style={{
            background: `
              radial-gradient(ellipse 80% 55% at 50% 25%, oklch(0.68 0.16 155 / 0.06) 0%, transparent 60%),
              radial-gradient(ellipse 100% 50% at 50% 100%, rgba(3, 7, 18, 0.8) 0%, transparent 60%)
            `,
          }}
        />
      </div>

      {/* ===== 内容区 ===== */}
      <div className="relative z-10 mx-auto max-w-5xl px-6 text-center">
        {/* 标签徽章 */}
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.55, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
          className="mb-8 inline-flex items-center gap-2 rounded-full border border-primary/15 bg-primary/[0.04] px-5 py-2 backdrop-blur-sm"
        >
          <span className="size-2 animate-pulse rounded-full bg-primary" style={{ boxShadow: '0 0 6px oklch(0.68 0.16 155 / 0.6)' }} />
          <span className="text-sm font-medium text-primary">Neural Network Visualization Platform</span>
        </motion.div>

        {/* 主标题 — 逐字跳跃翻转进入 */}
        <h1 className="mb-5 text-5xl font-bold leading-tight tracking-tight sm:text-6xl lg:text-7xl">
          <span className="block text-foreground">
            <FlipText text="神经网络" baseDelay={0.28} stagger={0.06} />
          </span>
          <span className="block">
            <FlipText
              text="训练可视化引擎"
              baseDelay={0.55}
              stagger={0.05}
              charClassName="bg-gradient-to-r from-primary via-[oklch(0.72_0.12_230)] to-[oklch(0.70_0.14_180)] bg-clip-text text-transparent"
            />
          </span>
        </h1>

        {/* 副标题 */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 1.15, ease: [0.22, 1, 0.36, 1] }}
          className="mx-auto mb-10 max-w-2xl text-lg leading-relaxed text-muted-foreground sm:text-xl"
        >
          实时洞察模型训练全流程，从 CNN 到 Transformer，
          <br className="hidden sm:block" />
          让每一次参数更新都清晰可见
        </motion.p>

        {/* CTA 按钮组 — 全部可交互 */}
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 1.32, ease: [0.22, 1, 0.36, 1] }}
          className="flex flex-col items-center justify-center gap-4 sm:flex-row"
        >
          <button
            onClick={scrollToShowcase}
            className="group relative inline-flex items-center gap-2 overflow-hidden rounded-xl bg-primary px-8 py-4 text-base font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:shadow-xl hover:shadow-primary/30 active:scale-[0.98]"
          >
            开始探索
            <svg className="size-4 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          <a
            href="/workbench"
            className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-8 py-4 text-base font-semibold text-foreground transition-all hover:border-primary/25 hover:bg-primary/[0.03] active:scale-[0.98]"
          >
            进入工作台
          </a>
        </motion.div>

        {/* 数据指标条 */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 1.55 }}
          className="mt-16 flex items-center justify-center gap-8 border-t border-white/[0.05] pt-8 sm:gap-16"
        >
          {[
            { label: '支持模型', value: '6+' },
            { label: '可视化维度', value: '3D' },
            { label: '实时分析', value: 'AI' },
            { label: '训练步骤', value: '∞' },
          ].map((item) => (
            <div key={item.label} className="group cursor-default text-center">
              <div className="text-2xl font-bold text-foreground tabular-nums transition-colors group-hover:text-primary">{item.value}</div>
              <div className="mt-1 text-xs text-muted-foreground">{item.label}</div>
            </div>
          ))}
        </motion.div>
      </div>

      {/* 底部渐隐遮罩 — 与下一屏平滑过渡 */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />

      {/* 向下滚动提示 — 点击也可跳转 */}
      <motion.button
        onClick={scrollToShowcase}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 cursor-pointer text-muted-foreground/40 transition-colors hover:text-muted-foreground"
        animate={{ y: [0, 7, 0] }}
        transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
        aria-label="向下滚动"
      >
        <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </motion.button>
    </section>
  );
}
