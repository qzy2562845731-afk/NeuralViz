import { motion } from 'framer-motion';

/* ============================================
   CTASection — 底部行动召唤区（增强版）
   双 CTA 按钮均可点击跳转
   动态光球 + 装饰动效
   ============================================ */

export function CTASection() {
  return (
    <section className="relative overflow-hidden py-28">
      {/* 背景：与视频融合的中央光晕 */}
      <div className="pointer-events-none absolute inset-0">
        <div
          className="absolute inset-0"
          style={{
            background: `
              radial-gradient(ellipse 75% 45% at 50% 35%, oklch(0.68 0.16 155 / 0.08), transparent 55%)
            `,
          }}
        />
      </div>

      <div className="relative z-10 mx-auto max-w-4xl px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 26 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.58 }}
        >
          {/* 主标题 */}
          <h2 className="text-3xl font-bold leading-tight text-foreground sm:text-4xl lg:text-5xl">
            准备好{' '}
            <span className="bg-gradient-to-r from-primary via-[oklch(0.72_0.12_230)] to-[oklch(0.70_0.14_180)] bg-clip-text text-transparent">
              洞察你的模型
            </span>{' '}
            了吗？
          </h2>

          <p className="mx-auto mt-6 max-w-xl text-base leading-relaxed text-muted-foreground sm:text-lg">
            立即开始使用，让每一次训练都变得透明、可控、可优化
          </p>

          {/* CTA 按钮组 — 全部可交互 */}
          <div className="mt-11 flex flex-col items-center justify-center gap-4 sm:flex-row">
            <a
              href="/workbench"
              className="group relative inline-flex items-center gap-2.5 rounded-xl bg-primary px-10 py-4 text-base font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:shadow-xl hover:shadow-primary/30 active:scale-[0.98]"
            >
              进入工作台
              <svg className="size-4 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
              </svg>
            </a>
            <button
              onClick={() => {
                const el = document.getElementById('showcase');
                if (el) window.scrollTo({ top: el.offsetTop - 64, behavior: 'smooth' });
              }}
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.02] px-10 py-4 text-base font-semibold text-foreground backdrop-blur-sm transition-all hover:border-white/18 hover:bg-white/[0.04] active:scale-[0.98]"
            >
              返回顶部预览
            </button>
          </div>

          {/* 底部辅助信息 */}
          <div className="mt-14 flex flex-wrap items-center justify-center gap-x-8 gap-y-3 border-t border-white/[0.04] pt-8 text-sm text-muted-foreground">
            {[
              { label: '无需安装', icon: '☁' },
              { label: '支持主流框架', icon: '⚡' },
              { label: '实时渲染', icon: '◉' },
              { label: '免费使用', icon: '✦' },
            ].map((item) => (
              <span key={item.label} className="flex items-center gap-1.5">
                <span className="text-primary">{item.icon}</span>
                {item.label}
              </span>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
