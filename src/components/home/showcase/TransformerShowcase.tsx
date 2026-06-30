import { motion } from 'framer-motion';

/* ============================================
   TransformerShowcase — Transformer 专题展示（最强科技感）
   左侧：标题+描述+能力点+CTA
   右侧：Attention 矩阵 + 架构模块 + 动态连接线
   动画：矩阵连接、节点激活、线条扩散
   ============================================ */

const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.07, delayChildren: 0.12 } },
};

const item = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
};

const blocks = [
  { label: 'Multi-Head Attention', sub: 'Q · K · V', icon: '◈' },
  { label: 'Add & Layer Norm', sub: '残差连接', icon: '⊕' },
  { label: 'Feed Forward', sub: 'MLP × 2', icon: '⇅' },
  { label: 'Positional Encoding', sub: 'Sin / Cos', icon: '⟳' },
];

export function TransformerShowcase() {
  return (
    <div className="mx-auto grid max-w-6xl gap-12 px-6 py-20 lg:grid-cols-2 lg:items-center">
      {/* ===== 左侧 ===== */}
      <motion.div variants={stagger} initial="hidden" animate="visible" className="space-y-5 order-2 lg:order-1">
        <motion.div variants={item}>
          <span className="inline-block rounded-full border border-violet-500/15 bg-violet-500/[0.05] px-3 py-1 text-[11px] font-bold tracking-wider text-violet-400 uppercase">
            Transformer
          </span>
        </motion.div>

        <motion.h2 variants={item} className="text-3xl font-bold leading-tight text-foreground sm:text-4xl">
          <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent">
            Transformer
          </span>{' '}
          架构解析
        </motion.h2>

        <motion.p variants={item} className="text-base leading-relaxed text-muted-foreground">
          自注意力机制的完整可视化。多头注意力权重矩阵、位置编码分布、FFN 激活模式，全局依赖关系一目了然。
        </motion.p>

        <motion.ul variants={stagger} className="space-y-2.5" initial="hidden" animate="visible">
          {[
            { label: 'Attention Matrix', desc: '8×8 多头注意力权重热力图交互式探索' },
            { label: 'QKV 投影', desc: 'Query/Key/Value 向量空间的可视化映射' },
            { label: '残差流追踪', desc: 'Skip Connection 路径上的信息流动分析' },
          ].map((f) => (
            <motion.li key={f.label} variants={item} className="flex items-start gap-2.5 group">
              <span className="mt-1 size-1.5 shrink-0 rounded-full bg-violet-400 transition-transform group-hover:scale-125" />
              <div>
                <div className="text-sm font-semibold text-foreground">{f.label}</div>
                <div className="mt-0.5 text-xs text-muted-foreground">{f.desc}</div>
              </div>
            </motion.li>
          ))}
        </motion.ul>

        <motion.div variants={item}>
          <a href="/workbench?model=transformer" className="group inline-flex items-center gap-2 rounded-xl bg-violet-500 px-7 py-3.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/20 transition-all hover:shadow-xl hover:shadow-violet-500/30 active:scale-[0.98]">
            进入 Transformer 工作台
            <svg className="size-4 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
          </a>
        </motion.div>
      </motion.div>

      {/* ===== 右侧：架构 + Attention 矩阵可视化 ===== */}
      <motion.div
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.55, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
        className="order-1 lg:order-2 space-y-5"
      >
        {/* Encoder Block 模块 */}
        <div className="relative overflow-hidden rounded-2xl border border-violet-500/12 bg-gradient-to-b from-violet-500/[0.05] to-[#080c14]/50 p-6 shadow-lg shadow-black/20 backdrop-blur-md">
          {/* 动态光效 */}
          <motion.div
            className="absolute -left-8 -top-8 h-40 w-40 rounded-full"
            style={{
              background: 'radial-gradient(circle, oklch(0.62 0.18 290 / 0.12), transparent 65%)',
              filter: 'blur(30px)',
            }}
            animate={{ scale: [1, 1.25, 1], opacity: [0.4, 0.7, 0.4] }}
            transition={{ duration: 5, repeat: Infinity }}
          />

          <div className="relative z-10 grid grid-cols-2 gap-3 sm:grid-cols-4">
            {blocks.map((block, i) => (
              <motion.div
                key={block.label}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.28 + i * 0.1, duration: 0.45 }}
                whileHover={{ y: -5, transition: { duration: 0.2 } }}
                className="group relative rounded-xl border border-white/[0.07] bg-[#080c14]/60 p-4 text-center shadow-sm shadow-black/20 backdrop-blur-md transition-all hover:border-violet-500/30 hover:bg-[#0a0f1c]/70"
              >
                <div className="mb-2 text-2xl opacity-60 group-hover:opacity-100 transition-opacity">{block.icon}</div>
                <h4 className="text-[11px] font-bold text-foreground tracking-wide leading-tight">{block.label}</h4>
                <p className="mt-1 text-[9px] text-muted-foreground font-mono">{block.sub}</p>
                {/* 底部高亮线 */}
                <div className="absolute bottom-0 left-1/2 h-[1.5px] w-0 rounded-full bg-violet-400/50 transition-all duration-300 group-hover:w-3/4 -translate-x-1/2" />
              </motion.div>
            ))}
          </div>

          {/* 流程箭头 */}
          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ delay: 0.85, duration: 0.5 }}
            className="relative z-10 mt-5 flex items-center justify-center gap-2 text-[11px] text-muted-foreground"
          >
            <span>Embed</span>
            <svg className="size-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
            <span className="font-semibold text-violet-400">Encoder × N</span>
            <svg className="size-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
            <span>Output</span>
          </motion.div>
        </div>

        {/* Self-Attention 矩阵预览 */}
        <div className="overflow-hidden rounded-2xl border border-white/[0.07] bg-[#080c14]/60 p-6 shadow-lg shadow-black/20 backdrop-blur-md">
          <h3 className="mb-4 flex items-center gap-2 text-xs font-semibold text-foreground">
            <span className="size-1.5 rounded-full bg-violet-400 animate-pulse" />
            Self-Attention Matrix Preview
          </h3>
          <div className="grid grid-cols-8 gap-[2px]">
            {Array.from({ length: 64 }).map((_, idx) => {
              const row = Math.floor(idx / 8);
              const col = idx % 8;
              // 模拟注意力模式：对角线强 + 随机噪声
              const diag = 1 - Math.abs(row - col) / 7;
              const noise = Math.sin(idx * 0.47 + col * 0.31) * 0.15;
              const intensity = Math.max(0, Math.min(1, diag * 0.7 + noise + 0.15));

              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{
                    delay: 0.4 + idx * 0.012,
                    duration: 0.18,
                    ease: 'easeOut',
                  }}
                  className="aspect-square rounded-[1px]"
                  style={{
                    backgroundColor: `oklch(${0.52 + intensity * 0.26} ${0.10 + intensity * 0.08} 280 / ${0.25 + intensity * 0.6})`,
                  }}
                />
              );
            })}
          </div>
          <p className="mt-3 text-center text-[10px] text-muted-foreground">
            8×8 注意力权重矩阵 — 实际支持 N×N 可交互热力图
          </p>
        </div>
      </motion.div>
    </div>
  );
}
