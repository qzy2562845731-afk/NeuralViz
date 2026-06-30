import { motion } from 'framer-motion';

/* ============================================
   RNNShowcase — 循环神经网络专题展示
   左侧：标题+描述+能力点+CTA
   右侧：时序展开可视化（隐藏状态流动）
   动画：逐步串联、流动推进
   ============================================ */

const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.07, delayChildren: 0.12 } },
};

const item = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
};

export function RNNShowcase() {
  return (
    <div className="mx-auto grid max-w-6xl gap-12 px-6 py-20 lg:grid-cols-2 lg:items-center">
      {/* ===== 左侧 ===== */}
      <motion.div variants={stagger} initial="hidden" animate="visible" className="space-y-5 order-2 lg:order-1">
        <motion.div variants={item}>
          <span className="inline-block rounded-full border border-emerald-500/15 bg-emerald-500/[0.05] px-3 py-1 text-[11px] font-bold tracking-wider text-emerald-400 uppercase">
            RNN
          </span>
        </motion.div>

        <motion.h2 variants={item} className="text-3xl font-bold leading-tight text-foreground sm:text-4xl">
          循环神经网络{' '}
          <span className="text-emerald-400">时序分析</span>
        </motion.h2>

        <motion.p variants={item} className="text-base leading-relaxed text-muted-foreground">
          捕捉时间序列中的隐藏依赖关系。可视化隐藏状态演化轨迹、记忆衰减曲线、门控机制动态。
        </motion.p>

        <motion.ul variants={stagger} className="space-y-2.5" initial="hidden" animate="visible">
          {[
            { label: '隐藏状态轨迹', desc: '高维隐状态在训练过程中的降维投影与演化' },
            { label: '梯度爆炸检测', desc: '实时监控梯度范数，预警数值不稳定风险' },
            { label: '序列注意力热力图', desc: '模型对输入序列各位置的关注度分布' },
          ].map((f) => (
            <motion.li key={f.label} variants={item} className="flex items-start gap-2.5 group">
              <span className="mt-1 size-1.5 shrink-0 rounded-full bg-emerald-400 transition-transform group-hover:scale-125" />
              <div>
                <div className="text-sm font-semibold text-foreground">{f.label}</div>
                <div className="mt-0.5 text-xs text-muted-foreground">{f.desc}</div>
              </div>
            </motion.li>
          ))}
        </motion.ul>

        <motion.div variants={item}>
          <a href="/workbench?model=rnn" className="group inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-7 py-3.5 text-sm font-semibold text-black shadow-lg shadow-emerald-500/20 transition-all hover:shadow-xl hover:shadow-emerald-500/30 active:scale-[0.98]">
            进入 RNN 工作台
            <svg className="size-4 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
          </a>
        </motion.div>
      </motion.div>

      {/* ===== 右侧：时序流动可视化 ===== */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
        className="order-1 lg:order-2"
      >
        <div className="overflow-hidden rounded-2xl border border-emerald-500/12 bg-gradient-to-b from-emerald-500/[0.05] to-[#080c14]/50 p-8 shadow-lg shadow-black/20 backdrop-blur-md">
          <h3 className="mb-8 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Temporal Unrolling</h3>

          {/* 时间步序列 */}
          <div className="flex items-end justify-center gap-3 sm:gap-4">
            {[1, 2, 3, 4, 5].map((t) => (
              <motion.div
                key={t}
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 60 + t * 28, opacity: 1 }}
                transition={{
                  delay: 0.25 + (t - 1) * 0.12,
                  duration: 0.5,
                  ease: [0.22, 1, 0.36, 1],
                }}
                whileHover={{ scaleY: 1.08 }}
                className="relative flex w-14 flex-col items-center gap-2"
              >
                {/* 状态柱 */}
                <div
                  className="w-full rounded-t-lg transition-all"
                  style={{
                    height: `${60 + t * 28}px`,
                    minHeight: 44,
                    background: `linear-gradient(to top, oklch(0.68 0.16 155 / 0.2), oklch(0.58 0.18 155 / 0.55))`,
                  }}
                />
                {/* 时间标签 */}
                <span className="font-mono text-[10px] text-muted-foreground tabular-nums">t-{6 - t}</span>

                {/* 循环箭头装饰（非最后一个） */}
                {t < 5 && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 0.3 }}
                    transition={{ delay: 0.8 + t * 0.15 }}
                    className="absolute -right-2 top-[30%] hidden sm:block"
                  >
                    <svg className="size-4 text-emerald-400/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                    </svg>
                  </motion.div>
                )}
              </motion.div>
            ))}

            {/* 输出箭头 + 输出节点 */}
            <motion.div
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 1.0, type: 'spring' }}
              className="flex flex-col items-center gap-2"
            >
              <svg className="size-5 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
              </svg>
              <div className="flex size-16 items-center justify-center rounded-xl border-2 border-emerald-500/30 bg-emerald-500/[0.08] text-xs font-bold text-emerald-400 shadow-lg shadow-black/20 backdrop-blur-sm">
                Output
              </div>
            </motion.div>
          </div>

          {/* 说明文字 */}
          <p className="mt-8 text-center text-[11px] leading-relaxed text-muted-foreground">
            隐藏状态随时间步动态传递，每个时刻的输出依赖于全部历史信息。
            <br />
            可视化展示记忆衰减模式与梯度传播路径。
          </p>
        </div>
      </motion.div>
    </div>
  );
}
