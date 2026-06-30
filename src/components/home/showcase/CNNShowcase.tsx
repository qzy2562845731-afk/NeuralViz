import { motion } from 'framer-motion';

/* ============================================
   CNNShowcase — 卷积神经网络专题展示
   左侧：标题+描述+能力点+CTA
   右侧：CNN 层级结构可视化（卷积/池化/全连接）
   动画：从左滑入 + 分层展开
   ============================================ */

const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.07, delayChildren: 0.12 } },
};

const item = {
  hidden: { opacity: 0, x: -20 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
};

const layers = [
  { name: 'Input', shape: '224×224×3', color: '#6366f1', w: 14 },
  { name: 'Conv2D', shape: '112×112×64', color: '#8b5cf6', w: 16 },
  { name: 'Conv2D', shape: '56×56×128', color: '#a855f7', w: 18 },
  { name: 'MaxPool', shape: '28×28×128', color: '#d946ef', w: 20 },
  { name: 'Conv2D', shape: '14×14×256', color: '#ec4899', w: 22 },
  { name: 'GlobalAvgPool', shape: '256', color: '#f43f5e', w: 24 },
  { name: 'Dense', shape: '10 classes', color: '#ef4444', w: 26 },
];

export function CNNShowcase() {
  return (
    <div className="mx-auto grid max-w-6xl gap-12 px-6 py-20 lg:grid-cols-2 lg:items-center">
      {/* ===== 左侧 ===== */}
      <motion.div variants={stagger} initial="hidden" animate="visible" className="space-y-5 order-2 lg:order-1">
        <motion.div variants={item}>
          <span className="inline-block rounded-full border border-blue-500/15 bg-blue-500/[0.05] px-3 py-1 text-[11px] font-bold tracking-wider text-blue-400 uppercase">
            CNN
          </span>
        </motion.div>

        <motion.h2 variants={item} className="text-3xl font-bold leading-tight text-foreground sm:text-4xl">
          卷积神经网络{' '}
          <span className="text-blue-400">可视化</span>
        </motion.h2>

        <motion.p variants={item} className="text-base leading-relaxed text-muted-foreground">
          从输入图像到分类输出，逐层追踪特征提取过程。查看卷积核激活模式、特征图空间分布、梯度流向。
        </motion.p>

        <motion.ul variants={stagger} className="space-y-2.5" initial="hidden" animate="visible">
          {[
            { label: '卷积核可视化', desc: '逐通道查看滤波器模式与激活响应' },
            { label: '特征图热力图', desc: '每层输出的空间特征分布一目了然' },
            { label: '梯度流追踪', desc: '反向传播路径的完整可视化呈现' },
          ].map((f) => (
            <motion.li key={f.label} variants={item} className="flex items-start gap-2.5 group">
              <span className="mt-1 size-1.5 shrink-0 rounded-full bg-blue-400 transition-transform group-hover:scale-125" />
              <div>
                <div className="text-sm font-semibold text-foreground">{f.label}</div>
                <div className="mt-0.5 text-xs text-muted-foreground">{f.desc}</div>
              </div>
            </motion.li>
          ))}
        </motion.ul>

        <motion.div variants={item}>
          <a href="/workbench?model=cnn" className="group inline-flex items-center gap-2 rounded-xl bg-blue-500 px-7 py-3.5 text-sm font-semibold text-white shadow-lg shadow-blue-500/20 transition-all hover:shadow-xl hover:shadow-blue-500/30 active:scale-[0.98]">
            进入 CNN 工作台
            <svg className="size-4 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
          </a>
        </motion.div>
      </motion.div>

      {/* ===== 右侧：CNN 层级结构可视化 ===== */}
      <motion.div
        initial={{ opacity: 0, x: 60 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.55, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
        className="order-1 lg:order-2"
      >
        <div className="overflow-hidden rounded-2xl border border-blue-500/12 bg-gradient-to-b from-blue-500/[0.05] to-[#080c14]/50 p-8 shadow-lg shadow-black/20 backdrop-blur-md">
          <h3 className="mb-6 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Network Architecture</h3>

          {/* 层级流 */}
          <div className="relative flex flex-col gap-4">
            {layers.map((layer, i) => (
              <motion.div
                key={layer.name}
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.25 + i * 0.1, duration: 0.45, ease: 'easeOut' }}
                whileHover={{ x: 6, transition: { duration: 0.2 } }}
                className="group relative flex items-center gap-4"
              >
                {/* 连接线 */}
                {i > 0 && (
                  <div className="absolute left-[19px] -top-4 h-4 w-[2px] bg-gradient-to-b from-white/10 to-white/5" />
                )}

                {/* 节点圆圈 */}
                <div
                  className="relative z-10 flex shrink-0 items-center justify-center rounded-xl text-xs font-bold text-white shadow-lg"
                  style={{
                    width: `${layer.w * 3}px`,
                    height: `${layer.w * 3}px`,
                    minWidth: 40,
                    minHeight: 40,
                    background: `linear-gradient(135deg, ${layer.color}aa, ${layer.color}44)`,
                    boxShadow: `0 4px 20px ${layer.color}30`,
                    maxWidth: 72,
                    maxHeight: 72,
                  }}
                >
                  {i + 1}
                </div>

                {/* 层信息 */}
                <div className="flex-1 rounded-lg border border-white/[0.07] bg-[#080c14]/60 px-4 py-2.5 transition-colors group-hover:border-blue-500/20 group-hover:bg-blue-500/[0.05] backdrop-blur-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-foreground">{layer.name}</span>
                    <span className="font-mono text-[10px] text-muted-foreground tabular-nums">{layer.shape}</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* 底部说明 */}
          <p className="mt-6 text-center text-[11px] text-muted-foreground">
            点击任意层级查看详细参数、激活图和梯度分布
          </p>
        </div>
      </motion.div>
    </div>
  );
}
