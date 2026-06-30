import { motion } from 'framer-motion';

/* ============================================
   CustomModelShowcase — 自定义模型入口展示
   左侧：标题+描述+能力点+CTA
   右侧：配置面板示意 + 模块组合可视化
   动画：面板展开、模块组合、拖拽感
   ============================================ */

const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.07, delayChildren: 0.12 } },
};

const item = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
};

const formats = ['PyTorch', 'ONNX', 'TensorFlow', 'JAX'];

export function CustomModelShowcase() {
  return (
    <div className="mx-auto grid max-w-6xl gap-12 px-6 py-20 lg:grid-cols-2 lg:items-center">
      {/* ===== 左侧 ===== */}
      <motion.div variants={stagger} initial="hidden" animate="visible" className="space-y-5 order-2 lg:order-1">
        <motion.div variants={item}>
          <span className="inline-block rounded-full border border-cyan-500/15 bg-cyan-500/[0.05] px-3 py-1 text-[11px] font-bold tracking-wider text-cyan-400 uppercase">
            Custom Model
          </span>
        </motion.div>

        <motion.h2 variants={item} className="text-3xl font-bold leading-tight text-foreground sm:text-4xl">
          构建{' '}
          <span className="bg-gradient-to-r from-cyan-400 to-teal-400 bg-clip-text text-transparent">
            你的模型
          </span>
        </motion.h2>

        <motion.p variants={item} className="text-base leading-relaxed text-muted-foreground">
          上传自定义模型配置，即刻获得专属的可视化分析工作台。支持主流框架格式，灵活配置超参数与可视化模块。
        </motion.p>

        <motion.ul variants={stagger} className="space-y-2.5" initial="hidden" animate="visible">
          {[
            { label: '自定义超参数', desc: '学习率、Batch Size、Optimizer 一键配置' },
            { label: '模块化面板', desc: '按需启用 3D 结构、曲线图、混淆矩阵等' },
            { label: '导出分析报告', desc: '生成 PDF / HTML 格式的完整训练报告' },
          ].map((f) => (
            <motion.li key={f.label} variants={item} className="flex items-start gap-2.5 group">
              <span className="mt-1 size-1.5 shrink-0 rounded-full bg-cyan-400 transition-transform group-hover:scale-125" />
              <div>
                <div className="text-sm font-semibold text-foreground">{f.label}</div>
                <div className="mt-0.5 text-xs text-muted-foreground">{f.desc}</div>
              </div>
            </motion.li>
          ))}
        </motion.ul>

        <motion.div variants={item} className="flex flex-wrap gap-3">
          <a href="/workbench?model=custom" className="group inline-flex items-center gap-2 rounded-xl bg-cyan-500 px-7 py-3.5 text-sm font-semibold text-black shadow-lg shadow-cyan-500/20 transition-all hover:shadow-xl hover:shadow-cyan-500/30 active:scale-[0.98]">
            上传并开始
            <svg className="size-4 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
          </a>
          <a href="/workbench" className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-7 py-3.5 text-sm font-semibold text-foreground transition-all hover:border-cyan-500/25 hover:bg-cyan-500/[0.03] active:scale-[0.98]">
            先看演示
          </a>
        </motion.div>
      </motion.div>

      {/* ===== 右侧：上传 + 配置面板示意 ===== */}
      <motion.div
        initial={{ opacity: 0, filter: 'blur(8px)' }}
        animate={{ opacity: 1, filter: 'blur(0px)' }}
        transition={{ duration: 0.55, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
        className="order-1 lg:order-2"
      >
        {/* 主上传卡片 */}
        <div className="group relative overflow-hidden rounded-2xl border border-cyan-500/14 bg-gradient-to-br from-cyan-500/[0.05] to-[#080c14]/50 p-8 sm:p-10 shadow-lg shadow-black/20 backdrop-blur-md">
          {/* 背景网格纹理 */}
          <div
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage:
                'linear-gradient(oklch(0.68 0.16 180 / 0.4) 1px, transparent 1px), linear-gradient(90deg, oklch(0.68 0.16 180 / 0.4) 1px, transparent 1px)',
              backgroundSize: '28px 28px',
            }}
          />

          {/* 动态光球 */}
          <motion.div
            className="absolute -right-6 -top-6 h-36 w-36 rounded-full"
            style={{
              background: 'radial-gradient(circle, oklch(0.68 0.18 195 / 0.14), transparent 65%)',
              filter: 'blur(25px)',
            }}
            animate={{ scale: [1, 1.3, 1], x: [0, 12, 0] }}
            transition={{ duration: 6, repeat: Infinity }}
          />

          <div className="relative z-10 flex flex-col items-center gap-6">
            {/* 上传图标 */}
            <motion.div
              whileHover={{ rotate: 90, scale: 1.08 }}
              transition={{ duration: 0.35 }}
              className="flex size-20 items-center justify-center rounded-2xl border border-cyan-500/25 bg-cyan-500/[0.08] text-cyan-400 shadow-lg shadow-black/20 backdrop-blur-sm"
            >
              <svg className="size-9" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 16v-8m0 0l-3 3m3-3l3 3M3 16.25V18.75A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.25M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
              </svg>
            </motion.div>

            {/* 文字说明 */}
            <div className="text-center">
              <h3 className="text-lg font-bold text-foreground">导入自定义模型</h3>
              <p className="mt-1.5 text-sm text-muted-foreground">支持 PyTorch .pt / .pth、ONNX、TensorFlow SavedModel</p>
            </div>

            {/* 支持格式标签 */}
            <div className="flex flex-wrap justify-center gap-2">
              {formats.map((fmt) => (
                <span key={fmt} className="rounded-full border border-white/[0.08] bg-[#080c14]/60 px-3 py-1 text-xs text-muted-foreground transition-colors group-hover:border-cyan-500/25 group-hover:text-foreground group-hover:bg-[#0a0f1c]/80">
                  {fmt}
                </span>
              ))}
            </div>

            {/* 上传按钮 */}
            <button className="group/btn inline-flex items-center gap-2 rounded-xl bg-cyan-500 px-8 py-3.5 text-sm font-semibold text-black shadow-lg shadow-cyan-500/15 transition-all hover:shadow-cyan-500/30 hover:brightness-110 active:scale-[0.98]">
              <svg className="size-4 transition-transform group-hover/btn:-translate-y-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
              选择文件上传
            </button>
          </div>
        </div>

        {/* 配置选项条 */}
        <div className="mt-4 grid grid-cols-3 gap-3">
          {[
            { icon: '⚙', title: '参数配置', sub: '一键设置' },
            { icon: '▣', title: '面板选择', sub: '按需启用' },
            { icon: '📄', title: '报告导出', sub: 'PDF/HTML' },
          ].map((opt, i) => (
            <motion.div
              key={opt.title}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.55 + i * 0.08, duration: 0.4 }}
              whileHover={{ y: -3 }}
              className="rounded-xl border border-white/[0.07] bg-[#080c14]/60 p-4 text-center shadow-sm shadow-black/20 transition-all hover:border-cyan-500/20 hover:bg-[#0a0f1c]/70"
            >
              <div className="mb-1.5 text-lg">{opt.icon}</div>
              <h4 className="text-[11px] font-semibold text-foreground">{opt.title}</h4>
              <p className="text-[9px] text-muted-foreground">{opt.sub}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
