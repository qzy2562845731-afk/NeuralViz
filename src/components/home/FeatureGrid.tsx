import { motion } from 'framer-motion';

/* ============================================
   FeatureGrid — 产品功能亮点区（增强版）
   6 张功能卡片，统一 hover 效果，分组入场
   上滑返回时有轻微回收效果
   ============================================ */

interface Feature {
  icon: React.ReactNode;
  title: string;
  description: string;
  tag?: string;
  color?: string; // CTA 按钮颜色
}

const features: Feature[] = [
  {
    tag: '核心',
    color: 'bg-primary',
    icon: (
      <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5" />
      </svg>
    ),
    title: '模型训练可视化',
    description:
      '实时渲染 Loss、Accuracy、学习率等关键指标曲线，支持多实验对比与区间缩放分析。',
  },
  {
    tag: 'AI',
    color: 'bg-violet-500',
    icon: (
      <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
      </svg>
    ),
    title: 'AI 智能诊断',
    description:
      '基于规则引擎自动检测过拟合、梯度异常、学习率衰减不当等问题，并给出优化建议。',
  },
  {
    color: 'bg-blue-500',
    icon: (
      <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
      </svg>
    ),
    title: '模型结构展示',
    description:
      '交互式 3D 神经网络架构图，支持层级展开、参数查看、连接关系探索。',
  },
  {
    tag: 'Multi',
    color: 'bg-emerald-500',
    icon: (
      <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 018 20.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" />
      </svg>
    ),
    title: '多模型支持',
    description:
      '原生支持 CNN、RNN、Transformer、BERT 等主流架构，每种模型提供专属视角与分析工具。',
  },
  {
    color: 'bg-orange-500',
    icon: (
      <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
      </svg>
    ),
    title: '实时回放控制',
    description:
      '训练过程完整录制，支持步进、变速回放、跳转到任意 Epoch，像视频一样复盘训练全过程。',
  },
  {
    color: 'bg-pink-500',
    icon: (
      <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
      </svg>
    ),
    title: '一键导出报告',
    description:
      '将训练全过程的可视化数据、诊断结论、优化建议打包为结构化报告，便于分享与存档。',
  },
];

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08, delayChildren: 0.15 } },
};

const cardVariants = {
  hidden: { opacity: 0, y: 28 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.52, ease: [0.22, 1, 0.36, 1] },
  },
};

export function FeatureGrid() {
  return (
    <section id="features" className="relative py-28 overflow-hidden">
      {/* 背景：与视频融合的顶部光晕 */}
      <div className="pointer-events-none absolute inset-0">
        <div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(ellipse at 50% 0%, oklch(0.68 0.16 155 / 0.05) 0%, transparent 55%)',
          }}
        />
      </div>

      <div className="relative z-10 mx-auto max-w-6xl px-6">
        {/* 标题 */}
        <motion.div
          initial={{ opacity: 0, y: 22 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.55 }}
          className="mb-14 text-center"
        >
          <span className="mb-3 inline-block rounded-full border border-primary/12 bg-primary/[0.04] px-4 py-1 text-[11px] font-semibold text-primary tracking-wider uppercase">
            Core Features
          </span>
          <h2 className="mt-4 text-3xl font-bold text-foreground sm:text-4xl">
            全方位{' '}
            <span className="bg-gradient-to-r from-primary to-[oklch(0.72_0.12_230)] bg-clip-text text-transparent">
              可视化能力
            </span>
          </h2>
          <p className="mt-4 text-muted-foreground">从训练到部署，覆盖深度学习全生命周期的每一个关键环节</p>
        </motion.div>

        {/* 卡片网格 */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3"
        >
          {features.map((feature) => (
            <motion.a
              key={feature.title}
              href="#showcase"
              variants={cardVariants}
              whileHover={{
                y: -8,
                transition: { duration: 0.25 },
              }}
              className="group relative overflow-hidden rounded-2xl border border-white/[0.08] bg-[#080c14]/70 p-7 shadow-sm shadow-black/20 backdrop-blur-md transition-all duration-300 hover:border-primary/25 hover:bg-[#0a101a]/80"
            >
              {/* Hover 光效 */}
              <div className="absolute -right-6 -top-6 size-32 rounded-full bg-primary/[0.04] blur-2xl opacity-0 transition-opacity duration-500 group-hover:opacity-100" />

              {/* 标签 */}
              {feature.tag && (
                <span className="mb-4 inline-block rounded-md bg-primary/[0.06] px-2 py-0.5 text-[9px] font-bold text-primary tracking-wider uppercase">
                  {feature.tag}
                </span>
              )}

              {/* 图标 */}
              <div
                className={`mb-4 flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary transition-colors group-hover:bg-primary/14`}
              >
                {feature.icon}
              </div>

              {/* 文字内容 */}
              <h3 className="mb-2 text-base font-semibold text-foreground">{feature.title}</h3>
              <p className="text-sm leading-relaxed text-muted-foreground">{feature.description}</p>

              {/* 箭头指示（hover 时显示） */}
              <div className="mt-5 flex items-center gap-1.5 text-xs font-medium text-primary opacity-0 transition-all duration-300 group-hover:opacity-100">
                了解详情
                <svg className="size-3.5 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                </svg>
              </div>

              {/* 底部高亮线 */}
              <div className="absolute bottom-0 left-0 h-[2px] w-0 rounded-full bg-gradient-to-r from-primary to-transparent transition-all duration-400 group-hover:w-full" />
            </motion.a>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
