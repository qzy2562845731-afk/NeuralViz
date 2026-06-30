import { motion } from 'framer-motion';

/* ============================================
   WorkflowSection — 使用流程区（增强版）
   横向时间线 + 依次延迟入场
   连接线动画 + 步骤编号角标
   ============================================ */

const steps = [
  {
    number: '01',
    title: '选择模型类型',
    description: '从 CNN、RNN、Transformer、BERT 或自定义模型中选择目标架构',
    icon: (
      <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 18.75a60.07 60.07 0 0115.797 2.101c.727.198 1.453-.342 1.453-1.096V18.75M3.75 4.5v.75A.75.75 0 013 6h-.75m0 0v-.375c0-.621.504-1.125 1.125-1.125H20.25M2.25 6v9m18-10.5v.75c0 .414.336.75.75.75h.75m-1.5-1.5h.375c.621 0 1.125.504 1.125 1.125v9.75c0 .621-.504 1.125-1.125 1.125h-.375m1.5-1.5H21a.75.75 0 00-.75.75v.75m0 0H3.75m0 0h-.375a1.125 1.125 0 01-1.125-1.125V15m1.5 1.5v-.75A.75.75 0 003 15h-.75M15 10.5a3 3 0 11-6 0 3 3 0 016 0zm3 0h.008v.008H18V10.5zm-12 0h.008v.008H6V10.5z" />
      </svg>
    ),
  },
  {
    number: '02',
    title: '上传训练数据',
    description: '导入训练日志、权重快照或实时连接训练进程获取数据流',
    icon: (
      <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
    ),
  },
  {
    number: '03',
    title: '启动可视化引擎',
    description: '系统自动解析模型结构，生成 3D 架构图、训练曲线和诊断面板',
    icon: (
      <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
      </svg>
    ),
  },
  {
    number: '04',
    title: '探索与分析',
    description: '在交互式工作台中逐层检查参数、查看 AI 诊断结论、导出分析报告',
    icon: (
      <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
      </svg>
    ),
  },
];

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.16, delayChildren: 0.2 } },
};

const stepVariants = {
  hidden: { opacity: 0, y: 28, scale: 0.96 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] },
  },
};

export function WorkflowSection() {
  return (
    <section id="workflow" className="relative overflow-hidden py-28">
      {/* 背景：与视频融合的底部光晕 */}
      <div className="pointer-events-none absolute inset-0">
        <div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(ellipse at 80% 100%, oklch(0.72 0.12 230 / 0.06) 0%, transparent 55%)',
          }}
        />
      </div>

      <div className="relative z-10 mx-auto max-w-6xl px-6">
        {/* 标题 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.55 }}
          className="mb-16 text-center"
        >
          <span className="mb-3 inline-block rounded-full border border-primary/12 bg-primary/[0.04] px-4 py-1 text-[11px] font-semibold text-primary tracking-wider uppercase">
            Workflow
          </span>
          <h2 className="mt-4 text-3xl font-bold text-foreground sm:text-4xl">
            四步开启{' '}
            <span className="bg-gradient-to-r from-primary to-[oklch(0.70_0.14_180)] bg-clip-text text-transparent">
              可视化之旅
            </span>
          </h2>
          <p className="mt-4 text-muted-foreground">从选择模型到深度分析，极简流程</p>
        </motion.div>

        {/* 时间线步骤 */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-80px' }}
          className="relative"
        >
          {/* 横向连接线（桌面端） */}
          <div className="absolute top-14 left-[calc(8%+36px)] right-[calc(8%+36px)] hidden h-[2px] bg-gradient-to-r from-white/[0.06] via-white/[0.08] to-white/[0.04] md:block">
            <motion.div
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1.2, delay: 0.45, ease: 'easeOut' }}
              className="h-full origin-left bg-gradient-to-r from-primary/40 to-primary/12"
            />
          </div>

          <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
            {steps.map((step) => (
              <motion.div
                key={step.number}
                variants={stepVariants}
                whileHover={{ y: -6, transition: { duration: 0.22 } }}
                className="group relative flex flex-col items-center text-center"
              >
                {/* 圆形步骤节点 */}
                <div className="relative mb-5 flex size-16 items-center justify-center rounded-2xl border border-white/[0.09] bg-[#080c14]/80 shadow-lg shadow-black/20 backdrop-blur-md transition-all duration-300 group-hover:border-primary/30 group-hover:shadow-primary/10">
                  <div className="text-primary">{step.icon}</div>

                  {/* 编号角标 */}
                  <span className="absolute -right-1.5 -top-1.5 flex size-6 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-primary-foreground tabular-nums shadow-sm shadow-primary/30">
                    {step.number}
                  </span>

                  {/* hover 内发光 */}
                  <div className="absolute inset-0 rounded-2xl opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                    style={{ boxShadow: 'inset 0 0 20px oklch(0.68 0.16 155 / 0.06)' }}
                  />
                </div>

                {/* 文字内容 */}
                <h3 className="mb-1.5 text-base font-semibold text-foreground">{step.title}</h3>
                <p className="max-w-[200px] text-sm leading-relaxed text-muted-foreground">
                  {step.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
