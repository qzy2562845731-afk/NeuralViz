import { motion } from 'framer-motion';

/* ============================================
   BERTShowcase — NLP / BERT 专题展示
   左侧：标题+描述+能力点+CTA
   右侧：Token 序列 + 语义高亮 + 层级编码
   动画：文本块亮起、语义高亮、层级展开
   ============================================ */

const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.06, delayChildren: 0.12 } },
};

const item = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
};

const tokens = [
  { text: '[CLS]', isSpecial: true },
  { text: '神经', highlight: true },
  { text: '网络' },
  { text: '训练', highlight: true },
  { text: '是' },
  { text: '非常' },
  { text: '有趣' },
  { text: '的', highlight: true },
  { text: '[SEP]', isSpecial: true },
];

export function BERTShowcase() {
  return (
    <div className="mx-auto grid max-w-6xl gap-12 px-6 py-20 lg:grid-cols-2 lg:items-center">
      {/* ===== 左侧 ===== */}
      <motion.div variants={stagger} initial="hidden" animate="visible" className="space-y-5 order-2 lg:order-1">
        <motion.div variants={item}>
          <span className="inline-block rounded-full border border-orange-500/15 bg-orange-500/[0.05] px-3 py-1 text-[11px] font-bold tracking-wider text-orange-400 uppercase">
            BERT
          </span>
        </motion.div>

        <motion.h2 variants={item} className="text-3xl font-bold leading-tight text-foreground sm:text-4xl">
          自然语言理解{' '}
          <span className="text-orange-400">深度剖析</span>
        </motion.h2>

        <motion.p variants={item} className="text-base leading-relaxed text-muted-foreground">
          双向上下文编码的可视化。词嵌入空间分布、Token 级注意力权重、掩码语言模型预测概率、层间表征漂移。
        </motion.p>

        <motion.ul variants={stagger} className="space-y-2.5" initial="hidden" animate="visible">
          {[
            { label: '词向量空间', desc: 't-SNE 降维后的词嵌入聚类，语义相近自然聚集' },
            { label: 'Token Attention', desc: '点击任意 Token 查看与其他位置的关注度分布' },
            { label: 'MLM 预测', desc: '掩码位置候选词 Top-K 概率排序与置信度' },
          ].map((f) => (
            <motion.li key={f.label} variants={item} className="flex items-start gap-2.5 group">
              <span className="mt-1 size-1.5 shrink-0 rounded-full bg-orange-400 transition-transform group-hover:scale-125" />
              <div>
                <div className="text-sm font-semibold text-foreground">{f.label}</div>
                <div className="mt-0.5 text-xs text-muted-foreground">{f.desc}</div>
              </div>
            </motion.li>
          ))}
        </motion.ul>

        <motion.div variants={item}>
          <a href="/workbench?model=bert" className="group inline-flex items-center gap-2 rounded-xl bg-orange-500 px-7 py-3.5 text-sm font-semibold text-black shadow-lg shadow-orange-500/20 transition-all hover:shadow-xl hover:shadow-orange-500/30 active:scale-[0.98]">
            进入 BERT 工作台
            <svg className="size-4 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
          </a>
        </motion.div>
      </motion.div>

      {/* ===== 右侧：Token 可视化 + 分析面板 ===== */}
      <motion.div
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.55, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
        className="order-1 lg:order-2 space-y-5"
      >
        {/* Token 序列 */}
        <div className="overflow-hidden rounded-2xl border border-orange-500/12 bg-gradient-to-b from-orange-500/[0.05] to-[#080c14]/50 p-6 shadow-lg shadow-black/20 backdrop-blur-md">
          <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Token-Level Visualization</h3>

          <div className="flex flex-wrap items-center gap-2">
            {tokens.map((token, i) => (
              <motion.span
                key={i}
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{
                  delay: 0.2 + i * 0.06,
                  duration: 0.35,
                  ease: [0.22, 1, 0.36, 1],
                }}
                whileHover={{
                  scale: 1.08,
                  y: -3,
                  transition: { duration: 0.18 },
                }}
                className={`cursor-pointer rounded-lg px-3.5 py-2 text-sm font-medium transition-all shadow-sm shadow-black/10 ${
                  token.isSpecial
                    ? 'border-orange-500/30 bg-orange-500/[0.10] text-orange-400 hover:bg-orange-500/[0.14]'
                    : token.highlight
                      ? 'border-white/[0.10] bg-[#080c14]/60 text-foreground hover:border-orange-500/30 hover:bg-orange-500/[0.08]'
                      : 'border-white/[0.07] bg-[#080c14]/50 text-muted-foreground hover:border-white/12 hover:text-foreground hover:bg-[#080c14]/70'
                }`}
              >
                {token.text}
              </motion.span>
            ))}
          </div>

          <p className="mt-4 text-[11px] text-muted-foreground">
            点击任意 Token 查看其与其他位置的注意力权重分布
          </p>
        </div>

        {/* 能力网格 */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {[
            { title: '词向量空间', desc: 't-SNE 聚类分布' },
            { title: '层间漂移', desc: '表征空间渐进变化' },
            { title: 'Attention Head', desc: '12 头语言学模式' },
            { title: 'NSP 任务', desc: '句子对关联强度' },
            { title: 'MLM 概率', desc: 'Top-K 预测置信度' },
            { title: 'Fine-tune 对比', desc: '预训练 vs 微调差异' },
          ].map((card, i) => (
            <motion.div
              key={card.title}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.55 + i * 0.07, duration: 0.4 }}
              whileHover={{ y: -3, borderColor: 'rgba(249,115,22,0.15)' }}
              className="rounded-xl border border-white/[0.07] bg-[#080c14]/60 p-3.5 shadow-sm shadow-black/20 transition-all hover:border-orange-500/20 hover:bg-[#0a0f1c]/70"
            >
              <h4 className="text-[11px] font-semibold text-foreground">{card.title}</h4>
              <p className="mt-1 text-[10px] text-muted-foreground leading-relaxed">{card.desc}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
