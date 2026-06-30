import { motion, AnimatePresence } from 'framer-motion';
import { OverviewShowcase } from './showcase/OverviewShowcase';
import { CNNShowcase } from './showcase/CNNShowcase';
import { RNNShowcase } from './showcase/RNNShowcase';
import { TransformerShowcase } from './showcase/TransformerShowcase';
import { BERTShowcase } from './showcase/BERTShowcase';
import { CustomModelShowcase } from './showcase/CustomModelShowcase';

/* ============================================
   DynamicShowcase — 模型展示主舞台容器
   使用 AnimatePresence 实现平滑切换
   每个 Tab 有独立的入场动画变体
   ============================================ */

/* 各 Tab 的入场动画变体 */
const tabVariants: Record<string, {
  initial: object;
  animate: object;
  exit: object;
}> = {
  overview: {
    initial: { opacity: 0, scale: 0.96 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.96 },
  },
  cnn: {
    initial: { opacity: 0, x: -60 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 60 },
  },
  rnn: {
    initial: { opacity: 0, y: 40 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -30 },
  },
  transformer: {
    initial: { opacity: 0, scale: 0.9, rotateY: -4 },
    animate: { opacity: 1, scale: 1, rotateY: 0 },
    exit: { opacity: 0, scale: 0.94, rotateY: 3 },
  },
  bert: {
    initial: { opacity: 0, x: 60 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -50 },
  },
  custom: {
    initial: { opacity: 0, y: 30, filter: 'blur(8px)' },
    animate: { opacity: 1, y: 0, filter: 'blur(0px)' },
    exit: { opacity: 0, filter: 'blur(6px)' },
  },
};

/* 面板注册表 */
const SHOWCASE_MAP: Record<string, React.ComponentType> = {
  overview: OverviewShowcase,
  cnn: CNNShowcase,
  rnn: RNNShowcase,
  transformer: TransformerShowcase,
  bert: BERTShowcase,
  custom: CustomModelShowcase,
};

interface DynamicShowcaseProps {
  activeTab: string;
}

export function DynamicShowcase({ activeTab }: DynamicShowcaseProps) {
  const PanelComponent = SHOWCASE_MAP[activeTab] ?? OverviewShowcase;
  const variants = tabVariants[activeTab] ?? tabVariants.overview;

  return (
    <section id="showcase" className="relative min-h-[70vh] rounded-b-3xl">
      {/* 内容切换区域 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          variants={variants}
          initial="initial"
          animate="animate"
          exit="exit"
          transition={{
            duration: activeTab === 'transformer' ? 0.55 : 0.45,
            ease: [0.22, 1, 0.36, 1],
          }}
        >
          <PanelComponent />
        </motion.div>
      </AnimatePresence>
    </section>
  );
}
