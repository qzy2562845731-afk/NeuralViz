import { useRef, useEffect, useState } from 'react';
import { motion, useInView } from 'framer-motion';

/* ============================================
   SectionWrapper — 双向动效区域包装器
   下滑进入：内容从下方/侧方淡入浮现
   上滑返回：内容有重新聚焦的微动效
   ============================================ */

interface SectionWrapperProps {
  id: string;
  children: React.ReactNode;
  className?: string;
  /** 入场动画方向: 'up'(从下往上) | 'left' | 'right' | 'fade'(纯淡入) */
  enterFrom?: 'up' | 'left' | 'right' | 'fade' | 'scale';
  /** 返回时的微动效类型 */
  returnEffect?: 'refocus' | 'slide-down' | 'none';
}

const enterVariants = {
  up: {
    hidden: { opacity: 0, y: 60 },
    visible: { opacity: 1, y: 0 },
    exitUp: { opacity: 0.4, y: -15 },
  },
  left: {
    hidden: { opacity: 0, x: -60 },
    visible: { opacity: 1, x: 0 },
    exitUp: { opacity: 0.4, x: 10 },
  },
  right: {
    hidden: { opacity: 0, x: 60 },
    visible: { opacity: 1, x: 0 },
    exitUp: { opacity: 0.4, x: -10 },
  },
  fade: {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
    exitUp: { opacity: 0.3 },
  },
  scale: {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { opacity: 1, scale: 1 },
    exitUp: { opacity: 0.5, scale: 0.98 },
  },
};

export function SectionWrapper({
  id,
  children,
  className = '',
  enterFrom = 'up',
  returnEffect = 'refocus',
}: SectionWrapperProps) {
  const ref = useRef<HTMLElement>(null);
  const isInView = useInView(ref, { once: false, margin: '-80px' });
  const [hasEntered, setHasEntered] = useState(false);
  const [isExiting, setIsExiting] = useState(false);

  // 进入视口时标记
  useEffect(() => {
    if (isInView && !hasEntered) {
      setHasEntered(true);
      setIsExiting(false);
    }
  }, [isInView, hasEntered]);

  // 离开视口向上滚动时触发返回效果
  useEffect(() => {
    if (!isInView && hasEntered && returnEffect !== 'none') {
      setIsExiting(true);
      // 短暂显示返回效果后重置
      const timer = setTimeout(() => setIsExiting(false), 400);
      return () => clearTimeout(timer);
    }
  }, [isInView, hasEntered, returnEffect]);

  const variant = enterVariants[enterFrom];

  return (
    <motion.section
      ref={ref}
      id={id}
      className={`relative ${className}`}
      initial="hidden"
      animate={isExiting ? 'exitUp' : isInView ? 'visible' : hasEntered ? 'exitUp' : 'hidden'}
      variants={variant}
      transition={{
        duration: isExiting ? 0.35 : 0.6,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      {children}
    </motion.section>
  );
}
