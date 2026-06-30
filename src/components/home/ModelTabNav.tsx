import { motion } from 'framer-motion';
import { useRef, useCallback } from 'react';

/* ============================================
   ModelTabNav — 高端产品栏目切换导航
   - 粘性定位在主展示区顶部
   - layoutId 滑动高亮指示器
   - hover 发光 + active 强调
   ============================================ */

export interface ModelTab {
  id: string;
  label: string;
  subtitle?: string;
}

export const MODEL_TABS: ModelTab[] = [
  { id: 'overview', label: '神经网络训练可视化', subtitle: 'AI 分析' },
  { id: 'cnn', label: 'CNN' },
  { id: 'rnn', label: 'RNN' },
  { id: 'transformer', label: 'Transformer' },
  { id: 'bert', label: 'BERT' },
  { id: 'custom', label: '自定义模型' },
];

interface ModelTabNavProps {
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export function ModelTabNav({ activeTab, onTabChange }: ModelTabNavProps) {
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 鼠标悬停时自动切换，带短暂延迟避免快速划过导致闪烁
  const handleMouseEnter = useCallback(
    (tabId: string) => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
      hoverTimeoutRef.current = setTimeout(() => {
        onTabChange(tabId);
      }, 80);
    },
    [onTabChange]
  );

  const handleMouseLeave = useCallback(() => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
  }, []);

  return (
    <div className="sticky top-0 z-30 border-b border-white/[0.05] bg-background/80 backdrop-blur-xl">
      <div className="mx-auto max-w-6xl px-6">
        <nav
          className="relative flex items-center gap-1 overflow-x-auto py-4 scrollbar-none"
          role="tablist"
          aria-label="模型类型选择"
        >
          {MODEL_TABS.map((tab) => {
            const isActive = activeTab === tab.id;

            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={isActive}
                onMouseEnter={() => handleMouseEnter(tab.id)}
                onMouseLeave={handleMouseLeave}
                className={`group relative shrink-0 cursor-default whitespace-nowrap rounded-lg px-5 py-2.5 text-sm font-medium transition-colors duration-200 ${
                  isActive ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {/* 文字内容 */}
                <span className="relative z-10 flex items-center gap-2">
                  <span>{tab.label}</span>
                  {tab.subtitle && (
                    <span
                      className={`hidden rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wider lg:inline-block ${
                        isActive
                          ? 'bg-primary/15 text-primary'
                          : 'bg-white/[0.04] text-muted-foreground'
                      }`}
                    >
                      {tab.subtitle}
                    </span>
                  )}
                </span>

                {/* Hover 光晕（非激活态） */}
                {!isActive && (
                  <motion.div
                    className="absolute inset-0 -z-0 rounded-lg opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                    style={{
                      background:
                        'radial-gradient(ellipse at center, oklch(0.68 0.16 155 / 0.08), transparent 70%)',
                    }}
                  />
                )}

                {/* Active 底部滑动高亮条 */}
                {isActive && (
                  <motion.div
                    layoutId="showcaseActiveIndicator"
                    className="absolute bottom-0 left-3 right-3 h-[2px] rounded-full bg-primary"
                    style={{
                      boxShadow:
                        '0 0 8px oklch(0.68 0.16 155 / 0.55), 0 0 18px oklch(0.68 0.16 155 / 0.2)',
                    }}
                    transition={{ type: 'spring', stiffness: 380, damping: 28 }}
                  />
                )}

                {/* Active 背景渐变 */}
                {isActive && (
                  <motion.div
                    layoutId="showcaseActiveGlow"
                    className="absolute inset-0 rounded-lg opacity-100"
                    style={{
                      background:
                        'linear-gradient(180deg, oklch(0.18 0.03 155 / 0.12) 0%, transparent 70%)',
                    }}
                    transition={{ type: 'spring', stiffness: 380, damping: 28 }}
                  />
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </div>
  );
}
