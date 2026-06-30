import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

/* ============================================
   FeatureEntryCards — 首页功能入口卡片化重构
   - 核心主入口：CNN 可视化工作台（大卡片）
   - 常用功能组：快速训练、历史实验、AI 分析设置
   - 帮助引导组：使用教程、常见问题
   - 最近使用快捷区域
   - 卡片 hover 交互：上浮 + 阴影加深
   ============================================ */

interface RecentItem {
  name: string;
  path: string;
  time: string;
}

const RECENT_STORAGE_KEY = 'nn-recent-entries';

function loadRecent(): RecentItem[] {
  try {
    const raw = localStorage.getItem(RECENT_STORAGE_KEY);
    if (raw) return JSON.parse(raw) as RecentItem[];
  } catch {
    // ignore
  }
  return [];
}

function saveRecent(items: RecentItem[]) {
  try {
    localStorage.setItem(RECENT_STORAGE_KEY, JSON.stringify(items));
  } catch {
    // ignore
  }
}

/** 记录访问入口到「最近使用」 */
export function recordRecentEntry(name: string, path: string) {
  const items = loadRecent().filter(i => i.path !== path);
  items.unshift({
    name,
    path,
    time: new Date().toISOString(),
  });
  saveRecent(items.slice(0, 4));
}

function formatRelativeTime(iso: string): string {
  const date = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return '刚刚';
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} 天前`;
  return date.toLocaleDateString('zh-CN');
}

export function FeatureEntryCards() {
  const [recent, setRecent] = useState<RecentItem[]>([]);

  useEffect(() => {
    setRecent(loadRecent());
  }, []);

  const handleClick = (name: string, path: string) => {
    recordRecentEntry(name, path);
    setRecent(loadRecent());
  };

  return (
    <section className="relative z-10 mx-auto max-w-6xl px-6 py-16">
      {/* 最近使用快捷区域 */}
      {recent.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-10"
        >
          <div className="mb-3 flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted-foreground">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">最近使用</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {recent.map((item, idx) => (
              <Link
                key={idx}
                to={item.path}
                onClick={() => handleClick(item.name, item.path)}
                className="group flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-[11px] text-foreground/70 transition-all hover:border-primary/25 hover:bg-primary/[0.04] hover:text-foreground"
              >
                <span className="font-medium">{item.name}</span>
                <span className="text-[10px] text-muted-foreground/60">{formatRelativeTime(item.time)}</span>
              </Link>
            ))}
          </div>
        </motion.div>
      )}

      {/* 核心主入口 - 大卡片 */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="mb-6"
      >
        <Link
          to="/workbench"
          onClick={() => handleClick('CNN 可视化工作台', '/workbench')}
          className="group relative block overflow-hidden rounded-2xl border border-primary/20 bg-gradient-to-br from-primary/[0.08] via-[#0c0e17] to-[#0c0e17] p-8 transition-all duration-300 hover:-translate-y-1 hover:border-primary/40 hover:shadow-2xl hover:shadow-primary/10"
        >
          <div className="flex items-center gap-6">
            {/* 图标 */}
            <div className="flex h-20 w-20 flex-shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-primary/5 ring-1 ring-primary/20">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-primary">
                <rect x="3" y="3" width="7" height="7" rx="1" />
                <rect x="14" y="3" width="7" height="7" rx="1" />
                <rect x="3" y="14" width="7" height="7" rx="1" />
                <rect x="14" y="14" width="7" height="7" rx="1" />
                <path d="M10 6.5h4M10 17.5h4M6.5 10v4M17.5 10v4" />
              </svg>
            </div>
            {/* 文案 */}
            <div className="flex-1">
              <div className="mb-1 flex items-center gap-2">
                <h3 className="text-xl font-bold text-foreground">CNN 可视化工作台</h3>
                <span className="rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">核心</span>
              </div>
              <p className="text-sm text-muted-foreground">3D 可视化神经网络结构，实时观察训练过程，AI 辅助诊断模型问题</p>
              <div className="mt-3 flex items-center gap-1 text-[12px] font-semibold text-primary opacity-80 transition-opacity group-hover:opacity-100">
                开始使用
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="transition-transform group-hover:translate-x-1">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          </div>
          {/* 装饰光效 */}
          <div className="pointer-events-none absolute -right-20 -top-20 h-40 w-40 rounded-full bg-primary/10 blur-3xl transition-opacity duration-500 group-hover:opacity-150" />
        </Link>
      </motion.div>

      {/* 常用功能组 */}
      <div className="mb-6">
        <div className="mb-3 flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted-foreground">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M9 9h6v6H9z" />
          </svg>
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">常用功能</span>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {[
            {
              to: '/workbench/training-log',
              name: '训练日志',
              desc: '查看训练进度、Loss/Acc 曲线、关键节点高亮',
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="8" y1="13" x2="16" y2="13" />
                  <line x1="8" y1="17" x2="16" y2="17" />
                </svg>
              ),
              color: 'text-emerald-400',
              bg: 'from-emerald-400/10',
            },
            {
              to: '/experiments',
              name: '我的历史实验',
              desc: '管理历史训练记录，对比不同实验效果',
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M3 3v18h18" />
                  <path d="M7 16l4-6 4 3 5-8" />
                </svg>
              ),
              color: 'text-violet-400',
              bg: 'from-violet-400/10',
            },
            {
              to: '/workbench/ai-settings',
              name: 'AI 分析设置',
              desc: '配置 DeepSeek、智谱等主流大模型 API',
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
              ),
              color: 'text-amber-400',
              bg: 'from-amber-400/10',
            },
          ].map((card, idx) => (
            <motion.div
              key={card.to}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: idx * 0.08 }}
            >
              <Link
                to={card.to}
                onClick={() => handleClick(card.name, card.to)}
                className={`group relative block h-full overflow-hidden rounded-xl border border-white/[0.06] bg-gradient-to-br ${card.bg} via-[#0c0e17] to-[#0c0e17] p-5 transition-all duration-300 hover:-translate-y-1 hover:border-white/[0.12] hover:shadow-xl hover:shadow-black/30`}
              >
                <div className={`mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-white/[0.03] ${card.color}`}>
                  {card.icon}
                </div>
                <h4 className="mb-1 text-sm font-bold text-foreground">{card.name}</h4>
                <p className="text-[11px] leading-relaxed text-muted-foreground">{card.desc}</p>
                <div className={`mt-3 flex items-center gap-1 text-[11px] font-medium ${card.color} opacity-0 transition-opacity group-hover:opacity-100`}>
                  进入
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="transition-transform group-hover:translate-x-0.5">
                    <path d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>

      {/* 帮助引导组 */}
      <div>
        <div className="mb-3 flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted-foreground">
            <circle cx="12" cy="12" r="10" />
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">帮助引导</span>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {[
            {
              name: '使用教程',
              desc: '从模型导入到训练可视化的完整入门指南',
              icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                  <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
                </svg>
              ),
              href: '#showcase-section',
            },
            {
              name: '常见问题',
              desc: 'API 配置、模型格式、训练异常等问题解答',
              icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                  <circle cx="12" cy="12" r="10" />
                </svg>
              ),
              href: '#showcase-section',
            },
          ].map((card, idx) => (
            <motion.div
              key={card.name}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: idx * 0.08 }}
            >
              <a
                href={card.href}
                className="group flex items-center gap-3 rounded-xl border border-white/[0.04] bg-white/[0.01] p-4 transition-all duration-300 hover:-translate-y-0.5 hover:border-white/[0.1] hover:bg-white/[0.03]"
              >
                <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-white/[0.03] text-muted-foreground transition-colors group-hover:text-foreground">
                  {card.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="text-[13px] font-semibold text-foreground/90">{card.name}</h4>
                  <p className="text-[10px] text-muted-foreground">{card.desc}</p>
                </div>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted-foreground/40 transition-all group-hover:translate-x-0.5 group-hover:text-foreground/80">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </a>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
